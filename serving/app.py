"""Model serving API (FastAPI + ONNX Runtime): one endpoint per activity model, always the registry's @champion.

    uvicorn serving.app:app --port 8000          # or: docker compose up -d serving

    GET  /health                          the process is up (liveness)
    GET  /ready                           every configured model is loaded (readiness)
    GET  /v1/models                       loaded models: registry version, architecture, threshold, test scores
    GET  /v1/models/{activity}            one model's settings (the client reads window, threshold, pairing rules)
    POST /v1/models/{activity}/predict    {"instances": [window, ...]}, window = T x F features, or the compact form
                                          {"shape": [batch, T, F], "data_b64": <float32 bytes, base64>}
                                          -> logits, probabilities, above_threshold (per window)
    POST /admin/reload                    re-check the registry now (otherwise every RELOAD_SECONDS)
    GET  /metrics                         Prometheus metrics (requests, latency, scores, model version)

Settings (environment variables, see .env.example):
    SERVING_MODELS     activity=registered-model pairs; empty = every model activity of activities.yaml
                       (a new model activity is one more entry there, or one more pair here)
    MODEL_SOURCE       registry (default) | local
    MLFLOW_TRACKING_URI, MODEL_ALIAS (champion), MODEL_DIR (local source: <activity>_model.onnx + .json)
    RELOAD_SECONDS     how often to look for a new champion (default 60, 0 = never)
    SERVING_ADMIN_TOKEN  if set, /admin/reload needs "Authorization: Bearer <token>"
    MODEL_FALLBACK_DIR the release model in the image (models/release), served while the registry has no champion
    SHADOW_ALIAS       registry alias scored in the shadow of the champion (default "challenger", "" = off)
"""
import base64
import hmac
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from serving.store import ModelStore, sigmoid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("serving")


class _QuietProbes(logging.Filter):
    """Health checks and Prometheus scrapes every few seconds would bury the useful access log lines."""
    def filter(self, record):
        return not any(f'"GET {p} ' in record.getMessage() for p in ("/ready", "/health", "/metrics"))


logging.getLogger("uvicorn.access").addFilter(_QuietProbes())

# ---------------------------------------------------------------------------- metrics (scraped by Prometheus)
REQUESTS = Counter("serving_requests_total", "Prediction requests", ["activity", "status"])
LATENCY = Histogram("serving_request_seconds", "Prediction latency (validation + inference)", ["activity"],
                    buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1))
WINDOWS = Counter("serving_windows_total", "Windows scored", ["activity"])
SCORES = Histogram("serving_probability", "Predicted probability per window (prediction drift)", ["activity"],
                   buckets=[i / 10 for i in range(1, 11)])
ABOVE = Counter("serving_above_threshold_total", "Windows scored above the model's threshold", ["activity"])
MODEL_INFO = Gauge("serving_model_info", "Model currently served (value 1)", ["activity", "version", "arch"])
RELOADS = Counter("serving_model_reloads_total", "Hot reloads after the registry alias moved", ["activity"])
# shadow model (the challenger): scored on the same windows after the response, never returned
SHADOW_INFO = Gauge("serving_shadow_model_info", "Model scored in shadow (value 1)", ["activity", "version", "arch"])
SHADOW_WINDOWS = Counter("serving_shadow_windows_total", "Windows also scored by the shadow model", ["activity"])
SHADOW_AGREE = Counter("serving_shadow_agreement_total", "Shadow vs served decision per window (each model with "
                       "its own threshold)", ["activity", "agree"])
SHADOW_ABOVE = Counter("serving_shadow_above_threshold_total", "Windows above the shadow model's threshold",
                       ["activity"])
SHADOW_DIFF = Histogram("serving_shadow_abs_diff", "|shadow probability - served probability| per window",
                        ["activity"], buckets=(.01, .02, .05, .1, .2, .3, .5, 1))
SHADOW_DROPPED = Counter("serving_shadow_dropped_total", "Windows not shadow-scored because it was busy", ["activity"])
SHADOW_MAX_PENDING = 8                  # shadow scoring never delays or queues up behind real requests


def parse_models(text):
    pairs = [p.split("=", 1) for p in text.split(",") if p.strip()]
    return {a.strip(): n.strip() for a, n in pairs}


def default_models():
    """Every model activity of activities.yaml (activity -> registered model); fight only if the file is missing."""
    try:
        from rover_ai import activities
        return activities.models() or {"fight": "fight-detector"}
    except (ImportError, OSError):
        return {"fight": "fight-detector"}


def make_store():
    text = os.environ.get("SERVING_MODELS", "").strip()
    return ModelStore(parse_models(text) if text else default_models(),
                      source=os.environ.get("MODEL_SOURCE", "registry"),
                      registry_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                      alias=os.environ.get("MODEL_ALIAS", "champion"),
                      model_dir=os.environ.get("MODEL_DIR", "models"),
                      cache_dir=os.environ.get("MODEL_CACHE", "/tmp/serving-cache"),
                      threads=int(os.environ.get("ORT_THREADS", "2")),
                      fallback_dir=os.environ.get("MODEL_FALLBACK_DIR") or None,
                      shadow_alias=os.environ.get("SHADOW_ALIAS", "challenger"))


_published = {}                         # activity -> labels currently exported in serving_model_info


def publish_model_info(store, changed):
    for activity in changed:
        m = store.get(activity)
        if activity in _published:
            MODEL_INFO.remove(*_published[activity])
        _published[activity] = (activity, str(m.version), m.meta["arch"].get("type", "gru"))
        MODEL_INFO.labels(*_published[activity]).set(1)


def publish_shadow_info(store):
    SHADOW_INFO.clear()
    for activity, m in store.shadow.items():
        SHADOW_INFO.labels(activity, str(m.version), m.meta["arch"].get("type", "gru")).set(1)


class ShadowScorer:
    """Scores windows with the shadow model in one background thread, after the champion's response."""

    def __init__(self):
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="shadow")
        self.pending = 0
        self.lock = threading.Lock()

    def submit(self, activity, shadow, x, probs, threshold):
        with self.lock:
            if self.pending >= SHADOW_MAX_PENDING:
                SHADOW_DROPPED.labels(activity).inc(len(x))
                return
            self.pending += 1
        self.pool.submit(self._score, activity, shadow, x, probs, threshold)

    def _score(self, activity, shadow, x, probs, threshold):
        try:
            ps = sigmoid(shadow.predict(x))
            above = ps >= float(shadow.meta["threshold"])
            agree = above == (probs >= threshold)
            SHADOW_WINDOWS.labels(activity).inc(len(x))
            SHADOW_AGREE.labels(activity, "true").inc(int(agree.sum()))
            SHADOW_AGREE.labels(activity, "false").inc(int((~agree).sum()))
            SHADOW_ABOVE.labels(activity).inc(int(above.sum()))
            for d in np.abs(ps - probs):
                SHADOW_DIFF.labels(activity).observe(float(d))
        except Exception as e:                  # noqa: BLE001 - the shadow never affects serving
            log.warning("shadow scoring failed: %s: %s", type(e).__name__, e)
        finally:
            with self.lock:
                self.pending -= 1

    def flush(self):
        self.pool.submit(lambda: None).result()      # one worker: returns once everything queued before is done


def create_app(store=None, reload_seconds=None):
    store = store or make_store()
    reload_seconds = float(os.environ.get("RELOAD_SECONDS", "60")) if reload_seconds is None else reload_seconds
    max_batch = int(os.environ.get("MAX_BATCH", "256"))

    def on_change(changed):
        publish_model_info(store, changed)
        for a in changed:
            RELOADS.labels(a).inc()

    @asynccontextmanager
    async def lifespan(_):
        publish_model_info(store, store.refresh())
        if reload_seconds > 0:
            store.watch(reload_seconds, on_change)       # also retries models that failed to load at start
        yield
        store.stop()

    app = FastAPI(title="Rover model serving", version="1.0", lifespan=lifespan)
    app.state.store = store
    app.state.shadow = shadow = ShadowScorer()

    class PredictRequest(BaseModel):
        instances: list[list[list[float]]] | None = Field(None, description="windows, each T x F features (JSON)")
        shape: list[int] | None = Field(None, description="with data_b64: [batch, T, F]")
        data_b64: str | None = Field(None, description="the same windows as base64 little-endian float32 (compact)")

    def decode(req):
        if req.instances is not None:
            return np.asarray(req.instances, np.float32)
        if req.data_b64 is None or req.shape is None or len(req.shape) != 3:
            raise HTTPException(422, "send either instances, or shape [batch, T, F] + data_b64")
        try:
            raw = base64.b64decode(req.data_b64, validate=True)
        except ValueError:
            raise HTTPException(422, "data_b64 is not valid base64") from None
        if len(raw) != 4 * int(np.prod(req.shape)):
            raise HTTPException(422, f"data_b64 has {len(raw)} bytes, shape {req.shape} needs {4 * int(np.prod(req.shape))}")
        return np.frombuffer(raw, "<f4").reshape(req.shape)

    def model_or_error(activity):
        if activity not in store.models:
            raise HTTPException(404, f"unknown activity {activity!r}; served: {sorted(store.models)}")
        m = store.get(activity)
        if m is None:
            raise HTTPException(503, f"{activity} model not loaded yet: {store.errors.get(activity, 'loading')}")
        return m

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response):
        ok = store.ready()
        if not ok:
            response.status_code = 503
        return {"ready": ok, "loaded": sorted(store.loaded), "errors": store.errors}

    @app.get("/v1/models")
    def models():
        return {"models": [m.info() for m in store.loaded.values()], "errors": store.errors,
                "reloads": store.reloads, "shadow": [m.info() for m in store.shadow.values()]}

    @app.get("/v1/models/{activity}")
    def model(activity: str):
        return model_or_error(activity).info()

    @app.post("/v1/models/{activity}/predict")
    def predict(activity: str, req: PredictRequest):
        t0 = time.perf_counter()
        try:
            m = model_or_error(activity)              # the model object is fixed for this request (hot reload safe)
            x = decode(req)
            if x.ndim != 3 or x.shape[1:] != (m.window, m.n_feat):
                raise HTTPException(422, f"instances must have shape (batch, {m.window}, {m.n_feat}), got {x.shape}")
            if not 1 <= len(x) <= max_batch:
                raise HTTPException(422, f"batch size must be 1..{max_batch}")
            if not np.isfinite(x).all():
                raise HTTPException(422, "instances contain NaN or infinity")
            logits = m.predict(x)
            probs = sigmoid(logits)
        except HTTPException as e:
            REQUESTS.labels(activity, str(e.status_code)).inc()
            raise
        thr = float(m.meta["threshold"])
        REQUESTS.labels(activity, "200").inc()
        LATENCY.labels(activity).observe(time.perf_counter() - t0)
        WINDOWS.labels(activity).inc(len(x))
        ABOVE.labels(activity).inc(int((probs >= thr).sum()))
        for p in probs:
            SCORES.labels(activity).observe(float(p))
        s = store.shadow.get(activity)
        if s is not None and (s.window, s.n_feat) == (m.window, m.n_feat):
            shadow.submit(activity, s, x, probs, thr)
        return {"activity": activity, "model_version": m.version, "threshold": thr,
                "logits": logits.round(5).tolist(), "probabilities": probs.round(5).tolist(),
                "above_threshold": (probs >= thr).tolist()}

    @app.post("/admin/reload")
    def reload(authorization: str = Header(default="")):
        token = os.environ.get("SERVING_ADMIN_TOKEN")
        if token and not hmac.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(401, "invalid admin token")
        changed = store.refresh()
        on_change(changed)
        return {"changed": changed, "models": [m.info() for m in store.loaded.values()], "errors": store.errors,
                "shadow": [m.info() for m in store.shadow.values()]}

    @app.get("/metrics")
    def metrics():
        publish_shadow_info(store)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()          # nothing is loaded until the server starts (lifespan)
