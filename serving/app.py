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
    SERVING_MODELS     activity=registered-model pairs, default "fight=fight-detector"
                       (a future activity model is one more pair, e.g. "fight=fight-detector,theft=theft-detector")
    MODEL_SOURCE       registry (default) | local
    MLFLOW_TRACKING_URI, MODEL_ALIAS (champion), MODEL_DIR (local source: <activity>_model.onnx + .json)
    RELOAD_SECONDS     how often to look for a new champion (default 60, 0 = never)
    SERVING_ADMIN_TOKEN  if set, /admin/reload needs "Authorization: Bearer <token>"
"""
import base64
import hmac
import logging
import os
import time
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


def parse_models(text):
    pairs = [p.split("=", 1) for p in text.split(",") if p.strip()]
    return {a.strip(): n.strip() for a, n in pairs}


def make_store():
    return ModelStore(parse_models(os.environ.get("SERVING_MODELS", "fight=fight-detector")),
                      source=os.environ.get("MODEL_SOURCE", "registry"),
                      registry_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                      alias=os.environ.get("MODEL_ALIAS", "champion"),
                      model_dir=os.environ.get("MODEL_DIR", "models"),
                      cache_dir=os.environ.get("MODEL_CACHE", "/tmp/serving-cache"),
                      threads=int(os.environ.get("ORT_THREADS", "2")))


_published = {}                         # activity -> labels currently exported in serving_model_info


def publish_model_info(store, changed):
    for activity in changed:
        m = store.get(activity)
        if activity in _published:
            MODEL_INFO.remove(*_published[activity])
        _published[activity] = (activity, str(m.version), m.meta["arch"].get("type", "gru"))
        MODEL_INFO.labels(*_published[activity]).set(1)


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
                "reloads": store.reloads}

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
        return {"changed": changed, "models": [m.info() for m in store.loaded.values()], "errors": store.errors}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()          # nothing is loaded until the server starts (lifespan)
