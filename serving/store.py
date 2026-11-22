"""Model store of the serving API: one ONNX model per activity, loaded from the MLflow registry (or a folder),
swapped in place when the registry's alias moves (promotion or rollback), without dropping requests.

Sources
    registry   the version with alias @champion of the registered model, downloaded over MLflow's REST API
               (no mlflow package needed: the image stays small)
    local      <MODEL_DIR>/<activity>_model.onnx + .json (offline use, or a cloud VM without MLflow)

A model version is downloaded once into a cache folder; a reload only swaps a reference, so in-flight requests
finish on the model they started with.
"""
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("serving")
ARTIFACT_DIR = "fight_model"            # run artifact folder written by training (model.onnx, model.json)


@dataclass
class LoadedModel:
    activity: str
    session: object                     # onnxruntime.InferenceSession
    input_name: str
    meta: dict
    source: str                         # "registry" | "local"
    version: str | None = None          # registry version
    run_id: str | None = None
    loaded_at: float = field(default_factory=time.time)

    @property
    def window(self):
        return int(self.meta["window"])

    @property
    def n_feat(self):
        return int(self.session.get_inputs()[0].shape[2])     # ONNX input (batch, T, F)

    def info(self):
        m = self.meta
        return dict(activity=self.activity, source=self.source, version=self.version, run_id=self.run_id,
                    registered_model=m.get("registry", {}).get("model"), arch=m.get("arch", {}).get("type", "gru"),
                    window=self.window, n_feat=self.n_feat, threshold=m.get("threshold"),
                    consecutive=m.get("consecutive"), proximity=m.get("proximity"), min_cover=m.get("min_cover"),
                    test_f1=(m.get("test_learned") or {}).get("f1"),
                    test_roc_auc=(m.get("test_learned") or {}).get("roc_auc"),
                    loaded_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.loaded_at)))

    def predict(self, x):
        """x (B, T, F) float32 -> logits (B,)"""
        return self.session.run(None, {self.input_name: x})[0].reshape(-1)


def _session(path, threads):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class MlflowRegistry:
    """The few MLflow REST calls serving needs."""

    def __init__(self, uri, timeout=30):
        self.uri = uri.rstrip("/")
        self.timeout = timeout

    def _get(self, endpoint, query):
        url = f"{self.uri}{endpoint}?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=self.timeout) as r:      # noqa: S310 - URL from our own settings
            return r.read()

    def version_by_alias(self, name, alias):
        """-> (version, run_id), or None if no version has this alias."""
        try:
            mv = json.loads(self._get("/api/2.0/mlflow/registered-models/alias", {"name": name, "alias": alias}))["model_version"]
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            raise
        return str(mv["version"]), mv["run_id"]

    def download(self, run_id, artifact, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(self._get("/get-artifact", {"path": artifact, "run_uuid": run_id}))
        tmp.replace(dest)


class ModelStore:
    def __init__(self, models, source="registry", registry_uri=None, alias="champion", model_dir="models",
                 cache_dir="/tmp/serving-cache", threads=2):
        self.models = dict(models)                  # activity -> registered model name
        self.source = source
        self.alias = alias
        self.registry = MlflowRegistry(registry_uri) if registry_uri else None
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.threads = threads
        self.loaded = {}                            # activity -> LoadedModel
        self.errors = {}                            # activity -> last load error
        self.reloads = 0
        self._lock = threading.Lock()               # one reload at a time; readers never wait
        self._stop = threading.Event()

    # ------------------------------------------------------------------ loading
    def _from_registry(self, activity, name):
        found = self.registry.version_by_alias(name, self.alias)
        if found is None:
            raise LookupError(f"{name}: no version has the alias @{self.alias}")
        version, run_id = found
        current = self.loaded.get(activity)
        if current and current.source == "registry" and current.version == version:
            return None                             # unchanged
        folder = self.cache_dir / name / f"v{version}"
        for f in ("model.onnx", "model.json"):
            if not (folder / f).exists():
                self.registry.download(run_id, f"{ARTIFACT_DIR}/{f}", folder / f)
        meta = json.loads((folder / "model.json").read_text())
        meta["registry"] = dict(model=name, version=int(version), alias=self.alias, run_id=run_id)
        session = _session(folder / "model.onnx", self.threads)
        return LoadedModel(activity, session, session.get_inputs()[0].name, meta, "registry", version, run_id)

    def _from_local(self, activity):
        path = self.model_dir / f"{activity}_model.onnx"
        meta = json.loads(path.with_suffix(".json").read_text())
        current = self.loaded.get(activity)
        stamp = path.stat().st_mtime
        if current and current.source == "local" and current.meta.get("_mtime") == stamp:
            return None
        meta["_mtime"] = stamp
        session = _session(path, self.threads)
        reg = meta.get("registry", {})
        return LoadedModel(activity, session, session.get_inputs()[0].name, meta, "local",
                           str(reg["version"]) if reg else None, reg.get("run_id"))

    def refresh(self):
        """Load every model that is missing or whose champion changed. Returns the activities that changed."""
        changed = []
        with self._lock:
            for activity, name in self.models.items():
                try:
                    new = (self._from_registry(activity, name) if self.source == "registry"
                           else self._from_local(activity))
                except Exception as e:                      # keep serving the model we have
                    self.errors[activity] = f"{type(e).__name__}: {e}"
                    log.warning("could not load %s: %s", activity, self.errors[activity])
                    continue
                self.errors.pop(activity, None)
                if new is None:
                    continue
                old = self.loaded.get(activity)
                self.loaded[activity] = new                  # atomic swap
                if old is not None:
                    self.reloads += 1
                changed.append(activity)
                log.info("%s: serving %s v%s (%s)%s", activity, name, new.version, new.meta["arch"].get("type"),
                         f", was v{old.version}" if old else "")
        return changed

    def ready(self):
        return all(a in self.loaded for a in self.models)

    def get(self, activity):
        return self.loaded.get(activity)

    # ------------------------------------------------------------------ background reload
    def watch(self, every, on_change=None):
        """Re-check the registry every `every` seconds in a daemon thread (promotion / rollback -> hot reload)."""
        def loop():
            while not self._stop.wait(every):
                changed = self.refresh()
                if changed and on_change:
                    on_change(changed)
        t = threading.Thread(target=loop, name="model-watch", daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop.set()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
