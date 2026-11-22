"""Model store of the serving API: one ONNX model per activity, loaded from the MLflow registry (or a folder),
swapped in place when the registry's alias moves (promotion or rollback), without dropping requests.

Sources
    registry   the version with alias @champion of the registered model, downloaded over MLflow's REST API
               (no mlflow package needed: the image stays small)
    local      <MODEL_DIR>/<activity>_model.onnx + .json (offline use, or a cloud VM without MLflow)
    bundled    the release model baked into the image (models/release/): used only while the registry has no
               champion (a freshly deployed server) or can't be reached at start-up; replaced as soon as it has one

Shadow model (model rollout): with a shadow alias (default "challenger"), the registry's challenger is loaded next to
the champion. The API scores every request with both but only answers with the champion; the app records how often
they agree (Prometheus), so a challenger is seen on live traffic before anyone promotes it.

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
    source: str                         # "registry" | "local" | "bundled"
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
                 cache_dir="/tmp/serving-cache", threads=2, fallback_dir=None, shadow_alias=None):
        self.models = dict(models)                  # activity -> registered model name
        self.source = source
        self.alias = alias
        self.registry = MlflowRegistry(registry_uri) if registry_uri else None
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.threads = threads
        self.fallback_dir = Path(fallback_dir) if fallback_dir else None
        self.shadow_alias = shadow_alias or None
        self.loaded = {}                            # activity -> LoadedModel
        self.shadow = {}                            # activity -> LoadedModel scored in the shadow of the champion
        self.errors = {}                            # activity -> last load error
        self.reloads = 0
        self._lock = threading.Lock()               # one reload at a time; readers never wait
        self._stop = threading.Event()

    # ------------------------------------------------------------------ loading
    def _from_registry(self, activity, name, alias, current):
        found = self.registry.version_by_alias(name, alias)
        if found is None:
            raise LookupError(f"{name}: no version has the alias @{alias}")
        version, run_id = found
        if current and current.source == "registry" and current.version == version:
            return None                             # unchanged
        folder = self.cache_dir / name / f"v{version}"
        for f in ("model.onnx", "model.json"):
            if not (folder / f).exists():
                self.registry.download(run_id, f"{ARTIFACT_DIR}/{f}", folder / f)
        meta = json.loads((folder / "model.json").read_text())
        meta["registry"] = dict(model=name, version=int(version), alias=alias, run_id=run_id)
        session = _session(folder / "model.onnx", self.threads)
        return LoadedModel(activity, session, session.get_inputs()[0].name, meta, "registry", version, run_id)

    def _from_folder(self, activity, folder, source, current):
        path = Path(folder) / f"{activity}_model.onnx"
        meta = json.loads(path.with_suffix(".json").read_text())
        stamp = path.stat().st_mtime
        if current and current.source == source and current.meta.get("_mtime") == stamp:
            return None
        meta["_mtime"] = stamp
        session = _session(path, self.threads)
        reg = meta.get("registry", {})
        return LoadedModel(activity, session, session.get_inputs()[0].name, meta, source,
                           str(reg["version"]) if reg else None, reg.get("run_id"))

    def _load(self, activity, name):
        """-> (a new model or None if unchanged, error message or None). Raises if nothing can be served."""
        current = self.loaded.get(activity)
        if self.source != "registry":
            return self._from_folder(activity, self.model_dir, "local", current), None
        try:
            return self._from_registry(activity, name, self.alias, current), None
        except Exception as e:
            if self.fallback_dir is None or (current is not None and current.source != "bundled"):
                raise                                    # keep the registry model we already have
            error = f"{type(e).__name__}: {e} (serving the bundled release model)"
            return self._from_folder(activity, self.fallback_dir, "bundled", current), error

    def _refresh_shadow(self, activity, name):
        """The shadow (challenger) next to the champion; dropped when it is the champion itself or gone."""
        champion, current = self.loaded.get(activity), self.shadow.get(activity)
        try:
            found = self.registry.version_by_alias(name, self.shadow_alias)
            if found is None or champion is None or champion.version == found[0]:
                if self.shadow.pop(activity, None) is not None:
                    log.info("%s: shadow model removed", activity)
                return
            new = self._from_registry(activity, name, self.shadow_alias, current)
        except Exception as e:                           # a shadow problem never affects the champion
            log.warning("%s: shadow model not loaded: %s: %s", activity, type(e).__name__, e)
            return
        if new is not None:
            self.shadow[activity] = new
            log.info("%s: shadow-scoring %s v%s next to v%s", activity, name, new.version, champion.version)

    def refresh(self):
        """Load every model that is missing or whose champion changed. Returns the activities that changed."""
        changed = []
        with self._lock:
            for activity, name in self.models.items():
                try:
                    new, error = self._load(activity, name)
                except Exception as e:                      # keep serving the model we have
                    new, error = None, f"{type(e).__name__}: {e}"
                    log.warning("could not load %s: %s", activity, error)
                if error:
                    self.errors[activity] = error
                else:
                    self.errors.pop(activity, None)
                if new is not None:
                    old = self.loaded.get(activity)
                    self.loaded[activity] = new              # atomic swap
                    if old is not None:
                        self.reloads += 1
                    changed.append(activity)
                    log.info("%s: serving %s v%s (%s, %s)%s", activity, name, new.version,
                             new.meta["arch"].get("type"), new.source, f", was v{old.version}" if old else "")
                if self.shadow_alias and self.source == "registry":
                    self._refresh_shadow(activity, name)
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
