"""Model serving API: predictions, input checks, registry hot reload, admin token, and the detector's remote client."""
import base64
import json
import shutil
import socket
import threading
import time

import numpy as np
import pytest
import torch

pytest.importorskip("prometheus_client")
from fastapi.testclient import TestClient  # noqa: E402

from rover_ai.learned import N_FEAT, OnnxModel, RemoteModel  # noqa: E402
from rover_ai.nets import build_model  # noqa: E402
from serving.app import create_app  # noqa: E402
from serving.store import ModelStore  # noqa: E402

T = 20


def export(folder, kind, seed, threshold=0.5):
    """A small untrained model as training would leave it: <folder>/model.onnx + model.json"""
    from training.train_fight import export_onnx
    torch.manual_seed(seed)
    folder.mkdir(parents=True, exist_ok=True)
    arch = {"type": kind, **({"hidden": 8, "layers": 1} if kind == "gru" else {"channels": 8, "levels": 2})}
    export_onnx(build_model(arch).eval(), folder / "model.onnx", T)
    meta = {"arch": arch, "window": T, "threshold": threshold, "consecutive": 2, "proximity": 5.0, "min_cover": 0.3,
            "test_learned": {"f1": 0.6, "roc_auc": 0.7}}
    (folder / "model.json").write_text(json.dumps(meta))
    return folder


class FakeRegistry:
    """Stands in for MLflow: versions in folders, an alias that can move."""

    def __init__(self, versions):
        self.versions = versions            # version -> folder
        self.champion = None
        self.down = False

    def version_by_alias(self, name, alias):
        if self.down:
            raise OSError("registry unreachable")
        return None if self.champion is None else (self.champion, f"run{self.champion}")

    def download(self, run_id, artifact, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.versions[run_id[3:]] / artifact.split("/")[-1], dest)


@pytest.fixture(autouse=True)
def no_admin_token(monkeypatch):
    monkeypatch.delenv("SERVING_ADMIN_TOKEN", raising=False)       # other tests may have loaded .env


@pytest.fixture
def local_store(tmp_path):
    src = export(tmp_path / "m", "tcn", 0)
    (tmp_path / "models").mkdir()
    for f in ("onnx", "json"):
        shutil.copy(src / f"model.{f}", tmp_path / "models" / f"fight_model.{f}")
    return ModelStore({"fight": "fight-detector"}, source="local", model_dir=tmp_path / "models")


def windows_scored(client):
    line = [ln for ln in client.get("/metrics").text.splitlines() if ln.startswith('serving_windows_total{activity="fight"}')]
    return float(line[0].split()[-1]) if line else 0.0


def x_batch(n=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, T, N_FEAT)).astype(np.float32)


def test_predict_matches_onnx_runtime_in_both_payload_forms(local_store, tmp_path):
    with TestClient(create_app(local_store, reload_seconds=0)) as client:
        before = windows_scored(client)
        assert client.get("/ready").json()["ready"]
        info = client.get("/v1/models/fight").json()
        assert (info["window"], info["n_feat"], info["arch"], info["threshold"]) == (T, N_FEAT, "tcn", 0.5)
        x = x_batch()
        a = client.post("/v1/models/fight/predict", json={"instances": x.tolist()}).json()
        b = client.post("/v1/models/fight/predict",
                        json={"shape": list(x.shape), "data_b64": base64.b64encode(x.astype("<f4").tobytes()).decode()}
                        ).json()
        expected = OnnxModel(tmp_path / "models" / "fight_model.onnx")(x).numpy()
        assert np.allclose(a["logits"], expected, atol=1e-4) and np.allclose(b["logits"], expected, atol=1e-4)
        assert a["above_threshold"] == [p >= 0.5 for p in a["probabilities"]]
        assert windows_scored(client) - before == 6
        assert 'serving_model_info{activity="fight",arch="tcn"' in client.get("/metrics").text


def test_bad_requests_are_rejected(local_store):
    with TestClient(create_app(local_store, reload_seconds=0)) as client:
        post = client.post
        assert post("/v1/models/theft/predict", json={"instances": x_batch().tolist()}).status_code == 404
        assert post("/v1/models/fight/predict", json={"instances": x_batch()[:, :10].tolist()}).status_code == 422
        assert post("/v1/models/fight/predict", json={"shape": [1, T, N_FEAT], "data_b64": "AAAA"}).status_code == 422
        assert post("/v1/models/fight/predict", json={"shape": [1, T, N_FEAT], "data_b64": "@@"}).status_code == 422
        assert post("/v1/models/fight/predict", json={}).status_code == 422
        nan = x_batch(1)
        nan[0, 0, 0] = np.nan
        assert post("/v1/models/fight/predict",
                    json={"shape": [1, T, N_FEAT], "data_b64": base64.b64encode(nan.tobytes()).decode()}
                    ).status_code == 422


def test_registry_champion_is_hot_reloaded_and_failures_keep_the_old_model(tmp_path, monkeypatch):
    reg = FakeRegistry({"1": export(tmp_path / "v1", "gru", 1), "2": export(tmp_path / "v2", "tcn", 2, 0.7)})
    store = ModelStore({"fight": "fight-detector"}, source="registry", registry_uri="http://unused",
                       cache_dir=tmp_path / "cache")
    store.registry = reg
    with TestClient(create_app(store, reload_seconds=0)) as client:
        assert client.get("/ready").status_code == 503                       # no champion yet
        assert client.post("/v1/models/fight/predict", json={"instances": x_batch().tolist()}).status_code == 503
        reg.champion = "1"
        assert client.post("/admin/reload").json()["changed"] == ["fight"]
        assert client.get("/v1/models/fight").json()["version"] == "1"
        assert client.post("/admin/reload").json()["changed"] == []          # unchanged: nothing reloaded
        reg.champion = "2"                                                    # promotion
        client.post("/admin/reload")
        info = client.get("/v1/models/fight").json()
        assert (info["version"], info["arch"], info["threshold"]) == ("2", "tcn", 0.7)
        reg.down = True                                                       # registry outage
        r = client.post("/admin/reload").json()
        assert r["changed"] == [] and "fight" in r["errors"]
        assert client.post("/v1/models/fight/predict", json={"instances": x_batch().tolist()}).status_code == 200
        assert 'serving_model_info{activity="fight",arch="tcn",version="2"} 1.0' in client.get("/metrics").text
        assert 'version="1"' not in client.get("/metrics").text.split("serving_model_info")[-1]


def test_admin_reload_needs_the_token_when_set(local_store, monkeypatch):
    monkeypatch.setenv("SERVING_ADMIN_TOKEN", "s3cret")
    with TestClient(create_app(local_store, reload_seconds=0)) as client:
        assert client.post("/admin/reload").status_code == 401
        assert client.post("/admin/reload", headers={"Authorization": "Bearer s3cret"}).status_code == 200


@pytest.fixture
def live_server(tmp_path):
    """The serving app on a real port (the detector's client speaks plain HTTP)."""
    import uvicorn
    reg = FakeRegistry({"1": export(tmp_path / "v1", "gru", 1), "2": export(tmp_path / "v2", "tcn", 2, 0.7)})
    reg.champion = "1"
    store = ModelStore({"fight": "fight-detector"}, source="registry", registry_uri="http://unused",
                       cache_dir=tmp_path / "cache")
    store.registry = reg
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(store, reload_seconds=0), port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", reg, store, tmp_path
    server.should_exit = True
    thread.join(5)


def test_detector_remote_model_follows_the_served_version(live_server):
    url, reg, store, tmp_path = live_server
    model = RemoteModel(url + "/v1/models/fight")
    assert model.meta["arch"]["type"] == "gru" and model.meta["window"] == T
    x = torch.from_numpy(x_batch(4))
    assert np.allclose(model(x).numpy(), OnnxModel(tmp_path / "v1" / "model.onnx")(x).numpy(), atol=1e-4)
    seen = []
    model.on_change.append(seen.append)
    reg.champion = "2"
    store.refresh()                                                   # what the watcher thread does
    out = model(x)                                                    # answered by v2; the client notices
    assert np.allclose(out.numpy(), OnnxModel(tmp_path / "v2" / "model.onnx")(x).numpy(), atol=1e-4)
    assert seen and seen[0]["threshold"] == 0.7 and model.meta["arch"]["type"] == "tcn"


def test_detector_survives_serving_outage(tmp_path):
    from rover_ai.learned import LearnedFightDetector
    with pytest.raises(OSError):
        RemoteModel("http://127.0.0.1:9/v1/models/fight", timeout=0.5)        # nothing listens on port 9
    det = object.__new__(LearnedFightDetector)                                 # judge() alone, model down
    det.failed_at = None

    def down(_):
        raise OSError("connection refused")
    det.model = down
    k, b = np.zeros((T, 17, 3), np.float32), np.tile(np.array([0, 0, 10, 20], np.float32), (T, 1))
    assert det.judge([(k, b, k, b)], t=1.0) is None and det.failed_at == 1.0
