"""Fight model code: architectures, ONNX export, augmentation, live-style clip scoring, registry promotion."""
import json

import numpy as np
import pytest
import torch

from rover_ai.learned import N_FEAT, load_model
from rover_ai.nets import ARCHS, build_model, feature_groups, pair_adjacency
from training.fightlib import augment, clip_score, metrics, roc_auc

ARCH = {"gru": {"hidden": 16, "layers": 2, "dropout": 0.1}, "tcn": {"channels": 16, "levels": 2},
        "stgcn": {"channels": 8, "blocks": 2}}


@pytest.mark.parametrize("kind", sorted(ARCHS))
def test_every_architecture_gives_one_logit_per_window(kind):
    m = build_model({"type": kind, **ARCH[kind]}).eval()
    with torch.no_grad():
        assert m(torch.randn(3, 20, N_FEAT)).shape == (3,)


@pytest.mark.parametrize("kind", sorted(ARCHS))
def test_saved_model_loads_as_pt_and_onnx_with_same_output(kind, tmp_path):
    from training.train_fight import export_onnx
    arch = {"type": kind, **ARCH[kind]}
    m = build_model(arch).eval()
    torch.save(m.state_dict(), tmp_path / "m.pt")
    export_onnx(m, tmp_path / "m.onnx", 20)
    (tmp_path / "m.json").write_text(json.dumps({"arch": arch, "window": 20, "threshold": 0.5}))
    x = torch.randn(5, 20, N_FEAT)
    a = load_model(tmp_path / "m.pt")[0](x).detach().numpy()
    b = load_model(tmp_path / "m.onnx")[0](x).numpy()
    assert np.abs(a - b).max() < 1e-4


def test_old_models_without_type_are_grus():
    assert type(build_model({"n_feat": N_FEAT, "hidden": 8, "layers": 1, "dropout": 0})).__name__ == "FightGRU"
    with pytest.raises(ValueError):
        build_model({"type": "transformer"})


def test_graph_is_symmetric_and_connects_the_two_people():
    a = pair_adjacency()
    assert a.shape == (34, 34) and np.allclose(a, a.T)
    assert a[9, 17 + 0] > 0          # A's left wrist <-> B's nose (contact edge)


def test_joint_dropout_removes_whole_joints_consistently():
    g = feature_groups()
    x = augment(torch.ones(256, 20, N_FEAT), {"joint_dropout": 0.5}, np.random.default_rng(0))
    vis, pos = x[:, :, g["vis"]], x[:, :, g["pos"]]          # (B, T, 2, 17), (B, T, 2, 17, 2)
    assert 0.4 < float((vis == 0).float().mean()) < 0.6
    assert torch.equal((vis == 0), (pos[..., 0] == 0)) and torch.equal(pos[..., 0], pos[..., 1])
    assert torch.equal(vis[:, 0], vis[:, -1])                   # same joints missing for the whole window
    assert torch.all(x[:, :, 170:] == 1)                       # pair features untouched


def test_time_stretch_slows_motion_down():
    g = feature_groups()
    x = torch.zeros(8, 20, N_FEAT)
    x[:, :, g["vel"].ravel()] = 1.0
    x[:, :, 170] = torch.arange(20.0)                          # a feature that grows over time
    y = augment(x.clone(), {"time_stretch": 1.0}, np.random.default_rng(0))
    assert torch.all(y[:, :, g["vel"].ravel()] < 1.0)           # velocities shrink
    assert torch.all(y[:, -1, 170] == 19) and torch.all(y[:, 0, 170] >= 1)   # replays the end of the window


def test_clip_score_follows_the_live_alert_rule():
    starts = np.array([0, 1, 2, 3, 30, 31])
    p = np.array([0.9, 0.2, 0.8, 0.7, 0.95, 0.9])
    assert clip_score({0: (starts, p)}, 2) == pytest.approx(0.9)   # frames 30,31 (min 0.9)
    assert clip_score({0: (starts, p)}, 3) == pytest.approx(0.2)   # the gap at frame 4..29 resets the run
    assert clip_score(None, 2) == 0.0


def test_metrics_and_auc():
    y = np.array([1, 1, 0, 0, 1])
    s = np.array([0.9, 0.4, 0.3, 0.4, 0.8])
    m = metrics(y, s >= 0.5, s)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 0, 2)
    pairs = [(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]]
    assert roc_auc(y, s) == pytest.approx(np.mean(pairs))


def test_promotion_needs_a_real_improvement(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from mlops import registry
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "params", lambda: {"model_name": "fight-test", "metric": "f1",
                                                     "min_improvement": 0.01, "max_auc_drop": 0.02})
    _, client = registry.connect()
    mlflow.set_experiment("t")
    client.create_registered_model("fight-test")

    def new_version(f1, auc):
        with mlflow.start_run() as run:
            mlflow.log_metrics({"test_f1": f1, "test_roc_auc": auc})
        v = client.create_model_version("fight-test", source=f"runs:/{run.info.run_id}/model", run_id=run.info.run_id)
        client.set_registered_model_alias("fight-test", "challenger", v.version)
        return registry.promote(report=tmp_path / "p.json")

    assert new_version(0.60, 0.70)["promoted"]                 # first model: no champion yet
    assert not new_version(0.605, 0.75)["promoted"]            # +0.005 F1: not enough
    assert not new_version(0.70, 0.60)["promoted"]             # better F1 but AUC collapsed
    d = new_version(0.70, 0.71)
    assert d["promoted"] and d["champion"] == 1
    assert str(client.get_model_version_by_alias("fight-test", "champion").version) == "4"
