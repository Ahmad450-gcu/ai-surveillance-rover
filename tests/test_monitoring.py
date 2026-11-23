"""Drift monitoring: PSI maths, drift features, per-camera drift, the Alertmanager webhook (retraining requests)."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from monitoring.drift import FEATURES, collect, describe, frames_from_records, level, proportions, psi
from monitoring.monitor import Monitor
from tools.frames import drift, read

FRAMES = Path(__file__).parent / "data" / "fight_frames.jsonl.gz"


@pytest.fixture(scope="module")
def records():
    return read(FRAMES)


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    ref = describe(rng.normal(0, 1, 20000))
    assert abs(sum(ref["proportions"]) - 1) < 1e-6 and len(ref["edges"]) == 9
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.02
    assert 0.1 < psi(ref, rng.normal(0.4, 1, 5000)) < 0.4
    assert psi(ref, rng.normal(2, 1, 5000)) > 1
    assert [level(v) for v in (0.05, 0.15, 0.5)] == ["stable", "moderate", "drift"]


def test_discrete_features_bin_the_same_way_live_as_in_the_reference():
    """Regression: rounded edges must not move values like 16/17 to another bin (gave PSI 2.7 on unchanged data)."""
    values = np.random.default_rng(1).integers(0, 18, 10000) / 17
    ref = json.loads(json.dumps(describe(values)))                   # stored as JSON, like the real reference
    assert psi(ref, values) < 1e-6
    assert np.allclose(proportions(values, ref["edges"]), ref["proportions"], atol=1e-6)


def test_frame_features_on_recorded_frames(records):
    v = collect(frames_from_records(records))
    assert set(v) == set(FEATURES)
    assert len(v["people"]) == len(records) and v["people"].max() >= 2
    assert 0 < v["kp_conf"].mean() < 1 and 0 < v["visible_joints"].mean() <= 1
    assert 0 < v["body_size"].mean() < 1 and len(v["pair_distance"]) > 0


def test_monitor_flags_only_the_drifted_camera_features(records):
    m = Monitor(min_samples=50, min_frames=100)
    base = collect(frames_from_records(records))
    m.reference = {"features": {f: describe(base[f]) for f in FEATURES}}
    normal = m.data_drift(records)
    assert all(d["level"] in ("stable", "moderate") for d in normal.values())
    dark = m.data_drift([drift(copy.deepcopy(r), "dark") for r in records])
    assert dark["kp_conf"]["level"] == "drift" and dark["visible_joints"]["level"] == "drift"
    assert dark["body_size"]["level"] == "stable" and dark["people"]["psi"] < 1e-6
    far = m.data_drift([drift(copy.deepcopy(r), "far") for r in records])
    assert far["body_size"]["level"] == "drift" and far["kp_conf"]["psi"] < 1e-6
    for few in (Monitor(min_samples=10**6, min_frames=1), Monitor(min_samples=1, min_frames=len(records) + 1)):
        few.reference = m.reference            # too few values, or too few frames from this camera
        assert few.data_drift(records)["kp_conf"]["level"] == "insufficient_data"


class FakeDB:
    """Just enough of a psycopg connection for the webhook."""

    def __init__(self):
        self.events, self.requests = [], []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=()):
        db = self

        class Result:
            def fetchone(self):
                if sql.startswith("SELECT id FROM retrain_requests"):
                    return next(((r["id"],) for r in db.requests
                                 if r["status"] == "pending" and r["activity"] == params[0]), None)
                return (db.requests[-1]["id"],)
        if sql.startswith("INSERT INTO ops_events"):
            self.events.append(params)
        elif sql.startswith("INSERT INTO retrain_requests"):
            self.requests.append({"id": len(self.requests) + 1, "reason": params[0], "status": "pending",
                                  "activity": params[2]})
        return Result()


def test_alertmanager_webhook_stores_alerts_and_asks_for_retraining_once():
    db = FakeDB()
    started = []
    m = Monitor(db=lambda: db, trigger=lambda rid, dag: started.append(rid),
                retrain_targets={"fight": "fight_retraining"})
    drift_alert = {"status": "firing", "labels": {"alertname": "DataDriftDetected", "camera": "cam-2",
                                                  "severity": "warning", "action": "retrain"},
                   "annotations": {"summary": "Data drift on camera cam-2"}, "startsAt": "2026-11-22T10:00:00Z"}
    down = {"status": "firing", "labels": {"alertname": "ServiceDown", "job": "serving", "severity": "critical"}}
    r = m.handle_alertmanager({"status": "firing", "alerts": [drift_alert, down]})
    assert r == {"stored": 2, "retrain_requests": [1]} and len(db.events) == 2
    assert m.handle_alertmanager({"alerts": [drift_alert]})["retrain_requests"] == []     # one is still pending
    db.requests[0]["status"] = "done"
    assert m.handle_alertmanager({"alerts": [drift_alert]})["retrain_requests"] == [2]
    resolved = dict(drift_alert, status="resolved")
    assert m.handle_alertmanager({"alerts": [resolved]})["retrain_requests"] == []
    assert started == [1, 2]                                   # the retraining DAG was started for each request

    def airflow_down(*_):
        raise OSError("connection refused")
    m.trigger = airflow_down
    db.requests[-1]["status"] = "done"
    assert m.handle_alertmanager({"alerts": [drift_alert]})["retrain_requests"] == [3]   # still recorded


def test_drift_asks_every_model_activity_that_retrains_on_drift():
    db = FakeDB()
    started = []
    m = Monitor(db=lambda: db, trigger=lambda rid, dag: started.append((rid, dag)),
                retrain_targets={"fight": "fight_retraining", "theft": "theft_retraining"})
    alert = {"status": "firing", "labels": {"alertname": "DataDriftDetected", "action": "retrain"}}
    assert m.handle_alertmanager({"alerts": [alert]})["retrain_requests"] == [1, 2]
    assert started == [(1, "fight_retraining"), (2, "theft_retraining")]
    assert [r["activity"] for r in db.requests] == ["fight", "theft"]
    db.requests[1]["status"] = "done"                       # theft finished, fight still pending
    assert m.handle_alertmanager({"alerts": [alert]})["retrain_requests"] == [3]
    assert db.requests[-1]["activity"] == "theft"


def test_drift_targets_come_from_the_activity_registry():
    from monitoring.monitor import drift_retrain_targets
    assert drift_retrain_targets() == {"fight": "fight_retraining"}
