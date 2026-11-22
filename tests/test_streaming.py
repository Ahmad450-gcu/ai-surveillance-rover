"""Streaming pipeline logic (no Kafka / Spark needed): frame records, stateful stream scoring, the alert bridge."""
import json
import random
from pathlib import Path

import numpy as np
import pytest

from rover_ai.learned import LearnedFightDetector
from rover_ai.streaming import persons_from_record, to_record
from streaming import scoring
from streaming.bridge import Seen, to_backend
from tools.frames import read

FRAMES = Path(__file__).parent / "data" / "fight_frames.jsonl.gz"     # tools/frames.py capture tests/videos/fight.avi


@pytest.fixture(scope="module")
def records():
    return read(FRAMES)


def make_detector(tmp_path, threshold):
    """A small untrained model (no registry needed); threshold 0 = every judgement is "above"."""
    import torch

    from rover_ai.nets import build_model
    from training.train_fight import export_onnx
    torch.manual_seed(0)
    arch = {"type": "tcn", "channels": 8, "levels": 2}
    export_onnx(build_model(arch).eval(), tmp_path / "m.onnx", 20)
    (tmp_path / "m.json").write_text(json.dumps({"arch": arch, "window": 20, "threshold": threshold,
                                                 "consecutive": 2, "proximity": 5.0, "min_cover": 0.3}))
    cfg = {"model": "m.onnx", "model_threshold": None, "model_consecutive": None, "proximity": 3.5,
           "window_seconds": 0.6}
    return lambda: LearnedFightDetector(cfg, tmp_path)


def test_record_round_trip_keeps_people():
    from types import SimpleNamespace
    k = np.random.default_rng(0).uniform(0, 100, (17, 3)).astype(np.float32)
    k[:, 2] = 0.5
    p = SimpleNamespace(track_id=7, box=np.array([1, 2, 30, 60.0]), conf=0.9, kpts=k)
    untracked = SimpleNamespace(track_id=-1, box=p.box, conf=0.5, kpts=k)
    rec = json.loads(json.dumps(to_record("cam", "s1", 3, 0.3, (480, 640, 3), [p, untracked])))
    assert (rec["width"], rec["height"], rec["frame"], len(rec["people"])) == (640, 480, 3, 1)
    back = persons_from_record(rec)[0]
    assert back.track_id == 7 and np.allclose(back.kpts, k, atol=0.01) and back.kpts.shape == (17, 3)


def test_micro_batches_give_the_same_result_as_one_pass(records, tmp_path):
    """What Spark does: random micro-batches, state through JSON, repeated and late frames."""
    new = make_detector(tmp_path, threshold=0.5)
    seq, seq_alerts, _, _ = scoring.score_frames(new(), None, records)
    assert len(seq) > 100
    rng = random.Random(0)
    state, preds, alerts, skipped, i = "", [], [], 0, 0
    while i < len(records):
        n = rng.randint(1, 15)
        batch = records[i:i + n] + ([records[max(0, i - 1)]] if i and rng.random() < 0.5 else [])   # + a repeat
        rng.shuffle(batch)
        i += n
        p, a, st, sk = scoring.score_frames(new(), scoring.loads(state), batch)
        preds, alerts, skipped, state = preds + p, alerts + a, skipped + sk, scoring.dumps(st)
    key = lambda r: (r["frame"], r["person_a"], r["person_b"])          # noqa: E731
    assert [key(r) for r in sorted(seq, key=key)] == [key(r) for r in sorted(preds, key=key)]
    assert np.allclose([r["probability"] for r in sorted(seq, key=key)],
                       [r["probability"] for r in sorted(preds, key=key)], atol=1e-6)
    assert [a["frame"] for a in seq_alerts] == [a["frame"] for a in alerts]
    assert skipped > 0                                                   # the repeats were dropped
    assert len(json.loads(state)["frames"]) < 120                        # state keeps only the last 3 s


def test_alert_cooldowns_follow_stream_time(records, tmp_path):
    """Threshold 0: every close pair qualifies; still one fight alert per camera per 10 s of stream time."""
    det = make_detector(tmp_path, threshold=0.0)()
    preds, alerts, _, _ = scoring.score_frames(det, None, records)
    assert len(alerts) == 1 and alerts[0]["message"].startswith("FIGHT")
    later = [dict(r, t=r["t"] + 11.0, frame=r["frame"] + 1000) for r in records]       # 11 s later
    _, more, _, _ = scoring.score_frames(det, None, records + later)
    assert len(more) == 2
    assert all(p["above"] for p in preds)


def test_bridge_maps_alerts_for_the_backend_and_drops_repeats():
    alert = {"id": "cam-s-12-3-9", "time": "2026-11-22T10:00:00", "type": "fight", "message": "FIGHT: ID 3 and ID 9",
             "track_ids": [3, 9], "score": 0.8, "model_version": "fight-detector v4", "rover_id": "cam",
             "source": "stream", "session": "s"}
    body = to_backend(alert)
    assert set(body) <= {"time", "type", "message", "track_ids", "snapshot", "clip", "score", "model_version",
                         "rover_id"}
    assert body["message"].endswith("[stream]") and body["score"] == 0.8
    seen = Seen(size=2)
    assert seen.add("a") and not seen.add("a") and seen.add("b") and seen.add("c") and seen.add("a")   # bounded
