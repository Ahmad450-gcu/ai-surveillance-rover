"""Human feedback -> training clips (training/feedback.py): evidence parsing, clip resampling, split, the stage."""
import io
import json

import numpy as np
import pytest

from training import feedback

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def person(pid, x):
    kpts = [[x + j, 100.0 + 5 * j, 0.9] for j in range(17)]
    return {"id": pid, "box": [x, 100.0, x + 40, 220.0], "conf": 0.9, "kpts": kpts}


def frame_rows(session="s1", n=80, fps=10.0, people=2, start=0):
    return [{"session": session, "frame": start + i, "t": (start + i) / fps, "width": 640, "height": 480,
             "people": json.dumps([person(k, 100.0 + 60 * k) for k in range(people)])} for i in range(n)]


class FakeLake:
    """Just what feedback.build uses: list_keys + client().get_object / put_object."""

    def __init__(self, parquet_by_key):
        self.objects = {("validated", k): v for k, v in parquet_by_key.items()}
        self.put = {}

    def list_keys(self, bucket, prefix):
        return [{"Key": k} for (b, k) in self.objects if b == bucket and k.startswith(prefix)]

    def client(self):
        lake = self

        class C:
            def get_object(self, Bucket, Key):
                return {"Body": io.BytesIO(lake.objects[(Bucket, Key)])}

            def put_object(self, Bucket, Key, Body, **_):
                lake.put[(Bucket, Key)] = Body
        return C()


def parquet(rows):
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf)
    return buf.getvalue()


def test_parse_evidence():
    assert feedback.parse_evidence("stream://rover-1/4111d5339b83/537") == ("rover-1", "4111d5339b83", 537)
    assert feedback.parse_evidence("stream://cam/with/slash/s/7") == ("cam/with/slash", "s", 7)


def test_clip_is_resampled_to_10fps_around_the_alert():
    rows = {r["frame"]: r for r in frame_rows(n=200, fps=25.0)}          # a 25 fps stream
    clip = feedback.clip_from_frames(rows, t_alert=6.0)
    # 4 s before + 1 s after, at 10 fps, both ends included
    assert clip["n_frames"] == int((feedback.CLIP_BEFORE + feedback.CLIP_AFTER) * feedback.FPS) + 1
    assert clip["kpts"].shape == (2 * int(clip["n_frames"]), 17, 3)
    assert clip["box"].shape[1] == 4 and set(np.unique(clip["frame"])) == set(range(int(clip["n_frames"])))
    assert int(clip["width"]) == 640 and int(clip["height"]) == 480


def test_clip_skips_ticks_without_a_nearby_frame():
    rows = {r["frame"]: r for r in frame_rows(n=100) if not 30 <= r["frame"] < 40}    # 1 s gap at 3..4 s
    clip = feedback.clip_from_frames(rows, t_alert=6.0)                                 # window 2..7 s
    assert clip["n_frames"] == 51 - 10


def test_split_keeps_everything_for_training_until_there_are_enough_labels():
    assert {feedback.split_of(i, feedback.MIN_FOR_TEST - 1) for i in range(50)} == {"train"}
    splits = [feedback.split_of(i, 500) for i in range(500)]
    assert 0.1 < splits.count("val") / 500 < 0.3                       # ~1 in 5, deterministic per alert
    assert splits == [feedback.split_of(i, 500) for i in range(500)]


def test_build_turns_labels_into_classed_clips(tmp_path, monkeypatch):
    manifest = tmp_path / "feedback_labels.json"
    manifest.write_text(json.dumps({"alerts": [
        {"alert_id": 1, "label": "real", "evidence": "stream://cam-a/s1/60"},
        {"alert_id": 2, "label": "false_alarm", "evidence": "stream://cam-a/s2/60"},
        {"alert_id": 3, "label": "real", "evidence": "stream://cam-a/s1/9999"},        # frames gone
        {"alert_id": 4, "label": "real", "evidence": "stream://cam-a/s3/60"},          # only one person
    ]}))
    monkeypatch.setattr(feedback, "MANIFEST", manifest)
    monkeypatch.setattr(feedback, "OUT", tmp_path / "rover")
    lake = FakeLake({
        "stream/pose/date=2026-11-22/camera_id=cam-a/part-0.parquet": parquet(frame_rows("s1") + frame_rows("s2")),
        "stream/pose/date=2026-11-22/camera_id=cam-a/part-1.parquet": parquet(frame_rows("s3", people=1)),
        "stream/pose/date=2026-11-22/camera_id=cam-b/part-0.parquet": parquet(frame_rows("s1", start=1000)),
    })
    summary = feedback.build(lake)
    assert (tmp_path / "rover/train/Fight/alert-1.npz").exists()
    assert (tmp_path / "rover/train/NonFight/alert-2.npz").exists()
    assert {s["alert_id"]: s["reason"] for s in summary["skipped"]} == {
        3: "frames not in the lake", 4: "too few frames with two people"}
    assert summary["by_split_class"]["train/Fight"] == 1 and summary["by_split_class"]["train/NonFight"] == 1
    assert set(lake.put) == {("labelled", "fight/rover/Fight/alert-1.npz"),
                             ("labelled", "fight/rover/NonFight/alert-2.npz")}
    d = np.load(tmp_path / "rover/train/Fight/alert-1.npz")
    assert d["kpts"].shape[1:] == (17, 3) and int(d["n_frames"]) == 51
    assert json.loads((tmp_path / "rover/feedback.json").read_text())["clips"] == 2


def test_build_without_labels_still_writes_the_stage_output(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "MANIFEST", tmp_path / "missing.json")
    monkeypatch.setattr(feedback, "OUT", tmp_path / "rover")
    assert feedback.build(FakeLake({}))["clips"] == 0
    assert (tmp_path / "rover/feedback.json").exists()
