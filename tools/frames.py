"""Frame records (what the detector sends to Kafka) to and from files: build test fixtures and replay demos.

    python tools/frames.py capture tests/videos/fight.avi data/streams/fight.jsonl.gz   # run pose + tracking once
    python tools/frames.py replay data/streams/fight.jsonl.gz                           # send to Kafka at real speed
    python tools/frames.py replay data/streams/fight.jsonl.gz --speed 0 --camera demo-2 # as fast as possible
    python tools/frames.py from-clips data/validated/pose/rwf2000/val data/streams/val100.jsonl.gz --clips 100
    python tools/frames.py replay data/streams/val100.jsonl.gz --speed 0 --drift dark  # simulate a camera problem

from-clips turns validated pose clips (unseen RWF-2000 val clips: the same population as the training data) into
one long frame stream, so the drift monitor sees "normal" data. --drift changes the frames like a camera problem:
    dark   keypoint confidences x0.55 (night, low light, dirty lens): fewer joints detected
    far    everyone half the size, towards the frame centre (camera moved back / other lens)

Replaying needs neither a camera nor YOLO, so the streaming pipeline can be demonstrated and load-tested
with recorded skeletons. Every replay is a new session (new session id, fresh timestamps).
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def capture(video, out, max_frames=None):
    import cv2

    from rover_ai.settings import load_config
    from rover_ai.streaming import to_record
    from rover_ai.vision import Detector
    cfg = load_config()
    cfg["theft"]["enabled"] = False
    det = Detector(cfg, ROOT)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(out, "wt", encoding="utf-8") as f:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            persons, items, _ = det(frame)
            f.write(json.dumps(to_record("capture", "capture", n, n / fps, frame.shape, persons, items, ts=0.0),
                               separators=(",", ":")) + "\n")
            n += 1
    print(f"{n} frames -> {out}")


def from_clips(folder, out, clips, seed=0):
    """Validated clips (npz) -> one frame stream: clips one after another, 5 s apart (so tracks don't mix)."""
    import numpy as np
    paths = sorted(Path(folder).glob("*/*.npz"))
    paths = sorted(np.random.default_rng(seed).choice(paths, min(clips, len(paths)), replace=False).tolist())
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_off, n = 0.0, 0
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for path in paths:
            d = np.load(path)
            fps, width, height = 10.0, int(d["width"]), int(d["height"])
            for i in range(int(d["n_frames"])):
                sel = d["frame"] == i
                people = [{"id": int(tid), "box": [round(float(v), 2) for v in box], "conf": round(float(c), 3),
                           "kpts": [[round(float(x), 2), round(float(y), 2), round(float(cf), 3)] for x, y, cf in k]}
                          for tid, box, c, k in zip(d["track"][sel], d["box"][sel], d["conf"][sel], d["kpts"][sel])]
                rec = {"schema": 1, "camera_id": "clips", "session": "clips", "frame": n,
                       "t": round(t_off + i / fps, 4),
                       "ts": 0.0, "width": width, "height": height, "people": people, "objects": []}
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n += 1
            t_off += int(d["n_frames"]) / fps + 5.0
    print(f"{len(paths)} clips, {n} frames -> {out}")


def drift(record, kind):
    """Simulate a camera problem on one frame record."""
    if kind == "dark":
        for p in record["people"]:
            p["kpts"] = [[x, y, round(c * 0.55, 3)] for x, y, c in p["kpts"]]
    elif kind == "far":
        cx, cy = record["width"] / 2, record["height"] / 2
        for p in record["people"]:
            p["kpts"] = [[round(cx + (x - cx) * 0.5, 2), round(cy + (y - cy) * 0.5, 2), c] for x, y, c in p["kpts"]]
            p["box"] = [round(c + (v - c) * 0.5, 2) for v, c in zip(p["box"], (cx, cy, cx, cy))]
    return record


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def replay(path, bootstrap, topic, camera, speed, drift_kind=None):
    import uuid

    from confluent_kafka import Producer
    records = read(path)
    session = uuid.uuid4().hex[:12]
    p = Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True, "acks": "all", "linger.ms": 20})
    t0, start = records[0]["t"], time.time()
    for r in records:
        if speed > 0:
            wait = start + (r["t"] - t0) / speed - time.time()
            if wait > 0:
                time.sleep(wait)
        r.update(camera_id=camera, session=session, ts=round(time.time(), 3))
        if drift_kind:
            drift(r, drift_kind)
        p.produce(topic, key=camera, value=json.dumps(r, separators=(",", ":")))
        p.poll(0)
    left = p.flush(10)
    print(f"replayed {len(records) - left}/{len(records)} frames as camera {camera!r}, session {session}"
          + (f", simulated drift: {drift_kind}" if drift_kind else ""))


def main():
    from rover_ai.settings import load_config
    cfg = load_config()["streaming"]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("video")
    c.add_argument("out")
    c.add_argument("--max-frames", type=int)
    r = sub.add_parser("replay")
    r.add_argument("file")
    r.add_argument("--bootstrap", default=cfg["bootstrap"])
    r.add_argument("--topic", default=cfg["topic"])
    r.add_argument("--camera", default="replay-1")
    r.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = as fast as possible")
    r.add_argument("--drift", choices=["dark", "far"], help="simulate a camera problem")
    c = sub.add_parser("from-clips")
    c.add_argument("folder")
    c.add_argument("out")
    c.add_argument("--clips", type=int, default=100)
    args = ap.parse_args()
    if args.cmd == "capture":
        capture(args.video, args.out, args.max_frames)
    elif args.cmd == "from-clips":
        from_clips(args.folder, args.out, args.clips)
    else:
        replay(args.file, args.bootstrap, args.topic, args.camera, args.speed, args.drift)


if __name__ == "__main__":
    main()
