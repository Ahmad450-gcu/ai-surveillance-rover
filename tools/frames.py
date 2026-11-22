"""Frame records (what the detector sends to Kafka) to and from files: build test fixtures and replay demos.

    python tools/frames.py capture tests/videos/fight.avi data/streams/fight.jsonl.gz   # run pose + tracking once
    python tools/frames.py replay data/streams/fight.jsonl.gz                           # send to Kafka at real speed
    python tools/frames.py replay data/streams/fight.jsonl.gz --speed 0 --camera demo-2 # as fast as possible

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


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def replay(path, bootstrap, topic, camera, speed):
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
        p.produce(topic, key=camera, value=json.dumps(r, separators=(",", ":")))
        p.poll(0)
    left = p.flush(10)
    print(f"replayed {len(records) - left}/{len(records)} frames as camera {camera!r}, session {session}")


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
    args = ap.parse_args()
    if args.cmd == "capture":
        capture(args.video, args.out, args.max_frames)
    else:
        replay(args.file, args.bootstrap, args.topic, args.camera, args.speed)


if __name__ == "__main__":
    main()
