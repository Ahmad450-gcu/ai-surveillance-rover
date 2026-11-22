"""Edge side of the streaming pipeline: every processed frame's people (tracked skeletons) and objects go to Kafka.

Record (JSON, topic rover.pose-frames, key = camera id so one camera's frames stay in order):

    {"schema": 1, "camera_id": "rover-1", "session": "<uuid of this run>", "frame": 123,
     "t": 12.3,                 # detector clock in seconds (video time for files, wall time for live cameras)
     "ts": 1795400000.12,       # wall-clock capture time (epoch seconds): event time for lateness checks
     "width": 640, "height": 480,
     "people":  [{"id": 5, "box": [x1, y1, x2, y2], "conf": 0.91, "kpts": [[x, y, c] x 17]}],
     "objects": [{"id": 2, "name": "backpack", "box": [...], "conf": 0.8}]}

The stream (streaming/spark_job.py) validates it, archives it in the lake and scores it with the served model.
If Kafka can't be reached, the publisher reports it (healthy = False) and run.py scores fights locally instead.
"""
import json
import time
import uuid
from types import SimpleNamespace

import numpy as np

SCHEMA_VERSION = 1


def to_record(camera_id, session, frame, t, shape, persons, items=(), ts=None):
    h, w = shape[:2]
    return {"schema": SCHEMA_VERSION, "camera_id": camera_id, "session": session, "frame": int(frame),
            "t": round(float(t), 4), "ts": round(time.time() if ts is None else ts, 3), "width": int(w), "height": int(h),
            "people": [{"id": int(p.track_id), "box": [round(float(v), 2) for v in p.box], "conf": round(float(p.conf), 3),
                        "kpts": [[round(float(x), 2), round(float(y), 2), round(float(c), 3)] for x, y, c in p.kpts]}
                       for p in persons if p.track_id >= 0],
            "objects": [{"id": int(i.track_id), "name": i.name, "box": [round(float(v), 2) for v in i.box],
                         "conf": round(float(i.conf), 3)} for i in items]}


def persons_from_record(record):
    """-> objects with the attributes PeopleState needs (track_id, box, conf, kpts), like vision.Person."""
    return [SimpleNamespace(track_id=p["id"], box=np.asarray(p["box"], np.float32), conf=p["conf"],
                            kpts=np.asarray(p["kpts"], np.float32)) for p in record["people"]]


class FramePublisher:
    """Kafka producer for the frame records. Never blocks the video loop: sends are queued and confirmed
    in the background; `healthy` turns False when no broker answers and back to True on the next success."""

    def __init__(self, cfg, camera_id):
        from confluent_kafka import KafkaError, Producer
        self.topic = cfg["topic"]
        self.camera_id = camera_id
        self.session = uuid.uuid4().hex[:12]
        self.fight_on_server = cfg.get("fight_on_server", True)
        self.healthy = False
        self.sent = self.failed = 0
        self._all_down = KafkaError._ALL_BROKERS_DOWN
        self.producer = Producer({
            "bootstrap.servers": cfg["bootstrap"],
            "client.id": f"detector-{camera_id}",
            "enable.idempotence": True,          # broker drops duplicates from producer retries
            "acks": "all",
            "linger.ms": 20,
            "message.timeout.ms": 10000,         # give up on a frame after 10 s (old frames are useless)
            "queue.buffering.max.messages": 20000,
            "error_cb": self._on_error,
            "log_level": 3,                      # only errors on the console (outages are reported by error_cb)
        })
        try:
            self.producer.list_topics(self.topic, timeout=3)
            self.healthy = True
        except Exception as e:
            print(f"Streaming: Kafka at {cfg['bootstrap']} not reachable ({e})")

    def _on_error(self, err):
        if err.code() == self._all_down:
            if self.healthy:
                print("Streaming: Kafka unreachable, scoring fights locally until it is back")
            self.healthy = False

    def _on_delivery(self, err, _msg):
        if err is None:
            if not self.healthy:
                print("Streaming: Kafka reachable again, fights are scored by the stream")
            self.healthy = True
            self.sent += 1
        else:
            self.failed += 1

    def publish(self, frame, t, shape, persons, items=()):
        record = to_record(self.camera_id, self.session, frame, t, shape, persons, items)
        try:
            self.producer.produce(self.topic, key=self.camera_id, value=json.dumps(record, separators=(",", ":")),
                                  on_delivery=self._on_delivery)
        except BufferError:                      # local queue full (Kafka down for long): drop this frame
            self.failed += 1
        self.producer.poll(0)                    # run delivery / error callbacks

    @property
    def server_scores_fights(self):
        return self.fight_on_server and self.healthy

    def close(self, timeout=5):
        self.producer.flush(timeout)
