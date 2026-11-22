"""End-to-end check of the running streaming pipeline (docker compose --profile streaming up -d).

    python tools/stream_smoke_test.py

1. sends one bad record per validation rule  -> each must reach rover.pose-frames.dlq with the right reason
2. replays the recorded fight clip (tests/data/fight_frames.jsonl.gz) as a new camera session
   -> Spark scores it through the serving API: predictions in PostgreSQL, the FIGHT alert in topic rover.alerts,
      and the alert bridge stores it in the backend's alerts table
Prints PASS / FAIL per check; exit code 1 if anything failed.
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rover_ai.settings import load_config, load_dotenv  # noqa: E402
from tools.frames import read  # noqa: E402

FRAMES = ROOT / "tests" / "data" / "fight_frames.jsonl.gz"


def bad_records(good):
    """(expected reason, raw value) for every validation rule."""
    def edit(**changes):
        r = json.loads(json.dumps(good))
        for k, v in changes.items():
            r[k] = v
        return json.dumps(r)
    person = good["people"][0]
    short = dict(person, kpts=person["kpts"][:16])
    flipped = dict(person, box=[person["box"][2], person["box"][1], person["box"][0], person["box"][3]])
    far = dict(person, kpts=[[50000.0, 10.0, 0.9]] + person["kpts"][1:])
    now = time.time()
    return [("malformed", "{not json"),
            ("malformed", edit(frame="twelve")),
            ("unsupported_schema", edit(schema=2)),
            ("missing_camera", edit(camera_id="")),
            ("missing_time", edit(t=None)),
            ("bad_frame_size", edit(width=0)),
            ("missing_people", edit(people=None)),
            ("bad_person_shape", edit(people=[short])),
            ("bad_box", edit(people=[flipped])),
            ("keypoint_out_of_range", edit(people=[far])),
            ("too_late", edit(ts=now - 3600)),
            ("clock_skew", edit(ts=now + 3600))]


def main():
    import psycopg
    from confluent_kafka import Consumer, Producer
    load_dotenv()
    cfg = load_config()["streaming"]
    boot = cfg["bootstrap"]
    db = (f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@127.0.0.1:"
          f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'rover')}")
    records = read(FRAMES)
    camera, session = "smoke-test", uuid.uuid4().hex[:12]
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    # 1. invalid records -> dead-letter topic
    dlq = Consumer({"bootstrap.servers": boot, "group.id": f"smoke-{session}", "auto.offset.reset": "latest"})
    dlq.subscribe(["rover.pose-frames.dlq"])
    t_end = time.time() + 10
    while not dlq.assignment() and time.time() < t_end:           # join the group before sending
        dlq.poll(0.5)
    good = dict(records[60], camera_id=camera, session=session + "-bad", ts=round(time.time(), 3))
    bad = bad_records(good)
    p = Producer({"bootstrap.servers": boot})
    for _, value in bad:
        p.produce(cfg["topic"], key=camera, value=value)
    p.flush(10)
    reasons, t_end = [], time.time() + 60                          # quality query runs every 10 s
    while len(reasons) < len(bad) and time.time() < t_end:
        msg = dlq.poll(1.0)
        if msg is not None and not msg.error():
            d = json.loads(msg.value())
            if session in d["record"] or d["record"] == "{not json":
                reasons.append(d["reason"])
    dlq.close()
    expected = sorted(r for r, _ in bad)
    check("every invalid record reaches the dead-letter topic with its reason", sorted(reasons) == expected,
          f"got {sorted(reasons)}")

    # 2. fight clip -> predictions + alert
    t0 = time.time()
    for i, r in enumerate(records):
        r.update(camera_id=camera, session=session, ts=round(t0 + r["t"], 3))
        p.produce(cfg["topic"], key=camera, value=json.dumps(r))
        if i % 10 == 0:
            p.poll(0)
            time.sleep(0.05)
    p.flush(10)
    n_pred = n_alert = 0
    stored = None
    t_end = time.time() + 90
    with psycopg.connect(db, autocommit=True) as conn:
        while time.time() < t_end and (n_pred < 400 or stored is None):
            n_pred, n_alert = conn.execute("SELECT count(*), count(*) FILTER (WHERE alert) FROM predictions "
                                           "WHERE session = %s", (session,)).fetchone()
            stored = conn.execute("SELECT message, score, model_version FROM alerts WHERE rover_id = %s "
                                  "AND message LIKE '%%[stream]' AND received > to_timestamp(%s) ORDER BY id LIMIT 1",
                                  (camera, t0)).fetchone()
            time.sleep(2)
    check("Spark scored every close pair of every frame (predictions in PostgreSQL)", n_pred >= 400,
          f"{n_pred} judgements")
    check("exactly one FIGHT alert for the clip (like the edge)", n_alert == 1, f"{n_alert} alerts")
    check("the alert bridge stored it in the backend", stored is not None and "ID 5 and ID 9" in stored[0],
          f"{stored}" + (f", {time.time() - t0:.0f} s after the first frame" if stored else ""))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
