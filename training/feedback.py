"""Human feedback -> training data: alerts that people labelled in the dashboard / app become labelled clips.

    python -m training.feedback export     # labelled stream alerts (PostgreSQL) -> data/manifests/feedback_labels.json
    python -m training.feedback build      # pipeline stage "feedback": manifest -> data/validated/pose/rover/

A stream alert keeps a reference to the frames it was raised on (alerts.evidence = stream://camera/session/frame);
the stream archived those frames' skeletons in the lake (s3://validated/stream/pose). So a label turns the alert
into a training clip without any new video processing:

    label "real"        -> Fight      the model was right: more examples like this
    label "false_alarm" -> NonFight   the model was wrong: exactly the cases it needs to learn

Each clip is the alert's 2-second window plus context (CLIP_BEFORE s before, CLIP_AFTER s after), resampled to 10 fps,
in the same .npz format as training/extract_pose.py writes, so the features stage reads it like an RWF-2000 clip
(params.yaml features.extra_data). Copies go to the lake's labelled zone (s3://labelled/fight/rover/).

The manifest is small and kept in git: which alerts, with which labels, built which version of the training data.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "data" / "manifests" / "feedback_labels.json"
OUT = ROOT / "data" / "validated" / "pose" / "rover"
CLASS = {"real": "Fight", "false_alarm": "NonFight"}
CLIP_BEFORE, CLIP_AFTER, FPS = 4.0, 1.0, 10.0
MIN_FOR_TEST = 10            # with fewer labelled clips, all of them are used for training (too few to test on)


def rel(path):
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


# ----------------------------------------------------------------------------------------------- export
def export(db_url):
    """Labelled fight alerts that have stream evidence -> manifest (sorted, so an unchanged set gives the same file)."""
    import psycopg
    with psycopg.connect(db_url) as conn:
        rows = conn.execute("SELECT id, time, label, evidence, rover_id, score, model_version FROM alerts "
                            "WHERE type = 'fight' AND label IN ('real', 'false_alarm') AND evidence LIKE 'stream://%%' "
                            "ORDER BY id").fetchall()
    items = [dict(alert_id=r[0], time=r[1], label=r[2], evidence=r[3], camera=r[4], score=r[5], model_version=r[6])
             for r in rows]
    old = json.loads(MANIFEST.read_text())["alerts"] if MANIFEST.exists() else None
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if items != old:
        MANIFEST.write_text(json.dumps({"alerts": items}, indent=1) + "\n")
    counts = {lab: sum(i["label"] == lab for i in items) for lab in CLASS}
    print(f"feedback: {len(items)} labelled stream alerts ({counts}) -> {rel(MANIFEST)}"
          + ("" if items != old else " (unchanged)"))
    return items


# ----------------------------------------------------------------------------------------------- build
def parse_evidence(evidence):
    camera, session, frame = evidence[len("stream://"):].rsplit("/", 2)
    return camera, session, int(frame)


def session_frames(lake, camera, session):
    """All archived frames of one camera session: {frame: row} (rows: t, width, height, people JSON)."""
    import pyarrow.parquet as pq
    rows = {}
    for obj in lake.list_keys("validated", "stream/pose/"):
        if f"camera_id={camera}/" not in obj["Key"]:
            continue
        t = pq.read_table(io.BytesIO(lake.client().get_object(Bucket="validated", Key=obj["Key"])["Body"].read()),
                          columns=["session", "frame", "t", "width", "height", "people"])
        for r in t.to_pylist():
            if r["session"] == session:
                rows[r["frame"]] = r
    return rows


def clip_from_frames(rows, t_alert):
    """Frames around the alert -> extract_pose-style arrays at 10 fps (nearest archived frame for each tick)."""
    frames = sorted(rows.values(), key=lambda r: r["t"])
    times = np.array([r["t"] for r in frames])
    ticks = np.arange(t_alert - CLIP_BEFORE, t_alert + CLIP_AFTER + 1e-9, 1 / FPS)
    out = {"frame": [], "track": [], "box": [], "conf": [], "kpts": []}
    n = 0
    for tick in ticks:
        i = int(np.argmin(np.abs(times - tick)))
        if abs(times[i] - tick) > 0.6 / FPS:              # no frame close to this tick: skip it
            continue
        for p in json.loads(frames[i]["people"]):
            out["frame"].append(n)
            out["track"].append(p["id"])
            out["box"].append(p["box"])
            out["conf"].append(p["conf"])
            out["kpts"].append(p["kpts"])
        n += 1
    return dict(frame=np.array(out["frame"], np.int16), track=np.array(out["track"], np.int32),
                box=np.array(out["box"], np.float32).reshape(-1, 4), conf=np.array(out["conf"], np.float32),
                kpts=np.array(out["kpts"], np.float32).reshape(-1, 17, 3), n_frames=np.int32(n),
                width=np.int32(frames[0]["width"]), height=np.int32(frames[0]["height"]))


def split_of(alert_id, n_total):
    if n_total < MIN_FOR_TEST:
        return "train"
    return "val" if int(hashlib.sha1(str(alert_id).encode()).hexdigest(), 16) % 5 == 0 else "train"


def build(lake):
    items = json.loads(MANIFEST.read_text())["alerts"] if MANIFEST.exists() else []
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    built, skipped, cache = [], [], {}
    for it in items:
        camera, session, frame = parse_evidence(it["evidence"])
        if (camera, session) not in cache:
            cache[(camera, session)] = session_frames(lake, camera, session)
        rows = cache[(camera, session)]
        if frame not in rows:
            skipped.append(dict(alert_id=it["alert_id"], reason="frames not in the lake"))
            continue
        clip = clip_from_frames(rows, rows[frame]["t"])
        people_frames = np.bincount(clip["frame"], minlength=int(clip["n_frames"])) if len(clip["frame"]) else []
        if clip["n_frames"] < 20 or (np.asarray(people_frames) >= 2).sum() < 10:      # quality gate, like validate
            skipped.append(dict(alert_id=it["alert_id"], reason="too few frames with two people"))
            continue
        split = split_of(it["alert_id"], len(items))
        path = OUT / split / CLASS[it["label"]] / f"alert-{it['alert_id']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **clip)
        lake.client().put_object(Bucket="labelled", Key=f"fight/rover/{CLASS[it['label']]}/alert-{it['alert_id']}.npz",
                                 Body=path.read_bytes())
        built.append(dict(alert_id=it["alert_id"], label=it["label"], split=split, frames=int(clip["n_frames"])))
    summary = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), labelled=len(items), clips=len(built),
                   skipped=skipped, by_split_class={f"{s}/{c}": sum(b["split"] == s and CLASS[b["label"]] == c
                                                                   for b in built)
                                                    for s in ("train", "val") for c in CLASS.values()},
                   built=built)
    (OUT / "feedback.json").write_text(json.dumps(summary, indent=1) + "\n")      # the stage output is never empty
    print(f"feedback: {len(built)} labelled clips -> {rel(OUT)} {summary['by_split_class']}"
          + (f", {len(skipped)} skipped" if skipped else ""))
    return summary


def main():
    from rover_ai.settings import load_dotenv
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "build"])
    args = ap.parse_args()
    load_dotenv()
    if args.cmd == "export":
        url = os.environ.get("DATABASE_URL") or (
            f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@"
            f"{os.environ.get('POSTGRES_HOST', '127.0.0.1')}:{os.environ.get('POSTGRES_PORT', '5432')}/"
            f"{os.environ.get('POSTGRES_DB', 'rover')}")
        export(url.replace("postgresql+psycopg://", "postgresql://"))
    else:
        from mlops import lake
        build(lake)


if __name__ == "__main__":
    main()
