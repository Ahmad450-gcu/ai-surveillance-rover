"""Stateful fight scoring of a frame stream (one camera session at a time), used by the Spark job.

It runs the SAME code as live detection on the laptop (rover_ai.tracks.PeopleState + rover_ai.learned.
LearnedFightDetector), fed from the frame records instead of the camera, with the model behind the serving API.
So the stream's judgements equal the edge's for the same frames.

Between micro-batches the state is plain JSON (Spark keeps it in its fault-tolerant state store):
    frames    the last KEEP_SECONDS of frame records (enough to rebuild every track's 2 s history)
    det       each pair's recent judgements (alert rule: N in a row above the threshold)
    last_t    newest frame time processed (older or repeated frames are skipped)
    alerted   pair -> stream time of its last alert, and "*" -> the camera's last fight alert of any pair
              (the edge's two cooldowns, measured in stream time so results don't depend on processing speed)
"""
import json
from collections import defaultdict

from rover_ai.streaming import persons_from_record
from rover_ai.tracks import PeopleState

KEEP_SECONDS = 3.0
ALERT_COOLDOWN = 10.0            # same pair: alerts.cooldown_seconds on the edge
TYPE_COOLDOWN = 10.0             # any pair of this camera: alerts.type_cooldown_seconds (3 people fighting = 1 alert)


def new_state():
    return {"frames": [], "det": {}, "last_t": None, "alerted": {}}


def score_frames(detector, state, records, cooldown=ALERT_COOLDOWN, type_cooldown=TYPE_COOLDOWN):
    """records: frame dicts of one camera session (any order). -> (predictions, alerts, state, skipped)"""
    state = state or new_state()
    people = PeopleState()
    for rec in state["frames"]:                          # rebuild the tracks (no judging)
        people.update(persons_from_record(rec), rec["t"])
    detector.set_state(state["det"])
    last_t = state["last_t"]
    predictions, alerts, kept, skipped = [], [], list(state["frames"]), 0
    version = str(detector.meta.get("registry", {}).get("version"))
    for rec in sorted(records, key=lambda r: r["t"]):
        t = rec["t"]
        if last_t is not None and t <= last_t:           # duplicate or out of order
            skipped += 1
            continue
        people.update(persons_from_record(rec), t)
        events, _ = detector.update(people.visible(t), t, defaultdict(set))
        alerting = {tuple(e.track_ids) for e in events}
        for (a, b), p in detector.probs.items():
            key = f"{a}-{b}"
            alert = ((a, b) in alerting and t - state["alerted"].get(key, -1e9) >= cooldown
                     and t - state["alerted"].get("*", -1e9) >= type_cooldown)
            if alert:
                state["alerted"][key] = state["alerted"]["*"] = t
                alerts.append(dict(camera_id=rec["camera_id"], session=rec["session"], frame=rec["frame"], t=t,
                                   ts=rec["ts"], track_ids=[a, b], probability=p, model_version=version,
                                   message=f"FIGHT: ID {a} and ID {b} ({p:.0%})"))
            predictions.append(dict(camera_id=rec["camera_id"], session=rec["session"], frame=rec["frame"], t=t,
                                    ts=rec["ts"], person_a=a, person_b=b, probability=p,
                                    threshold=float(detector.thr), above=bool(p >= detector.thr), alert=alert,
                                    model_version=version))
        kept.append(rec)
        last_t = t
    if last_t is not None:
        kept = [r for r in kept if r["t"] >= last_t - KEEP_SECONDS]
        state["alerted"] = {k: v for k, v in state["alerted"].items() if last_t - v < max(cooldown, type_cooldown)}
    state.update(frames=kept, det=detector.get_state(), last_t=last_t)
    return predictions, alerts, state, skipped


def dumps(state):
    return json.dumps(state, separators=(",", ":"))


def loads(text):
    return json.loads(text) if text else new_state()
