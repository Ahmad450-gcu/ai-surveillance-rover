"""Data-drift features and the Population Stability Index (PSI), shared by the reference stage and the monitor.

The features describe what the camera sees, not the model, so they show *why* a model might get worse:

    kp_conf        mean keypoint confidence of a person      (darker, blurrier, more occluded -> lower)
    visible_joints share of a person's 17 joints detected    (occlusion, people cut off by the frame)
    body_size      body width / frame height                 (camera further away or closer, other lens)
    people         tracked people per frame                  (crowded vs empty scenes)
    pair_distance  closest pair's distance in body widths    (how close people stand)

PSI compares a live distribution with the reference, using the reference's deciles as bins:
    PSI = sum over bins (live% - ref%) * ln(live% / ref%)
    < 0.1 stable, 0.1 - 0.2 moderate shift, > 0.2 significant drift (the usual rule of thumb in model monitoring)
"""
import numpy as np

from rover_ai.geometry import KP_CONF
from rover_ai.learned import body_centers, body_scales

FEATURES = ("kp_conf", "visible_joints", "body_size", "people", "pair_distance")
PSI_MODERATE, PSI_DRIFT = 0.1, 0.2
EPS = 1e-4


def frame_features(kpts, boxes, height):
    """One frame: kpts (n,17,3), boxes (n,4) of its tracked people -> {feature: list of values}."""
    out = {f: [] for f in FEATURES}
    n = len(kpts)
    out["people"].append(float(n))
    if n == 0:
        return out
    k, b = np.asarray(kpts, np.float32), np.asarray(boxes, np.float32)
    out["kp_conf"] += k[:, :, 2].mean(1).tolist()
    out["visible_joints"] += (k[:, :, 2] > KP_CONF).mean(1).tolist()
    scales = body_scales(k, b)
    out["body_size"] += (scales / max(float(height), 1.0)).tolist()
    if n >= 2:
        c = body_centers(k, b)
        d = np.linalg.norm(c[:, None] - c[None], axis=2) / ((scales[:, None] + scales[None]) / 2)
        out["pair_distance"].append(float(d[np.triu_indices(n, 1)].min()))
    return out


def collect(frames):
    """frames: iterable of (kpts, boxes, height) -> {feature: np.array of values}"""
    acc = {f: [] for f in FEATURES}
    for kpts, boxes, height in frames:
        for f, v in frame_features(kpts, boxes, height).items():
            acc[f] += v
    return {f: np.asarray(v, np.float64) for f, v in acc.items()}


def frames_from_records(records):
    """Kafka / lake frame records (dicts, or rows with JSON 'people') -> (kpts, boxes, height) per frame."""
    import json
    for r in records:
        people = r["people"] if not isinstance(r["people"], str) else json.loads(r["people"])
        kpts = [p["kpts"] for p in people]
        boxes = [p["box"] for p in people]
        yield (np.asarray(kpts, np.float32).reshape(-1, 17, 3), np.asarray(boxes, np.float32).reshape(-1, 4),
               r["height"])


# ----------------------------------------------------------------------------------------------- reference + PSI
def describe(values, bins=10):
    """Reference distribution of one feature: decile edges and the share of values in each bin."""
    values = values[np.isfinite(values)]
    # bins from the stored (rounded) edges, so live values are binned exactly like the reference
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]).round(6))
    return {"edges": edges.tolist(), "proportions": proportions(values, edges).round(6).tolist(),
            "n": int(len(values)), "mean": round(float(values.mean()), 6), "std": round(float(values.std()), 6)}


def proportions(values, edges):
    idx = np.searchsorted(np.asarray(edges), values, side="right")       # bin 0 .. len(edges)
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def psi(reference, values):
    """PSI of `values` against a reference made by describe()."""
    expected = np.maximum(np.asarray(reference["proportions"]), EPS)
    actual = np.maximum(proportions(values[np.isfinite(values)], reference["edges"]), EPS)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def level(value):
    return "drift" if value > PSI_DRIFT else "moderate" if value > PSI_MODERATE else "stable"
