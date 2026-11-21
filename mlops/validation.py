"""Data quality checks for pose clips (one .npz per video clip, written by training/extract_pose.py).

Two levels, like a data contract:
  - clip level:    schema, shapes, value ranges, consistency. An ERROR rejects the clip (it goes to the
                   rejected zone with the reason); a WARNING keeps it but is counted in the report.
  - dataset level: class balance, train/test leakage, minimum size, reject rate. A failure stops the
                   pipeline (quality gate), so a model is never trained on broken data.

The per-clip statistics are also aggregated into a data profile: the reference distribution that the
drift monitor later compares live data against.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

REQUIRED = ("frame", "track", "box", "conf", "kpts", "n_frames", "width", "height")
SPLITS = ("train", "val")
LABELS = ("Fight", "NonFight")


@dataclass
class ClipReport:
    path: str
    split: str
    label: str
    errors: list = field(default_factory=list)      # [(code, message)]
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def valid(self):
        return not self.errors

    def to_dict(self):
        return {"path": self.path, "split": self.split, "label": self.label, "valid": self.valid,
                "errors": [{"code": c, "message": m} for c, m in self.errors],
                "warnings": [{"code": c, "message": m} for c, m in self.warnings], "stats": self.stats}


def validate_clip(data, split, label, path="", rules=None):
    """data: mapping with the npz arrays. rules: params.yaml section data_validation."""
    rules = rules or {}
    r = ClipReport(path, split, label)
    err, warn = r.errors.append, r.warnings.append

    if split not in SPLITS:
        err(("bad_split", f"split '{split}' is not one of {SPLITS}"))
    if label not in LABELS:
        err(("bad_label", f"label '{label}' is not one of {LABELS}"))
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        err(("missing_fields", f"missing {missing}"))
        return r

    frame, track, box = np.asarray(data["frame"]), np.asarray(data["track"]), np.asarray(data["box"])
    conf, kpts = np.asarray(data["conf"]), np.asarray(data["kpts"])
    n_frames, width, height = int(data["n_frames"]), float(data["width"]), float(data["height"])
    n = len(frame)

    # ---- schema / shapes
    if not (len(track) == len(conf) == len(box) == len(kpts) == n):
        err(("length_mismatch", f"frame {n}, track {len(track)}, box {len(box)}, conf {len(conf)}, kpts {len(kpts)}"))
        return r
    if n and (box.shape[1:] != (4,) or kpts.shape[1:] != (17, 3)):
        err(("bad_shape", f"box {box.shape}, kpts {kpts.shape} (expected (N,4) and (N,17,3))"))
        return r

    # ---- clip-level values
    min_frames = rules.get("min_frames", 20)
    if n_frames < min_frames:
        err(("too_short", f"{n_frames} frames at 10 fps, need at least {min_frames}"))
    if width <= 0 or height <= 0:
        err(("bad_frame_size", f"{width}x{height}"))
    if n:
        if not (np.isfinite(box).all() and np.isfinite(kpts).all() and np.isfinite(conf).all()):
            err(("non_finite", "NaN or infinite values"))
            return r
        if frame.min() < 0 or frame.max() >= n_frames:
            err(("frame_out_of_range", f"frame index {frame.min()}..{frame.max()} outside 0..{n_frames - 1}"))
        if conf.min() < 0 or conf.max() > 1 or kpts[..., 2].min() < 0 or kpts[..., 2].max() > 1:
            err(("confidence_out_of_range", "detection or keypoint confidence outside [0, 1]"))
        if ((box[:, 2] <= box[:, 0]) | (box[:, 3] <= box[:, 1])).any():
            err(("invalid_box", "box with x2 <= x1 or y2 <= y1"))
        tol = rules.get("box_tolerance", 0.05)
        if width > 0 and height > 0:
            outside = ((box[:, [0, 2]] < -tol * width) | (box[:, [0, 2]] > (1 + tol) * width)).any() or \
                      ((box[:, [1, 3]] < -tol * height) | (box[:, [1, 3]] > (1 + tol) * height)).any()
            if outside:
                err(("box_outside_frame", f"box more than {tol:.0%} outside the {width:.0f}x{height:.0f} frame"))
        pairs = frame.astype(np.int64) * 1_000_000 + track.astype(np.int64)
        if len(np.unique(pairs)) != n:
            err(("duplicate_detection", "the same track appears twice in one frame"))

    # ---- statistics (also the data profile for drift monitoring)
    per_frame = np.bincount(frame.astype(int), minlength=max(n_frames, 1)) if n else np.zeros(max(n_frames, 1))
    vis = kpts[..., 2] if n else np.zeros((0, 17))
    r.stats = {
        "n_frames": n_frames,
        "n_detections": n,
        "n_tracks": int(len(np.unique(track))) if n else 0,
        "people_per_frame": float(per_frame.mean()),
        "max_people": int(per_frame.max()) if n else 0,
        "detection_conf": float(conf.mean()) if n else 0.0,
        "keypoint_conf": float(vis.mean()) if n else 0.0,
        "box_height_frac": float(np.median((box[:, 3] - box[:, 1]) / height)) if n and height > 0 else 0.0,
        "width": width, "height": height,
    }

    # ---- warnings (kept, but reported)
    if n == 0:
        warn(("no_people", "no person detected in the whole clip"))
    elif r.stats["max_people"] < 2:
        warn(("no_pair", "never two people at once (the fight model needs a pair)"))
    if n and r.stats["keypoint_conf"] < rules.get("low_keypoint_conf", 0.3):
        warn(("low_keypoint_conf", f"mean keypoint confidence {r.stats['keypoint_conf']:.2f}"))
    return r


def check_dataset(reports, rules=None):
    """Dataset-level gate. Returns (passed, checks) where checks is a list of dicts."""
    rules = rules or {}
    checks = []

    def add(name, ok, detail, blocking=True):
        checks.append({"check": name, "passed": bool(ok), "blocking": blocking, "detail": detail})

    total = len(reports)
    rejected = sum(not r.valid for r in reports)
    max_reject = rules.get("max_reject_rate", 0.05)
    rate = rejected / total if total else 1.0
    add("reject_rate", total and rate <= max_reject, f"{rejected}/{total} rejected ({rate:.1%}, limit {max_reject:.0%})")

    valid = [r for r in reports if r.valid]
    counts = Counter((r.split, r.label) for r in valid)
    min_clips = rules.get("min_clips_per_class", 50)
    for split in SPLITS:
        for label in LABELS:
            add(f"min_clips[{split}/{label}]", counts[(split, label)] >= min_clips,
                f"{counts[(split, label)]} valid clips (need {min_clips})")
        n_split = counts[(split, "Fight")] + counts[(split, "NonFight")]
        frac = counts[(split, "Fight")] / n_split if n_split else 0.0
        lo, hi = rules.get("class_balance", [0.3, 0.7])
        add(f"class_balance[{split}]", lo <= frac <= hi, f"Fight share {frac:.1%} (allowed {lo:.0%}-{hi:.0%})")

    names = defaultdict(set)
    for r in valid:
        names[r.split].add(r.path.replace("\\", "/").rsplit("/", 1)[-1])
    leaked = names["train"] & names["val"]
    add("no_train_test_leakage", not leaked, f"{len(leaked)} clip(s) in both train and val" +
        (f", e.g. {sorted(leaked)[:3]}" if leaked else ""))

    passed = all(c["passed"] for c in checks if c["blocking"])
    return passed, checks


PROFILE_STATS = ("people_per_frame", "max_people", "detection_conf", "keypoint_conf", "box_height_frac", "n_frames")


def profile(reports, bins=10):
    """Reference distribution of the validated data: quantile bin edges + bin shares per statistic.
    The drift monitor (Phase 6) bins live data with the same edges and computes the PSI."""
    valid = [r for r in reports if r.valid and r.stats.get("n_detections", 0) > 0]
    out = {"clips": len(valid), "stats": {}}
    for name in PROFILE_STATS:
        values = np.array([r.stats[name] for r in valid], dtype=float)
        if not len(values):
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        hist = np.histogram(values, bins=edges)[0] if len(edges) > 1 else np.array([len(values)])
        out["stats"][name] = {"mean": float(values.mean()), "std": float(values.std()),
                              "edges": edges.round(6).tolist(), "shares": (hist / hist.sum()).round(6).tolist()}
    return out
