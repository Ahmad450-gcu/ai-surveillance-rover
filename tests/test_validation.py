"""Data quality rules (mlops/validation.py) on synthetic pose clips."""
import numpy as np

from mlops.validation import check_dataset, profile, validate_clip

RULES = {"min_frames": 20, "box_tolerance": 0.05, "low_keypoint_conf": 0.3, "max_reject_rate": 0.05,
         "min_clips_per_class": 2, "class_balance": [0.3, 0.7]}


def clip(n_frames=30, people=2, width=640, height=480, kp_conf=0.8):
    """Two people (track 1 and 2) standing side by side in every frame."""
    rows = [(f, t) for f in range(n_frames) for t in range(1, people + 1)]
    n = len(rows)
    box = np.array([[100 + 150 * t, 100, 200 + 150 * t, 400] for _, t in rows], np.float32).reshape(-1, 4)
    kpts = np.zeros((n, 17, 3), np.float32)
    kpts[..., 0] = (box[:, None, 0] + box[:, None, 2]) / 2
    kpts[..., 1] = 250
    kpts[..., 2] = kp_conf
    return {"frame": np.array([r[0] for r in rows], np.int16), "track": np.array([r[1] for r in rows], np.int32),
            "box": box, "conf": np.full(n, 0.9, np.float32), "kpts": kpts,
            "n_frames": n_frames, "width": width, "height": height}


def codes(report):
    return {c for c, _ in report.errors}


def test_good_clip_is_valid_with_stats():
    r = validate_clip(clip(), "train", "Fight", rules=RULES)
    assert r.valid and not r.warnings
    assert r.stats["max_people"] == 2 and r.stats["n_tracks"] == 2 and abs(r.stats["keypoint_conf"] - 0.8) < 1e-6


def test_rejections():
    d = clip(); d["kpts"][0, 0, 2] = 1.5
    assert "confidence_out_of_range" in codes(validate_clip(d, "train", "Fight", rules=RULES))
    d = clip(); d["box"][3] = [300, 100, 200, 400]
    assert "invalid_box" in codes(validate_clip(d, "train", "Fight", rules=RULES))
    d = clip(); d["box"][0, 2] = 2000
    assert "box_outside_frame" in codes(validate_clip(d, "train", "Fight", rules=RULES))
    d = clip(); d["kpts"][1, 1, 0] = np.nan
    assert "non_finite" in codes(validate_clip(d, "train", "Fight", rules=RULES))
    d = clip(); d["track"][1] = d["track"][0]
    assert "duplicate_detection" in codes(validate_clip(d, "train", "Fight", rules=RULES))
    d = clip(); d["conf"] = d["conf"][:-1]
    assert "length_mismatch" in codes(validate_clip(d, "train", "Fight", rules=RULES))
    assert "too_short" in codes(validate_clip(clip(n_frames=10), "train", "Fight", rules=RULES))
    assert "bad_label" in codes(validate_clip(clip(), "train", "Dancing", rules=RULES))
    d = clip(); del d["kpts"]
    assert "missing_fields" in codes(validate_clip(d, "train", "Fight", rules=RULES))


def test_warnings_keep_the_clip():
    r = validate_clip(clip(people=1, kp_conf=0.1), "val", "NonFight", rules=RULES)
    assert r.valid and {c for c, _ in r.warnings} == {"no_pair", "low_keypoint_conf"}
    r = validate_clip(clip(people=0), "val", "NonFight", rules=RULES)
    assert r.valid and {c for c, _ in r.warnings} == {"no_people"}


def make_reports(extra=()):
    reports = []
    for split in ("train", "val"):
        for label in ("Fight", "NonFight"):
            for i in range(3):
                reports.append(validate_clip(clip(), split, label, f"{split}/{label}/{split}_{label}_{i}.npz", RULES))
    return reports + list(extra)


def test_dataset_gate_passes_on_clean_data():
    passed, checks = check_dataset(make_reports(), RULES)
    assert passed, checks


def test_dataset_gate_catches_leakage_and_reject_rate():
    leak = validate_clip(clip(), "val", "Fight", "val/Fight/train_Fight_0.npz", RULES)
    passed, checks = check_dataset(make_reports([leak]), RULES)
    assert not passed and not next(c for c in checks if c["check"] == "no_train_test_leakage")["passed"]

    broken = [validate_clip(clip(n_frames=5), "train", "Fight", f"train/Fight/bad{i}.npz", RULES) for i in range(3)]
    passed, checks = check_dataset(make_reports(broken), RULES)
    assert not passed and not next(c for c in checks if c["check"] == "reject_rate")["passed"]


def test_profile_has_bins_that_sum_to_one():
    p = profile(make_reports())
    assert p["clips"] == 12
    for stat in p["stats"].values():
        assert abs(sum(stat["shares"]) - 1) < 1e-4
