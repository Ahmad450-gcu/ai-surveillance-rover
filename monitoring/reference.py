"""Pipeline stage "drift_reference": what the training data looks like, for data-drift monitoring.

    python -m monitoring.reference             # params.yaml "monitoring"; writes reports/drift_reference.json
    python -m monitoring.reference --upload    # also s3://processed/monitoring/drift_reference.json (read by the monitor)

It reads the validated pose clips of the TRAIN split (the data the model learned from) and stores, for each drift
feature (monitoring/drift.py), its decile bins and the share of values in each. The drift monitor compares live
frames with exactly this. It is a DVC stage, so the reference is rebuilt and versioned whenever the training data
changes: a model and the reference of its data always belong together.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monitoring.drift import FEATURES, collect, describe  # noqa: E402

OUT = ROOT / "reports" / "drift_reference.json"
LAKE_KEY = "monitoring/drift_reference.json"            # bucket "processed"


def clip_frames(path):
    """A validated clip (npz from extract_pose) -> (kpts, boxes, height) for every frame."""
    d = np.load(path)
    frame, height = d["frame"], int(d["height"])
    for f in range(int(d["n_frames"])):
        sel = frame == f
        yield d["kpts"][sel], d["box"][sel], height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "params.yaml").read_text())["monitoring"]
    clips = sorted((ROOT / cfg["data"] / "train").glob("*/*.npz"))
    rng = np.random.default_rng(0)
    if cfg.get("max_clips") and len(clips) > cfg["max_clips"]:
        clips = sorted(rng.choice(clips, cfg["max_clips"], replace=False).tolist())
    t0 = time.time()
    values = collect(f for c in clips for f in clip_frames(c))
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text()) if (ROOT / "dvc.lock").exists() else {}
    validated = next((o["md5"] for o in lock.get("stages", {}).get("validate", {}).get("outs", [])
                      if o["path"].startswith(cfg["data"])), None)
    ref = {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": f"{cfg['data']}/train",
           "clips": len(clips), "data_md5": validated, "bins": cfg["bins"],
           "features": {f: describe(values[f], cfg["bins"]) for f in FEATURES}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ref, indent=1) + "\n")
    print(f"drift reference from {len(clips)} training clips in {time.time() - t0:.0f} s -> "
          f"{OUT.relative_to(ROOT)}")
    for f in FEATURES:
        r = ref["features"][f]
        print(f"  {f:15s} n={r['n']:7d}  mean {r['mean']:.3f}  std {r['std']:.3f}")
    if args.upload:
        from mlops import lake
        lake.client().put_object(Bucket="processed", Key=LAKE_KEY, Body=OUT.read_bytes(),
                                 ContentType="application/json")
        print(f"lake: s3://processed/{LAKE_KEY}")


if __name__ == "__main__":
    main()
