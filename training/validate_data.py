"""Pipeline stage "validate": raw pose clips -> validated zone (+ rejected zone), quality report, data profile.

    python training/validate_data.py              # settings: params.yaml section "data_validation"
    python training/validate_data.py --upload     # also write the zones to the data lake

Input   data/raw/pose/<dataset>/<split>/<Fight|NonFight>/<clip>.npz         (stage extract_pose)
Output  data/validated/pose/<dataset>/...                                   valid clips (training input)
        reports/data_validation.json     summary: counts, reject reasons, dataset checks (DVC metric)
        reports/data_validation_clips.jsonl   one line per clip with its errors/warnings/stats
        reports/data_profile.json        reference distribution for drift monitoring
Lake    s3://validated/pose/<dataset>/...  and  s3://rejected/pose/<dataset>/<clip>.npz + .reason.json

Exits with an error if a blocking dataset check fails (quality gate): nothing downstream is retrained.
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlops.validation import check_dataset, profile, validate_clip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="write validated / rejected zones to the lake")
    ap.add_argument("--input", help="override data_validation.input (e.g. a small sample)")
    ap.add_argument("--output", help="override data_validation.output")
    ap.add_argument("--reports", default="reports", help="folder for the reports")
    args = ap.parse_args()
    rules = yaml.safe_load((ROOT / "params.yaml").read_text())["data_validation"]
    src, dst = ROOT / (args.input or rules["input"]), ROOT / (args.output or rules["output"])
    dataset = src.name

    reports = []
    for path in sorted(src.glob("*/*/*.npz")):
        split, label = path.parts[-3], path.parts[-2]
        rel = path.relative_to(src).as_posix()
        try:
            with np.load(path) as d:
                rep = validate_clip({k: d[k] for k in d.files}, split, label, rel, rules)
        except Exception as e:  # noqa: BLE001 - an unreadable file is a data error, not a crash
            from mlops.validation import ClipReport
            rep = ClipReport(rel, split, label, errors=[("unreadable", f"{type(e).__name__}: {e}")])
        reports.append(rep)
    if not reports:
        sys.exit(f"no clips found under {src.relative_to(ROOT)} (run the extract_pose stage first)")

    # validated zone: a clean copy of the valid clips (rebuilt every run, so removed clips disappear)
    if dst.exists():
        shutil.rmtree(dst)
    for r in reports:
        if r.valid:
            out = dst / r.path
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / r.path, out)

    passed, checks = check_dataset(reports, rules)
    reasons = Counter(c for r in reports for c, _ in r.errors)
    warnings = Counter(c for r in reports for c, _ in r.warnings)
    per_split = Counter(f"{r.split}/{r.label}" for r in reports if r.valid)
    summary = {
        "dataset": dataset,
        "clips": len(reports),
        "valid": sum(r.valid for r in reports),
        "rejected": sum(not r.valid for r in reports),
        "valid_per_split": dict(sorted(per_split.items())),
        "reject_reasons": dict(reasons.most_common()),
        "warnings": dict(warnings.most_common()),
        "passed": passed,
        "checks": checks,
    }
    reports_dir = ROOT / args.reports
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "data_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (reports_dir / "data_validation_clips.jsonl").open("w") as f:
        for r in reports:
            f.write(json.dumps(r.to_dict()) + "\n")
    (reports_dir / "data_profile.json").write_text(json.dumps({"dataset": dataset, **profile(reports)}, indent=2) + "\n")

    print(f"{dataset}: {summary['valid']} valid, {summary['rejected']} rejected of {summary['clips']} clips")
    for reason, n in reasons.most_common():
        print(f"  rejected  {reason}: {n}")
    for w, n in warnings.most_common():
        print(f"  warning   {w}: {n}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")

    if args.upload:
        upload(reports, src, dst, dataset)
    if not passed:
        sys.exit("data quality gate FAILED: fix the data (see reports/data_validation.json) before training")


def upload(reports, src, dst, dataset):
    from mlops import lake
    prefix = f"pose/{dataset}"
    up, same = lake.sync_dir(dst, "validated", prefix, "*/*/*.npz")
    print(f"lake: s3://validated/{prefix}/  {up} uploaded, {same} unchanged")
    bad = [r for r in reports if not r.valid]
    for r in bad:
        key = f"{prefix}/{r.path}"
        if (src / r.path).exists():
            lake.client().upload_file(str(src / r.path), "rejected", key)
        lake.put_json("rejected", key + ".reason.json", r.to_dict())
    print(f"lake: s3://rejected/{prefix}/  {len(bad)} clip(s) with reasons")


if __name__ == "__main__":
    main()
