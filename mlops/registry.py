"""Model registry (MLflow): versions, the champion / challenger aliases, promotion and deployment.

    python -m mlops.registry list                       # every version of the fight model with its test metrics
    python -m mlops.registry promote                    # challenger -> champion if it is better (params.yaml registry)
    python -m mlops.registry pull                       # download the champion into models/ (what the rover runs)
    python -m mlops.registry pull --alias challenger --dest models/try
    python -m mlops.registry release                    # champion -> models/release/ (in git, baked into the serving
                                                        # image: what a freshly deployed server starts with)
    python -m mlops.registry seed                       # empty registry (new server) -> models/release becomes v1 @champion

Lifecycle of a model version:
    train_fight.py  -> new version, alias "challenger" (MLflow run with params, metrics, plots, data version)
    promote         -> compares challenger vs champion on the same test clips (the challenger's training run re-scores
                       the champion on its test set, which matters once new data was added); on success the alias "champion"
                       moves (the previous champion keeps its version, so a rollback is one command:
                       python -m mlops.registry set-alias champion <version>)
    pull            -> models/fight_model.{onnx,pt,json}, used by run.py (and later the serving API)
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACT_DIR = "fight_model"        # run artifact folder holding model.onnx, model.pt, model.json


def params():
    return yaml.safe_load((ROOT / "params.yaml").read_text())["registry"]


def connect():
    import mlflow

    from rover_ai.settings import load_dotenv
    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI") or (ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    if uri.startswith("http"):                      # artifacts go through the server to the lake (bucket mlflow)
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "300")
    return mlflow, mlflow.MlflowClient()


def version_by_alias(client, name, alias):
    from mlflow.exceptions import MlflowException
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def run_metrics(client, version):
    return client.get_run(version.run_id).data.metrics


def register(mlflow, client, onnx_path, name, tags):
    """Inside an active run: log the ONNX model, register it as a new version and make it the challenger."""
    import numpy as np
    import onnx
    from mlflow.models import infer_signature
    model = onnx.load(str(onnx_path))
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]     # [batch, T, F]
    example = np.zeros((1, dims[1], dims[2]), np.float32)
    info = mlflow.onnx.log_model(model, artifact_path="model", registered_model_name=name, await_registration_for=120,
                                 signature=infer_signature(example, np.zeros(1, np.float32)))
    version = info.registered_model_version or client.search_model_versions(
        f"name='{name}' and run_id='{mlflow.active_run().info.run_id}'")[0].version
    for k, v in tags.items():
        client.set_model_version_tag(name, version, k, str(v))
    client.set_registered_model_alias(name, "challenger", version)
    return int(version)


def promote(name=None, force=False, report=ROOT / "reports" / "promotion.json"):
    """Move "champion" to the challenger if it wins. Returns the decision (also written to reports/promotion.json)."""
    cfg = params()
    name = name or cfg["model_name"]
    _, client = connect()
    challenger = version_by_alias(client, name, "challenger")
    champion = version_by_alias(client, name, "champion")
    if challenger is None:
        sys.exit(f"{name}: no version has the alias 'challenger' (train a model first)")
    metric = cfg["metric"]
    new = run_metrics(client, challenger)
    decision = dict(model=name, challenger=int(challenger.version), metric=metric,
                    challenger_metric=new.get(f"test_{metric}"), challenger_auc=new.get("test_roc_auc"),
                    time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if champion is None:
        promote_ok, reason = True, "no champion yet"
    elif champion.version == challenger.version:
        promote_ok, reason = False, "the challenger already is the champion"
    else:
        old = run_metrics(client, champion)
        compared = client.get_run(challenger.run_id).data.tags.get("champion_version_compared")
        if compared == str(champion.version) and f"champion_test_{metric}" in new:
            # the challenger's run re-scored this champion on the challenger's test clips: same data, fair comparison
            old = {f"test_{metric}": new[f"champion_test_{metric}"], "test_roc_auc": new.get("champion_test_roc_auc")}
            decision["compared_on"] = "challenger's test set (champion re-scored)"
        else:
            decision["compared_on"] = "each version's own test set"
        decision.update(champion=int(champion.version), champion_metric=old.get(f"test_{metric}"),
                        champion_auc=old.get("test_roc_auc"))
        gain = new.get(f"test_{metric}", 0) - old.get(f"test_{metric}", 0)
        auc_drop = old.get("test_roc_auc", 0) - new.get("test_roc_auc", 0)
        if gain < cfg["min_improvement"]:
            promote_ok, reason = False, f"{metric} {gain:+.4f} (needs +{cfg['min_improvement']})"
        elif auc_drop > cfg["max_auc_drop"]:
            promote_ok, reason = False, f"{metric} {gain:+.4f} but ROC AUC dropped {auc_drop:.4f}"
        else:
            promote_ok, reason = True, f"{metric} {gain:+.4f}, ROC AUC {-auc_drop:+.4f}"
    if force and not promote_ok:
        promote_ok, reason = True, f"forced ({reason})"
    decision.update(promoted=promote_ok, reason=reason)
    client.set_model_version_tag(name, challenger.version, "promotion", f"{'promoted' if promote_ok else 'rejected'}: {reason}")
    if promote_ok:
        if champion is not None and champion.version != challenger.version:
            client.set_model_version_tag(name, champion.version, "retired", decision["time"])
        client.set_registered_model_alias(name, "champion", challenger.version)
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps(decision, indent=2) + "\n")
    print(f"{name} v{challenger.version}: {'PROMOTED to champion' if promote_ok else 'not promoted'} ({reason})")
    return decision


def pull(name=None, alias="champion", dest="models", stem="fight_model"):
    """Download a version's model files to <dest>/<stem>.{onnx,pt,json}."""
    mlflow, client = connect()
    name = name or params()["model_name"]
    v = version_by_alias(client, name, alias)
    if v is None:
        sys.exit(f"{name}: no version has the alias {alias!r}")
    dest = ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(run_id=v.run_id, artifact_path=stem, dst_path=tmp))
        for f in local.iterdir():
            shutil.copy2(f, dest / f"{stem}{f.suffix}")
    meta = json.loads((dest / f"{stem}.json").read_text())
    meta["registry"] = dict(model=name, version=int(v.version), alias=alias, run_id=v.run_id)
    (dest / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"{name} v{v.version} ({alias}, {meta['arch']['type']}) -> {(dest / stem).relative_to(ROOT)}.onnx/.pt/.json")
    return int(v.version)


def release(dest="models/release"):
    """The champion -> models/release/fight_model.{onnx,json}: committed with the code that ships it, copied into the
    serving image (fallback while a new server's registry is empty) and the source of `seed`."""
    version = pull(dest=dest)
    (ROOT / dest / "fight_model.pt").unlink(missing_ok=True)       # serving and seeding need the ONNX model only
    return version


def seed(src="models/release", name=None):
    """A registry without a champion (a freshly deployed server) gets the release model as its champion, so the first
    retraining there has a model to beat instead of being promoted unchecked. Does nothing if there is a champion."""
    mlflow, client = connect()
    name = name or params()["model_name"]
    current = version_by_alias(client, name, "champion")
    if current is not None:
        print(f"{name}: v{current.version} is the champion already; nothing to seed")
        return None
    src = ROOT / src
    meta = json.loads((src / "fight_model.json").read_text())
    mlflow.set_experiment("fight-model")
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name="seed-release"):
        for f in ("onnx", "json"):
            shutil.copy(src / f"fight_model.{f}", Path(tmp) / f"model.{f}")
        mlflow.set_tags({"source": "release-bundle", "origin": json.dumps(meta.get("registry", {}))})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in (meta.get("test_learned") or {}).items()
                            if isinstance(v, (int, float))})
        mlflow.log_artifacts(tmp, artifact_path=ARTIFACT_DIR)
        version = register(mlflow, client, Path(tmp) / "model.onnx", name, tags={
            "arch": meta["arch"]["type"], "source": "release-bundle",
            "origin_version": meta.get("registry", {}).get("version", "?")})
    client.set_registered_model_alias(name, "champion", version)
    client.delete_registered_model_alias(name, "challenger")
    print(f"{name}: release model (origin v{meta.get('registry', {}).get('version', '?')}) seeded as v{version} @champion")
    return version


def list_versions(name=None):
    _, client = connect()
    name = name or params()["model_name"]
    versions = sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))
    aliases = {}
    for alias in ("champion", "challenger"):
        v = version_by_alias(client, name, alias)
        if v:
            aliases.setdefault(v.version, []).append(alias)
    print(f"{'ver':>3}  {'arch':6} {'test f1':>7} {'auc':>6} {'recall':>6}  aliases / promotion")
    for v in versions:
        m = run_metrics(client, v)
        print(f"{v.version:>3}  {v.tags.get('arch', '?'):6} {m.get('test_f1', float('nan')):7.4f} "
              f"{m.get('test_roc_auc', float('nan')):6.4f} {m.get('test_recall', float('nan')):6.4f}  "
              f"{','.join(aliases.get(v.version, []))} {v.tags.get('promotion', '')}")


def model_of(activity):
    """activities.yaml: activity -> (registered model name, promotion report path)."""
    from rover_ai import activities
    acts = activities.load()
    if activity not in acts or not acts[activity].is_model:
        sys.exit(f"{activity!r} is not a model activity in activities.yaml: "
                 f"{', '.join(activities.models(acts)) or 'none'}")
    report = ROOT / "reports" / ("promotion.json" if activity == "fight" else f"promotion_{activity}.json")
    return acts[activity].model.registered_name, report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity", default="fight", help="model activity of activities.yaml (default: fight)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("promote")
    p.add_argument("--force", action="store_true", help="promote even if the challenger is not better")
    p = sub.add_parser("pull")
    p.add_argument("--alias", default="champion")
    p.add_argument("--dest", default="models")
    sub.add_parser("release")
    sub.add_parser("seed")
    p = sub.add_parser("set-alias", help="e.g. roll back: set-alias champion 1")
    p.add_argument("alias")
    p.add_argument("version")
    args = ap.parse_args(argv)
    name, report = model_of(args.activity)
    if args.cmd == "list":
        list_versions(name)
    elif args.cmd == "promote":
        promote(name, force=args.force, report=report)
    elif args.cmd == "pull":
        pull(name, alias=args.alias, dest=args.dest, stem=f"{args.activity}_model")
    elif args.cmd == "release":
        release()
    elif args.cmd == "seed":
        seed()
    else:
        _, client = connect()
        client.set_registered_model_alias(name, args.alias, args.version)
        print(f"{name}: alias {args.alias} -> v{args.version}")


if __name__ == "__main__":
    main()
