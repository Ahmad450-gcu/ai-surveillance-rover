"""Pipeline stage "train_fight": feature windows -> trained fight model, evaluated, registered in MLflow.

    python training/train_fight.py                    # settings: params.yaml section "fight_model"
    python training/train_fight.py --arch stgcn       # quick experiment with another architecture
    python training/train_fight.py --no-register      # do not create a registry version

Data   (from the "features" stage, data/processed/fight/rwf2000)
    RWF-2000 train split (1600 clips) -> 5-fold cross-validation chooses the alert rule (threshold + judgements
                                         in a row); the final model is fitted on 90% with 10% dev for early stopping
    RWF-2000 val split   (400 clips)  -> final test, used once, never for training or tuning; scored like live
Output
    models/candidate/model.{pt,onnx,json}   weights, ONNX export (checked against PyTorch), settings + metrics
    reports/fight_metrics.json, reports/*.png
    MLflow: run in experiment "fight-model" + new version of the registered model, alias "challenger"
    (python -m mlops.registry promote  then decides whether it replaces the champion)
"""
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rover_ai.learned import FPS, N_FEAT, PAIR_GAP  # noqa: E402
from rover_ai.nets import build_model, count_params  # noqa: E402
from training.features import load_features  # noqa: E402
from training.fightlib import (  # noqa: E402
    arch_config,
    choose_rule,
    clip_score,
    metrics,
    pair_probs,
    seed_all,
    train_model,
    windows_of,
)

ALL = yaml.safe_load((ROOT / "params.yaml").read_text())
CANDIDATE = ROOT / "models" / "candidate"


def cross_validate(P, arch, data, train_ids, labels, log=True, trial=None):
    """K-fold CV on the train split: every train clip is scored by a model that never saw it.
    Returns out-of-fold pair probabilities per clip and the dev AUC of each fold."""
    K = P["folds"]
    fold = np.arange(len(train_ids)) % K
    oof, aucs = {}, []
    t0 = time.time()
    for k in range(K):
        rest = train_ids[fold != k]
        n_dev = len(rest) // 10
        model, hist, auc = train_model(P, arch, *windows_of(data["train"], rest[n_dev:]),
                                       *windows_of(data["train"], rest[:n_dev]), log=False)
        aucs.append(auc)
        oof.update(pair_probs(model, data["eval"], train_ids[fold == k]))
        if log:
            print(f"fold {k + 1}/{K}: {len(hist)} epochs, dev window AUC {auc:.3f} | {time.time() - t0:.0f}s", flush=True)
        if trial is not None:
            import optuna
            trial.report(float(np.mean(aucs)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    probs = [oof.get(int(i)) for i in train_ids]
    return probs, aucs


def rules_baseline(paths):
    """Replay the test skeletons through the original hand-tuned rules (rover_ai/fight.py)."""
    from rover_ai.fight import FightDetector
    from rover_ai.tracks import PeopleState
    from rover_ai.vision import Person
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["fight"]
    preds = []
    for path in paths:
        d = np.load(ROOT / path)
        people, det, fired = PeopleState(), FightDetector(cfg), False
        for f in range(int(d["n_frames"])):
            sel = d["frame"] == f
            persons = [Person(int(i), b, float(c), k) for i, b, c, k in
                       zip(d["track"][sel], d["box"][sel], d["conf"][sel], d["kpts"][sel])]
            t = f / FPS
            people.update(persons, t)
            ev, _ = det.update(people.visible(t), t, defaultdict(set))
            fired = fired or bool(ev)
        preds.append(fired)
    return np.array(preds)


def export_onnx(model, path, T):
    """Export to ONNX (dynamic batch) and check that ONNX Runtime gives the same output as PyTorch."""
    import onnxruntime as ort
    dummy = torch.randn(4, T, N_FEAT)
    torch.onnx.export(model, dummy, str(path), input_names=["window"], output_names=["logit"],
                      dynamic_axes={"window": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17)
    x = torch.randn(64, T, N_FEAT)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"window": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    if diff > 1e-3:
        raise SystemExit(f"ONNX export differs from PyTorch by {diff}")
    return diff


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def data_version():
    """Which data the model was trained on: dataset checksum + DVC hash of the feature windows."""
    out = {}
    manifest = ROOT / "data" / "manifests" / "rwf2000.json"
    if manifest.exists():
        out["dataset_sha256"] = json.loads(manifest.read_text())["sha256"]
    lock = ROOT / "dvc.lock"
    if lock.exists():
        stages = yaml.safe_load(lock.read_text()).get("stages", {})
        for o in stages.get("features", {}).get("outs", []):
            out["features_md5"] = o.get("md5")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["gru", "tcn", "stgcn"], help="override fight_model.arch")
    ap.add_argument("--no-register", action="store_true", help="log to MLflow but do not create a registry version")
    ap.add_argument("--run-name", help="MLflow run name")
    args = ap.parse_args()
    P = dict(ALL["fight_model"])
    if args.arch:
        P["arch"] = args.arch
    arch = arch_config(P)
    seed_all(P["seed"])
    t0 = time.time()
    data = load_features()
    index, fcfg = data["index"], data["index"]["settings"]
    clips = index["clips"]
    labels = np.array([c["label"] for c in clips])
    n_train = index["n_train"]
    train_ids = np.random.default_rng(P["seed"]).permutation(n_train)
    test_ids = np.arange(n_train, len(clips))
    print(f"{arch['type']}: {count_params(build_model(arch)):,} parameters | "
          f"clips train {n_train}, test {len(test_ids)} | windows {len(data['train'][1])} | {time.time() - t0:.0f}s")

    # 1. cross-validation on the train split -> alert rule (judgements in a row + threshold)
    oof, fold_aucs = cross_validate(P, arch, data, train_ids, labels)
    need, thr, cv_scores = choose_rule(oof, labels[train_ids], P["consecutive_choices"])
    has = np.array([p is not None for p in oof])
    cv = metrics(labels[train_ids][has], cv_scores[has] >= thr, cv_scores[has])
    print(f"alert rule: {need} judgements in a row >= {thr:.2f} | CV (train clips with a close pair): {cv}")

    # 2. final model: 90% fit, 10% dev for early stopping
    n_dev = n_train // 10
    model, history, dev_auc = train_model(P, arch, *windows_of(data["train"], train_ids[n_dev:]),
                                          *windows_of(data["train"], train_ids[:n_dev]))

    # 3. final test on RWF-2000 val (never seen during training or rule choice), scored like live
    probs = pair_probs(model, data["eval"], test_ids)
    test_scores = np.array([clip_score(probs.get(int(i)), need) for i in test_ids])
    has_pair = np.array([int(i) in probs for i in test_ids])
    test_y = labels[test_ids]
    rule_pred = rules_baseline([clips[i]["path"] for i in test_ids])
    learned = metrics(test_y, test_scores >= thr, test_scores)
    rules = metrics(test_y, rule_pred)
    learned_pair = metrics(test_y[has_pair], test_scores[has_pair] >= thr, test_scores[has_pair])
    rules_pair = metrics(test_y[has_pair], rule_pred[has_pair])
    fight = test_y.astype(bool)
    coverage = dict(test_clips=len(test_ids), with_close_pair=int(has_pair.sum()),
                    fight_with_pair=int((fight & has_pair).sum()), nonfight_with_pair=int((~fight & has_pair).sum()))
    print("TEST learned:", learned)
    print("TEST rules  :", rules)
    print("coverage:", coverage)
    print("TEST (clips with a close pair) learned:", learned_pair)
    print("TEST (clips with a close pair) rules  :", rules_pair)

    # 4. candidate model files
    CANDIDATE.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CANDIDATE / "model.pt")
    onnx_diff = export_onnx(model, CANDIDATE / "model.onnx", fcfg["window"])
    minutes = (time.time() - t0) / 60
    meta = dict(arch=arch, n_params=count_params(model), window=fcfg["window"], fps=FPS, threshold=thr,
                consecutive=need, pair_gap=PAIR_GAP, proximity=fcfg["proximity"],
                trained_on="RWF-2000 train split (pose from YOLO11n-pose + ByteTrack)", data_version=data_version(),
                clips=dict(train=n_train, fit=n_train - n_dev, dev=n_dev, test=len(test_ids), folds=P["folds"]),
                windows=int(len(windows_of(data["train"], train_ids[n_dev:])[1])), augment=P.get("augment"),
                cv_with_pair=cv, cv_fold_dev_auc=[round(a, 4) for a in fold_aucs], test_learned=learned,
                test_rules=rules, coverage=coverage, test_learned_with_pair=learned_pair,
                test_rules_with_pair=rules_pair, onnx_max_diff=onnx_diff, git_commit=git_commit(),
                train_minutes=round(minutes, 1), created=time.strftime("%Y-%m-%d %H:%M"))
    (CANDIDATE / "model.json").write_text(json.dumps(meta, indent=2))

    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    (rep / "fight_metrics.json").write_text(json.dumps(
        dict(arch=arch["type"], threshold=thr, consecutive=need, cv_with_pair=cv, test_learned=learned,
             test_rules=rules, coverage=coverage, test_learned_with_pair=learned_pair, test_rules_with_pair=rules_pair,
             dev_window_auc=dev_auc, test_fight_clips_without_close_pair=int((fight & ~has_pair).sum()),
             history=history), indent=2))
    plots(rep, history, test_y[has_pair], test_scores[has_pair], thr, learned_pair, rules_pair, arch["type"].upper())
    version = log_mlflow(P, meta, history, register=not args.no_register, run_name=args.run_name)
    print(f"saved {CANDIDATE.relative_to(ROOT)}/ and reports/ in {minutes:.1f} min"
          + (f" | registry: version {version} = challenger" if version else ""))


def log_mlflow(P, meta, history, register=True, run_name=None):
    """MLflow run (experiment "fight-model"): params, per-epoch curves, CV + test metrics, plots, model files;
    plus a new registry version with the alias "challenger". Tracking server from MLFLOW_TRACKING_URI (.env),
    or a local ./mlruns folder."""
    from mlops import registry
    mlflow, client = registry.connect()
    mlflow.set_experiment("fight-model")
    flat = {"arch": P["arch"], **{f"{P['arch']}.{k}": v for k, v in P[P["arch"]].items()},
            **{f"augment.{k}": v for k, v in (P.get("augment") or {}).items()},
            **{k: v for k, v in P.items() if not isinstance(v, dict) and k != "arch"}}
    tags = {"git_commit": meta["git_commit"], "activity": "fight", "arch": P["arch"], **meta["data_version"]}
    with mlflow.start_run(run_name=run_name or f"{P['arch']}-{time.strftime('%m%d-%H%M')}", tags=tags):
        mlflow.log_params(flat)
        for h in history:
            mlflow.log_metrics({"loss": h["loss"], "dev_auc": h["dev_auc"]}, step=h["epoch"])
        m = {f"test_{k}": v for k, v in meta["test_learned"].items()}
        m.update({f"test_pair_{k}": v for k, v in meta["test_learned_with_pair"].items()})
        m.update({f"cv_{k}": v for k, v in meta["cv_with_pair"].items()})
        m.update(threshold=meta["threshold"], consecutive=meta["consecutive"], n_params=meta["n_params"],
                 cv_dev_auc_mean=float(np.mean(meta["cv_fold_dev_auc"])), train_minutes=meta["train_minutes"],
                 onnx_max_diff=meta["onnx_max_diff"])
        mlflow.log_metrics({k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and v == v})
        mlflow.log_artifacts(str(CANDIDATE), artifact_path=registry.ARTIFACT_DIR)
        for f in ("fight_metrics.json", "fight_training.png", "fight_roc.png", "fight_confusion.png"):
            mlflow.log_artifact(str(ROOT / "reports" / f), artifact_path="reports")
        if not register:
            return None
        name = ALL["registry"]["model_name"]
        return registry.register(mlflow, client, CANDIDATE / "model.onnx", name, tags={
            "arch": P["arch"], "test_f1": meta["test_learned"]["f1"], "test_roc_auc": meta["test_learned"]["roc_auc"],
            "threshold": meta["threshold"], "consecutive": meta["consecutive"], **meta["data_version"]})


# ----------------------------------------------------------------------------- plots
def plot_training(rep, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.9))
    a1.plot(ep, [h["loss"] for h in history], color="#2a6fdb", lw=2)
    a1.set_title("Training loss", fontsize=10); a1.set_xlabel("epoch")
    a2.plot(ep, [h["dev_auc"] for h in history], color="#2a6fdb", lw=2)
    best = max(history, key=lambda h: h["dev_auc"])
    a2.scatter([best["epoch"]], [best["dev_auc"]], color="#e07b00", zorder=3, s=40)
    a2.annotate(f"kept: epoch {best['epoch']}", (best["epoch"], best["dev_auc"]), textcoords="offset points",
                xytext=(8, -4), fontsize=9, va="top")
    a2.set_title("Dev window ROC AUC", fontsize=10); a2.set_xlabel("epoch")
    for a in (a1, a2):
        a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout(); fig.savefig(rep / "fight_training.png", dpi=200)
    plt.close(fig)


def plots(rep, history, y, scores, thr, learned, rules, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_training(rep, history)

    y, s = np.asarray(y, bool), np.asarray(scores)
    ths = np.unique(np.concatenate([[0, 1.01], s]))
    tpr = [((s >= t) & y).sum() / y.sum() for t in ths]
    fpr = [((s >= t) & ~y).sum() / (~y).sum() for t in ths]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, color="#2a6fdb", lw=2, label=f"trained {name} (AUC {learned['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    ax.scatter([rules["fp"] / max(rules["fp"] + rules["tn"], 1)], [rules["recall"]], color="#e07b00", zorder=3, s=40,
               label="hand-tuned rules")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.legend(loc="lower right", frameon=False)
    ax.set_title(f"Test clips with a close pair (n = {len(y)})", fontsize=10)
    fig.tight_layout(); fig.savefig(rep / "fight_roc.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, (title, m) in zip(axes, ((f"Trained {name}", learned), ("Hand-tuned rules", rules))):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum() / 2)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.sum() / 4 else "black")
        ax.set_xticks([0, 1], ["normal", "fight"]); ax.set_yticks([0, 1], ["normal", "fight"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(f"{title}\naccuracy {m['accuracy']:.1%}", fontsize=10)
    fig.suptitle(f"RWF-2000 test clips where two people were close together (n = {len(y)})", fontsize=10)
    fig.tight_layout(); fig.savefig(rep / "fight_confusion.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
