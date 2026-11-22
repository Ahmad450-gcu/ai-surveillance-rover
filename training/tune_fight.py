"""Hyperparameter search (Optuna) for a fight model architecture, logged to MLflow.

    python training/tune_fight.py --arch gru --trials 30
    python training/tune_fight.py --arch tcn --trials 30 --timeout 90      # stop after 90 minutes

Objective: cross-validated ROC AUC of whole clips on the TRAIN split only, each clip scored like live detection
(the test split is never looked at, so the final test stays an unbiased estimate).
Every trial is a nested MLflow run under one parent run; trials that are clearly worse than the median are
stopped early (pruning). The best settings are written to reports/tuning_<arch>.json; copy them into params.yaml
(fight_model) and run  dvc repro train_fight  to train, test and register the model.
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
from training.features import load_features  # noqa: E402
from training.fightlib import arch_config, clip_score, roc_auc, seed_all  # noqa: E402
from training.train_fight import cross_validate  # noqa: E402

SPACES = {
    "gru": lambda t: {"hidden": t.suggest_categorical("hidden", [32, 64, 96, 128]),
                      "layers": t.suggest_int("layers", 1, 3),
                      "dropout": t.suggest_float("dropout", 0.0, 0.5)},
    "tcn": lambda t: {"channels": t.suggest_categorical("channels", [32, 64, 96, 128]),
                      "levels": t.suggest_int("levels", 2, 4),
                      "kernel": t.suggest_categorical("kernel", [3, 5]),
                      "dropout": t.suggest_float("dropout", 0.0, 0.5)},
    "stgcn": lambda t: {"channels": t.suggest_categorical("channels", [16, 32, 48]),
                        "blocks": t.suggest_int("blocks", 2, 4),
                        "kernel": t.suggest_categorical("kernel", [3, 5]),
                        "dropout": t.suggest_float("dropout", 0.0, 0.5)},
}


def suggest(trial, base, kind):
    P = json.loads(json.dumps(base))
    P["arch"] = kind
    P[kind] = SPACES[kind](trial)
    P["lr"] = trial.suggest_float("lr", 2e-4, 5e-3, log=True)
    P["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    P["batch"] = trial.suggest_categorical("batch", [64, 128, 256])
    P["augment"] = {"noise": trial.suggest_float("noise", 0.0, 0.15),
                    "joint_dropout": trial.suggest_float("joint_dropout", 0.0, 0.2),
                    "time_stretch": trial.suggest_float("time_stretch", 0.0, 0.5)}
    return P


def main():
    import optuna

    from mlops import registry
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(SPACES), default="gru")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=float, help="minutes")
    ap.add_argument("--folds", type=int, default=3, help="CV folds per trial (fewer = faster)")
    ap.add_argument("--epochs", type=int, default=25, help="max epochs per fit")
    args = ap.parse_args()

    base = yaml.safe_load((ROOT / "params.yaml").read_text())["fight_model"]
    base.update(folds=args.folds, epochs=args.epochs, patience=min(base["patience"], 6))
    data = load_features()
    labels = np.array([c["label"] for c in data["index"]["clips"]])
    n_train = data["index"]["n_train"]
    train_ids = np.random.default_rng(base["seed"]).permutation(n_train)
    y = labels[train_ids]

    mlflow, _ = registry.connect()
    mlflow.set_experiment("fight-model-tuning")
    t0 = time.time()

    def objective(trial):
        P = suggest(trial, base, args.arch)
        seed_all(P["seed"])
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params({"arch": args.arch, **trial.params})
            try:
                probs, aucs = cross_validate(P, arch_config(P), data, train_ids, labels, log=False, trial=trial)
            except optuna.TrialPruned:
                mlflow.set_tag("state", "pruned")
                raise
            has = np.array([p is not None for p in probs])
            scores = np.array([clip_score(p, 2) for p in probs])
            auc = roc_auc(y[has], scores[has])
            mlflow.log_metrics({"cv_clip_auc": auc, "cv_window_auc": float(np.mean(aucs))})
        print(f"trial {trial.number:3d}: clip AUC {auc:.4f} (window AUC {np.mean(aucs):.4f}) "
              f"{trial.params} | {(time.time() - t0) / 60:.0f} min", flush=True)
        return auc

    study = optuna.create_study(direction="maximize", study_name=f"fight-{args.arch}",
                                sampler=optuna.samplers.TPESampler(seed=base["seed"]),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with mlflow.start_run(run_name=f"tune-{args.arch}-{time.strftime('%m%d-%H%M')}", tags={"arch": args.arch}):
        mlflow.log_params({"arch": args.arch, "trials": args.trials, "folds": args.folds, "epochs": args.epochs})
        study.optimize(objective, n_trials=args.trials, timeout=args.timeout * 60 if args.timeout else None)
        best = study.best_trial
        mlflow.log_metric("best_cv_clip_auc", best.value)
        mlflow.log_params({f"best.{k}": v for k, v in best.params.items()})
        P = suggest(optuna.trial.FixedTrial(best.params), base, args.arch)
        result = dict(arch=args.arch, best_cv_clip_auc=round(best.value, 4), best_trial=best.number,
                      params={args.arch: P[args.arch], "lr": P["lr"], "weight_decay": P["weight_decay"],
                              "batch": P["batch"], "augment": P["augment"]},
                      trials=[dict(number=t.number, state=t.state.name, value=t.value, params=t.params)
                              for t in study.trials], minutes=round((time.time() - t0) / 60, 1))
        out = ROOT / "reports" / f"tuning_{args.arch}.json"
        out.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(out))
    done = sum(t.state.name == "COMPLETE" for t in study.trials)
    pruned = sum(t.state.name == "PRUNED" for t in study.trials)
    print(f"\nbest {args.arch}: clip AUC {best.value:.4f} (trial {best.number}; {done} complete, {pruned} pruned)")
    print("params.yaml fight_model settings:\n" + yaml.safe_dump(result["params"], sort_keys=False))


if __name__ == "__main__":
    main()
