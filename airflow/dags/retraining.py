"""Retraining pipelines: one Airflow DAG per model activity with a `retraining` section in activities.yaml
(today: "fight_retraining"). A new model activity gets its own DAG by adding that section; nothing here changes.

When a DAG runs
    * event: the drift monitor triggers it when Alertmanager reports drift (DataDriftDetected, PredictionDrift,
      FalseAlarmRateHigh -> a row in retrain_requests for this activity -> POST /api/v1/dags/<dag>/dagRuns)
    * on its schedule (daily at 03:00) it checks whether retraining is due: a pending retraining request (e.g. Airflow
      was down when it was created), enough new labelled alerts since the last training, or the model is too old
    * by hand: Airflow UI "Trigger DAG" (conf {"force": true} skips the check)

Steps (each one a task: own logs, retries, timing in the Airflow UI); for the fight model:
    decide           -> why retrain? (stop here if there is no reason); marks the requests "running"
    export_feedback  -> retraining.prepare: labelled stream alerts from the backend -> data/manifests/feedback_labels.json
    feedback         -> DVC stage feedback: labelled clips from the stream frames in the lake
    features         -> DVC stage features: training windows from RWF-2000 + the feedback clips
    train            -> DVC stage train_fight (model.train_stage): CV, training, test; the champion is re-scored on the
                        same test clips; new registry version = challenger
    drift_reference  -> DVC stage drift_reference: the new training data becomes the drift monitor's reference
    promote          -> promotion gate (params.yaml registry): challenger -> champion only if clearly better
    verify_serving   -> if promoted: the serving API hot-reloads the new champion -- wait until it does
    push_data        -> dvc push: every new data version is stored in the lake (bucket dvc)
    finish           -> retraining requests "done" with the decision; reports/retraining_last.json

Everything runs in /opt/rover (the project, mounted) with /opt/trainer (training environment). DVC only reruns
what changed. The new dvc.lock, the feedback manifest and the reports are left in the project folder, ready to be
committed (they record exactly which data and code produced the new model).
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator

PROJECT = Path(os.environ.get("PROJECT_DIR", "/opt/rover"))
sys.path.insert(0, str(PROJECT))
from rover_ai import activities  # noqa: E402  (the registry: which activities have a model to retrain)

PY = os.environ.get("TRAINER", "/opt/trainer/bin") + "/python"
DVC = os.environ.get("TRAINER", "/opt/trainer/bin") + "/dvc"
DB_URL = os.environ.get("ROVER_DATABASE_URL", "")
SERVING = os.environ.get("SERVING_URL", "http://serving:8000/v1/models/fight").rsplit("/v1/models", 1)[0]


def env_number(name, default):
    """RETRAIN_MIN_NEW_LABELS / RETRAIN_MAX_AGE_DAYS in .env override activities.yaml for every activity."""
    v = os.environ.get(name, "").strip()
    return float(v) if v else float(default)


def connect():
    import psycopg2
    return psycopg2.connect(DB_URL.replace("postgresql+psycopg://", "postgresql://"))


def mark_requests(ids, status, run_id, decision=None):
    if not ids:
        return
    with connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE retrain_requests SET status = %s, run_id = %s, handled_at = CASE WHEN %s IN "
                    "('done', 'failed') THEN now() ELSE handled_at END, details = details || %s::jsonb "
                    "WHERE id = ANY(%s)", (status, run_id, status, json.dumps({"decision": decision} if decision
                                                                               else {}), list(ids)))


def on_failure(context):
    """A failed step: the requests of this run become "failed" (the next drift alert creates a new one)."""
    ids = context["ti"].xcom_pull(task_ids="decide", key="request_ids") or []
    mark_requests(ids, "failed", context["run_id"], {"failed_task": context["ti"].task_id})


def dvc_stage(stage, train_stage, timeout_min):
    # --single-item: only this stage (its deps are checked, upstream stages such as the 12 GB raw videos are not)
    # PATH: the stages' commands ("python training/...") must use the training environment, not Airflow's Python
    return BashOperator(task_id="train" if stage == train_stage else stage,
                        bash_command=f"export PATH={Path(PY).parent}:$PATH && cd {PROJECT} && "
                                     f"{DVC} repro --single-item {stage}",
                        execution_timeout=timedelta(minutes=timeout_min), append_env=True,
                        env={"PYTHONUNBUFFERED": "1"})


def build(act):
    """The retraining DAG of one model activity."""
    spec, model = act.retraining, act.model
    suffix = "" if act.name == "fight" else f"_{act.name}"          # fight keeps its original file names
    last = PROJECT / "reports" / f"retraining_last{suffix}.json"
    promotion = PROJECT / "reports" / f"promotion{suffix}.json"
    manifest = PROJECT / "data" / "manifests" / f"feedback_labels{suffix}.json"
    min_labels = env_number("RETRAIN_MIN_NEW_LABELS", spec.min_new_labels)
    max_age = env_number("RETRAIN_MAX_AGE_DAYS", spec.max_age_days)
    serving_url = f"{SERVING}/v1/models/{act.name}"

    @dag(dag_id=spec.dag, schedule=spec.schedule, start_date=datetime(2026, 11, 1), catchup=False,
         max_active_runs=1, tags=["rover", "retraining", act.name], on_failure_callback=on_failure,
         default_args={"retries": 1, "retry_delay": timedelta(minutes=2), "on_failure_callback": on_failure},
         doc_md=f"**{act.title}** ({model.registered_name}): {act.description}\n\n" + __doc__)
    def retraining():

        @task
        def decide(**context):
            """Reasons to retrain now; none -> the rest of the run is skipped."""
            conf = context["dag_run"].conf or {}
            reasons, ids = [], []
            with connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT to_regclass('retrain_requests') IS NOT NULL")
                if cur.fetchone()[0]:
                    cur.execute("ALTER TABLE retrain_requests ADD COLUMN IF NOT EXISTS activity TEXT NOT NULL "
                                "DEFAULT 'fight'")
                    cur.execute("UPDATE retrain_requests SET status = 'running', run_id = %s "
                                "WHERE status = 'pending' AND activity = %s RETURNING id, reason",
                                (context["run_id"], act.name))
                    rows = cur.fetchall()
                    ids = [r[0] for r in rows]
                    reasons += sorted({f"alert: {r[1]}" for r in rows})
                cur.execute("SELECT count(*) FROM alerts WHERE type = %s AND label IN ('real', 'false_alarm') "
                            "AND evidence LIKE 'stream://%%'", (act.alert_type,))
                labelled = cur.fetchone()[0]
            used = len(json.loads(manifest.read_text())["alerts"]) if manifest.exists() else 0
            if labelled - used >= min_labels:
                reasons.append(f"feedback: {labelled - used} new labelled alerts")
            if last.exists():
                age = datetime.now(timezone.utc) - datetime.fromisoformat(json.loads(last.read_text())["finished"])
                if age > timedelta(days=max_age):
                    reasons.append(f"scheduled: last retraining {age.days} days ago")
            if conf.get("force"):
                reasons.append("manual")
            context["ti"].xcom_push(key="request_ids", value=ids)
            if not reasons:
                raise AirflowSkipException("nothing to do: no retraining request, too few new labels, model is recent")
            print(f"retraining {act.name} because:", "; ".join(reasons))
            return reasons

        @task
        def promote():
            import subprocess
            subprocess.run([PY, "-m", "mlops.registry", "--activity", act.name, "promote"], cwd=PROJECT, check=True)
            return json.loads(promotion.read_text())

        @task
        def verify_serving(decision):
            """The serving API follows @champion by itself (it polls the registry): wait until it serves the new one."""
            import time
            import urllib.request
            if not decision.get("promoted"):
                raise AirflowSkipException("not promoted: serving keeps the current champion")
            want, info = str(decision["challenger"]), {}
            for _ in range(36):                                   # up to 3 minutes (serving checks every 60 s)
                try:
                    info = json.loads(urllib.request.urlopen(serving_url, timeout=5).read())
                    if str(info.get("version")) == want:
                        print(f"serving runs v{want} ({info.get('arch')}, threshold {info.get('threshold')})")
                        return info
                except OSError as e:
                    print("serving not reachable:", e)
                time.sleep(5)
            raise RuntimeError(f"serving still runs v{info.get('version')} instead of v{want}")

        # none_failed_min_one_success: runs after a promotion and after a rejection, not when decide found nothing
        push_data = BashOperator(task_id="push_data", trigger_rule="none_failed_min_one_success",
                                 bash_command=f"cd {PROJECT} && {DVC} push -r lake-docker")

        @task(trigger_rule="none_failed_min_one_success")
        def finish(reasons, decision, **context):
            ids = context["ti"].xcom_pull(task_ids="decide", key="request_ids") or []
            summary = {"activity": act.name, "run_id": context["run_id"], "reasons": reasons,
                       "promoted": decision.get("promoted"), "challenger": decision.get("challenger"),
                       "champion_before": decision.get("champion"), "metric": decision.get("metric"),
                       "challenger_metric": decision.get("challenger_metric"),
                       "champion_metric": decision.get("champion_metric"), "compared_on": decision.get("compared_on"),
                       "reason": decision.get("reason"), "finished": datetime.now(timezone.utc).isoformat()}
            mark_requests(ids, "done", context["run_id"], summary)
            last.write_text(json.dumps(summary, indent=1) + "\n")
            print(json.dumps(summary, indent=1))
            return summary

        reasons = decide()
        steps = []
        if spec.prepare:
            steps.append(BashOperator(task_id="export_feedback", append_env=True,
                                      bash_command=f"cd {PROJECT} && {spec.prepare.replace('python ', PY + ' ', 1)}",
                                      env={"DATABASE_URL": DB_URL}))
        for stage in spec.stages:
            steps.append(dvc_stage(stage, model.train_stage, 120 if stage == model.train_stage else 60
                                   if stage == "features" else 30))
        decision = promote()
        chain = [reasons, *steps, decision]
        for a, b in zip(chain, chain[1:]):
            a >> b
        verified = verify_serving(decision)
        [decision, verified] >> push_data
        push_data >> finish(reasons, decision)

    return retraining()


for _act in activities.retrained():
    globals()[_act.retraining.dag] = build(_act)
