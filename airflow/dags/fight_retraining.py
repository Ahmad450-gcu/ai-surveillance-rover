"""Retraining pipeline of the fight model (Airflow DAG "fight_retraining").

When it runs
    * event: the drift monitor triggers it when Alertmanager reports drift (DataDriftDetected, PredictionDrift,
      FalseAlarmRateHigh -> a row in retrain_requests -> POST /api/v1/dags/fight_retraining/dagRuns)
    * daily at 03:00, it checks whether retraining is due: a pending retraining request (e.g. Airflow was down
      when it was created), enough new labelled alerts since the last training, or the model is older than a week
    * by hand: Airflow UI "Trigger DAG" (conf {"force": true} skips the check)

Steps (each one a task: own logs, retries, timing in the Airflow UI)
    decide           -> why retrain? (stop here if there is no reason); marks the requests "running"
    export_feedback  -> labelled stream alerts from the backend -> data/manifests/feedback_labels.json
    build_feedback   -> DVC stage feedback: labelled clips from the stream frames in the lake
    features         -> DVC stage features: training windows from RWF-2000 + the feedback clips
    train            -> DVC stage train_fight: CV, training, test; the champion is re-scored on the same test clips;
                        new registry version = challenger
    drift_reference  -> DVC stage drift_reference: the new training data becomes the drift monitor's reference
    promote          -> promotion gate (params.yaml registry): challenger -> champion only if clearly better
    verify_serving   -> if promoted: the serving API hot-reloads the new champion (Phase 4) -- wait until it does
    push_data        -> dvc push: every new data version is stored in the lake (bucket dvc)
    finish           -> retraining requests "done" with the decision; reports/retraining_last.json

Everything runs in /opt/rover (the project, mounted) with /opt/trainer (training environment). DVC only reruns
what changed. The new dvc.lock, the feedback manifest and the reports are left in the project folder, ready to be
committed (they record exactly which data and code produced the new model).
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator

PROJECT = Path(os.environ.get("PROJECT_DIR", "/opt/rover"))
PY = os.environ.get("TRAINER", "/opt/trainer/bin") + "/python"
DVC = os.environ.get("TRAINER", "/opt/trainer/bin") + "/dvc"
DB_URL = os.environ.get("ROVER_DATABASE_URL", "")
MIN_NEW_LABELS = int(os.environ.get("RETRAIN_MIN_NEW_LABELS", "5"))
MAX_AGE_DAYS = float(os.environ.get("RETRAIN_MAX_AGE_DAYS", "7"))
SERVING_URL = os.environ.get("SERVING_URL", "http://serving:8000/v1/models/fight")
LAST = PROJECT / "reports" / "retraining_last.json"


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


def dvc_stage(stage, timeout_min):
    # --single-item: only this stage (its deps are checked, upstream stages such as the 12 GB raw videos are not)
    # PATH: the stages' commands ("python training/...") must use the training environment, not Airflow's Python
    return BashOperator(task_id=stage if stage != "train_fight" else "train",
                        bash_command=f"export PATH={Path(PY).parent}:$PATH && cd {PROJECT} && "
                                     f"{DVC} repro --single-item {stage}",
                        execution_timeout=timedelta(minutes=timeout_min), append_env=True,
                        env={"PYTHONUNBUFFERED": "1"})


@dag(dag_id="fight_retraining", schedule="0 3 * * *", start_date=datetime(2026, 11, 1), catchup=False,
     max_active_runs=1, tags=["rover", "retraining"], on_failure_callback=on_failure,
     default_args={"retries": 1, "retry_delay": timedelta(minutes=2), "on_failure_callback": on_failure},
     doc_md=__doc__)
def fight_retraining():

    @task
    def decide(**context):
        """Reasons to retrain now; none -> the rest of the run is skipped."""
        conf = context["dag_run"].conf or {}
        reasons, ids = [], []
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('retrain_requests') IS NOT NULL")
            if cur.fetchone()[0]:
                cur.execute("UPDATE retrain_requests SET status = 'running', run_id = %s "
                            "WHERE status = 'pending' RETURNING id, reason", (context["run_id"],))
                rows = cur.fetchall()
                ids = [r[0] for r in rows]
                reasons += sorted({f"alert: {r[1]}" for r in rows})
            cur.execute("SELECT count(*) FROM alerts WHERE type = 'fight' AND label IN ('real', 'false_alarm') "
                        "AND evidence LIKE 'stream://%%'")
            labelled = cur.fetchone()[0]
        manifest = PROJECT / "data" / "manifests" / "feedback_labels.json"
        used = len(json.loads(manifest.read_text())["alerts"]) if manifest.exists() else 0
        if labelled - used >= MIN_NEW_LABELS:
            reasons.append(f"feedback: {labelled - used} new labelled alerts")
        if LAST.exists():
            age = datetime.now(timezone.utc) - datetime.fromisoformat(json.loads(LAST.read_text())["finished"])
            if age > timedelta(days=MAX_AGE_DAYS):
                reasons.append(f"scheduled: last retraining {age.days} days ago")
        if conf.get("force"):
            reasons.append("manual")
        context["ti"].xcom_push(key="request_ids", value=ids)
        if not reasons:
            raise AirflowSkipException("nothing to do: no retraining request, too few new labels, model is recent")
        print("retraining because:", "; ".join(reasons))
        return reasons

    export_feedback = BashOperator(task_id="export_feedback", append_env=True,
                                   bash_command=f"cd {PROJECT} && {PY} -m training.feedback export",
                                   env={"DATABASE_URL": DB_URL})

    @task
    def promote():
        import subprocess
        subprocess.run([PY, "-m", "mlops.registry", "promote"], cwd=PROJECT, check=True)
        return json.loads((PROJECT / "reports" / "promotion.json").read_text())

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
                info = json.loads(urllib.request.urlopen(SERVING_URL, timeout=5).read())
                if str(info.get("version")) == want:
                    print(f"serving runs v{want} ({info.get('arch')}, threshold {info.get('threshold')})")
                    return info
            except OSError as e:
                print("serving not reachable:", e)
            time.sleep(5)
        raise RuntimeError(f"serving still runs v{info.get('version')} instead of v{want}")

    # none_failed_min_one_success: runs after a promotion and after a rejection, but not when decide found nothing to do
    push_data = BashOperator(task_id="push_data", trigger_rule="none_failed_min_one_success",
                             bash_command=f"cd {PROJECT} && {DVC} push -r lake-docker")

    @task(trigger_rule="none_failed_min_one_success")
    def finish(reasons, decision, **context):
        ids = context["ti"].xcom_pull(task_ids="decide", key="request_ids") or []
        summary = {"run_id": context["run_id"], "reasons": reasons, "promoted": decision.get("promoted"),
                   "challenger": decision.get("challenger"), "champion_before": decision.get("champion"),
                   "metric": decision.get("metric"), "challenger_metric": decision.get("challenger_metric"),
                   "champion_metric": decision.get("champion_metric"), "compared_on": decision.get("compared_on"),
                   "reason": decision.get("reason"), "finished": datetime.now(timezone.utc).isoformat()}
        mark_requests(ids, "done", context["run_id"], summary)
        LAST.write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))
        return summary

    reasons = decide()
    build_feedback = dvc_stage("feedback", 30)
    features = dvc_stage("features", 60)
    train = dvc_stage("train_fight", 120)
    reference = dvc_stage("drift_reference", 30)
    decision = promote()
    reasons >> export_feedback >> build_feedback >> features >> train >> reference >> decision
    verified = verify_serving(decision)
    [decision, verified] >> push_data
    push_data >> finish(reasons, decision)


fight_retraining()
