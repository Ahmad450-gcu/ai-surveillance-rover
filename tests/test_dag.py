"""The retraining DAG imports cleanly and has the intended shape (runs where Airflow is installed: the Airflow image).

    docker compose --profile mlops exec -w /tmp airflow-scheduler python -m pytest /opt/rover/tests/test_dag.py -q -p no:cacheprovider
(CI runs it in a fresh container of the image, after "airflow db migrate".)
"""
from pathlib import Path

import pytest

pytest.importorskip("airflow.models")   # (the project's own airflow/ folder is not Airflow)
from airflow.models import DagBag  # noqa: E402

DAGS = Path(__file__).resolve().parents[1] / "airflow" / "dags"


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(DAGS), include_examples=False)
    assert bag.import_errors == {}
    return bag.get_dag("fight_retraining")


def test_steps_run_in_order(dag):
    order = ["decide", "export_feedback", "feedback", "features", "train", "drift_reference", "promote"]
    for a, b in zip(order, order[1:]):
        extra = {"finish"} if a == "decide" else set()           # finish reports decide's reasons
        assert dag.get_task(a).downstream_task_ids == {b} | extra, a
    assert dag.get_task("promote").downstream_task_ids == {"verify_serving", "push_data", "finish"}
    assert dag.get_task("push_data").downstream_task_ids == {"finish"}


def test_rejected_challenger_still_finishes(dag):
    # verify_serving is skipped when the gate rejects the challenger: data is still pushed, requests still closed
    for t in ("push_data", "finish"):
        assert dag.get_task(t).trigger_rule == "none_failed_min_one_success"


def test_schedule_and_safety(dag):
    assert dag.schedule_interval == "0 3 * * *" and not dag.catchup and dag.max_active_runs == 1
    assert dag.get_task("train").execution_timeout.total_seconds() == 120 * 60
    assert "repro --single-item train_fight" in dag.get_task("train").bash_command
    for stage in ("feedback", "features", "train", "drift_reference"):     # stages run in the training venv (torch)
        assert dag.get_task(stage).bash_command.startswith("export PATH=/opt/trainer/bin:"), stage
    assert all(t.on_failure_callback for t in dag.tasks)          # a failed step marks the requests failed
