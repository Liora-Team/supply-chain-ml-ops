"""DagBag sanity checks for the Card 3.1 data DAG.

Airflow is only in the opt-in `orchestration` group (it runs in its own container), so
these tests skip wherever it isn't installed. CI runs them in a dedicated step that syncs
that group (.github/workflows/ci.yml); run locally with `uv sync --group orchestration`.
"""

from datetime import timedelta

import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder="dags")


def test_dagbag_has_no_import_errors(dagbag):
    assert dagbag.import_errors == {}


def test_data_pipeline_shape(dagbag):
    # dagbag.dags is the parse result; get_dag() would need the metadata DB.
    dag = dagbag.dags.get("data_pipeline")
    assert dag is not None
    assert {t.task_id for t in dag.tasks} == {"ingest", "preprocess", "dvc_version"}
    assert dag.get_task("preprocess").upstream_task_ids == {"ingest"}
    assert dag.get_task("dvc_version").upstream_task_ids == {"preprocess"}


def test_data_pipeline_is_manual_only_with_retries(dagbag):
    dag = dagbag.dags.get("data_pipeline")
    assert type(dag.timetable).__name__ == "NullTimetable"  # schedule=None → manual only
    for t in dag.tasks:
        assert t.retries == 2


def test_model_pipeline_shape(dagbag):
    dag = dagbag.dags.get("model_pipeline")

    assert dag is not None
    assert {t.task_id for t in dag.tasks} == {
        "pull_data",
        "train",
        "evaluate",
        "gate",
    }
    assert dag.get_task("train").upstream_task_ids == {"pull_data"}
    assert dag.get_task("evaluate").upstream_task_ids == {"train"}
    assert dag.get_task("gate").upstream_task_ids == {"evaluate"}


def test_model_pipeline_is_manual_only_with_retries_and_alerts(dagbag):
    dag = dagbag.dags.get("model_pipeline")

    assert type(dag.timetable).__name__ == "NullTimetable"

    for task in dag.tasks:
        assert task.retries == 2
        assert task.retry_delay == timedelta(minutes=2)
        assert [callback.__name__ for callback in task.on_failure_callback] == ["_alert"]


def test_auto_retrain_pipeline_shape(dagbag):
    dag = dagbag.dags.get("auto_retrain_pipeline")
    assert dag is not None
    assert {t.task_id for t in dag.tasks} == {
        "decide",
        "record_state",
        "trigger_data_pipeline",
        "trigger_model_pipeline",
    }
    assert dag.get_task("record_state").upstream_task_ids == {"decide"}
    assert dag.get_task("trigger_data_pipeline").upstream_task_ids == {"record_state"}
    assert dag.get_task("trigger_model_pipeline").upstream_task_ids == {"trigger_data_pipeline"}


def test_auto_retrain_pipeline_is_scheduled_single_run(dagbag):
    dag = dagbag.dags.get("auto_retrain_pipeline")
    assert type(dag.timetable).__name__ != "NullTimetable"  # poller, not manual
    assert dag.max_active_runs == 1
