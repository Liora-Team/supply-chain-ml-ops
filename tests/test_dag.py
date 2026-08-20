"""DagBag sanity checks for the Card 3.1 data DAG.

Airflow is only in the opt-in `orchestration` group (it runs in its own container), so
these tests skip wherever it isn't installed. CI runs them in a dedicated step that syncs
that group (.github/workflows/ci.yml); run locally with `uv sync --group orchestration`.
"""

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
