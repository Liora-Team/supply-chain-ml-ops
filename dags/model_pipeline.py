"""Model half of the ML pipeline (Card 3.2).

Consumes the versioned data/processed/ output from Card 3.1, runs the
existing MLflow-tracked training implementation, validates the candidate
metric, and promotes only when macro-F1 strictly beats Production.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from datetime import timedelta

from airflow.sdk import dag, task

logger = logging.getLogger(__name__)


def _alert(context):
    # Simplest orchestrator-native alert (no SMTP configured): loud log line + red task in UI.
    print(f"[ALERT] task {context['task_instance'].task_id} failed after retries - see log above")


default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": _alert,
}


EXPERIMENT_NAME = "trustpilot-reviews"


def _require_tracking_uri() -> str:
    """Return the configured MLflow URI or fail the task clearly."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must be set for the model pipeline.")
    return tracking_uri


def _find_candidate_run(
    client,
    *,
    schema: str,
    algorithm: str,
    started_after_ms: int,
) -> str:
    """Find the candidate MLflow run created by this training execution."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {EXPERIMENT_NAME!r} does not exist.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )

    for run in runs:
        if run.info.start_time < started_after_ms:
            break

        if run.data.tags.get("schema") == schema and run.data.tags.get("algorithm") == algorithm:
            return run.info.run_id

    raise RuntimeError(
        f"No new MLflow run found for schema={schema!r}, " f"algorithm={algorithm!r}."
    )


@dag(
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["model", "card-3.2"],
    params={
        "schema": "3-class",
        "algorithm": "LogReg",
        "train_subset": None,
    },
)
def model_pipeline():
    """Train, evaluate, and conditionally promote one candidate model."""

    @task.bash
    def pull_data() -> str:
        """Restore the versioned processed train/test data."""
        return "cd /app && " "dvc pull data/processed/train.csv.dvc " "data/processed/test.csv.dvc"

    @task
    def train(**context) -> int:
        """Run the existing MLflow-tracked training implementation."""
        started_after_ms = int(time.time() * 1000)

        tracking_uri = _require_tracking_uri()
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = tracking_uri
        env["MLFLOW_REQUIRED"] = "1"

        train_subset = context["params"].get("train_subset")

        if train_subset is not None:
            env["TRAIN_SUBSET"] = str(train_subset)
        else:
            env.pop("TRAIN_SUBSET", None)

        subprocess.run(
            ["uv", "run", "--no-sync", "python", "scripts/build_pipelines.py"],
            cwd="/app",
            check=True,
            env=env,
        )

        return started_after_ms

    @task
    def evaluate(started_after_ms: int, **context) -> str:
        """Locate and validate the candidate MLflow run."""
        import mlflow

        from src.promotion import get_run_macro_f1

        schema = str(context["params"]["schema"])
        algorithm = str(context["params"]["algorithm"])

        mlflow.set_tracking_uri(_require_tracking_uri())

        client = mlflow.MlflowClient()

        run_id = _find_candidate_run(
            client,
            schema=schema,
            algorithm=algorithm,
            started_after_ms=started_after_ms,
        )

        candidate_score = get_run_macro_f1(client, run_id)

        logger.info(
            "Candidate evaluation: run_id=%s schema=%s algorithm=%s macro_f1=%s",
            run_id,
            schema,
            algorithm,
            candidate_score,
        )

        return run_id

    @task
    def gate(candidate_run_id: str, **context) -> dict:
        """Promote the candidate only when it strictly beats Production."""
        import mlflow

        from src.promotion import evaluate_and_maybe_promote

        mlflow.set_tracking_uri(_require_tracking_uri())

        client = mlflow.MlflowClient()
        schema = str(context["params"]["schema"])

        result = evaluate_and_maybe_promote(
            client=client,
            schema=schema,
            candidate_run_id=candidate_run_id,
        )

        return {
            "run_id": result.run_id,
            "candidate_macro_f1": result.candidate_macro_f1,
            "production_macro_f1": result.production_macro_f1,
            "promoted": result.promoted,
            "new_version": result.new_version,
            "previous_version": result.previous_version,
        }

    data_ready = pull_data()
    candidate = train()
    candidate.set_upstream(data_ready)
    gate(evaluate(candidate))


model_pipeline()
