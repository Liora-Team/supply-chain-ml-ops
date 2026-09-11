"""Data half of the ML pipeline (Card 3.1) — ingest → preprocess → DVC version.

Interface with Card 3.2 (model half, `dags/model_pipeline.py`, branch
feature/di-dag-model): this DAG ends by producing a fresh, DVC-versioned
`data/processed/` — train.csv + test.csv with `dvc add` + `dvc push` done.
The refreshed `.dvc` pointer files appear on the host bind mount for a human
`git commit` (no git credentials ever enter the container). The model DAG
consumes exactly those files, runs on this same Airflow stack, and lives in
this same `dags/` tree.

Manual trigger only (schedule=None): Airflow UI → data_pipeline → Trigger
(optionally with {"sample": 500} for a smoke run), or `make dag-trigger`.
Progress: UI Grid view shows per-task status; logs per task instance.
"""

from datetime import timedelta

from airflow.sdk import dag, task

# data/raw/ is gitignored; the bind mount makes the handoff visible on the host too.
RAW_PATH = "/app/data/raw/trustpilot_reviews.parquet"


def _alert(context):
    # Simplest orchestrator-native alert (no SMTP configured): loud log line + red task in UI.
    print(f"[ALERT] task {context['task_instance'].task_id} failed after retries — see log above")


default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": _alert,
}


@dag(
    schedule=None,  # manual trigger only in M3
    catchup=False,
    default_args=default_args,
    tags=["data", "card-3.1"],
    params={"sample": 0, "category": None},  # category: Card 4.4 Option 3 slice replay
)
def data_pipeline():
    # Heavy imports live inside the task bodies so the DAG processor parses this file
    # without the data stack installed.

    @task
    def ingest(**context) -> str:
        from pathlib import Path

        from scripts.get_data import load_raw

        df = load_raw(
            sample=int(context["params"]["sample"]),
            category=context["params"].get("category"),
        )
        Path(RAW_PATH).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RAW_PATH)
        return RAW_PATH  # XCom carries only small paths, never DataFrames

    @task
    def preprocess(raw_path: str, **context) -> list[str]:
        import pandas as pd

        # aliased: an unaliased import would shadow this task's own name inside the body
        from scripts.get_data import (
            append_to_train_only,
            split_and_write,
        )
        from scripts.get_data import (
            preprocess as preprocess_df,
        )

        category = context["params"].get("category")
        df = preprocess_df(pd.read_parquet(raw_path))

        # Card 4.4 Option 3: when replaying a category slice, keep test.csv fixed to preserve
        # promotion-gate comparability (Card 3.2). Only train.csv is updated.
        if category is not None:
            train_path = append_to_train_only(df)
            return [str(train_path)]

        train_path, test_path = split_and_write(df)
        return [str(train_path), str(test_path)]

    @task.bash
    def dvc_version(paths: list[str]) -> str:
        # dvc add refreshes the .dvc pointers (visible on the host via the bind mount);
        # dvc push uploads to the DagsHub remote. A human commits the pointers afterwards.
        return f"cd /app && dvc add {' '.join(paths)} && dvc push"

    dvc_version(preprocess(ingest()))


data_pipeline()
