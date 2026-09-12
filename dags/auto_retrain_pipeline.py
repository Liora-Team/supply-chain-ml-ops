# dags/auto_retrain_pipeline.py
"""Automated retraining control loop (Card 4.4) — close the MLOps loop.

Poller pattern: runs on a short schedule, reads the Card 4.1 drift signal
(`monitoring/drift_status.json`, ADR 005), and only acts on *actionable* drift
that the persistent cooldown/dedupe guard permits. It then:

    1. triggers the existing data_pipeline with the next category slice
       (Option 3, genuinely new data), then
    2. triggers the existing model_pipeline, whose Card 3.2 gate stays the SOLE
       promotion authority.

It never promotes, never writes drift_status.json, never runs training itself.
All decision logic lives in the pure, unit-tested `src.retrain.decision` module.
It imports only stdlib + src.retrain (both present in every image target) — never
`monitoring` (not copied into images; see PR #38 review).

Anti-loop proof: a persistent drift cannot retrain forever — the same drift
timestamp is deduplicated and a cooldown window suppresses further triggers;
persistent drift after cooldown is surfaced (logged + state flag), not looped.
"""

from __future__ import annotations

import os
from datetime import timedelta

from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import dag, task

# Path resolution: DRIFT_STATUS_PATH takes priority over MONITORING_DIR fallback.
MONITORING_DIR = os.environ.get("MONITORING_DIR", "/app/monitoring")
DRIFT_STATUS_PATH = os.environ.get(
    "DRIFT_STATUS_PATH",
    f"{MONITORING_DIR}/drift_status.json",
)
RETRAIN_STATE_PATH = os.environ.get(
    "RETRAIN_STATE_PATH",
    f"{MONITORING_DIR}/retrain_state.json",
)

# Blank env var must fall back to defaults (not produce an empty list).
_raw_schemas = os.environ.get("RETRAIN_SCHEMAS", "").strip()
WATCHED_SCHEMAS = (
    [s.strip() for s in _raw_schemas.split(",") if s.strip()] if _raw_schemas else ["3-class"]
)

_raw_categories = os.environ.get("RETRAIN_CATEGORIES", "").strip()
RETRAIN_CATEGORIES = (
    [c.strip() for c in _raw_categories.split(",") if c.strip()]
    if _raw_categories
    else [
        "Education & Training",
        "Sports",
        "Shopping & Fashion",
        "Health & Medical",
    ]
)

COOLDOWN_HOURS = float(os.environ.get("RETRAIN_COOLDOWN_HOURS", "24"))
STALE_AFTER_HOURS = float(os.environ.get("RETRAIN_STALE_AFTER_HOURS", "48"))
POLL_MINUTES = int(os.environ.get("RETRAIN_POLL_MINUTES", "15"))


def _alert(context):
    print(
        f"[ALERT] auto_retrain task {context['task_instance'].task_id} "
        "failed after retries — see log above"
    )


default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": _alert,
}


@dag(
    schedule=timedelta(minutes=POLL_MINUTES),
    catchup=False,
    max_active_runs=1,  # never overlap the control loop
    default_args=default_args,
    tags=["monitoring", "retrain", "card-4.4"],
)
def auto_retrain_pipeline():
    """Read drift -> guard (cooldown/dedupe) -> next slice -> trigger M3."""

    @task
    def decide(**context) -> dict:
        """Evaluate drift + cooldown per watched schema (first actionable wins)."""
        import json
        from pathlib import Path

        from airflow.exceptions import AirflowSkipException

        from src.retrain.decision import Action, evaluate_schema, load_json

        drift_status = load_json(Path(DRIFT_STATUS_PATH))
        retrain_state = load_json(Path(RETRAIN_STATE_PATH)) or {}

        chosen = None
        state_updated = False
        for schema in WATCHED_SCHEMAS:
            d = evaluate_schema(
                schema,
                drift_status,
                retrain_state,
                categories=RETRAIN_CATEGORIES,
                cooldown_hours=COOLDOWN_HOURS,
                stale_after_hours=STALE_AFTER_HOURS,
            )
            print(f"[decide] schema={schema} action={d.action.value} " f"reason={d.reason}")
            if d.persistent_drift:
                print(f"[PERSISTENT-DRIFT] schema={schema}: {d.reason}")
                # Persist block flag so it survives scheduler restarts.
                if schema not in retrain_state or not isinstance(retrain_state[schema], dict):
                    retrain_state[schema] = {}
                if not retrain_state[schema].get("persistent_drift_blocked"):
                    retrain_state[schema]["persistent_drift_blocked"] = True
                    state_updated = True

            if d.action is Action.TRIGGER and chosen is None:
                chosen = d

        if state_updated:
            state_path = Path(RETRAIN_STATE_PATH)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(retrain_state, indent=2), encoding="utf-8")
            print("[decide] persistent_drift_blocked flagged as True on disk.")

        if chosen is None:
            raise AirflowSkipException("No actionable drift under cooldown/dedupe policy.")

        return {
            "schema": chosen.schema,
            "category": chosen.category,
            "slice_cursor": chosen.slice_cursor,
            "drift_timestamp": chosen.drift_timestamp,
        }

    @task
    def record_state(decision: dict) -> dict:
        """Persist cooldown + advanced cursor BEFORE triggering (idempotent guard)."""
        import json
        from pathlib import Path

        from src.retrain.decision import build_triggered_state, load_json

        state_path = Path(RETRAIN_STATE_PATH)
        state = load_json(state_path) or {}
        state[decision["schema"]] = build_triggered_state(
            decision["schema"],
            decision["category"],
            decision["slice_cursor"],
            decision["drift_timestamp"],
            cooldown_hours=COOLDOWN_HOURS,
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[record_state] cooldown + cursor written for " f"schema={decision['schema']}")
        return decision

    # BLOCKER FIX (PR #40 review — Marco): add fail_when_dag_is_paused=True
    # and execution_timeout to both TriggerDagRunOperator tasks. Without
    # these, a paused downstream DAG causes an infinite wait on a fresh
    # Airflow volume (DAGs are paused by default). The timeouts are generous
    # but finite; CI/demo environments will not hang.
    trigger_data = TriggerDagRunOperator(
        task_id="trigger_data_pipeline",
        trigger_dag_id="data_pipeline",
        conf={
            "sample": 0,
            "category": ("{{ ti.xcom_pull(task_ids='record_state')['category'] }}"),
        },
        wait_for_completion=True,
        poke_interval=30,
        reset_dag_run=True,
        fail_when_dag_is_paused=True,
        execution_timeout=timedelta(hours=1),
    )

    trigger_model = TriggerDagRunOperator(
        task_id="trigger_model_pipeline",
        trigger_dag_id="model_pipeline",
        conf={
            "schema": ("{{ ti.xcom_pull(task_ids='record_state')['schema'] }}"),
            "algorithm": "LogReg",
        },
        wait_for_completion=True,
        poke_interval=30,
        reset_dag_run=True,
        fail_when_dag_is_paused=True,
        execution_timeout=timedelta(hours=2),
    )

    recorded = record_state(decide())
    recorded >> trigger_data >> trigger_model


auto_retrain_pipeline()
