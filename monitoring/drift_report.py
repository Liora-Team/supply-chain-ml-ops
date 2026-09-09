"""Evidently drift-report job — the heart of Card 4.1.

Compares the reference (training) and current (recent request-store window) datasets
built by ``monitoring/datasets.py``, runs Evidently's data-drift + data-quality checks,
and writes the result to ``monitoring/drift_status.json`` — the single machine-readable
flag 4.2 (alerting) and 4.4 (retraining) consume (see ``docs/adr/005-drift-signal-format.md``).

Drift is judged on the numeric text-derived features (``text_length``, ``word_count``)
only — the raw ``text`` column is high-cardinality free text, and a naive per-value drift
test on it is noisy rather than informative. ``text`` still feeds the data-quality checks
(duplicates, missing values) via ``DataSummaryPreset``.

Runnable on demand (``scripts/run_drift_check.py``) or on a schedule (the orchestrator,
Card 3.1's DAG). Every run overwrites that schema's entry in the status file — including
back to ``drift_detected: false`` once drift clears, so 4.4 doesn't retrain forever on a
stale flag.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from evidently import Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from monitoring.datasets import load_current, load_reference

# Columns the drift-share verdict is computed on. Deliberately excludes raw `text`
# (see module docstring) — DataSummaryPreset still runs on the full frame below.
DRIFT_COLUMNS = ["text_length", "word_count"]

DEFAULT_STATUS_PATH = Path(os.environ.get("DRIFT_STATUS_PATH") or "monitoring/drift_status.json")
DEFAULT_REPORTS_DIR = Path(os.environ.get("DRIFT_REPORTS_DIR") or "monitoring/reports")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _update_status_file(path: Path, schema: str, entry: dict[str, Any]) -> None:
    """Merge ``entry`` into ``path`` under ``schema``, leaving other schemas untouched.

    The file holds one entry per label schema (``3-class`` / ``5-class``) since the
    service serves both; each schema's entry is fully overwritten on its own run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {}
    if path.exists():
        try:
            status = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            status = {}  # corrupt/partial file — start clean rather than fail the job
    status[schema] = entry
    path.write_text(json.dumps(status, indent=2))


def run_drift_report(
    schema: str,
    *,
    window_hours: int | None = None,
    max_rows: int | None = None,
    min_rows: int | None = None,
    status_path: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the drift check for one schema and update the status file.

    Returns the same dict written to ``drift_status.json[schema]``. Env-var defaults
    (``DRIFT_WINDOW_HOURS=24``, ``DRIFT_MAX_ROWS=500``, ``DRIFT_MIN_ROWS=50``) can be
    overridden per call for tests, or globally via the environment.
    """
    window_hours = window_hours if window_hours is not None else _env_int("DRIFT_WINDOW_HOURS", 24)
    max_rows = max_rows if max_rows is not None else _env_int("DRIFT_MAX_ROWS", 500)
    min_rows = min_rows if min_rows is not None else _env_int("DRIFT_MIN_ROWS", 50)
    status_path = status_path or DEFAULT_STATUS_PATH
    reports_dir = reports_dir or DEFAULT_REPORTS_DIR

    now = datetime.now(UTC)
    window_start = now - timedelta(hours=window_hours)
    current = load_current(schema=schema, window_hours=window_hours, max_rows=max_rows)

    base_entry: dict[str, Any] = {
        "timestamp": now.isoformat(),
        "schema": schema,
        "n_rows": len(current),
        "window_start": window_start.isoformat(),
        "window_end": now.isoformat(),
    }

    if len(current) < min_rows:
        # Too little traffic to say anything — neither a drift verdict nor a false
        # "all clear". 4.2/4.4 should treat this status as "skip", not as "no drift".
        entry = {
            **base_entry,
            "status": "insufficient_data",
            "drift_detected": False,
            "report_path": None,
        }
        _update_status_file(status_path, schema, entry)
        return entry

    reference = load_reference()

    report = Report([DataDriftPreset(columns=DRIFT_COLUMNS), DataSummaryPreset()])
    snapshot = report.run(Dataset.from_pandas(current), Dataset.from_pandas(reference))

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{schema}-{now.strftime('%Y%m%dT%H%M%S')}.html"
    snapshot.save_html(str(report_path))

    result = snapshot.dict()
    drifted_columns = next(
        (
            m
            for m in result["metrics"]
            if m.get("metric_name", "").startswith("DriftedColumnsCount")
        ),
        None,
    )
    if drifted_columns is None:
        # Evidently's internal metric id changed — fail loudly rather than silently
        # reporting "no drift" on a report we can't actually read.
        raise RuntimeError(
            "DriftedColumnsCount metric not found in the Evidently report; "
            "the installed evidently version may have changed its API."
        )

    drift_share = drifted_columns["value"]["share"]
    drift_share_threshold = drifted_columns["config"].get("drift_share", 0.5)

    entry = {
        **base_entry,
        "status": "ok",
        "drift_detected": drift_share >= drift_share_threshold,
        "drift_share": drift_share,
        "report_path": str(report_path),
    }
    _update_status_file(status_path, schema, entry)
    return entry
