"""SQLite-backed store of ``/predict`` requests — Card 4.1's "current" dataset.

Every successful ``/predict`` call is logged here (text, prediction, confidence, the
model that served it, and a timestamp). Card 4.1's drift job reads a recent window of
these rows as the "current" distribution to compare against the training ("reference")
distribution pulled via DVC (``monitoring/datasets.py``, next commit).

WAL mode + a busy timeout let multiple ``api`` replicas (``--scale api=N``) write
concurrently without corrupting the file. The database file lives under the
``./monitoring`` bind mount shared with the ``airflow`` service (``docker-compose.yml``),
so the scheduled drift job can read it without an HTTP call.

See ``docs/adr/005-drift-signal-format.md`` for the format/backend decisions and
``docs/MAINTENANCE.md`` for the retention note this store is responsible for.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Overridable for tests / alternate deployments; defaults to the shared compose mount.
DB_PATH = Path(os.environ.get("MONITORING_DB_PATH", "monitoring/request_store.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    text            TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence      REAL NOT NULL,
    model_version   TEXT NOT NULL,
    schema          TEXT NOT NULL,
    timestamp       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_schema_timestamp
    ON predictions (schema, timestamp);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Open a connection with WAL mode + a busy timeout for concurrent writers.

    Creates the parent directory and the table on first use, so neither the API nor
    the drift job need a separate init step.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=10000;")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_prediction(
    text: str,
    predicted_label: str,
    confidence: float,
    model_version: str,
    schema: str,
    timestamp: str | None = None,
) -> None:
    """Persist one ``/predict`` call.

    Designed to be called from a FastAPI ``BackgroundTask`` (api/main.py): it runs
    after the response is already sent, so a slow or failing write never adds latency
    to — or breaks — inference.
    """
    ts = timestamp or datetime.now(UTC).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO predictions "
            "(text, predicted_label, confidence, model_version, schema, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (text, predicted_label, confidence, model_version, schema, ts),
        )


def get_recent(schema: str, window_hours: int, max_rows: int) -> list[sqlite3.Row]:
    """Rows for ``schema`` from the last ``window_hours``, newest first, capped at ``max_rows``.

    Two independent limits, whichever is hit first — a quiet 24h returns fewer than
    ``max_rows`` rows; a traffic spike is truncated to the most recent ``max_rows``
    rather than growing the comparison set unbounded. Used by ``monitoring/datasets.py``
    (next commit) to build the "current" frame.
    """
    cutoff = (datetime.now(UTC) - timedelta(hours=window_hours)).isoformat()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM predictions "
            "WHERE schema = ? AND timestamp >= ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (schema, cutoff, max_rows),
        )
        return cursor.fetchall()


def purge_older_than(days: int) -> int:
    """Delete rows older than ``days`` days; returns the number of rows removed.

    Closes the retention/cleanup note flagged in docs/MAINTENANCE.md pending this
    card. Intended to run as a scheduled step in the orchestrator (Card 4.1's Evidently
    job / DAG), not on every request.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM predictions WHERE timestamp < ?", (cutoff,))
        return cursor.rowcount
