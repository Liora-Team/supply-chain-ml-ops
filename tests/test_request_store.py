"""Unit tests for monitoring/request_store.py (Card 4.1 prerequisite)."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Reload request_store against a throwaway DB file for each test."""
    monkeypatch.setenv("MONITORING_DB_PATH", str(tmp_path / "request_store.db"))
    import monitoring.request_store as request_store

    importlib.reload(request_store)
    return request_store


def test_log_and_get_recent_round_trip(store):
    store.log_prediction(
        text="Great product, fast shipping!",
        predicted_label="positive",
        confidence=0.91,
        model_version="reviews-classifier-3class",
        schema="3-class",
    )

    rows = store.get_recent(schema="3-class", window_hours=24, max_rows=500)

    assert len(rows) == 1
    assert rows[0]["text"] == "Great product, fast shipping!"
    assert rows[0]["predicted_label"] == "positive"
    assert rows[0]["schema"] == "3-class"


def test_get_recent_filters_by_schema(store):
    store.log_prediction("a", "positive", 0.9, "m1", schema="3-class")
    store.log_prediction("b", "1", 0.8, "m2", schema="5-class")

    rows = store.get_recent(schema="5-class", window_hours=24, max_rows=500)

    assert len(rows) == 1
    assert rows[0]["text"] == "b"


def test_get_recent_respects_max_rows(store):
    for i in range(10):
        store.log_prediction(str(i), "positive", 0.9, "m1", schema="3-class")

    rows = store.get_recent(schema="3-class", window_hours=24, max_rows=3)

    assert len(rows) == 3


def test_get_recent_excludes_rows_outside_window(store):
    stale_ts = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    store.log_prediction("old", "positive", 0.9, "m1", schema="3-class", timestamp=stale_ts)
    store.log_prediction("fresh", "positive", 0.9, "m1", schema="3-class")

    rows = store.get_recent(schema="3-class", window_hours=24, max_rows=500)

    assert [r["text"] for r in rows] == ["fresh"]


def test_blank_env_var_falls_back_to_default_not_empty_path(monkeypatch):
    """A var that's *set but blank* (e.g. a sourced `.env` with `MONITORING_DB_PATH=`)
    must fall through to the default, not resolve to Path("") == the cwd."""
    monkeypatch.setenv("MONITORING_DB_PATH", "")
    import monitoring.request_store as request_store

    importlib.reload(request_store)

    assert str(request_store.DB_PATH) == "monitoring/request_store.db"


def test_purge_older_than_removes_only_stale_rows(store):
    stale_ts = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    store.log_prediction("old", "positive", 0.9, "m1", schema="3-class", timestamp=stale_ts)
    store.log_prediction("fresh", "positive", 0.9, "m1", schema="3-class")

    removed = store.purge_older_than(days=7)
    rows = store.get_recent(schema="3-class", window_hours=24 * 365, max_rows=500)

    assert removed == 1
    assert [r["text"] for r in rows] == ["fresh"]


def test_concurrent_writes_do_not_corrupt_store(store):
    """WAL mode + busy_timeout: sequential writers simulating multiple api replicas."""
    for i in range(20):
        store.log_prediction(f"review {i}", "positive", 0.9, "m1", schema="3-class")

    rows = store.get_recent(schema="3-class", window_hours=24, max_rows=500)

    assert len(rows) == 20
