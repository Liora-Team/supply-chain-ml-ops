"""Unit tests for the Card 4.4 pure retraining-decision logic.

No Airflow, no Docker, no network: exercises every decision branch. This is the
machine-checkable proof that the cooldown/dedupe cannot loop.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.retrain.decision import (
    DEFAULT_CATEGORIES,
    Action,
    build_triggered_state,
    evaluate_schema,
    load_json,
    next_category,
)

CATEGORIES = ["electronics", "clothing", "home", "beauty"]
COOLDOWN_HOURS = 24.0
STALE_HOURS = 48.0
NOW = datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC)


def _drift(schema="3-class", *, detected=True, status="ok", ts=None):
    return {
        schema: {
            "status": status,
            "drift_detected": detected,
            "timestamp": (ts or NOW).isoformat(),
            "n_rows": 300,
            "window_start": (NOW - timedelta(hours=24)).isoformat(),
            "window_end": NOW.isoformat(),
            "report_path": "monitoring/reports/drift_report_3class.html",
        }
    }


def _eval(status, state=None, categories=CATEGORIES):
    return evaluate_schema(
        "3-class",
        status,
        state,
        categories=categories,
        cooldown_hours=COOLDOWN_HOURS,
        stale_after_hours=STALE_HOURS,
        now=NOW,
    )


def test_missing_status_file():
    assert _eval(None).action is Action.SKIP_MISSING_STATUS


def test_malformed_no_schema_entry():
    assert _eval({"5-class": {}}).action is Action.SKIP_MALFORMED_STATUS


def test_bad_timestamp_is_malformed():
    s = _drift()
    s["3-class"]["timestamp"] = "not-a-date"
    assert _eval(s).action is Action.SKIP_MALFORMED_STATUS


def test_insufficient_data_never_triggers():
    assert _eval(_drift(status="insufficient_data")).action is Action.SKIP_INSUFFICIENT_DATA


def test_insufficient_data_with_drift_false_still_skips_as_insufficient():
    # Card 4.1 writes drift_detected=False for insufficient_data too.
    s = _drift(status="insufficient_data", detected=False)
    assert _eval(s).action is Action.SKIP_INSUFFICIENT_DATA


def test_drift_false_skips():
    assert _eval(_drift(detected=False)).action is Action.SKIP_NO_DRIFT


def test_stale_report_skips():
    old = NOW - timedelta(hours=STALE_HOURS + 1)
    assert _eval(_drift(ts=old)).action is Action.SKIP_STALE


def test_actionable_drift_triggers_first_category():
    d = _eval(_drift())
    assert d.action is Action.TRIGGER
    assert d.schema == "3-class"
    assert d.category == "electronics"
    assert d.slice_cursor == 0


def test_zulu_timestamp_is_parsed():
    # Card 4.1 timestamps may end with 'Z'.
    s = _drift()
    s["3-class"]["timestamp"] = "2026-08-15T12:00:00Z"
    d = _eval(s)
    assert d.action is Action.TRIGGER


def test_same_event_already_handled_skips():
    state = {"3-class": {"last_handled_drift_ts": NOW.isoformat()}}
    assert _eval(_drift(ts=NOW), state).action is Action.SKIP_ALREADY_HANDLED


def test_cooldown_active_blocks_and_flags_persistent():
    state = {
        "3-class": {
            "last_handled_drift_ts": (NOW - timedelta(hours=1)).isoformat(),
            "cooldown_until": (NOW + timedelta(hours=5)).isoformat(),
        }
    }
    d = _eval(_drift(ts=NOW), state)
    assert d.action is Action.SKIP_COOLDOWN
    assert d.persistent_drift is True


def test_cooldown_expired_allows_new_trigger():
    state = {
        "3-class": {
            "last_handled_drift_ts": (NOW - timedelta(hours=30)).isoformat(),
            "cooldown_until": (NOW - timedelta(hours=6)).isoformat(),
            "slice_cursor": 1,
        }
    }
    d = _eval(_drift(ts=NOW), state)
    assert d.action is Action.TRIGGER
    assert d.category == "clothing"
    assert d.slice_cursor == 1


def test_build_triggered_state_starts_cooldown_and_advances_cursor():
    new = build_triggered_state(
        "3-class",
        "electronics",
        0,
        NOW.isoformat(),
        cooldown_hours=COOLDOWN_HOURS,
        now=NOW,
    )
    assert new["slice_cursor"] == 1
    assert new["last_category_used"] == "electronics"
    assert new["last_handled_drift_ts"] == NOW.isoformat()
    assert datetime.fromisoformat(new["cooldown_until"]) == NOW + timedelta(hours=COOLDOWN_HOURS)


def test_category_rotation_wraps():
    assert next_category(CATEGORIES, 0) == ("electronics", 1)
    assert next_category(CATEGORIES, 3) == ("beauty", 4)
    assert next_category(CATEGORIES, 4) == ("electronics", 5)


def test_default_categories_are_the_verified_real_ones():
    assert DEFAULT_CATEGORIES == [
        "Education & Training",
        "Sports",
        "Shopping & Fashion",
        "Health & Medical",
    ]


def test_empty_categories_raises():
    with pytest.raises(ValueError):
        next_category([], 0)


def test_load_json_missing_returns_none(tmp_path):
    assert load_json(tmp_path / "nope.json") is None


def test_load_json_malformed_returns_none(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_json(p) is None


def test_load_json_valid(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert load_json(p) == {"a": 1}
