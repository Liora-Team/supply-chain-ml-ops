# tests/test_retrain_decision.py
"""Unit tests for the Card 4.4 pure retraining-decision logic.

No Airflow, no Docker, no network: exercises every decision branch. This is the
machine-checkable proof that the cooldown/dedupe cannot loop.

Includes the regression test for Marco's PR #40 review correction:
persistent-drift policy must match Issue #23 / ADR 006 / MAINTENANCE:
  - During cooldown with new drift timestamp -> SKIP_COOLDOWN, persistent_drift=False
  - After cooldown expires with drift still true -> SKIP_COOLDOWN, persistent_drift=True
    (only when persistent_drift_blocked flag is set by the DAG)
  - Day-0 / Day-3 (clear) / Day-30 (new drift) -> TRIGGER / SKIP_NO_DRIFT / TRIGGER
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


# ---------------------------------------------------------------------------
# Basic skip cases
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# BLOCKER FIX: corrected cooldown / persistent-drift tests
# ---------------------------------------------------------------------------


def test_cooldown_active_suppresses_without_persistent_flag():
    """During active cooldown, persistent_drift must be False.

    BLOCKER FIX (PR #40 / Issue #23): a new drift timestamp arriving while
    the cooldown is still active must NOT be flagged as persistent drift.
    Card 4.1 runs every 24 h; the cooldown is also 24 h, so a new timestamp
    during the window is expected behaviour.
    """
    state = {
        "3-class": {
            "last_handled_drift_ts": (NOW - timedelta(hours=1)).isoformat(),
            "cooldown_until": (NOW + timedelta(hours=5)).isoformat(),
        }
    }
    d = _eval(_drift(ts=NOW), state)
    assert d.action is Action.SKIP_COOLDOWN
    # BLOCKER FIX: must be False (not True) during the active cooldown window.
    assert d.persistent_drift is False


def test_cooldown_expired_with_fresh_drift_triggers():
    """After cooldown expires with a NEW drift timestamp, TRIGGER (new episode).

    BLOCKER FIX (PR #40 / Issue #23 / ADR 006):
    A new drift timestamp after cooldown expiry means a new drift episode
    has started (e.g. drift cleared at Day 3, re-appeared at Day 30).
    The engine must TRIGGER, not block as persistent.
    The DAG is responsible for setting persistent_drift_blocked separately
    if it determines that retraining did not resolve the drift.
    """
    state = {
        "3-class": {
            "last_handled_drift_ts": (NOW - timedelta(hours=30)).isoformat(),
            "cooldown_until": (NOW - timedelta(hours=6)).isoformat(),
            "slice_cursor": 1,
        }
    }
    d = _eval(_drift(ts=NOW), state)
    # A new timestamp + expired cooldown = new episode = TRIGGER.
    assert d.action is Action.TRIGGER
    assert d.persistent_drift is False
    assert d.category == CATEGORIES[1]  # cursor=1 -> "clothing"
    assert d.slice_cursor == 1


def test_cooldown_expired_but_same_event_still_skips():
    """Anti-loop: an expired cooldown must not re-handle an already-handled report.

    A new retrain requires a *fresh* Card 4.1 report (new timestamp), so the
    15-min poller cannot re-trigger on every tick once the cooldown closes.
    This pins the guard order: dedupe is evaluated before the cooldown window.
    """
    state = {
        "3-class": {
            "last_handled_drift_ts": NOW.isoformat(),
            "cooldown_until": (NOW - timedelta(hours=6)).isoformat(),
            "slice_cursor": 2,
        }
    }
    assert _eval(_drift(ts=NOW), state).action is Action.SKIP_ALREADY_HANDLED


def test_persistent_drift_blocked_prevents_trigger():
    """An operator/system block must prevent retraining until manually reset.

    The DAG sets persistent_drift_blocked=True after detecting that a
    completed retrain did not clear the drift. The engine must respect this
    flag and block further automatic triggers.
    """
    state = {
        "3-class": {
            "persistent_drift_blocked": True,
        }
    }
    d = _eval(_drift(ts=NOW), state)
    assert d.action is Action.SKIP_COOLDOWN
    assert d.persistent_drift is True


# ---------------------------------------------------------------------------
# BLOCKER FIX: Day-0 / Day-3 / Day-30 regression (Marco's PR #40 review)
# ---------------------------------------------------------------------------


def test_day0_trigger_day3_clear_day30_new_trigger():
    """Regression: drift episode that clears must allow a new trigger later.

    Scenario (from Marco's review comment on PR #40):
      Day  0: drift=True  -> TRIGGER (cooldown_until set to Day 1).
      Day  3: drift=False -> SKIP_NO_DRIFT (drift cleared; no persistent block).
      Day 30: drift=True  -> TRIGGER again (new episode; persistent_drift=False).

    Previously broken: a new drift timestamp during the cooldown window was
    flagged as persistent, permanently disabling the loop after a single trigger.

    Root cause of old bug: the expired-cooldown branch returned persistent_drift=True
    for ANY new timestamp after cooldown, including genuine new episodes. The fix:
    persistent_drift_blocked is only set by the DAG (not by evaluate_schema itself),
    and the engine triggers on any new timestamp once the cooldown has expired.
    """
    day_0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    day_3 = day_0 + timedelta(days=3)
    day_30 = day_0 + timedelta(days=30)

    categories = CATEGORIES
    schema = "3-class"

    # Day 0: fresh drift -> TRIGGER
    drift_day0 = {
        schema: {
            "status": "ok",
            "drift_detected": True,
            "timestamp": day_0.isoformat(),
            "n_rows": 300,
            "window_start": (day_0 - timedelta(hours=24)).isoformat(),
            "window_end": day_0.isoformat(),
            "report_path": "monitoring/reports/drift_report_3class.html",
        }
    }
    d0 = evaluate_schema(
        schema,
        drift_day0,
        None,
        categories=categories,
        cooldown_hours=COOLDOWN_HOURS,
        stale_after_hours=STALE_HOURS * 30,
        now=day_0,
    )
    assert d0.action is Action.TRIGGER, f"Day 0 should TRIGGER, got {d0.action}"
    assert d0.persistent_drift is False

    # Persist state as the DAG would after a trigger.
    state_after_day0 = {
        schema: build_triggered_state(
            schema,
            d0.category,
            d0.slice_cursor,
            d0.drift_timestamp,
            cooldown_hours=COOLDOWN_HOURS,
            now=day_0,
        )
    }

    # Day 3: drift cleared -> SKIP_NO_DRIFT (no persistent flag)
    drift_day3 = {
        schema: {
            "status": "ok",
            "drift_detected": False,
            "timestamp": day_3.isoformat(),
            "n_rows": 300,
            "window_start": (day_3 - timedelta(hours=24)).isoformat(),
            "window_end": day_3.isoformat(),
            "report_path": None,
        }
    }
    d3 = evaluate_schema(
        schema,
        drift_day3,
        state_after_day0,
        categories=categories,
        cooldown_hours=COOLDOWN_HOURS,
        stale_after_hours=STALE_HOURS * 30,
        now=day_3,
    )
    assert d3.action is Action.SKIP_NO_DRIFT, f"Day 3 should SKIP_NO_DRIFT, got {d3.action}"
    assert d3.persistent_drift is False

    # Day 30: new drift episode -> must TRIGGER (not be blocked as persistent).
    # At day 30:
    #   - cooldown_until (Day 1) is long expired      -> not in active cooldown
    #   - last_handled_drift_ts = Day-0 ts            -> Day-30 ts != Day-0 ts
    #   - dedupe guard passes (different timestamp)
    #   - persistent_drift_blocked not set (never was)
    #   -> TRIGGER ✅
    drift_day30 = {
        schema: {
            "status": "ok",
            "drift_detected": True,
            "timestamp": day_30.isoformat(),
            "n_rows": 300,
            "window_start": (day_30 - timedelta(hours=24)).isoformat(),
            "window_end": day_30.isoformat(),
            "report_path": "monitoring/reports/drift_report_3class.html",
        }
    }
    d30 = evaluate_schema(
        schema,
        drift_day30,
        state_after_day0,
        categories=categories,
        cooldown_hours=COOLDOWN_HOURS,
        stale_after_hours=STALE_HOURS * 30,
        now=day_30,
    )
    assert d30.action is Action.TRIGGER, (
        f"Day 30 new drift should TRIGGER, got {d30.action} " f"(reason: {d30.reason})"
    )
    assert (
        d30.persistent_drift is False
    ), "Day 30 must not be flagged as persistent drift — drift cleared at day 3"


def test_drift_never_clears_becomes_persistent_after_cooldown():
    """Complementary: drift that never clears must block after the DAG flags it.

    Correct persistent-drift scenario (Issue #23): retraining fired but did
    not resolve the drift. The DAG sets persistent_drift_blocked=True. On the
    next tick, the engine sees the flag and blocks.

    Note: evaluate_schema itself does NOT set persistent_drift_blocked. That
    responsibility belongs to the DAG's decide() task, which inspects the
    Decision after each tick and writes the flag to retrain_state.json.
    """
    day_0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    day_30 = day_0 + timedelta(days=30)

    categories = CATEGORIES
    schema = "3-class"

    # Day 0: trigger.
    drift_day0 = {
        schema: {
            "status": "ok",
            "drift_detected": True,
            "timestamp": day_0.isoformat(),
            "n_rows": 300,
            "window_start": (day_0 - timedelta(hours=24)).isoformat(),
            "window_end": day_0.isoformat(),
            "report_path": "monitoring/reports/drift_report_3class.html",
        }
    }
    d0 = evaluate_schema(
        schema,
        drift_day0,
        None,
        categories=categories,
        cooldown_hours=COOLDOWN_HOURS,
        stale_after_hours=STALE_HOURS * 30,
        now=day_0,
    )
    assert d0.action is Action.TRIGGER

    # DAG persists state + sets persistent_drift_blocked=True after
    # determining that the day-30 drift (new ts) means retraining failed.
    state = {
        schema: {
            **build_triggered_state(
                schema,
                d0.category,
                d0.slice_cursor,
                d0.drift_timestamp,
                cooldown_hours=COOLDOWN_HOURS,
                now=day_0,
            ),
            # The DAG sets this flag when it detects persistent drift.
            "persistent_drift_blocked": True,
        }
    }

    # Day 30: drift still true (new timestamp), DAG has flagged persistent.
    drift_day30 = {
        schema: {
            "status": "ok",
            "drift_detected": True,
            "timestamp": day_30.isoformat(),
            "n_rows": 300,
            "window_start": (day_30 - timedelta(hours=24)).isoformat(),
            "window_end": day_30.isoformat(),
            "report_path": "monitoring/reports/drift_report_3class.html",
        }
    }
    d30 = evaluate_schema(
        schema,
        drift_day30,
        state,
        categories=categories,
        cooldown_hours=COOLDOWN_HOURS,
        stale_after_hours=STALE_HOURS * 30,
        now=day_30,
    )
    assert (
        d30.action is Action.SKIP_COOLDOWN
    ), f"Drift that never cleared should be blocked, got {d30.action}"
    assert (
        d30.persistent_drift is True
    ), "Drift that never cleared after cooldown must set persistent_drift=True"


# ---------------------------------------------------------------------------
# Unchanged structural tests
# ---------------------------------------------------------------------------


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
