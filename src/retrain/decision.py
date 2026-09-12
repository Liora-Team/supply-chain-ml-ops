# src/retrain/decision.py
"""Pure decision logic for the automated retraining loop (Card 4.4).

Consumes the Card 4.1 drift contract (`monitoring/drift_status.json`, per-schema,
ADR 005) and the Card 4.4 state file (`monitoring/retrain_state.json`), then
decides — deterministically, with no Airflow import and no `monitoring` import
(stdlib json only, so it stays importable in every image target) — whether a
retrain should run, on which schema, and with which category slice.

Verified against origin/pr/38 (Card 4.1). The per-schema entry contains:
`status` ("ok" | "insufficient_data"), `drift_detected` (bool), `timestamp`,
`n_rows`, `window_start`, `window_end`, `report_path`.

Answers to the Issue #23 cooldown questions:
    * What starts the cooldown?      a triggered retrain sets `cooldown_until`.
    * Where is it persisted?         a JSON file on the shared monitoring mount.
    * Survives restarts?             yes, on disk, not in memory.
    * Same event vs new event?       compare drift `timestamp` to
                                     `last_handled_drift_ts` (edge detection).
    * Drift persists after cooldown? flag `persistent_drift`, surface it, no loop.
    * Retrain on every tick?         impossible: dedupe + cooldown both block it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

# Verified top-volume categories from Kerassy/trustpilot-reviews-123k (~6k
# rows each, no comma in the name so RETRAIN_CATEGORIES env list splits
# safely).
DEFAULT_CATEGORIES = [
    "Education & Training",
    "Sports",
    "Shopping & Fashion",
    "Health & Medical",
]


class Action(StrEnum):
    """The single decision the control loop can make for one schema."""

    TRIGGER = "trigger"
    SKIP_NO_DRIFT = "skip_no_drift"
    SKIP_INSUFFICIENT_DATA = "skip_insufficient_data"
    SKIP_ALREADY_HANDLED = "skip_already_handled"
    SKIP_COOLDOWN = "skip_cooldown"
    SKIP_MISSING_STATUS = "skip_missing_status"
    SKIP_MALFORMED_STATUS = "skip_malformed_status"
    SKIP_STALE = "skip_stale"


@dataclass(frozen=True)
class Decision:
    """The outcome of evaluating one schema's drift status."""

    action: Action
    schema: str | None = None
    category: str | None = None
    slice_cursor: int | None = None
    drift_timestamp: str | None = None
    persistent_drift: bool = False
    reason: str = ""


def _now() -> datetime:
    """Timezone-aware current time (UTC). Overridable in tests."""
    return datetime.now(UTC)


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string to an aware datetime, or None if invalid.

    Handles the trailing-'Z' form (e.g. Card 4.1 timestamps) that older
    Python fromisoformat rejects, by normalising it to +00:00.
    """
    if not value:
        return None
    normalised = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalised)
    except (ValueError, TypeError):
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def load_json(path: Path) -> dict | None:
    """Read a JSON object, or None if missing/malformed. Never raises."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        logger.warning("Could not read/parse JSON at %s", path, exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def next_category(categories: list[str], cursor: int) -> tuple[str, int]:
    """Return the category at `cursor` (wrapping) and the advanced cursor."""
    if not categories:
        raise ValueError("RETRAIN_CATEGORIES is empty; cannot rotate slices.")
    return categories[cursor % len(categories)], cursor + 1


def evaluate_schema(
    schema: str,
    drift_status: dict | None,
    retrain_state: dict | None,
    *,
    categories: list[str],
    cooldown_hours: float,
    stale_after_hours: float,
    now: datetime | None = None,
) -> Decision:
    """Decide the action for one schema. Pure — no I/O, no Airflow.

    Precedence (safest first):
      missing/malformed -> insufficient_data -> no-drift -> stale ->
      already-handled -> persistent-block -> cooldown -> TRIGGER.

    Persistent-drift policy (Issue #23 / ADR 006 / MAINTENANCE):
      - During active cooldown with a new drift timestamp:
        SKIP_COOLDOWN, persistent_drift=False.
        A new timestamp during the cooldown window is expected (Card 4.1
        runs every 24 h; cooldown is also 24 h). Flagging it persistent
        here would permanently disable the loop after the first retrain.
      - persistent_drift_blocked flag set by the DAG (operator action
        required): SKIP_COOLDOWN, persistent_drift=True.
      - New drift episode after cooldown expires (drift cleared then
        re-appeared): TRIGGER, persistent_drift=False. This is the
        Day-0/Day-3/Day-30 scenario from PR #40 review.
    """
    now = now or _now()

    if drift_status is None:
        return Decision(
            Action.SKIP_MISSING_STATUS,
            schema=schema,
            reason="drift_status.json missing",
        )

    entry = drift_status.get(schema)
    if not isinstance(entry, dict):
        return Decision(
            Action.SKIP_MALFORMED_STATUS,
            schema=schema,
            reason=f"no valid entry for {schema!r}",
        )

    # Card 4.1 writes drift_detected=False for insufficient_data too;
    # check status before drift_detected.
    if entry.get("status") == "insufficient_data":
        return Decision(
            Action.SKIP_INSUFFICIENT_DATA,
            schema=schema,
            reason="insufficient_data",
        )

    if entry.get("drift_detected") is not True:
        return Decision(
            Action.SKIP_NO_DRIFT,
            schema=schema,
            reason="drift_detected is not true",
        )

    drift_ts_raw = entry.get("timestamp")
    drift_ts = _parse_iso(drift_ts_raw)
    if drift_ts is None:
        return Decision(
            Action.SKIP_MALFORMED_STATUS,
            schema=schema,
            reason="missing/invalid drift timestamp",
        )

    if now - drift_ts > timedelta(hours=stale_after_hours):
        return Decision(
            Action.SKIP_STALE,
            schema=schema,
            drift_timestamp=drift_ts_raw,
            reason="stale drift report",
        )

    state_entry = (retrain_state or {}).get(schema, {})
    if not isinstance(state_entry, dict):
        state_entry = {}

    # Deduplicate: same drift event already handled → skip regardless of
    # cooldown state. This guard runs before the cooldown check so that
    # an expired cooldown cannot re-trigger on the same drift timestamp.
    if state_entry.get("last_handled_drift_ts") == drift_ts_raw:
        return Decision(
            Action.SKIP_ALREADY_HANDLED,
            schema=schema,
            drift_timestamp=drift_ts_raw,
            reason="drift event already handled",
        )

    # Operator / persistent-drift block: set by the DAG when the
    # persistent_drift flag is raised. Requires manual operator reset.
    if state_entry.get("persistent_drift_blocked") is True:
        return Decision(
            Action.SKIP_COOLDOWN,
            schema=schema,
            drift_timestamp=drift_ts_raw,
            persistent_drift=True,
            reason=(
                "persistent drift block is active "
                "(retraining did not clear drift); "
                "operator attention required"
            ),
        )

    # Cooldown window check.
    cooldown_until = _parse_iso(state_entry.get("cooldown_until"))
    if cooldown_until is not None and now < cooldown_until:
        # BLOCKER FIX (PR #40 / Issue #23 / ADR 006):
        # During active cooldown, suppress retraining but do NOT flag
        # persistent drift. A new drift timestamp during the cooldown
        # window is expected (Card 4.1 runs every 24 h; cooldown is also
        # 24 h). Flagging it persistent here permanently disables the
        # loop after the first retrain — the inverse of the policy.
        return Decision(
            Action.SKIP_COOLDOWN,
            schema=schema,
            drift_timestamp=drift_ts_raw,
            persistent_drift=False,
            reason=f"cooldown active until {cooldown_until.isoformat()}",
        )

    # Cooldown has expired (or no prior cooldown) AND this is a new drift
    # timestamp (dedupe guard above confirmed it differs from the last
    # handled one). This means either:
    #   a) first-ever trigger for this schema (no prior state), or
    #   b) drift cleared after the last retrain and has now re-appeared
    #      with a new timestamp (Day-30 scenario from PR #40 review).
    # Both cases are new episodes that must TRIGGER.
    # The DAG is responsible for setting persistent_drift_blocked if it
    # determines that retraining did not clear the drift (separate path
    # via the decide task inspecting persistent_drift on a later tick).
    cursor = int(state_entry.get("slice_cursor", 0))
    category, _ = next_category(categories, cursor)
    return Decision(
        Action.TRIGGER,
        schema=schema,
        category=category,
        slice_cursor=cursor,
        drift_timestamp=drift_ts_raw,
        reason=f"actionable drift; selected category {category!r}",
    )


def build_triggered_state(
    schema: str,
    category: str,
    cursor: int,
    drift_ts_raw: str,
    *,
    cooldown_hours: float,
    now: datetime | None = None,
) -> dict:
    """Build the state dict to persist after a retrain is triggered.

    Advances the slice cursor and sets the cooldown window so the same
    event cannot trigger a second retrain (dedupe + cooldown double-guard).
    """
    now = now or _now()
    _, next_cursor = next_category([category], 0)  # advances by 1
    return {
        "last_handled_drift_ts": drift_ts_raw,
        "last_category_used": category,
        "slice_cursor": cursor + 1,
        "cooldown_until": (now + timedelta(hours=cooldown_hours)).isoformat(),
        "triggered_at": now.isoformat(),
        # Reset the persistent block on a new trigger so a later recovery
        # is possible.
        "persistent_drift_blocked": False,
    }
