# scripts/seed_drift_status.py
"""Seed a Card-4.1-shaped drift_status.json for Card 4.4 isolation testing.

This is NOT a drift detector and NOT a replacement for Card 4.1's tool.

The authoritative drift path is `scripts/simulate_drift.py` (Card 4.1, PR #38),
which replays a category through /predict -> SQLite -> Evidently and writes the
real drift_status.json (needs a running API + the orchestration group). By
contrast, this helper only *writes a small fixture file* matching the ADR 005
contract, so the Card 4.4 control loop can be exercised in isolation and in CI
without a network, an API, or Evidently.

Usage
-----
    uv run python scripts/seed_drift_status.py --drift
    uv run python scripts/seed_drift_status.py --no-drift
    uv run python scripts/seed_drift_status.py --insufficient-data
    uv run python scripts/seed_drift_status.py --drift --schema 5-class
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _resolve_status_path() -> Path:
    """Resolve the drift_status.json path using the same priority as the DAG.

    Priority: DRIFT_STATUS_PATH env var (explicit) > MONITORING_DIR env var
    > default. This matches the resolution in dags/auto_retrain_pipeline.py
    so the seeder always writes to the same file the control loop reads.
    """
    explicit = os.environ.get("DRIFT_STATUS_PATH", "").strip()
    if explicit:
        return Path(explicit)
    monitoring_dir = os.environ.get("MONITORING_DIR", "monitoring").strip() or "monitoring"
    return Path(monitoring_dir) / "drift_status.json"


def build_entry(*, drift: bool, insufficient: bool) -> dict:
    """Build one per-schema entry matching the ADR 005 drift-signal contract."""
    now = datetime.now(UTC)
    if insufficient:
        return {
            "status": "insufficient_data",
            "drift_detected": False,
            "timestamp": now.isoformat(),
            "n_rows": 12,
            "report_path": None,
        }
    return {
        "status": "ok",
        "drift_detected": drift,
        "timestamp": now.isoformat(),
        "n_rows": 300,
        "window_start": (now - timedelta(hours=24)).isoformat(),
        "window_end": now.isoformat(),
        "report_path": "monitoring/reports/drift_report_3class.html",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--drift", action="store_true", help="Write drift_detected=true.")
    grp.add_argument("--no-drift", action="store_true", help="Write drift_detected=false.")
    grp.add_argument(
        "--insufficient-data",
        action="store_true",
        help="Write status=insufficient_data.",
    )
    ap.add_argument("--schema", default="3-class", help="Schema key (default 3-class).")
    args = ap.parse_args()

    status_path = _resolve_status_path()

    # NIT FIX (PR #40 review — Marco): merge per-schema instead of
    # overwriting the whole file. Seeding 5-class must not erase the
    # existing 3-class entry (and vice versa).
    payload: dict = {}
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload = existing
        except (json.JSONDecodeError, OSError):
            # Existing file is malformed; start fresh but warn loudly.
            print(
                f"[seed_drift_status] WARNING: {status_path} is malformed;"
                " overwriting with new entry."
            )

    payload[args.schema] = build_entry(drift=args.drift, insufficient=args.insufficient_data)

    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {status_path} (schema={args.schema!r}) -> " f"{json.dumps(payload[args.schema])}")


if __name__ == "__main__":
    main()
