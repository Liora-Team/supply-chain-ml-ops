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
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

MONITORING_DIR = Path(os.environ.get("MONITORING_DIR", "monitoring"))
STATUS_PATH = MONITORING_DIR / "drift_status.json"


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
        "--insufficient-data", action="store_true", help="Write status=insufficient_data."
    )
    ap.add_argument("--schema", default="3-class", help="Schema key (default 3-class).")
    args = ap.parse_args()

    payload = {args.schema: build_entry(drift=args.drift, insufficient=args.insufficient_data)}
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {STATUS_PATH} -> {json.dumps(payload)}")


if __name__ == "__main__":
    main()
