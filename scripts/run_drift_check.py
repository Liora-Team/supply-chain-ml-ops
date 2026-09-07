"""CLI: run Card 4.1's drift report for one schema, or all.

Usage
-----
    uv run --group orchestration python scripts/run_drift_check.py --schema 3-class
    uv run --group orchestration python scripts/run_drift_check.py            # both schemas

Exits non-zero when drift was detected for any checked schema, so this doubles as an
Airflow task callable (Card 3.1's orchestrator) or a CI/cron gate — 4.4 can branch on
either the exit code or `monitoring/drift_status.json` directly (docs/adr/005).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `monitoring` importable when run as a script from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitoring.drift_report import run_drift_report  # noqa: E402

SCHEMAS = ("3-class", "5-class")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema",
        choices=(*SCHEMAS, "all"),
        default="all",
        help="Which label schema to check (default: all).",
    )
    args = parser.parse_args()

    schemas = SCHEMAS if args.schema == "all" else (args.schema,)
    any_drift = False

    for schema in schemas:
        result = run_drift_report(schema=schema)
        print(f"[{schema}] {json.dumps(result, indent=2)}")
        if result.get("drift_detected"):
            print(f"  -> DRIFT DETECTED for {schema}")
            any_drift = True

    return 1 if any_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
