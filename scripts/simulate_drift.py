"""Simulate drift by replaying a held-out category slice through `/predict`.

Card 4.1's proof-of-work step: shows the drift pipeline (request_store -> datasets ->
drift_report -> drift_status.json / Prometheus gauge) actually flips `drift_detected`
on a real covariate shift, not just on synthetic unit-test data.

Uses **Option 3** from docs/DATA_SOURCES.md — hold out one category from the raw
`Kerassy/trustpilot-reviews-123k` dataset and feed it back as a "batch arriving over
time". Note `data/processed/train.csv` (scripts/get_data.py) drops the `category`
column, so this script re-loads the raw HF dataset directly rather than reading the
processed CSV.

Usage
-----
    # See what categories exist and how large each is, to pick a good skewed slice.
    uv run --group data python scripts/simulate_drift.py --list-categories

    # Replay 150 reviews from one category through a locally running API, then
    # immediately run the drift check (rather than waiting for the schedule) and
    # print the result.
    export JWT_SECRET_KEY=dev-test-key   # must match the running API's secret
    uv run --group data python scripts/simulate_drift.py \\
        --category Electronics --n 150 --run-drift-check

No new dependencies: HTTP calls use the stdlib (`urllib.request`), matching Option 3's
"zero new deps" framing in docs/DATA_SOURCES.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Make `api`/`monitoring` importable when run as a script from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_ID = "Kerassy/trustpilot-reviews-123k"


def _load_dataset_df():
    """Lazy import so --list-categories/--category don't need `datasets` just to --help."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - guidance path
        raise SystemExit(
            "Missing dependency. Install the data group: `uv sync --group data`."
        ) from e

    print(f"Loading {DATASET_ID} …")
    return load_dataset(DATASET_ID, split="train").to_pandas()


def list_categories() -> None:
    df = _load_dataset_df()
    counts = df["category"].value_counts()
    print(f"{len(counts)} categories, {len(df):,} rows total:\n")
    for category, count in counts.items():
        print(f"  {category:<30} {count:>7,}")


def load_category_slice(category: str, n: int):
    """Return up to `n` review texts for `category`, oldest-loaded-order (deterministic)."""
    df = _load_dataset_df()
    if category not in set(df["category"]):
        raise SystemExit(
            f"Unknown category {category!r}. Run with --list-categories to see valid values."
        )
    slice_df = df[df["category"] == category].dropna(subset=["review"]).head(n)
    if slice_df.empty:
        raise SystemExit(f"Category {category!r} has no usable rows.")
    return slice_df["review"].tolist()


def mint_token(subject: str = "simulate-drift") -> str:
    """Sign a token with this process's JWT_SECRET_KEY (must match the running API's)."""
    from api.auth import create_access_token

    return create_access_token(subject)


def _post_once(req: urllib.request.Request) -> tuple[bool, int | None]:
    """One attempt. Returns (ok, retry_after_seconds) — retry_after is set on a 429."""
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True, None
    except urllib.error.HTTPError as e:
        if e.code == 429:
            retry_after = e.headers.get("Retry-After")
            return False, int(retry_after) if retry_after else 2
        raise


def replay(
    texts: list[str],
    *,
    base_url: str,
    token: str,
    schema: str,
    sleep: float,
    max_retries: int = 5,
) -> tuple[int, int]:
    """POST each text to `/predict`. Returns (n_ok, n_failed).

    Retries on HTTP 429 (rate limit — see the API's RATE_LIMIT_PER_MINUTE) with
    backoff honouring `Retry-After` when present, so a `--sleep` that's a bit too
    fast loses time, not data. `sleep` should already be >= 60 / RATE_LIMIT_PER_MINUTE
    (default 60/min -> 1.0s) for the replay to run cleanly without relying on retries.
    """
    ok = failed = 0
    for i, text in enumerate(texts, start=1):
        payload = json.dumps({"text": text, "schema": schema}).encode()
        req = urllib.request.Request(
            f"{base_url}/predict",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        succeeded = False
        try:
            for attempt in range(max_retries + 1):
                succeeded, retry_after = _post_once(req)
                if succeeded:
                    break
                if attempt < max_retries:
                    time.sleep(retry_after)
            if succeeded:
                ok += 1
            else:
                failed += 1
                print(f"  [{i}/{len(texts)}] gave up after {max_retries} retries (still 429)")
        except urllib.error.URLError as e:
            failed += 1
            print(f"  [{i}/{len(texts)}] request failed: {e}")
        if sleep:
            time.sleep(sleep)
        if i % 25 == 0 or i == len(texts):
            print(f"  sent {i}/{len(texts)} (ok={ok}, failed={failed})")
    return ok, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print categories and row counts, then exit.",
    )
    parser.add_argument("--category", help="Category to replay (see --list-categories).")
    parser.add_argument("--n", type=int, default=150, help="Max reviews to replay (default: 150).")
    parser.add_argument("--schema", default="3-class", choices=("3-class", "5-class"))
    parser.add_argument("--base-url", default="http://localhost:8000", help="Running API base URL.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.1,
        help="Delay between requests, in seconds. Default assumes the API's default "
        "RATE_LIMIT_PER_MINUTE=60 (i.e. ~1 req/sec); lower it if you raised that limit.",
    )
    parser.add_argument(
        "--run-drift-check",
        action="store_true",
        help="Run monitoring.drift_report.run_drift_report(schema) immediately after "
        "the replay and print the result, instead of waiting for the schedule.",
    )
    args = parser.parse_args()

    if args.list_categories:
        list_categories()
        return 0

    if not args.category:
        parser.error("--category is required (or pass --list-categories)")

    texts = load_category_slice(args.category, args.n)
    print(f"Replaying {len(texts)} '{args.category}' reviews against {args.base_url} …")

    token = mint_token()
    ok, failed = replay(
        texts, base_url=args.base_url, token=token, schema=args.schema, sleep=args.sleep
    )
    print(f"Done: {ok} ok, {failed} failed.")

    if args.run_drift_check:
        from monitoring.drift_report import run_drift_report

        print("Running the drift check now …")
        result = run_drift_report(schema=args.schema)
        print(json.dumps(result, indent=2))
        if result.get("drift_detected"):
            print(f"-> DRIFT DETECTED for {args.schema} (as expected for a skewed slice).")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
