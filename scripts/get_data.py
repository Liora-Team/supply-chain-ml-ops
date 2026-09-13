# scripts/get_data.py
"""Collect + preprocess the Trustpilot reviews dataset into a single canonical location.

Downloads `Kerassy/trustpilot-reviews-123k` from the Hugging Face hub, applies the SAME
preprocessing the models were trained with (`src.preprocessing.preprocess_text`), attaches
the 3-class sentiment label, and writes a stratified train/test split to `data/processed/`.

This is the Phase-1 "collect & preprocess data" entry point; downstream,
`scripts/build_pipelines.py` (make train) fits on the columns written here.

The steps are plain functions (`load_raw` -> `preprocess` -> `split_and_write`) so the
Airflow data DAG (Card 3.1, `dags/data_pipeline.py`) can import them as task callables
instead of shelling out to this script.

Usage
-----
    make data                              # full dataset
    uv run --group data python scripts/get_data.py --sample 500   # quick smoke test

The written CSVs are gitignored; version them with DVC (`dvc add` + `dvc push`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Make `src` importable when run as a script from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import collapse_to_3class, preprocess_text  # noqa: E402

if TYPE_CHECKING:  # pandas is a core dependency, but keep runtime imports lazy anyway
    import pandas as pd

DATASET_ID = "Kerassy/trustpilot-reviews-123k"
OUT_DIR = ROOT / "data" / "processed"


def load_raw(sample: int = 0, category: str | None = None) -> pd.DataFrame:
    """Download the raw dataset and return the cleaned (review, stars) frame.

    Card 4.4 (Option 3) may pass `category` to replay one held-out category
    slice, so each retrain genuinely sees new data. When `category` is None the
    behaviour is unchanged (full dataset).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - guidance path
        raise SystemExit(
            "Missing dependency. Install the data group: `uv sync --group data` "
            "(or run via `make data`)."
        ) from e

    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()

    if category is not None:
        if "category" not in df.columns:
            raise SystemExit(
                f"Cannot filter by category {category!r}: " "dataset has no 'category' column."
            )
        df = df[df["category"] == category].copy()
        print(f"  Filtered to category {category!r}: {len(df):,} rows")
        if df.empty:
            raise SystemExit(f"Category {category!r} matched 0 rows; " "check the category name.")

    if sample:
        df = df.head(sample).copy()
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    if not {"review", "stars"} <= set(df.columns):
        raise SystemExit(f"Unexpected dataset columns: {list(df.columns)} " "(need review, stars)")
    df = df[["review", "stars"]].dropna()
    df["stars"] = df["stars"].astype(int)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add the `review_lemma` and `label_3class` columns the models train on."""
    print("Preprocessing (clean -> stopwords -> lemmatise) ...")
    df = df.copy()
    df["review_lemma"] = df["review"].map(preprocess_text)
    df["label_3class"] = collapse_to_3class(df["stars"].to_numpy())
    return df


def split_and_write(
    df: pd.DataFrame,
    *,
    out_dir: Path = OUT_DIR,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Stratified train/test split written to `out_dir`; returns the two CSV paths."""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:  # pragma: no cover - guidance path
        raise SystemExit(
            "Missing dependency. Install the data group: `uv sync --group data` "
            "(or run via `make data`)."
        ) from e

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["stars"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(
        f"Wrote {len(train):,} train + {len(test):,} test rows to {out_dir}/"
        " (train.csv, test.csv). These are gitignored — version them with DVC."
    )
    return train_path, test_path


def append_to_train_only(
    df: pd.DataFrame,
    *,
    out_dir: Path = OUT_DIR,
    train_name: str = "train.csv",
    test_name: str = "test.csv",
    dedupe: bool = True,
) -> Path:
    """Append new rows to the existing training split while keeping test split fixed.

    Card 4.4 Option 3 category replay must preserve promotion-gate comparability
    (Card 3.2): candidate and production must be evaluated on the same fixed test
    set.

    Requires baseline {train.csv, test.csv} to already exist (via a prior full run
    or DVC pull). Rows whose `review` text already appears in test.csv are excluded
    before appending, to prevent evaluation-set leakage into the training set.

    Note: because the current baseline was split from the full dataset (which
    includes all RETRAIN_CATEGORIES), most/all replayed rows already exist in
    either train.csv or test.csv. After filtering and deduplication, a category
    replay may add few or zero genuinely new rows. Holding retrain categories out
    of the initial baseline split is a documented follow-up improvement (ADR 006).
    """
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / train_name
    test_path = out_dir / test_name

    # Baseline split is required; cannot append without it.
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(
            "Incremental category replay requires an existing baseline split: "
            f"{train_path} and {test_path}. "
            "Run the data pipeline once without a category (full split), "
            "or restore them via DVC (`make pull`)."
        )

    existing = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Ensure column compatibility and stable order.
    if set(existing.columns) != set(df.columns):
        raise SystemExit(
            "Column mismatch between baseline train.csv and incoming slice. "
            f"baseline={list(existing.columns)} incoming={list(df.columns)}"
        )

    # BLOCKER FIX (PR #40 review — Marco): protect the fixed Card 3.2
    # evaluation set. Incoming rows whose `review` text appears in test.csv
    # must not enter train.csv, because the promotion gate evaluates the
    # candidate on that same fixed test set.
    # Fail closed: if the `review` column is missing, we cannot guarantee
    # protection.
    if "review" not in existing.columns or "review" not in test_df.columns:
        raise SystemExit(
            "Cannot protect the fixed evaluation set: expected column 'review' "
            "in both baseline train.csv and test.csv. "
            "Refusing to append without the leakage guard."
        )
    if "review" not in df.columns:
        raise SystemExit(
            "Cannot protect the fixed evaluation set: expected column 'review' "
            "in the incoming slice. "
            "Refusing to append without the leakage guard."
        )

    test_reviews: set[str] = set(test_df["review"].dropna())
    n_before_filter = len(df)
    df_filtered = df[~df["review"].isin(test_reviews)].copy()
    n_excluded = n_before_filter - len(df_filtered)
    if n_excluded > 0:
        print(
            f"[append_to_train_only] Excluded {n_excluded:,} row(s) whose "
            "review text appears in test.csv (leakage guard)."
        )

    df_aligned = df_filtered[existing.columns]
    combined = pd.concat([existing, df_aligned], ignore_index=True)

    if dedupe:
        # Deduplicate on `review` text to avoid duplicating rows already in
        # train.csv.
        combined = combined.drop_duplicates(subset=["review"], keep="first")

    combined.to_csv(train_path, index=False)
    print(
        f"[append_to_train_only] "
        f"train_before={len(existing):,} "
        f"incoming={len(df_aligned):,} (after test-set filter) "
        f"train_after={len(combined):,} "
        f"test_unchanged={len(test_df):,}"
    )
    return train_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Use only the first N rows (0 = all).",
    )
    ap.add_argument(
        "--category",
        type=str,
        default=None,
        help=("Train on a single category slice (Card 4.4 Option 3). " "None = full dataset."),
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split.",
    )
    args = ap.parse_args()

    df = preprocess(load_raw(sample=args.sample, category=args.category))

    if args.category is not None:
        append_to_train_only(df, out_dir=OUT_DIR)
    else:
        split_and_write(df, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
    main()
