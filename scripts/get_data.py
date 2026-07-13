"""Collect + preprocess the Trustpilot reviews dataset into a single canonical location.

Downloads `Kerassy/trustpilot-reviews-123k` from the Hugging Face hub, applies the SAME
preprocessing the models were trained with (`src.preprocessing.preprocess_text`), attaches
the 3-class sentiment label, and writes a stratified train/test split to `data/processed/`.

This is the Phase-1 "collect & preprocess data" entry point; downstream,
`scripts/build_pipelines.py` (make train) fits on the columns written here.

Usage
-----
    make data                              # full dataset
    uv run --group data python scripts/get_data.py --sample 500   # quick smoke test

The written CSVs are gitignored; version them with DVC in Phase 2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when run as a script from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import collapse_to_3class, preprocess_text  # noqa: E402

DATASET_ID = "Kerassy/trustpilot-reviews-123k"
OUT_DIR = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=0, help="Use only the first N rows (0 = all).")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
        from sklearn.model_selection import train_test_split
    except ImportError as e:  # pragma: no cover - guidance path
        raise SystemExit(
            "Missing dependency. Install the data group: `uv sync --group data` "
            "(or run via `make data`)."
        ) from e

    print(f"Loading {DATASET_ID} …")
    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()
    if args.sample:
        df = df.head(args.sample).copy()
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    if not {"review", "stars"} <= set(df.columns):
        raise SystemExit(f"Unexpected dataset columns: {list(df.columns)} (need review, stars)")
    df = df[["review", "stars"]].dropna()
    df["stars"] = df["stars"].astype(int)

    print("Preprocessing (clean → stopwords → lemmatise) …")
    df["review_lemma"] = df["review"].map(preprocess_text)
    df["label_3class"] = collapse_to_3class(df["stars"].to_numpy())

    train, test = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["stars"]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(
        f"Wrote {len(train):,} train + {len(test):,} test rows to {OUT_DIR}/"
        " (train.csv, test.csv). These are gitignored — version them with DVC in Phase 2."
    )


if __name__ == "__main__":
    main()
