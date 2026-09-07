"""Reference vs. current datasets for Card 4.1 drift detection.

Reference = the training distribution (pulled via DVC into ``data/processed/train.csv``,
Card 3.1's data DAG or a manual ``dvc pull``). Current = a recent window of logged
``/predict`` requests (``monitoring/request_store.py``).

Both frames expose the same text-derived feature columns (``text``, ``text_length``,
``word_count``) so Evidently can compare them column-for-column
(``monitoring/drift_report.py``, next commit). Label/prediction columns are deliberately
excluded: Card 4.1 scopes this to input (covariate) drift on the review text itself, not
label or concept drift — see ``docs/TODO.md`` Card 4.1's subtasks.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from monitoring.request_store import get_recent

# Overridable for tests; defaults to the repo's canonical DVC-tracked location.
REFERENCE_DIR = Path(os.environ.get("REFERENCE_DATA_DIR", "data/processed"))
TRAIN_FILE = "train.csv"

FEATURE_COLUMNS = ["text", "text_length", "word_count"]


def _text_features(texts: pd.Series) -> pd.DataFrame:
    """Derive the shared feature columns from a raw-text series."""
    text = texts.astype(str)
    return pd.DataFrame(
        {
            "text": text,
            "text_length": text.str.len(),
            "word_count": text.str.split().map(len),
        }
    )


def load_reference(sample: int | None = None) -> pd.DataFrame:
    """Training distribution, read from the DVC-tracked ``train.csv``.

    Assumes ``dvc pull`` has already restored ``data/processed/train.csv`` — this
    function reads the file, it does not shell out to DVC itself (that stays an
    explicit orchestrator/CI step, same pattern as the rest of the repo).
    """
    path = REFERENCE_DIR / TRAIN_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `dvc pull data/processed/train.csv.dvc` "
            "(or `make pull`) before running the drift report."
        )

    df = pd.read_csv(path)
    if "review" not in df.columns:
        raise ValueError(f"{path} is missing the expected 'review' column: {list(df.columns)}")

    features = _text_features(df["review"])
    if sample:
        features = features.sample(n=min(sample, len(features)), random_state=42)
    return features.reset_index(drop=True)


def load_current(schema: str, window_hours: int, max_rows: int) -> pd.DataFrame:
    """Recent ``/predict`` requests for ``schema``, as the same text-derived features.

    Returns an empty (but correctly-columned) frame when the store has no rows yet —
    callers (the drift job) decide what "insufficient data" means, this function just
    doesn't fail.
    """
    rows = get_recent(schema=schema, window_hours=window_hours, max_rows=max_rows)
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    raw = pd.DataFrame([dict(row) for row in rows])
    return _text_features(raw["text"])
