"""EDA data access — load the compact artefacts built by build_eda_artifacts.py.

The app never touches the 82 MB featurised parquet. Headline aggregates come
from eda_summary.json (exact, full-dataset); row-level plots use the small
stratified eda_sample.parquet.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
SUMMARY_PATH = DATA / "eda_summary.json"
SAMPLE_PATH = DATA / "eda_sample.parquet"

# Star -> 3-class band, for colouring plots consistently with the model schema.
# Display-string twin of preprocessing.collapse_to_3class (which maps to the
# canonical ints 0/1/2 for training) — plots want names, not label ids.
STAR_BAND = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}


def _require(path: Path) -> Path:
    """Stop with a friendly message if a build artefact is missing."""
    if not path.exists():
        st.error(
            f"EDA artefact `{path.name}` not found — run "
            "`python scripts/build_eda_artifacts.py` to generate it."
        )
        st.stop()
    return path


@lru_cache(maxsize=1)
def summary() -> dict:
    """Load the exact full-dataset aggregate summary (cached)."""
    return json.loads(_require(SUMMARY_PATH).read_text())


@lru_cache(maxsize=1)
def sample() -> pd.DataFrame:
    """Load the stratified row-level sample for distribution plots (cached)."""
    df = pd.read_parquet(_require(SAMPLE_PATH))
    df["band"] = df["stars"].map(STAR_BAND)
    return df


def star_distribution() -> pd.DataFrame:
    """Exact review count per star rating."""
    s = summary()["star_counts"]
    return pd.DataFrame({"stars": [int(k) for k in s], "count": list(s.values())}).sort_values(
        "stars"
    )


def category_stats() -> pd.DataFrame:
    """Exact mean rating + review volume per category, best-rated first."""
    return pd.DataFrame(summary()["category_stats"])


def keyword_prevalence() -> pd.DataFrame:
    """Exact share of reviews flagged for each supply-chain keyword."""
    kp = summary()["keyword_prevalence"]
    return pd.DataFrame({"keyword": list(kp), "share": list(kp.values())})


def top_tokens_per_star(star: int) -> pd.DataFrame:
    """Most frequent lemmas for one star rating (exact, full corpus)."""
    rows = summary()["top_tokens_per_star"][str(star)]
    return pd.DataFrame(rows)
