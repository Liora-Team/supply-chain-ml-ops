"""Build the committed EDA artefacts from the full featurised dataset.

Produces `data/processed/eda_summary.json` (exact full-dataset aggregates) and
`data/processed/eda_sample.parquet` (stratified row sample for distribution
plots). Input is `data/processed/reviews_featurised.parquet` (~82 MB), which
is NOT committed and has no producer in this repo — it comes from the
exploratory data-science phase, which is out of scope here by design (see the
README scope note). That's why only the two small outputs are committed; this
script is kept as the executable record of how they were built.

Headline numbers (star distribution, per-category rating, keyword prevalence)
are exact; distribution plots use the representative sample.

Run:  make eda-artifacts   (only on a machine that has the featurised parquet)
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FULL = DATA / "reviews_featurised.parquet"
SAMPLE_N = 12_000
RANDOM_STATE = 42
KEYWORDS = ["delivery", "refund", "defective", "price"]


def main() -> None:
    df = pd.read_parquet(FULL)
    n = len(df)
    print(f"Loaded {n:,} rows")

    summary: dict = {
        "n_reviews": int(n),
        "n_categories": int(df["category"].nunique()),
        "n_companies": int(df["company"].nunique()),
    }

    # Star distribution — exact counts
    summary["star_counts"] = {
        int(k): int(v) for k, v in df["stars"].value_counts().sort_index().items()
    }

    # Per-category mean rating + volume — exact
    cat = df.groupby("category")["stars"].agg(["mean", "count"]).reset_index()
    cat = cat.sort_values("mean", ascending=False)
    summary["category_stats"] = [
        {"category": r.category, "mean_rating": round(float(r.mean), 3), "n": int(r.count)}
        for r in cat.itertuples()
    ]

    # Keyword prevalence — exact share of reviews flagged
    summary["keyword_prevalence"] = {kw: round(float(df[f"has_{kw}"].mean()), 4) for kw in KEYWORDS}

    # Top tokens per star — most frequent lemmas (cheap, exact over full corpus)
    top_tokens = {}
    for star, g in df.groupby("stars"):
        c = Counter()
        for txt in g["review_lemma"].dropna():
            c.update(txt.split())
        top_tokens[int(star)] = [{"token": t, "count": int(n)} for t, n in c.most_common(12)]
    summary["top_tokens_per_star"] = top_tokens

    (DATA / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote eda_summary.json ({(DATA / 'eda_summary.json').stat().st_size/1024:.0f} KB)")

    # Stratified sample for the word-count box plot — only the columns the EDA
    # page actually reads (pages/1_EDA.py derives its 'band' colour from stars).
    cols = ["stars", "word_count"]
    frac = min(1.0, SAMPLE_N / n)
    # groupby(...).sample keeps the grouping column ('stars'); groupby.apply
    # would drop it under pandas 2.x.
    sample = (
        df[cols].groupby("stars", group_keys=False).sample(frac=frac, random_state=RANDOM_STATE)
    )
    sample.to_parquet(DATA / "eda_sample.parquet", index=False)
    print(
        f"Wrote eda_sample.parquet: {len(sample):,} rows "
        f"({(DATA / 'eda_sample.parquet').stat().st_size/1024:.0f} KB)"
    )


if __name__ == "__main__":
    main()
