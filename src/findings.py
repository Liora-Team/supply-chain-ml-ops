"""Verified result constants that the app surfaces but cannot derive from the
shipped artifacts (the test set and cost JSON are not committed).

Every number is tagged with its source; model metrics are measured against
the served pipelines on the held-out test set.
Coefficients are NOT here — the Insights page reads them live from the served
joblib so they can never drift.

Mirrors how src/eda.py centralises EDA access: one import, one source of truth.
"""

from __future__ import annotations

# Per-class performance of the served Logistic Regression winner
# Source: the SERVED pipelines (models/pipelines/LogReg…joblib) predicting on the
# 24,637-row held-out test set (data/processed/test.csv, written by
# scripts/get_data.py).
# To regenerate after a retrain: load each LogReg joblib, predict on test.csv's
# `review_lemma` column (the same input scripts/build_pipelines.py fits on), and
# read sklearn.metrics.classification_report. macro_f1 equals the mean of the
# per-class F1 rows AND the Leaderboard's pipeline_macro_f1 — one source of truth
# (tests/test_findings.py pins this).
PER_CLASS = {
    "3-class": {
        "labels": ["negative", "neutral", "positive"],
        "rows": [
            # label, precision, recall, f1, support
            ("negative", 0.793, 0.735, 0.763, 8750),
            ("neutral", 0.380, 0.548, 0.449, 4253),
            ("positive", 0.899, 0.803, 0.848, 11634),
        ],
        "macro_f1": 0.687,
        "accuracy": 0.735,
    },
    "5-class": {
        "labels": ["1★", "2★", "3★", "4★", "5★"],
        "rows": [
            ("1★", 0.643, 0.679, 0.660, 4813),
            ("2★", 0.398, 0.410, 0.404, 3937),
            ("3★", 0.390, 0.376, 0.383, 4253),
            ("4★", 0.524, 0.486, 0.504, 5203),
            ("5★", 0.756, 0.771, 0.763, 6431),
        ],
        "macro_f1": 0.543,
        "accuracy": 0.567,
    },
}

# Business cost translation
# Source: ALL numbers below derive from ONE served LogReg 3-class confusion matrix
# on the test set (FN=2318 missed negatives, FP=1678 false alarms over 24,637 rows;
# read-all uses the true class mix: 8750 neg / 15887 non-neg). Triage = flag
# predicted-negative for a human; model cost = only its errors; read-all/10k =
# (n_neg*c_FN + n_pos*c_FP)*10000/N; model/10k = (FN*c_FN + FP*c_FP)*10000/N.
# The euro costs are WORKING ASSUMPTIONS, not signed off by the mentor.
COST_FN_EUR = 30.0  # missed negative review -> escalation / churn
COST_FP_EUR = 5.0  # false alarm -> a few minutes of agent verification

# Per 10,000 reviews, from the served LogReg confusion matrix.
COST_READ_ALL_PER_10K = 138_789  # routing every review to a human (true class mix)
COST_MODEL_PER_10K = 31_631  # letting the model triage (only its errors cost)
COST_SAVED_PER_10K = COST_READ_ALL_PER_10K - COST_MODEL_PER_10K  # ~107,158

# Within the model's errors (test set), missed-negatives dominate ~8.3x in euros.
FN_EUR_ON_TEST = 69_540  # 2,318 FN x 30
FP_EUR_ON_TEST = 8_390  # 1,678 FP x 5

# Sensitivity sweep, every row from the same confusion matrix. Columns:
# (c_FN, c_FP, send_all/10k, model/10k, savings/10k, FN-vs-FP euro ratio).
# The 30/5 base row reproduces the headline constants above exactly.
COST_SENSITIVITY = [
    (30, 5, 138_789, 31_631, 107_158, "8.3x"),
    (10, 5, 67_758, 12_814, 54_944, "2.8x"),
    (50, 10, 242_063, 53_854, 188_209, "6.9x"),
    (100, 20, 484_126, 107_708, 376_418, "6.9x"),
    (20, 20, 200_000, 32_439, 167_561, "1.4x"),
    (5, 5, 50_000, 8_110, 41_890, "1.4x"),
]
# Robustness: model beats read-all for every row above; missed-negatives dominate
# false-alarms whenever a miss is worth more than this fraction of a false alarm
# (= FP/FN count ratio on the served matrix = 1678/2318).
FN_DOMINATES_THRESHOLD = 0.72

# Vocabulary leakage finding
# Source: served LogReg 3-class joblib coefficients:
# `three star` is the #1 neutral-class feature (+3.70 / rank 1 of 10,000).
LEAKAGE_HEADLINE = (
    'Reviewers often type their rating into the text — *"giving this three '
    'stars because…"*. TF-IDF learned the phrase **`three star`** as the single '
    "strongest feature for the neutral class (and `two star` for negative). The "
    "model partly reads the rating off the text instead of inferring sentiment."
)
LEAKAGE_IMPLICATION = (
    "So the reported **0.69 macro-F1 is mildly optimistic** versus production "
    "reviews that don't state their rating. The fix is cheap and planned: strip "
    "the `{one…five} star` bigrams from the vocabulary, refit, and report both "
    "numbers — the gap is the size of the leak. The rest of the vocabulary is "
    "genuine sentiment (terrible, avoid, great, excellent), so the model learned "
    "the right lexicon; only these few features piggyback."
)

# EDA statistics not stored in eda_summary.json
# Source: full-dataset EDA; inputs not committed (see scripts/build_eda_artifacts.py)
LENGTH_STARS_SPEARMAN = -0.275  # review length vs stars (longer -> lower rating)
CATEGORY_ANOVA_F = 16.04
CATEGORY_ANOVA_P = "1.4e-26"
