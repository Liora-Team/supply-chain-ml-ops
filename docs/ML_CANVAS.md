# ML Canvas — Trustpilot Review Rating

The project framing, following the [Machine Learning Canvas v1.1](https://www.ownml.co)
(Louis Dorard) — the ten blocks below match the template 1:1. This is the **single source of
truth** for what we predict, why it's worth predicting, and how we'll know it keeps working.
Confirm/adjust as a team whenever scope shifts (last confirmed: **Jul 13, 2026**).

> **How to read this file:** each block answers the template's question for *our* project,
> then an *In plain terms* line restates it without jargon — this canvas doubles as the
> lesson on framing an ML system before building it.

## Prediction task

Multiclass text classification: review text → star rating. The entity is **one review**;
outcomes are **5-class** (1–5★) or **3-class** (0 = negative 1–2★, 1 = neutral 3★,
2 = positive 4–5★). There is no wait time before observing the truth: the star the user gave
arrives with the review.

*In plain terms:* given the words someone wrote, guess how many stars they gave.

## Decisions

Predictions route and flag reviews by predicted sentiment — e.g. escalate predicted-negative
reviews to a support queue. The operating threshold is tuned with the cost model in
**Impact simulation** below.

*In plain terms:* the model's output is only useful because someone acts on it — here, deciding
which reviews a human looks at first.

## Value proposition

The end-user is a **business or analyst tracking customer satisfaction**. Their objective is to
understand and react to feedback without reading thousands of free-text reviews. The ML system
auto-rates and triages reviews at scale, exposed as a REST API (plus an optional Streamlit UI
for exploration).

*In plain terms:* turn a pile of unread text into a sorted, actionable queue.

## Data collection

Initial training set: the 123k HuggingFace snapshot (see **Data sources**). Continuous update
(Phase 4): ingest new reviews — each arrives already labelled with its star rating, so there is
no labelling cost. Candidate streams for fresh data (live Trustpilot, timestamped public
datasets, category replay) are compared in [DATA_SOURCES.md](DATA_SOURCES.md).

*In plain terms:* we start from a fixed download; later we plug in a stream so the model can
be monitored and retrained on data it hasn't seen.

## Data sources

[`Kerassy/trustpilot-reviews-123k`](https://huggingface.co/datasets/Kerassy/trustpilot-reviews-123k)
(HuggingFace, ~123k reviews) — columns `review`, `stars`, `category`, `company`. Full detail
and future options → [DATA_SOURCES.md](DATA_SOURCES.md).

*In plain terms:* one public dataset is our whole raw-data world for now.

## Impact simulation

Performance is assessed on a held-out stratified test split. The business cost model
(`src/findings.py`): a missed negative review (**false negative**) costs **€30** (an unhappy
customer nobody called back); a false alarm (**false positive**) costs **€5** (a wasted agent
review). Class imbalance is the fairness concern — hence macro-F1 as the primary metric (see
**Objectives** below).

*In plain terms:* before deploying, we price the model's mistakes so "is it good enough?" has
a number, not a feeling.

## Making predictions

**Real-time**: `POST /predict` (`api/main.py`), milliseconds per call, on CPU — featurization
(lemmatise + TF-IDF) happens inside the served pipeline. Batch scoring is possible later via
the same pipelines.

*In plain terms:* one review in, one answer out, fast enough to sit behind a live app.

## Building models

**One served model per label schema** (3-class and 5-class). Classical baseline now
(TF-IDF + sklearn/XGBoost, `scripts/build_pipelines.py`); DistilBERT as a separate service
later. Models are retrained on drift or on schedule (Phase 4), which takes minutes on CPU for
the classical pipelines.

*In plain terms:* start with the simplest model that works, keep a path open to a fancier one,
and decide *when* to retrain before you need to.

## Features

Input representation available at prediction time: the raw review text, transformed by
`src/preprocessing.py` (clean → stopword-removed → lemmatised), then **TF-IDF** for classical
models or tokenised text for DistilBERT.

*In plain terms:* the model never sees raw text — it sees a cleaned, numeric summary of which
words matter.

## Monitoring

In production we track: **macro-F1 on labelled traffic** (value creation), **input drift**
(Evidently), and **latency/throughput/system health** (Prometheus + Grafana). Built in
Phase 4 — the concrete tasks live in [MILESTONES.md](../MILESTONES.md).

*In plain terms:* a deployed model degrades silently; monitoring is how we notice before the
users do.

---

## Objectives & key metrics

- **Goal:** serve reliable review-rating predictions and keep the model maintainable in production.
- **Primary metric: macro-F1** — averages per-class F1 equally, so a model only scores well if
  it also handles the rare ratings (unlike accuracy, which is fooled by the majority class).
- **Secondary:** per-class precision/recall/F1, weighted-F1, accuracy (leaderboard in `src/findings.py`).
- **Baseline to beat:** LogReg (TF-IDF, class-balanced) — **macro-F1 = 0.687 (3-class)**,
  rebuilt from a fresh `make data` pull under the pinned sklearn.
