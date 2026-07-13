"""About page — the full project story, readable on its own for the defense.

Static prose only (no models loaded): business context, problem framing, data,
methodology, metric choice, results, and limitations — the narrative the other
pages demonstrate interactively.
"""

import streamlit as st

st.title("ℹ️ About this project")

st.markdown("""
## Supply Chain — Customer Satisfaction (NLP)

*Team: Marco, Mykola, Dilshana, Luc — mentor: Kilyan.*

> This page is the full written story of the project. The other tabs are the
> interactive version of the same thing — try **Predict** and **Explain** first
> if you'd rather click than read.

### 1. Business context
In a supply chain (supplier → factory → warehouse → outlet → customer),
satisfaction is measured downstream from what customers *write*: reviews on
Trustpilot, marketplaces, social media. Reading that free-text feedback by hand
is slow, so teams usually sample it. The goal here is to **extract structured
signal from the raw text automatically**: turn a written review into a rating and
surface *why* it is positive or negative — useful for triaging dissatisfied
customers and spotting recurring problems (delivery, refunds, defects, price).

### 2. Problem framing
- **Task:** multi-class text classification — predict a review's rating from its
  text alone.
- **Two label schemas:**
  - **5-class** — the raw 1-5 star rating (granular).
  - **3-class** — stars collapsed to **negative {1,2} / neutral {3} / positive
    {4,5}**. This is the operational target: the three bands are what a triage
    workflow actually acts on.
- **Why two:** 5-class is harder (neighbouring stars overlap in language);
  3-class is the business-relevant view. We report both throughout.

### 3. Data
`Kerassy/trustpilot-reviews-123k` — **123,181 reviews across 22 categories**,
freely available on Hugging Face. Mild class imbalance (5★ largest, 2★ smallest),
which is the reason for the metric choice in §6. See the **EDA** page for the
distributions and statistical tests.

### 4. Preprocessing pipeline
Every model consumes text produced by one shared pipeline (`src/preprocessing.py`,
imported by the data script, the API, and this app, so the demo cannot drift
from training):

1. **Clean** — lowercase, strip punctuation / digits.
2. **Stopword removal** — drop high-frequency, low-information words (NLTK).
3. **Lemmatisation** — reduce each word to its dictionary form (WordNet).
4. **Vectorise (classical models only):** **TF-IDF** — turns text into a numeric
   vector where each word is weighted by how characteristic it is of a document
   versus the whole corpus. 10,000-term vocabulary, 1- and 2-word n-grams.

> **No data leakage:** the TF-IDF vocabulary and weights were fitted on the
> *training split only*; the app loads that fitted vectoriser, never re-fits it.

### 5. Models
| Family | What it is | Role |
|---|---|---|
| **Logistic Regression** | linear classifier on TF-IDF | **winner** (3-class macro-F1 ≈ 0.69) |
| **LinearSVC** | linear support-vector classifier | close second; softmaxed decision margins |
| **Random Forest** | bagged decision trees | comparison; weaker here |
| **XGBoost** | gradient-boosted trees | comparison; also supports SHAP TreeExplainer |
| **DistilBERT** | fine-tuned transformer | two-phase fine-tune (freeze → partial unfreeze) |

The **Leaderboard** page ranks them all; **Predict** runs any selection live.

### 6. Why macro-F1 is the primary metric
The classes are imbalanced, so plain **accuracy** would reward a model that just
predicts the majority class. **F1** balances precision and recall;
**macro-F1** averages the per-class F1 *equally*, so a model is only rewarded if
it does well on every class — including the rare ones. A Dummy "predict-majority"
baseline sets the floor (3-class ≈ 0.21, 5-class ≈ 0.08); lift above that floor
is what matters.

### 7. Interpretability — why a prediction happened
- **LIME** — perturbs the review (drops words) and watches how the prediction
  moves, giving each word a signed weight toward the predicted class.
  Model-agnostic; works for every model including DistilBERT.
- **SHAP (TreeExplainer)** — exact Shapley-value attribution per token for the
  tree models (Random Forest / XGBoost).

The **Explain** page runs these live on your own text: green words pushed the
prediction toward the shown class, red pushed away. The **Insights** page adds the
global view — per-class scores, the model's strongest words, and a vocabulary
**leakage** caveat we flag ourselves. The **Business** page translates the model's
errors into euros (the deployment case).

### 8. Reproducibility by construction
Each classical model ships as one **self-contained `Pipeline(TF-IDF → classifier)`**
(`models/pipelines/`, built by `scripts/build_pipelines.py`) fitted on text.
The vocabulary travels inside the same artifact as the weights, so the
vectoriser and classifier can never disagree — a single `joblib.load` gives a
model that predicts from text with no external state to keep in sync. The
build is deterministic: fixed TF-IDF settings + the stored `best_params`
reproduce the reported scores (details in the `scripts/build_pipelines.py`
docstring).
""")
