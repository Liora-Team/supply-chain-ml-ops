"""Leaderboard page — every trained model ranked by macro-F1.

Metrics come from the registry (checkpoint JSON sidecars + DistilBERT eval files);
no models are loaded, so the page is fast. For models the app serves live, the
registry surfaces the metrics measured on the bundled Pipeline(tfidf->clf) it runs.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import registry as R
from src import ui

st.title("🏆 Model leaderboard")
st.markdown(
    "Every trained model ranked by **macro-F1** — the per-class F1 averaged "
    "equally, so a model only scores well if it handles *every* class (the right "
    "metric for this imbalanced dataset; accuracy would flatter a majority-class "
    "guesser). On the committed baseline, Logistic Regression wins both schemas "
    "(the table below is live and would reflect any retrain). The **Served live** column "
    "marks the models you can run on the Predict / Explain pages."
)
st.caption(
    "One number hides a lot — the **Insights** page opens up the winner's "
    "per-class scores and the words it weighs."
)

schema = ui.sidebar_controls()

# One row per model for the chosen schema. For served models the metrics are
# the pipeline's own test scores (registry resolves this); others show their
# recorded scores.
rows = []
for e in R.all_models():
    if e.schema != schema:
        continue
    rows.append(
        {
            "Model": e.algo,
            "Variant": e.variant,
            "macro-F1": e.macro_f1,
            "weighted-F1": e.metrics.get("weighted_f1"),
            "accuracy": e.metrics.get("accuracy"),
            "Served live": "✓" if e.loadable else "",
        }
    )

df = pd.DataFrame(rows).sort_values("macro-F1", ascending=False, na_position="last")

# Bar chart of the headline metric
fig = px.bar(
    df,
    x="macro-F1",
    y="Model",
    color="Variant",
    orientation="h",
    title=f"macro-F1 by model ({schema})",
    range_x=[0, 1],
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, width="stretch")

# Full sortable table
st.subheader("All models")
st.dataframe(
    df.style.format(
        {"macro-F1": "{:.4f}", "weighted-F1": "{:.4f}", "accuracy": "{:.4f}"}, na_rep="—"
    ),
    width="stretch",
    hide_index=True,
    column_config={
        "macro-F1": st.column_config.Column(
            help="Average score across all ratings, each weighted equally — the "
            "primary metric. Rewards a model only if it handles the rare "
            "ratings too."
        ),
        "weighted-F1": st.column_config.Column(
            help="Same kind of score, but ratings that appear more often count "
            "more. Easier to score high on than macro-F1."
        ),
        "accuracy": st.column_config.Column(
            help="Plain percentage correct. Looks good here for the wrong reason "
            "— always guessing the most common rating already scores high."
        ),
        "Served live": st.column_config.Column(
            help="✓ = you can run this model yourself on the Predict / Explain " "pages."
        ),
    },
)

st.caption(
    "**Served live** = the ready-to-run model the app loads on the Predict / "
    "Explain pages; its row shows the score that model gets on held-out reviews "
    "it never saw during training. DistilBERT's scores are from its fine-tuning."
)
with st.expander("Note on XGBoost's score"):
    st.markdown(
        "XGBoost's live macro-F1 here is higher than the number first recorded "
        "during tuning. The early figure came from a GPU `multi:softmax` run; the "
        "CPU `multi:softprob` pipeline the app actually serves is the trustworthy "
        "one, so that is the figure shown."
    )
