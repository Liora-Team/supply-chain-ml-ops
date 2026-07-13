"""Predict page — run the user's review through every selected model.

Classical models are self-contained Pipeline(tfidf->clf) objects from models/pipelines/.
DistilBERT is loaded lazily only when the user opts in (it is ~255 MB and slow on
CPU). Results are shown as a comparison table + a grouped probability bar chart.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import registry as R
from src import ui
from src.inference import predict_bert, predict_classical

st.title("🎯 Predict a review's rating")
st.markdown(
    "Type a review and compare every model on the **same input**. Use the sidebar "
    "to switch between **3-class** (negative / neutral / positive) and **5-class** "
    "(1-5 stars). Classical models run instantly; **DistilBERT** is opt-in (it "
    "loads ~255 MB on first use, then is fast). Bars show each model's predicted "
    "class probabilities."
)

schema = ui.sidebar_controls()

# Model selection — loadable classical models for this schema, best F1 first.
# Default to the top 3 so the comparison stays readable; the full list is a click away.
loadable = [e for e in R.classical(schema) if e.loadable]
default_ids = [e.id for e in loadable if e.algo in ("LogReg", "LinearSVC", "XGBoost")][:3]
chosen = st.multiselect(
    "Classical models to compare",
    options=[e.id for e in loadable],
    default=default_ids,
    format_func=lambda i: R.get(i).display_name,
)

# DistilBERT is opt-in (lazy load — heavy)
bert_entry = R.bert(schema)
use_bert = False
if bert_entry and bert_entry.loadable:
    use_bert = st.checkbox("Include DistilBERT (slower — loads ~255 MB on first use)", value=False)

text = ui.review_input(key="predict")

# Run inference on click
if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Enter some review text first.")
        st.stop()
    if not chosen and not use_bert:
        st.warning("Select at least one model.")
        st.stop()

    preds = []
    for mid in chosen:
        preds.append(predict_classical(mid, text))
    if use_bert:
        with st.spinner("Running DistilBERT (first call also loads the model)…"):
            preds.append(predict_bert(schema, text))

    # Comparison table: one row per model
    rows = []
    for p in preds:
        e = R.get(p.model_id)
        top_p = max(p.probs.values()) if p.probs else None
        rows.append(
            {
                "Model": e.display_name,
                "Prediction": p.label_display,
                "Confidence": f"{top_p:.0%}" if top_p is not None else "—",
                "Score type": p.proba_kind,
            }
        )
    st.subheader("Predictions")
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Confidence": st.column_config.Column(
                help="How sure this model is about its top guess (0–100%)."
            ),
            "Score type": st.column_config.Column(
                help="How the confidence was computed. `proba` = a real "
                "probability. `softmax(...)` = a probability-like score for "
                "models that don't output true probabilities (LinearSVC, "
                "DistilBERT) — comparable across models, but not calibrated."
            ),
        },
    )

    # Grouped probability bars across models
    long = []
    for p in preds:
        short = R.get(p.model_id).algo
        for label, prob in p.probs.items():
            long.append({"Model": short, "Class": label, "Probability": prob})
    if long:
        fig = px.bar(
            pd.DataFrame(long),
            x="Class",
            y="Probability",
            color="Model",
            barmode="group",
            title="Predicted class probabilities",
            range_y=[0, 1],
        )
        st.plotly_chart(fig, width="stretch")

    st.caption(
        "Each bar is how likely a model thinks each class is; the bars for one "
        "model add up to 100%. Some models (LinearSVC, DistilBERT) don't give "
        "true probabilities, so their bars are a probability-like score — fine "
        "for comparing, but not exact odds. Hover the **Score type** column "
        "above to see which is which."
    )
