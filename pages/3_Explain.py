"""Explain page — LIME / SHAP token attribution on the user's own review.

Shows which words pushed the model toward its predicted class. LIME works for
every model (including DistilBERT); SHAP TreeExplainer is offered for the tree
models (RandomForest / XGBoost).
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import registry as R
from src import ui
from src.explain import lime_explain, shap_tree_explain

st.title("🔎 Why did the model decide that?")
st.markdown(
    "See which words made the model decide the way it did, for one review. "
    "**LIME** hides words one at a time and watches how the prediction shifts — "
    "it works for any model, including DistilBERT. **SHAP** fairly splits the "
    "credit among the words and is exact for the tree models (Random Forest / "
    "XGBoost). **Green** = the word pushed the prediction *toward* the shown "
    "rating, **red** = pushed it *away*."
)

schema = ui.sidebar_controls()

# Model choice — loadable classical + DistilBERT
options = [e.id for e in R.classical(schema) if e.loadable]
bert_entry = R.bert(schema)
if bert_entry and bert_entry.loadable:
    options.append(bert_entry.id)
model_id = st.selectbox("Model", options, format_func=lambda i: R.get(i).display_name)
entry = R.get(model_id)

# Method — SHAP only for tree models; LIME for everything
methods = ["LIME"]
if entry.algo in ("RandomForest", "XGBoost"):
    methods.append("SHAP (tree)")
method = st.radio("Attribution method", methods, horizontal=True)

text = ui.review_input(key="explain")

if st.button("Explain", type="primary"):
    if not text.strip():
        st.warning("Enter some review text first.")
        st.stop()

    slow = entry.kind == "bert"
    spinner = "Running DistilBERT + LIME (CPU, ~20-40s)…" if slow else "Computing attribution…"
    with st.spinner(spinner):
        if method == "SHAP (tree)":
            exp = shap_tree_explain(model_id, text)
        else:
            # fewer LIME samples for the slow BERT path
            exp = lime_explain(model_id, text, num_samples=200 if slow else 500)

    st.subheader(f"Prediction: **{exp.pred_display}**  ·  {exp.method}")

    # Signed horizontal bar — green pushes toward the class, red away
    df = pd.DataFrame(exp.tokens, columns=["token", "weight"])
    df["direction"] = df["weight"].apply(lambda w: "toward" if w >= 0 else "against")
    fig = px.bar(
        df,
        x="weight",
        y="token",
        orientation="h",
        color="direction",
        color_discrete_map={"toward": "#2ca02c", "against": "#d62728"},
        title=f"Token contribution toward “{exp.pred_display}”",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Bars are sorted by how much each word mattered. Green pushed the "
        "prediction toward the shown rating; red pushed it away. LIME and SHAP "
        "use different methods but should broadly agree on the top words."
    )
