"""Model insights — per-class performance, what the model weighs, and an honest
leakage caveat.

Three things the leaderboard's single macro-F1 number can't show:
1. *where* the score comes from (per-class precision / recall / F1),
2. *which words* drive each class (coefficients, read live from the served model),
3. a vocabulary-leakage finding we flag ourselves.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import findings as F
from src import registry as R
from src import ui
from src.inference import load_classical

st.title("🧠 Model insights")
st.markdown(
    "The leaderboard ranks models by one number. This page opens that number up: "
    "**how the winner scores on each class, which words it weighs, and one honest "
    "caveat** about its vocabulary."
)

schema = ui.sidebar_controls()

# 1. Per-class performance
st.subheader("1. Per-class performance (Logistic Regression winner)")
pc = F.PER_CLASS[schema]
df = pd.DataFrame(pc["rows"], columns=["class", "precision", "recall", "f1", "support"])

fig = px.bar(
    df,
    x="f1",
    y="class",
    orientation="h",
    range_x=[0, 1],
    text_auto=".2f",
    color="f1",
    color_continuous_scale="RdYlGn",
    title=f"Per-class F1 ({schema}) · macro-F1 = {pc['macro_f1']:.3f}",
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, width="stretch")

st.dataframe(
    df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "support": "{:,}"}),
    width="stretch",
    hide_index=True,
)
# Captions interpolate the same F.PER_CLASS numbers as the chart, so the prose
# can never disagree with the bars after a retrain.
f1 = {row[0]: row[3] for row in pc["rows"]}
if schema == "3-class":
    st.caption(
        f"The macro-F1 of {pc['macro_f1']:.2f} is an **average that hides a split**: "
        f"negative ({f1['negative']:.2f}) and positive ({f1['positive']:.2f}) are "
        "strong — exactly what a triage workflow acts on — while the **neutral 3★ "
        f"class drags it down at {f1['neutral']:.2f}**. That middle band is "
        "genuinely ambiguous text; it's hard for humans too."
    )
else:
    st.caption(
        f"On the harder 5-class task the extreme ratings (1★ {f1['1★']:.2f}, "
        f"5★ {f1['5★']:.2f}) are easiest; the middle ratings overlap in language "
        "and score lower. This is why we collapse to 3 classes for the "
        "operational view."
    )

# 2. What the model weighs (live coefficients)
st.subheader("2. What the model weighs — top words per class")
st.markdown(
    "Logistic Regression is a linear model: each class has one weight per word. "
    "The bars below are read **live from the served model**, so they are exactly "
    "what drives the predictions on the Predict page. **Green** = the word pushes "
    "*toward* that class, **red** = away."
)

# Load the served LogReg pipeline for this schema (best LogReg, loadable).
log_entry = R.best_loadable(schema)
if log_entry is None:
    st.info("Logistic Regression pipeline not available for this schema.")
else:
    pipe = load_classical(log_entry.id)
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    feats = tfidf.get_feature_names_out()
    class_labels = F.PER_CLASS[schema]["labels"]

    pick = st.selectbox(
        "Class", list(range(len(clf.classes_))), format_func=lambda i: class_labels[i]
    )
    row = clf.coef_[pick]
    order = row.argsort()
    top_pos = [(feats[i], float(row[i])) for i in order[::-1][:10]]
    top_neg = [(feats[i], float(row[i])) for i in order[:10]]
    cdf = pd.DataFrame(top_pos + top_neg, columns=["token", "weight"])
    cdf["direction"] = cdf["weight"].apply(lambda w: "toward" if w >= 0 else "against")

    fig = px.bar(
        cdf,
        x="weight",
        y="token",
        orientation="h",
        color="direction",
        color_discrete_map={"toward": "#2ca02c", "against": "#d62728"},
        title=f"Strongest words for / against “{class_labels[pick]}”",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=560)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Top 10 words pushing toward this class (green) and top 10 pushing away "
        "(red), straight from the model's coefficients — no sampling, exact."
    )

# 3. Honest caveat: vocabulary leakage
st.subheader("3. An honest caveat — vocabulary leakage")
st.warning(F.LEAKAGE_HEADLINE)
st.markdown(F.LEAKAGE_IMPLICATION)
if schema == "3-class" and log_entry is not None:
    st.caption(
        "You can see it above: switch the class selector to **neutral** — "
        "`three star` sits at the very top. That's the leak, in the open."
    )
