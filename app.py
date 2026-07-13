"""Trustpilot Review Star Predictor — Streamlit app entry point (Home).

Multi-page app (see pages/). This Home page pitches the project and shows a few
headline dataset / model numbers. The heavy lifting lives in src/ modules so the
API and the app share one preprocessing + inference path.

Run locally:  uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src import registry as R
from src import ui

st.set_page_config(
    page_title="Trustpilot Star Predictor",
    page_icon="⭐",
    layout="wide",
)


def home() -> None:
    """Home page — project pitch, headline numbers, and the shared glossary."""
    st.title("⭐ Trustpilot Review — Customer Satisfaction NLP")
    st.markdown("""
Predict a Trustpilot review's **star rating from its text**, compare classical
ML against a fine-tuned **DistilBERT**, and see *why* a model decided what it did
(SHAP / LIME).

This is the demo surface for the *Supply Chain — Customer Satisfaction* project.
The dataset is `Kerassy/trustpilot-reviews-123k`: 123k reviews
across 22 categories.

**In plain terms:** paste a customer review, get a predicted star rating, and see
which words drove that prediction. Pick a page from the sidebar to start — and
open the **Glossary** below any time a term is unfamiliar.
""")

    # Headline numbers, pulled live from the model registry so they never go stale.
    best3 = R.best_loadable("3-class")
    bert3 = R.bert("3-class")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", "123k")
    c2.metric("Categories", "22")
    if best3:
        c3.metric(
            "Best classical (3-class)",
            f"{best3.macro_f1:.3f} macro-F1",
            help=f"{best3.display_name}",
        )
    if bert3 and bert3.macro_f1:
        c4.metric("DistilBERT (3-class)", f"{bert3.macro_f1:.3f} macro-F1")

    st.divider()

    st.subheader("What's inside")
    st.markdown("""
| Page | What it does |
|---|---|
| **EDA** | Interactive exploration of the 123k-review dataset |
| **Predict** | Type a review → star prediction from every model, side by side |
| **Explain** | SHAP / LIME on *your* text — which words drove the prediction |
| **Leaderboard** | Every trained model ranked by macro-F1 |
| **Insights** | Per-class scores, what words the model weighs, and a leakage caveat |
| **Business** | Model errors translated into euros — the deployment case |
| **About** | Methodology, metric choice, and links |

Use the **sidebar** on each page to switch between *3-class* (neg / neu / pos)
and *5-class* (1-5 stars).
""")

    st.info(
        "Primary metric is **macro-F1** — the dataset is class-imbalanced, so plain "
        "accuracy would flatter a majority-class guesser.",
    )

    ui.glossary()  # plain-language glossary, rendered once here for all pages


pages = st.navigation(
    [
        st.Page(home, title="Home", icon="🏠", default=True),
        st.Page("pages/1_EDA.py", title="EDA", icon="📊"),
        st.Page("pages/2_Predict.py", title="Predict", icon="🎯"),
        st.Page("pages/3_Explain.py", title="Explain", icon="🔎"),
        st.Page("pages/4_Leaderboard.py", title="Leaderboard", icon="🏆"),
        st.Page("pages/5_Insights.py", title="Insights", icon="🧠"),
        st.Page("pages/6_Business.py", title="Business", icon="💶"),
        st.Page("pages/7_About.py", title="About", icon="ℹ️"),
    ]
)
pages.run()
