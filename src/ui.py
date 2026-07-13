"""Shared Streamlit UI helpers — sidebar controls and sample reviews.

Keeps the per-page code thin: every page calls `sidebar_controls()` to get the
selected label schema, and `SAMPLE_REVIEWS` gives a common dropdown of examples.
"""

from __future__ import annotations

import streamlit as st

# Canned reviews for the Predict / Explain dropdowns — one per sentiment band,
# plus a sarcastic one that classical models tend to miss (the DistilBERT demo).
SAMPLE_REVIEWS = {
    "— type your own —": "",
    "Clear negative": "Terrible service, package never arrived, refund denied.",
    "Clear neutral": "Decent, arrived on time, nothing special.",
    "Clear positive": "Absolutely brilliant, the team went above and beyond.",
    "Delivery complaint": "Item was defective and the delivery was two weeks late.",
    "Sarcastic (hard)": "Great, my package arrived in 3 weeks. Wonderful service.",
}

SCHEMAS = ["3-class", "5-class"]


# Plain-language definitions of the few terms that recur across every page.
# Rendered once on the Home page (via glossary()) so no page has to re-explain them.
GLOSSARY = {
    "Prediction detail (3-class / 5-class)": "How detailed the prediction is. **3-class** "
    "sorts a review into negative / neutral / positive; **5-class** guesses the exact 1–5 "
    "star rating. Switch with the control above.",
    "Confidence": "How sure a model is about its top guess, from 0 to 100%. Higher means "
    "the model is more certain.",
    "macro-F1": "The score we rank models by. It averages how well a model does on each "
    "rating equally, so it only looks good if it handles the rare ratings too "
    "— unlike plain accuracy, which can be fooled by always guessing the "
    "most common rating.",
    "TF-IDF": "The way review text is turned into numbers the classic models can read: "
    "each word gets a weight for how telling it is of that review.",
    "LIME / SHAP": "Two ways to show *why* a model decided what it did, by scoring which "
    "words pushed the prediction up or down. Used on the Explain page.",
}


def glossary() -> None:
    """Render a collapsed plain-language glossary in the main body (Home page)."""
    with st.expander("📖 Glossary"):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}** — {definition}")


def sidebar_controls() -> str:
    """Render the shared sidebar and return the selected schema string.

    3-class = negative / neutral / positive (the operational target).
    5-class = raw 1-5 star prediction (the granular schema).
    """
    st.sidebar.header("Settings")
    schema = st.sidebar.radio(
        "Prediction detail",
        SCHEMAS,
        index=0,
        help="3-class sorts reviews into negative / neutral / positive; "
        "5-class guesses the exact 1–5 star rating.",
    )
    st.sidebar.caption("Every page's models and scores follow this choice.")
    return schema


def review_input(key: str = "review") -> str:
    """Render a sample-review dropdown + text area, return the chosen text.

    A keyed ``text_area`` ignores its ``value=`` arg once session state exists,
    so we sync the sample into session state via the selectbox's on_change
    callback instead — picking a sample fills the box, and the user can edit it.
    """
    sel_key, txt_key = f"{key}_sel", f"{key}_txt"

    def _apply_sample() -> None:
        st.session_state[txt_key] = SAMPLE_REVIEWS[st.session_state[sel_key]]

    st.selectbox(
        "Pick a sample or type your own", list(SAMPLE_REVIEWS), key=sel_key, on_change=_apply_sample
    )
    return st.text_area("Review text", height=120, key=txt_key)
