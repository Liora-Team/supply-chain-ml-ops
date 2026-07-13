"""Business impact — translating model errors into euros.

macro-F1 is the model metric; euros are the deployment metric. This page turns the
winner's confusion matrix into a triage cost, compares it to reading every review
by hand, and stress-tests the conclusion against the (unsigned) cost assumptions.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import findings as F

st.title("💶 Business impact — euros, not F1")
st.markdown(
    "A jury cares about deployment, not a metric. So we cost the errors. Scenario: "
    "route negative-leaning reviews to a human agent. A **missed negative** (false "
    "negative) costs more than a **false alarm** (false positive) — so the model "
    "that catches the most negatives can win on euros even at slightly lower F1."
)

st.warning(
    "**Working assumptions, not yet signed off by the mentor.** The euro figures "
    "below show the *shape* of the cost case; real per-case costs would slot "
    "straight in. The conclusion (section 3) holds regardless."
)

# 1. The cost matrix
st.subheader("1. The cost matrix")
c1, c2 = st.columns(2)
c1.metric(
    "Missed negative (FN)",
    f"{F.COST_FN_EUR:.0f} €",
    help="A genuinely unhappy customer slips through — escalation / churn.",
)
c2.metric(
    "False alarm (FP)",
    f"{F.COST_FP_EUR:.0f} €",
    help="A fine review gets flagged — an agent spends a few minutes checking.",
)
# Ratios below are derived from the same F.* constants as the metrics beside
# them, so editing a cost assumption keeps the whole page consistent.
st.caption(
    f"Correct decisions (TP, TN) cost nothing. A miss is treated as "
    f"~{F.COST_FN_EUR / F.COST_FP_EUR:.0f}× a false alarm — losing a customer "
    "hurts more than a quick double-check."
)

# 2. Per 10,000 reviews
st.subheader("2. Cost per 10,000 reviews")
m1, m2, m3 = st.columns(3)
m1.metric("Read every review by hand", f"{F.COST_READ_ALL_PER_10K:,} €")
m2.metric("Let the model triage", f"{F.COST_MODEL_PER_10K:,} €")
m3.metric("Saved", f"{F.COST_SAVED_PER_10K:,} €", delta="cheaper", delta_color="normal")

split = pd.DataFrame(
    {
        "error type": ["Missed negatives (FN)", "False alarms (FP)"],
        "cost on test set (€)": [F.FN_EUR_ON_TEST, F.FP_EUR_ON_TEST],
    }
)
fig = px.bar(
    split,
    x="cost on test set (€)",
    y="error type",
    orientation="h",
    text_auto=",",
    color="error type",
    color_discrete_map={"Missed negatives (FN)": "#d62728", "False alarms (FP)": "#1f77b4"},
    title="Where the model's error cost lands",
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
st.plotly_chart(fig, width="stretch")
st.caption(
    f"Inside the model's mistakes, missed-negatives dominate the bill "
    f"~{F.FN_EUR_ON_TEST / F.FP_EUR_ON_TEST:.1f}×. That tells you which way to "
    "tune the decision threshold: catch more negatives, accept a few more false "
    "alarms."
)

# 3. Robustness
st.subheader("3. Does the conclusion survive different costs?")
sens = pd.DataFrame(
    F.COST_SENSITIVITY,
    columns=[
        "c_FN (€)",
        "c_FP (€)",
        "read-all /10k (€)",
        "model /10k (€)",
        "saved /10k (€)",
        "FN-vs-FP cost",
    ],
)
st.dataframe(
    sens.style.format(
        {"read-all /10k (€)": "{:,}", "model /10k (€)": "{:,}", "saved /10k (€)": "{:,}"}
    ),
    width="stretch",
    hide_index=True,
)
saved = [row[4] for row in F.COST_SENSITIVITY]  # savings/10k column
st.success(
    "**The conclusion is assumption-proof.** Across every cost pair tested "
    "(from 5€/5€ to 100€/20€), model-triage beats reading everything — savings "
    f"range ~{min(saved) / 1000:.0f}k–{max(saved) / 1000:.0f}k € per 10k. And "
    "missed-negatives outweigh false-alarms as long as a miss is worth more "
    f"than **{F.FN_DOMINATES_THRESHOLD:.2f}×** a false alarm — true in any "
    "realistic scenario. Only the headline euro figure moves; *“save money, "
    "catch negatives”* does not."
)
st.caption(
    "Every row is derived from the same served confusion matrix; the 30€/5€ row "
    "reproduces the headline figures above."
)
