"""EDA page — interactive exploration of the 123k-review dataset.

Renders the committed EDA artefacts (via src/eda.py) with Plotly. Headline
aggregates are exact (full dataset); the length boxplot uses the stratified
sample.
"""

import plotly.express as px
import streamlit as st

from src import eda
from src import findings as F

st.title("📊 Dataset exploration")
st.markdown(
    "What the 123k-review dataset looks like before modelling: rating balance, "
    "review length, how ratings vary by category, and the vocabulary that "
    "separates good reviews from bad. These observations motivate the metric "
    "choice and the model design."
)

s = eda.summary()
# Mean rating is derivable from the exact star counts (no need to store it).
_sc = s["star_counts"]
mean_rating = sum(int(k) * v for k, v in _sc.items()) / sum(_sc.values())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{s['n_reviews']:,}")
c2.metric("Categories", s["n_categories"])
c3.metric("Companies", f"{s['n_companies']:,}")
c4.metric(
    "Mean rating",
    f"{mean_rating:.2f}",
    help="Slightly positive midpoint; the distribution is imbalanced, not centred.",
)

# Star rating distribution
st.subheader("1. Star rating distribution")
dist = eda.star_distribution()
fig = px.bar(
    dist,
    x="stars",
    y="count",
    text="count",
    title="Reviews per star rating",
    color="stars",
    color_continuous_scale="RdYlGn",
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Mild class imbalance: 5★ is the largest class, 2★ the smallest — "
    "why macro-F1 (not accuracy) is the primary metric."
)

# Review length by star
st.subheader("2. Review length by star")
samp = eda.sample()
fig = px.box(
    samp,
    x="stars",
    y="word_count",
    color="band",
    title="Word count per star (representative 12,000-review sample)",
    points=False,
    range_y=[0, samp["word_count"].quantile(0.97)],
)
st.plotly_chart(fig, width="stretch")
st.caption(
    f"Lower ratings tend to run longer — unhappy customers explain more. "
    f"Spearman ρ(stars, word count) = {F.LENGTH_STARS_SPEARMAN} "
    f"(weak but significant inverse link)."
)

# Average rating per category
st.subheader("3. Average rating per category")
cat = eda.category_stats()
fig = px.bar(
    cat,
    x="mean_rating",
    y="category",
    orientation="h",
    color="mean_rating",
    color_continuous_scale="RdYlGn",
    hover_data=["n"],
    title="Mean star rating by category",
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Each bar is one business sector the reviewed companies belong to "
    "(e.g. electronics, travel). Average satisfaction varies a lot by "
    "sector — some are structurally happier than others, so the company's "
    "category is itself a useful signal. A one-way ANOVA confirms the "
    f"effect is real: F = {F.CATEGORY_ANOVA_F}, p ≈ {F.CATEGORY_ANOVA_P}."
)

# Supply-chain keyword prevalence
st.subheader("4. Supply-chain keyword prevalence")
kp = eda.keyword_prevalence()
fig = px.bar(
    kp,
    x="keyword",
    y="share",
    text_auto=".1%",
    title="Share of reviews mentioning each theme",
    range_y=[0, kp["share"].max() * 1.3],
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Share of reviews that mention each supply-chain theme (delivery / "
    "refund / defective / price). Flags match whole words only, so "
    "'price' isn't caught inside 'pricey'."
)

# Top tokens per star
st.subheader("5. Top tokens per star rating")
star = st.select_slider("Star rating", options=[1, 2, 3, 4, 5], value=1)
tok = eda.top_tokens_per_star(star)
fig = px.bar(
    tok, x="count", y="token", orientation="h", title=f"Most frequent lemmas in {star}★ reviews"
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, width="stretch")
st.caption(
    "Most common words in reviews at this rating, counted over the full "
    "dataset. Words are reduced to their base form (e.g. 'arrived' → "
    "'arrive') and common filler words are removed first."
)
