"""Ops page — internal operator view of project state.

Cross-schema cockpit for the team (not the demo audience): model registry with the
API's local fallback serving default, dataset + versioning state, and tracking
configuration.
Everything is read in-process (registry scan, committed EDA artifact, env vars, a
local git call) and every external read degrades to a note on a fresh clone, so
the page renders with nothing configured. No sidebar schema switch here — this
page always shows both schemas.
"""

import os
import subprocess
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src import eda
from src import registry as R

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ("3-class", "5-class")


def _serving_id(schema: str) -> str | None:
    """Local fallback model id for a schema, or None when nothing is loadable.

    Mirrors api/main.py::default_model_id — keep in sync. Replicated rather than
    imported so the app never depends on the `api` dependency group. When
    MLFLOW_TRACKING_URI is set the API prefers the MLflow @production alias
    (api/main.py::production_model), so this is its fallback, not always the
    model actually served.
    """
    try:
        override = os.environ.get(f"DEFAULT_MODEL_{schema.replace('-class', 'class').upper()}")
        if override:
            return override
        loadable = [e for e in R.classical(schema) if e.loadable]
        return loadable[0].id if loadable else None  # classical() sorts best macro-F1 first
    except Exception:
        return None


def _git_sha() -> str | None:
    """Short SHA of the checked-out commit, or None (no git binary, no .git, timeout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _dvc_configured() -> bool:
    """Whether DVC is initialised in this clone."""
    try:
        return (ROOT / ".dvc").is_dir()
    except Exception:
        return False


def _secret_status(name: str) -> str:
    """Presence marker for a secret env var — the value itself is never rendered."""
    return "✓ set" if os.environ.get(name) else "—"


st.title("🛠️ Ops — internal dashboard")
st.markdown(
    "Operator view of project state: which models the registry knows, what the API "
    "would serve, where the data stands, and how tracking is configured. This page "
    "is for the team — the other pages are the demo."
)

entries = R.all_models()
servable = [e for e in entries if e.loadable]
serving = {s: _serving_id(s) for s in SCHEMAS}
summary_ok = eda.SUMMARY_PATH.exists()
s = eda.summary() if summary_ok else None
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")

# Status strip
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Models registered", len(entries))
c2.metric("Servable", len(servable))
c3.metric("Serving (3-class)", serving["3-class"] or "—")
c4.metric("Dataset rows", f"{s['n_reviews']:,}" if summary_ok else "—")
c5.metric("MLflow", "configured" if mlflow_uri else "not set")

st.divider()

# Model registry — both schemas in one table, ★ marks the API's local fallback
# default per schema.
st.subheader("Model registry")
rows = []
for e in entries:
    rows.append(
        {
            "Model": e.algo,
            "Variant": e.variant,
            "Schema": e.schema,
            "macro-F1": e.macro_f1,
            "Servable": "✓" if e.loadable else "",
            "Serving": "★" if e.id == serving[e.schema] else "",
            "Kind": e.kind,
        }
    )
if rows:
    df = pd.DataFrame(rows).sort_values(
        ["Schema", "macro-F1"], ascending=[True, False], na_position="last"
    )
    st.dataframe(
        df.style.format({"macro-F1": "{:.4f}"}, na_rep="—"),
        width="stretch",
        hide_index=True,
        column_config={
            "Servable": st.column_config.Column(
                help="✓ = a ready-to-load pipeline (or DistilBERT weights) exists in this clone."
            ),
            "Serving": st.column_config.Column(
                help="★ = the API's local fallback default for that schema (env "
                "override, else best loadable classical). With MLflow configured, "
                "the API may serve the registry @production model instead."
            ),
        },
    )
if not servable:
    st.info(
        "No servable model in this clone — model weights are not checked out. "
        "Run `make pull` to restore them from the DVC remote."
    )

st.divider()

# Data & versioning
st.subheader("Data & versioning")
if summary_ok:
    d1, d2, d3 = st.columns(3)
    d1.metric("Reviews", f"{s['n_reviews']:,}")
    d2.metric("Categories", s["n_categories"])
    d3.metric("Companies", f"{s['n_companies']:,}")
    fig = px.bar(eda.star_distribution(), x="stars", y="count", title="Star balance")
    st.plotly_chart(fig, width="stretch")
else:
    st.info(
        "EDA summary artifact missing (data/processed/eda_summary.json) — "
        "run scripts/build_eda_artifacts.py to rebuild it."
    )

v1, v2 = st.columns(2)
sha = _git_sha()
v1.metric("Git commit", sha or "unavailable")
v2.metric("DVC", "configured" if _dvc_configured() else "not configured")
if not _dvc_configured():
    v2.caption("DVC not initialised in this clone.")

st.divider()

# MLflow & tracking — reflect env config only. No HTTP ping: that would need
# requests/httpx, which are deliberately absent from the `app` dependency group.
st.subheader("MLflow & tracking")
if mlflow_uri:
    st.success(f"Tracking URI: {mlflow_uri}")
else:
    st.info(
        "MLFLOW_TRACKING_URI is not set — experiment tracking lands with "
        "Cards 2.1 / 2.4. Runs would fall back to a local `mlruns/` directory."
    )

with st.expander("Environment / config"):
    shown = ["DEFAULT_MODEL_3CLASS", "DEFAULT_MODEL_5CLASS", "DISTILBERT_3CLASS"]
    shown += ["DISTILBERT_5CLASS", "MLFLOW_TRACKING_URI"]
    masked = ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
    env_rows = [{"Variable": v, "Value": os.environ.get(v) or "—"} for v in shown]
    env_rows += [{"Variable": v, "Value": _secret_status(v)} for v in masked]
    st.dataframe(pd.DataFrame(env_rows), width="stretch", hide_index=True)
    st.caption("Credentials are shown as set / unset only — values are never rendered.")

st.divider()

# Deep links — one home per tool; greyed until its phase lands.
st.subheader("Tools")
t1, t2, t3, t4 = st.columns(4)
t1.link_button("API docs", "http://localhost:8000/docs", width="stretch")
mlflow_url = mlflow_uri if mlflow_uri and mlflow_uri.startswith("http") else "http://localhost"
t2.link_button(
    "MLflow",
    mlflow_url,
    disabled=not (mlflow_uri and mlflow_uri.startswith("http")),
    width="stretch",
)
t3.link_button("Airflow", "http://localhost", disabled=True, width="stretch")
t4.link_button("Grafana", "http://localhost", disabled=True, width="stretch")
st.caption(
    "MLflow enables once a tracking URI is set; Airflow arrives in Phase 3, Grafana in Phase 4."
)
