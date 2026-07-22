"""Model registry — single source of truth for which models exist.

Scans `models/checkpoints/*.json` (per-model tuning metrics + best_params
sidecars) and pairs each with its `.joblib` estimator. Also
registers the two fine-tuned DistilBERT directories. Every page (Predict,
Leaderboard, Explain) reads from here so the UI never hard-codes a model list.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
# Self-contained Pipeline(tfidf -> clf) artifacts (scripts/build_pipelines.py).
# Each bundles its own vectoriser, so vocabulary and weights always travel
# together in one file. These are what the app loads.
PIPELINE_DIR = ROOT / "models" / "pipelines"

REGISTERED_MODEL_NAMES = {
    "3-class": "reviews-classifier-3class",
    "5-class": "reviews-classifier-5class",
}

logger = logging.getLogger(__name__)


# DistilBERT source per schema. Defaults to the local fine-tune dirs; on
# Hugging Face Spaces (where shipping 2×255 MB is heavy) set the env vars to a
# hub repo id, e.g. DISTILBERT_3CLASS=your-org/trustpilot-distilbert-3class — then
# transformers.from_pretrained loads from the hub instead. A `bert_dir` that is
# a plain string (not an existing local path) is treated as a hub id.
DISTILBERT_DIRS = {
    "3-class": os.environ.get(
        "DISTILBERT_3CLASS", str(ROOT / "models" / "distilbert_3class" / "final")
    ),
    "5-class": os.environ.get(
        "DISTILBERT_5CLASS", str(ROOT / "models" / "distilbert_5class" / "final")
    ),
}


@dataclass
class ModelEntry:
    """One trained model: where it lives + how it scored."""

    id: str  # unique key, e.g. "LogReg__grid_+_cw=balanced__3-class"
    algo: str  # LogReg, LinearSVC, RandomForest, XGBoost, Dummy, DistilBERT
    variant: str  # human label of the training variant
    schema: str  # "3-class" or "5-class"
    kind: str  # "classical" or "bert"
    metrics: dict = field(default_factory=dict)  # macro_f1, weighted_f1, accuracy, time_s, ...
    joblib_path: Path | None = None  # estimator file (classical only)
    bert_dir: str | None = None  # local dir OR hub id (bert only)

    @property
    def loadable(self) -> bool:
        """True if this model can be loaded for live inference.

        Classical models need a bundled pipeline (models/pipelines/); BERT needs
        its directory. Checkpoint-only rows (stale joblib) are leaderboard-only.
        """
        if self.kind == "bert":
            if not self.bert_dir:
                return False
            p = Path(self.bert_dir)
            # An absolute or explicitly-local path ('.'/'~'/'/') must exist on
            # disk — every absolute path contains '/', so the hub-id heuristic
            # below would otherwise mark a missing local model as loadable and
            # crash at load time with a "Repo id must be in the form..." error.
            if p.is_absolute() or self.bert_dir.startswith((".", "~", "/")):
                return p.exists()
            # Otherwise it looks like a hub id ("org/name"), resolved by
            # transformers at load time. p.exists() is re-checked because a bare
            # relative dir name (e.g. "models_local") is still a valid local path.
            return p.exists() or "/" in self.bert_dir
        return self.joblib_path is not None and self.joblib_path.exists()

    @property
    def macro_f1(self) -> float | None:
        return self.metrics.get("macro_f1")

    @property
    def display_name(self) -> str:
        return f"{self.algo} — {self.variant} ({self.schema})"


# Scan results, cached at module level on first call. A plain global (rather
# than functools.cache) keeps the cache visible and trivially resettable
# (`registry._CACHE = None`) after adding model files mid-session.
_CACHE: list[ModelEntry] | None = None


def _scan_classical() -> list[ModelEntry]:
    """Build a ModelEntry for every json sidecar that has a matching joblib."""
    entries: list[ModelEntry] = []
    for jf in sorted(CHECKPOINT_DIR.glob("*.json")):
        meta = json.loads(jf.read_text())
        # The bundled pipeline (if built) is what the app loads for inference.
        # Checkpoint joblibs are bare estimators without a vectoriser → never
        # loaded; their sidecars feed the leaderboard only.
        pipe_jl = PIPELINE_DIR / f"{jf.stem}.joblib"
        pipe_json = PIPELINE_DIR / f"{jf.stem}.json"
        metrics = {
            k: meta.get(k) for k in ("macro_f1", "weighted_f1", "accuracy", "time_s", "cv_macro_f1")
        }
        # For served models, the pipeline's own test metrics are authoritative
        # (measured on the in-pipeline vectoriser). Overwrite macro/weighted/acc
        # with them; keep the tuning-time macro under `tuning_macro_f1` for
        # provenance (not displayed).
        if pipe_json.exists():
            pm = json.loads(pipe_json.read_text())
            metrics["tuning_macro_f1"] = metrics.get("macro_f1")
            for src_key, dst_key in (
                ("pipeline_macro_f1", "macro_f1"),
                ("pipeline_weighted_f1", "weighted_f1"),
                ("pipeline_accuracy", "accuracy"),
            ):
                if pm.get(src_key) is not None:
                    metrics[dst_key] = pm[src_key]
        entries.append(
            ModelEntry(
                id=jf.stem,
                algo=meta.get("model", "?"),
                variant=meta.get("variant", "?"),
                schema=meta.get("schema", "?"),
                kind="classical",
                metrics=metrics,
                joblib_path=pipe_jl if pipe_jl.exists() else None,
            )
        )
    return entries


def _scan_bert() -> list[ModelEntry]:
    """Register the two DistilBERT directories, reading their eval_metrics.json."""
    entries: list[ModelEntry] = []
    for schema, src in DISTILBERT_DIRS.items():
        d = Path(src)
        is_local = d.exists()
        if not is_local and "/" not in src:
            continue  # neither a local dir nor a hub id — nothing to register
        # Leaderboard metrics are read independently of where weights load from,
        # so the score still shows when DISTILBERT_* points at the Hub. Look in
        # models/pipelines/ first (the small committed copy), then the canonical local
        # fine-tune dir.
        ev = {}
        n = schema.replace("-class", "class")  # "3-class" -> "3class"
        for ev_path in (
            PIPELINE_DIR / f"distilbert_{schema}_eval.json",
            d / "eval_metrics.json",
            ROOT / "models" / f"distilbert_{n}" / "final" / "eval_metrics.json",
        ):
            if ev_path.exists():
                raw = json.loads(ev_path.read_text())
                phase2 = raw.get("phases", {}).get("phase_2_partial_unfreeze", {})
                ev = {
                    "macro_f1": phase2.get("test_macro_f1"),
                    "weighted_f1": phase2.get("test_weighted_f1"),
                    "accuracy": phase2.get("test_accuracy"),
                }
                break
        entries.append(
            ModelEntry(
                id=f"DistilBERT__{schema}",
                algo="DistilBERT",
                variant="2-phase fine-tune",
                schema=schema,
                kind="bert",
                metrics=ev,
                bert_dir=src,
            )
        )
    return entries


def all_models() -> list[ModelEntry]:
    """Return every registered model (classical + bert). Cached."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _scan_classical() + _scan_bert()
    return _CACHE


def get(model_id: str) -> ModelEntry:
    """Look up one model by id; raise if unknown."""
    for e in all_models():
        if e.id == model_id:
            return e
    raise KeyError(f"Unknown model id: {model_id}")


def classical(schema: str | None = None) -> list[ModelEntry]:
    """Classical models, optionally filtered to one schema, best macro-F1 first."""
    out = [
        e for e in all_models() if e.kind == "classical" and (schema is None or e.schema == schema)
    ]
    return sorted(out, key=lambda e: e.macro_f1 or -1, reverse=True)


def _load_local_pipeline(schema: str):
    """Load the best available local classical pipeline for a schema."""
    entry = next((e for e in classical(schema) if e.loadable), None)

    if entry is None or entry.joblib_path is None:
        raise RuntimeError(f"No loadable local model found for schema {schema!r}.")

    return joblib.load(entry.joblib_path), "local_joblib"


def load_production(schema: str):
    """Load the production pipeline for a supported label schema."""
    if schema not in REGISTERED_MODEL_NAMES:
        raise ValueError(f"Unsupported schema: {schema!r}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        model_name = REGISTERED_MODEL_NAMES[schema]
        model_uri = f"models:/{model_name}@production"
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            pipeline = mlflow.sklearn.load_model(model_uri)
            return pipeline, "registry"

        except Exception:
            logger.warning(
                "Could not load %s from MLflow; falling back to local joblib.",
                model_uri,
                exc_info=True,
            )

    return _load_local_pipeline(schema)


def best_loadable(schema: str, algo: str = "LogReg") -> ModelEntry | None:
    """Best-scoring *servable* classical model of one algo (used by Home and
    Insights to fetch the winning LogReg pipeline)."""
    return next((e for e in classical(schema) if e.algo == algo and e.loadable), None)


def bert(schema: str) -> ModelEntry | None:
    """The DistilBERT entry for a schema, or None if not present on disk."""
    for e in all_models():
        if e.kind == "bert" and e.schema == schema:
            return e
    return None
