"""Inference — one unified predict interface over classical and DistilBERT models.

Both model families consume the SAME preprocessed (lemmatised) text, because
both were trained on the `review_lemma` column written by scripts/get_data.py.
So every path here runs the user's input through
`preprocessing.preprocess_text` first.

The DistilBERT path is OPTIONAL: it activates only when the DISTILBERT_3CLASS /
DISTILBERT_5CLASS env vars point at a Hugging Face hub id or a local fine-tune
directory (see src/registry.py). Without them the app runs classical-only and
never imports torch.

Label conventions:
  - classical 5-class: predicts raw stars 1-5
  - classical 3-class: predicts {0=neg, 1=neu, 2=pos}
  - DistilBERT 5-class: output index k -> star (k+1)   (labels were trained as star-1)
  - DistilBERT 3-class: output index k -> {0,1,2}        (same as classical)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache

import joblib
import numpy as np

from . import registry
from .preprocessing import label_name, preprocess_text


@dataclass
class Prediction:
    """Result of one inference call, ready for the UI."""

    model_id: str
    schema: str
    label: int  # integer label in the model's own space
    label_display: str  # human name, e.g. "Negative" or "4★"
    probs: dict  # {label_display: probability}
    proba_kind: str  # "proba", "softmax(decision)", or "softmax(logits)"


def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis (1-D vector or 2-D rows)."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# Classical
@cache
def load_classical(model_id: str):
    """Load and cache one classical Pipeline(tfidf -> clf) by registry id.

    The pipeline bundles its own vectoriser, so inference takes lemmatised
    TEXT directly.
    """
    entry = registry.get(model_id)
    return joblib.load(entry.joblib_path)


def _star_shift(classes: list[int], schema: str) -> int:
    """Return +1 if a 5-class model is 0-indexed (XGBoost), else 0.

    Most 5-class models predict stars 1-5 directly. XGBoost is fit on 0-indexed
    labels (0-4) because XGBClassifier requires it, so its class column order
    must be shifted by +1 to recover the star value. Mirror image of
    scripts/build_pipelines.py, which fits the 5-class XGBoost on `stars - 1`.
    """
    return 1 if (schema == "5-class" and classes and min(classes) == 0) else 0


def _predict_with_pipeline(
    model_id: str,
    text: str,
    pipe,
    schema: str,
) -> Prediction:
    """Predict using an already-loaded classical pipeline."""
    clean = preprocess_text(text)
    X = [clean]

    raw_classes = [int(c) for c in pipe.classes_]
    shift = _star_shift(raw_classes, schema)
    classes = [c + shift for c in raw_classes]
    label = int(pipe.predict(X)[0]) + shift

    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(X)[0]
        kind = "proba"
    else:
        scores = pipe.decision_function(X)[0]
        scores = np.atleast_1d(scores)
        p = _softmax(scores)
        kind = "softmax(decision)"

    probs = {label_name(c, schema): float(pi) for c, pi in zip(classes, p, strict=False)}

    return Prediction(
        model_id=model_id,
        schema=schema,
        label=label,
        label_display=label_name(label, schema),
        probs=probs,
        proba_kind=kind,
    )


def predict_classical(model_id: str, text: str) -> Prediction:
    """Lemmatise -> Pipeline(tfidf -> clf) predict. Input is TEXT, not a matrix."""
    entry = registry.get(model_id)
    pipe = load_classical(model_id)

    return _predict_with_pipeline(
        model_id=model_id,
        text=text,
        pipe=pipe,
        schema=entry.schema,
    )


def predict_loaded_classical(
    model_id: str,
    text: str,
    *,
    pipe,
    schema: str,
) -> Prediction:
    """Predict using a classical pipeline that has already been loaded."""
    return _predict_with_pipeline(
        model_id=model_id,
        text=text,
        pipe=pipe,
        schema=schema,
    )


# Tokenizer truncation length used at fine-tune time — inference must match it.
BERT_MAX_LENGTH = 160


def _install_transformers_log_filter() -> None:
    """Silence the spurious "Accessing `__path__`" warnings.

    Streamlit's file watcher probes ``hasattr(m, "__path__")`` on every module
    in ``sys.modules`` each rerun; transformers' backward-compat alias modules
    log a warning on any attribute access, flooding the console. Filtering on
    the handlers keeps every other transformers warning visible.
    """
    import logging

    def _drop(record: logging.LogRecord) -> bool:
        return "Accessing `__path__`" not in record.getMessage()

    for handler in logging.getLogger("transformers").handlers:
        handler.addFilter(_drop)


def _bert_device() -> str:
    """Pick the fastest available device: Apple Metal (MPS) > CUDA > CPU.

    On a Mac this uses the Metal GPU; on a CUDA box the GPU; on Hugging Face
    Spaces (CPU-only) it resolves to "cpu". Inference is correct on all three.
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=2)
def load_bert(schema: str):
    """Load and cache a DistilBERT (tokenizer, model) pair for a schema.

    Lazy heavy imports keep classical-only sessions light. Pre-warms nothing;
    the first predict call pays the load cost (Streamlit wraps this in a spinner).
    The model is moved to the best available device (MPS / CUDA / CPU).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _install_transformers_log_filter()

    entry = registry.bert(schema)
    if entry is None:
        raise KeyError(f"No DistilBERT model registered for {schema}")
    tok = AutoTokenizer.from_pretrained(str(entry.bert_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(entry.bert_dir))
    model.eval()
    model.to(_bert_device())
    return tok, model


def _bert_index_to_label(idx: int, schema: str) -> int:
    """Map a DistilBERT output index to the schema's integer label."""
    return idx + 1 if schema == "5-class" else idx  # 5-class trained as star-1


def predict_bert(schema: str, text: str, max_length: int = BERT_MAX_LENGTH) -> Prediction:
    """Preprocess -> tokenise -> DistilBERT logits -> softmax probabilities."""
    import torch

    tok, model = load_bert(schema)
    clean = preprocess_text(text)  # same lemmatised input as training
    enc = tok(
        clean, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0].cpu().numpy()

    p = _softmax(logits)
    top_idx = int(p.argmax())
    label = _bert_index_to_label(top_idx, schema)

    probs = {
        label_name(_bert_index_to_label(i, schema), schema): float(pi) for i, pi in enumerate(p)
    }
    return Prediction(
        model_id=f"DistilBERT__{schema}",
        schema=schema,
        label=label,
        label_display=label_name(label, schema),
        probs=probs,
        proba_kind="softmax(logits)",
    )
