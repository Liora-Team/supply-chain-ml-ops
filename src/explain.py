"""Interpretability — why a model predicted what it did, on the user's own text.

Primary method is **LIME** (lime.lime_text): model-agnostic, needs no background
dataset, and works uniformly for every classical model *and* DistilBERT. It
perturbs the raw review (dropping words) and watches how the prediction moves,
returning a signed weight per token toward the predicted class.

Bonus: **SHAP TreeExplainer** for RandomForest / XGBoost (no background data
needed). LIME already explains the linear models well, so SHAP is offered only
for the tree models.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import registry

# Underscore names are package-internal to src/ — shared between inference and
# explain on purpose, just not part of the public surface used by pages/ or api/.
from .inference import (
    BERT_MAX_LENGTH,
    _bert_index_to_label,
    _softmax,
    _star_shift,
    load_bert,
    load_classical,
)
from .preprocessing import label_name, preprocess_text


@dataclass
class Explanation:
    """Token-level attribution toward the predicted class."""

    method: str  # "LIME" or "SHAP"
    pred_display: str  # predicted class name
    tokens: list  # [(token, weight)] sorted by |weight| desc
    class_names: list  # ordered class display names


def _classical_class_names(model, schema: str) -> list[str]:
    """Class display names in the model's predict_proba column order."""
    raw = [int(c) for c in model.classes_]
    shift = _star_shift(raw, schema)
    return [label_name(c + shift, schema) for c in raw]


def _bert_class_names(model, schema: str) -> list[str]:
    """Class display names in DistilBERT's output-index order."""
    n = model.config.num_labels
    return [label_name(_bert_index_to_label(i, schema), schema) for i in range(n)]


def _classical_proba_fn(pipe):
    """LIME-compatible proba function: list[str] -> (n_texts, n_classes) array.

    Each perturbed text is lemmatised, then the bundled pipeline vectorises and
    scores it with its own TF-IDF step.
    """

    def f(texts):
        X = [preprocess_text(t) for t in texts]
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(X)
        d = np.atleast_2d(pipe.decision_function(X))
        return _softmax(d)

    return f


def _bert_proba_fn(tok, model, max_length=BERT_MAX_LENGTH):
    """LIME-compatible proba function over DistilBERT logits."""
    import torch

    def f(texts):
        clean = [preprocess_text(t) for t in texts]
        # padding=True (pad to longest in batch) vs padding="max_length" in
        # predict_bert (single text): both cap at the same BERT_MAX_LENGTH.
        enc = tok(
            clean, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        return _softmax(logits)

    return f


@lru_cache(maxsize=1)
def _lime_explainer(class_names_key: tuple):
    """Build a LimeTextExplainer (cached per class-name tuple)."""
    from lime.lime_text import LimeTextExplainer

    return LimeTextExplainer(class_names=list(class_names_key))


def lime_explain(
    model_id: str, text: str, *, num_features: int = 10, num_samples: int = 500
) -> Explanation:
    """LIME token attribution toward the predicted class, for any model."""
    if model_id.startswith("DistilBERT__"):
        schema = registry.get(model_id).schema
        tok, model = load_bert(schema)
        proba = _bert_proba_fn(tok, model)
        class_names = _bert_class_names(model, schema)
    else:
        entry = registry.get(model_id)
        schema = entry.schema
        model = load_classical(model_id)
        proba = _classical_proba_fn(model)
        class_names = _classical_class_names(model, schema)

    explainer = _lime_explainer(tuple(class_names))
    probs = proba([text])[0]
    pred_col = int(np.argmax(probs))

    exp = explainer.explain_instance(
        text,
        proba,
        num_features=num_features,
        labels=[pred_col],
        num_samples=num_samples,
    )
    tokens = sorted(exp.as_list(label=pred_col), key=lambda kv: abs(kv[1]), reverse=True)
    return Explanation(
        method="LIME", pred_display=class_names[pred_col], tokens=tokens, class_names=class_names
    )


def shap_tree_explain(model_id: str, text: str, *, top_k: int = 12) -> Explanation:
    """SHAP TreeExplainer token attribution for RandomForest / XGBoost.

    Returns the signed SHAP value of each token PRESENT in the review, toward the
    predicted class. No background dataset needed (tree path-dependent SHAP).
    """
    import shap

    entry = registry.get(model_id)
    if entry.algo not in ("RandomForest", "XGBoost"):
        raise ValueError("shap_tree_explain only supports RandomForest / XGBoost")

    # Pull the fitted vectoriser + classifier out of the bundled pipeline.
    pipe = load_classical(model_id)
    vectoriser = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    X = vectoriser.transform([preprocess_text(text)])  # 1 x vocab-size sparse row
    dense = X.toarray()
    names = vectoriser.get_feature_names_out()

    pred_col = int(np.argmax(clf.predict_proba(X)))
    class_names = _classical_class_names(clf, entry.schema)

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(dense)
    # shap_values shape handling: list[n_classes] of (1, n_feat), or (1, n_feat, n_classes)
    if isinstance(sv, list):
        vals = np.asarray(sv[pred_col])[0]
    else:
        arr = np.asarray(sv)
        vals = arr[0, :, pred_col] if arr.ndim == 3 else arr[0]

    nz = dense[0].nonzero()[0]  # tokens present in the review
    pairs = [(names[i], float(vals[i])) for i in nz]
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return Explanation(
        method="SHAP (tree)",
        pred_display=class_names[pred_col],
        tokens=pairs[:top_k],
        class_names=class_names,
    )
