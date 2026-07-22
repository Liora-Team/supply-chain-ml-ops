"""FastAPI inference service — predict a Trustpilot review's star rating from its text.

Torch-free: serves only the classical TF-IDF + sklearn/XGBoost pipelines via
``src.inference.predict_classical``. The DistilBERT path is intentionally out of scope
for this service (it becomes its own container in Phase 2).

Run locally:  ``make api``  (→ http://localhost:8000/docs)
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src import registry
from src.inference import predict_classical, predict_loaded_classical

SCHEMAS = ("3-class", "5-class")

app = FastAPI(
    title="Trustpilot Review Rating API",
    version="0.1.0",
    description="Predict a review's star rating from its text (classical models).",
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    text: str = Field(..., min_length=1, description="Raw review text.")
    label_schema: str = Field(
        "3-class", alias="schema", description="Label schema: '3-class' or '5-class'."
    )
    model_id: str | None = Field(
        None,
        description="Optional explicit local model id. When omitted, the API uses the "
        "production MLflow alias with local joblib fallback.",
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    model_id: str
    label_schema: str = Field(alias="schema", serialization_alias="schema")
    label: int
    label_display: str
    probs: dict[str, float]
    proba_kind: str


# Default model
@lru_cache(maxsize=len(SCHEMAS))
def default_model_id(schema: str) -> str:
    """Best loadable classical model for a schema, or a `DEFAULT_MODEL_*` env override."""
    # "3-class" -> DEFAULT_MODEL_3CLASS, "5-class" -> DEFAULT_MODEL_5CLASS (.env.example)
    override = os.environ.get(f"DEFAULT_MODEL_{schema.replace('-class', 'class').upper()}")
    if override:
        return override
    loadable = [e for e in registry.classical(schema) if e.loadable]
    if not loadable:
        raise RuntimeError(f"No loadable classical model registered for schema {schema!r}.")
    return loadable[0].id  # registry.classical() is sorted best macro-F1 first


@lru_cache(maxsize=len(SCHEMAS))
def production_model(schema: str):
    """Load and cache the production pipeline and its source for one schema."""
    return registry.load_production(schema)


# Routes
@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/models")
def models(schema: str | None = None) -> dict:
    """List loadable classical models (optionally filtered to one schema)."""
    schemas = [schema] if schema else list(SCHEMAS)
    out = []
    for sc in schemas:
        if sc not in SCHEMAS:
            raise HTTPException(status_code=422, detail=f"schema must be one of {SCHEMAS}")
        _, source = production_model(sc)
        default = default_model_id(sc)
        for e in registry.classical(sc):
            if e.loadable:
                out.append(
                    {
                        "id": e.id,
                        "schema": e.schema,
                        "algo": e.algo,
                        "macro_f1": e.macro_f1,
                        "default": e.id == default,
                        "source": source,
                    }
                )
    return {"models": out}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict the star rating (or 3-class sentiment) of one review."""
    if req.label_schema not in SCHEMAS:
        raise HTTPException(status_code=422, detail=f"schema must be one of {SCHEMAS}")

    if req.model_id:
        model_id = req.model_id
        try:
            p = predict_classical(model_id, req.text)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown model_id {model_id!r}.",
            ) from None
    else:
        pipe, _source = production_model(req.label_schema)
        model_id = registry.REGISTERED_MODEL_NAMES[req.label_schema]
        p = predict_loaded_classical(
            model_id,
            req.text,
            pipe=pipe,
            schema=req.label_schema,
        )
    return PredictResponse(
        model_id=p.model_id,
        label_schema=p.schema,
        label=p.label,
        label_display=p.label_display,
        probs=p.probs,
        proba_kind=p.proba_kind,
    )
