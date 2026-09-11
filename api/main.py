"""FastAPI inference service — predict a Trustpilot review's star rating from its text.

Torch-free: serves only the classical TF-IDF + sklearn/XGBoost pipelines via
``src.inference.predict_classical``. The DistilBERT path is intentionally out of scope
for this service (it becomes its own container in Phase 2).

Run locally:  ``make api``  (→ http://localhost:8000/docs)
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from api.auth import enforce_rate_limit, verify_token
from monitoring.drift_metrics import register as register_drift_metrics
from monitoring.request_store import log_prediction
from src import registry
from src.inference import predict_classical, predict_loaded_classical

SCHEMAS = ("3-class", "5-class")

app = FastAPI(
    title="Trustpilot Review Rating API",
    version="0.1.0",
    description="Predict a review's star rating from its text (classical models).",
    openapi_tags=[
        {"name": "health", "description": "Liveness probe (unauthenticated)."},
        {"name": "models", "description": "Inspect the models the service can serve."},
        {"name": "inference", "description": "Rating prediction from raw review text."},
        {
            "name": "monitoring",
            "description": "Prometheus metrics (unauthenticated; the compose nginx proxy "
            "returns 403 so it is reachable only inside the compose network).",
        },
    ],
)

# Prometheus /metrics with the instrumentator's default HTTP metrics
# (http_requests_total, http_request_duration_seconds, request/response sizes),
# plus Card 4.1's drift_detected gauge (read fresh from drift_status.json per scrape).
Instrumentator().instrument(app).expose(app, tags=["monitoring"])
register_drift_metrics()

# Auth/rate-limit error shapes shared by every protected route (see api/auth.py).
AUTH_RESPONSES: dict[int | str, dict] = {
    401: {"description": "Missing bearer token."},
    403: {"description": "Invalid or expired token."},
    429: {"description": "Rate limit exceeded (RATE_LIMIT_PER_MINUTE requests/min per IP)."},
}


class PredictRequest(BaseModel):
    """One review to classify."""

    model_config = ConfigDict(
        protected_namespaces=(),
        populate_by_name=True,
        json_schema_extra={
            "examples": [{"text": "Fast delivery and great support, would buy again!"}]
        },
    )

    text: str = Field(..., min_length=1, max_length=5000, description="Raw review text.")
    label_schema: str = Field(
        "3-class", alias="schema", description="Label schema: '3-class' or '5-class'."
    )
    model_id: str | None = Field(
        None,
        description="Optional explicit local model id. When omitted, the API uses the "
        "production MLflow alias with local joblib fallback.",
    )


class PredictResponse(BaseModel):
    """Prediction for one review."""

    model_config = ConfigDict(
        protected_namespaces=(),
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "model_id": "reviews-classifier-3class",
                    "schema": "3-class",
                    "label": 2,
                    "label_display": "positive",
                    "probs": {"negative": 0.03, "neutral": 0.12, "positive": 0.85},
                    "proba_kind": "proba",
                }
            ]
        },
    )

    model_id: str = Field(description="Model that actually served the prediction.")
    label_schema: str = Field(
        alias="schema", serialization_alias="schema", description="Label schema used."
    )
    label: int = Field(description="Predicted class index.")
    label_display: str = Field(description="Human-readable label (e.g. 'positive' or '5').")
    probs: dict[str, float] = Field(description="Per-class probabilities keyed by display label.")
    proba_kind: str = Field(
        description="How probabilities were produced: 'proba', 'softmax(decision)' "
        "or 'softmax(logits)'."
    )


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
    return registry.load_production(schema, default_model_id(schema))


# Routes
@app.get("/health", tags=["health"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get(
    "/models",
    tags=["models"],
    dependencies=[Depends(verify_token), Depends(enforce_rate_limit)],
    responses={
        **AUTH_RESPONSES,
        200: {
            "description": "Loadable models, best macro-F1 first per schema.",
            "content": {
                "application/json": {
                    "example": {
                        "models": [
                            {
                                "id": "reviews-classifier-3class",
                                "schema": "3-class",
                                "algo": None,
                                "macro_f1": None,
                                "default": True,
                                "source": "registry",
                            }
                        ]
                    }
                }
            },
        },
    },
)
def models(schema: str | None = None) -> dict:
    """List loadable classical models (optionally filtered to one schema)."""
    schemas = [schema] if schema else list(SCHEMAS)
    out = []
    for sc in schemas:
        if sc not in SCHEMAS:
            raise HTTPException(status_code=422, detail=f"schema must be one of {SCHEMAS}")

        _, production_source = production_model(sc)
        local_default = default_model_id(sc)
        if production_source == "registry":
            out.append(
                {
                    "id": registry.REGISTERED_MODEL_NAMES[sc],
                    "schema": sc,
                    "algo": None,
                    "macro_f1": None,
                    "default": True,
                    "source": "registry",
                }
            )
        for e in registry.classical(sc):
            if e.loadable:
                out.append(
                    {
                        "id": e.id,
                        "schema": e.schema,
                        "algo": e.algo,
                        "macro_f1": e.macro_f1,
                        "default": (production_source == "local_joblib" and e.id == local_default),
                        "source": "local_joblib",
                    }
                )
    return {"models": out}


@app.post(
    "/predict",
    tags=["inference"],
    response_model=PredictResponse,
    dependencies=[Depends(verify_token), Depends(enforce_rate_limit)],
    responses={
        **AUTH_RESPONSES,
        404: {"description": "Unknown model_id."},
    },
)
def predict(req: PredictRequest, background_tasks: BackgroundTasks) -> PredictResponse:
    """Predict the star rating (or 3-class sentiment) of one review."""
    if req.label_schema not in SCHEMAS:
        raise HTTPException(
            status_code=422,
            detail=f"schema must be one of {SCHEMAS}",
        )

    registered_id = registry.REGISTERED_MODEL_NAMES[req.label_schema]

    if req.model_id:
        if req.model_id == registered_id:
            pipe, source = production_model(req.label_schema)

            # Report the model that was actually served.
            model_id = registered_id if source == "registry" else default_model_id(req.label_schema)

            p = predict_loaded_classical(
                model_id,
                req.text,
                pipe=pipe,
                schema=req.label_schema,
            )
        else:
            try:
                p = predict_classical(req.model_id, req.text)
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown model_id {req.model_id!r}.",
                ) from None
    else:
        pipe, source = production_model(req.label_schema)

        # MLflow success registered-model ID.
        # Local fallback configured local default ID.
        model_id = registered_id if source == "registry" else default_model_id(req.label_schema)

        p = predict_loaded_classical(
            model_id,
            req.text,
            pipe=pipe,
            schema=req.label_schema,
        )

    # Card 4.1: log this call so it becomes part of the "current" window the drift job
    # compares against the training distribution. Runs after the response is sent —
    # never adds latency to, or can fail, the prediction itself.
    background_tasks.add_task(
        log_prediction,
        text=req.text,
        predicted_label=p.label_display,
        confidence=max(p.probs.values()),
        model_version=p.model_id,
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
