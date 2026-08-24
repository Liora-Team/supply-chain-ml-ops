"""Unit tests for the FastAPI inference service."""

import os

from fastapi.testclient import TestClient

from api.main import app, default_model_id, production_model
from src import registry

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")

client = TestClient(app)


def test_api_falls_back_without_tracking_uri(monkeypatch, auth_headers):
    """Without MLflow configuration, /predict serves the local joblib model."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    production_model.cache_clear()

    response = client.post(
        "/predict",
        json={
            "text": "Fantastic, the best I ever bought!",
            "schema": "3-class",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    _, source = production_model("3-class")
    assert source == "local_joblib"


def test_local_fallback_respects_configured_default(monkeypatch):
    """Local fallback loads the model selected by DEFAULT_MODEL_3CLASS."""
    loadable = [entry for entry in registry.classical("3-class") if entry.loadable]
    configured = loadable[1]  # Deliberately choose a model that is not the best one.
    loaded_paths = []

    def fake_joblib_load(path):
        loaded_paths.append(path)
        return object()

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("DEFAULT_MODEL_3CLASS", configured.id)
    monkeypatch.setattr(registry.joblib, "load", fake_joblib_load)
    default_model_id.cache_clear()
    production_model.cache_clear()

    try:
        _, source = production_model("3-class")

        assert source == "local_joblib"
        assert loaded_paths == [configured.joblib_path]
    finally:
        default_model_id.cache_clear()
        production_model.cache_clear()


def test_api_uses_registry_when_available(monkeypatch, auth_headers):
    """When MLflow succeeds, /predict serves the production-alias pipeline."""
    import sys
    from types import SimpleNamespace

    local_pipeline, _ = registry._load_local_pipeline("3-class")
    loaded_uris = []

    def fake_load_model(model_uri):
        loaded_uris.append(model_uri)
        return local_pipeline

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        sklearn=SimpleNamespace(load_model=fake_load_model),
    )

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    production_model.cache_clear()

    response = client.post(
        "/predict",
        json={
            "text": "Fantastic, the best I ever bought!",
            "schema": "3-class",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    _, source = production_model("3-class")
    assert source == "registry"
    assert loaded_uris == ["models:/reviews-classifier-3class@production"]


def test_api_falls_back_when_registry_fails(monkeypatch, auth_headers):
    """When MLflow loading fails, /predict cleanly uses local joblib."""
    import sys
    from types import SimpleNamespace

    attempted_uris = []

    def fake_load_model(model_uri):
        attempted_uris.append(model_uri)
        raise RuntimeError("Registry unavailable")

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        sklearn=SimpleNamespace(load_model=fake_load_model),
    )

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    production_model.cache_clear()

    response = client.post(
        "/predict",
        json={
            "text": "Fantastic, the best I ever bought!",
            "schema": "3-class",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    _, source = production_model("3-class")
    assert source == "local_joblib"
    assert attempted_uris == ["models:/reviews-classifier-3class@production"]


def test_models_lists_a_default(auth_headers):
    r = client.get("/models", params={"schema": "3-class"}, headers=auth_headers)
    assert r.status_code == 200
    models = r.json()["models"]
    assert models, "expected at least one loadable model"
    assert sum(m["default"] for m in models) == 1


def test_models_reports_local_joblib_source(monkeypatch, auth_headers):
    """Without MLflow configuration, /models reports the local serving path."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    production_model.cache_clear()

    response = client.get("/models", params={"schema": "3-class"}, headers=auth_headers)

    assert response.status_code == 200
    models = response.json()["models"]
    assert models, "expected at least one loadable model"
    assert all(model["source"] == "local_joblib" for model in models)


def test_models_reports_registry_source(monkeypatch, auth_headers):
    """When registry loading succeeds, /models reports the registry path."""
    local_pipeline, _ = registry._load_local_pipeline("3-class")
    monkeypatch.setattr(
        registry,
        "load_production",
        lambda schema, fallback_model_id=None: (local_pipeline, "registry"),
    )
    production_model.cache_clear()

    try:
        response = client.get("/models", params={"schema": "3-class"}, headers=auth_headers)

        assert response.status_code == 200
        models = response.json()["models"]
        assert models, "expected at least one loadable model"
        registry_models = [model for model in models if model["source"] == "registry"]
        local_models = [model for model in models if model["source"] == "local_joblib"]

        assert len(registry_models) == 1
        assert registry_models[0]["id"] == registry.REGISTERED_MODEL_NAMES["3-class"]
        assert registry_models[0]["default"] is True
        assert local_models
        assert all(model["default"] is False for model in local_models)
    finally:
        production_model.cache_clear()


def test_predict_positive_review(auth_headers):
    payload = {"text": "Fantastic, the best I ever bought!", "schema": "3-class"}
    r = client.post("/predict", json=payload, headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["label_display"] == "Positive"
    assert abs(sum(body["probs"].values()) - 1.0) < 1e-6


def test_predict_rejects_bad_schema(auth_headers):
    r = client.post("/predict", json={"text": "hi", "schema": "7-class"}, headers=auth_headers)
    assert r.status_code == 422


def test_predict_unknown_model_id(auth_headers):
    r = client.post(
        "/predict",
        json={"text": "hi", "schema": "3-class", "model_id": "nope"},
        headers=auth_headers,
    )
    assert r.status_code == 404


def test_predict_accepts_registered_model_id(monkeypatch, auth_headers):
    """A registered-model ID returned by the API can be sent back explicitly."""
    registered_id = registry.REGISTERED_MODEL_NAMES["3-class"]
    local_pipeline, _ = registry._load_local_pipeline("3-class")

    monkeypatch.setattr(
        "api.main.production_model",
        lambda schema: (local_pipeline, "registry"),
    )

    response = client.post(
        "/predict",
        json={
            "text": "Fantastic service!",
            "schema": "3-class",
            "model_id": registered_id,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["model_id"] == registered_id


def test_predict_requires_auth():
    r = client.post("/predict", json={"text": "hi", "schema": "3-class"})
    assert r.status_code == 401


def test_predict_rejects_bad_token():
    r = client.post(
        "/predict",
        json={"text": "hi", "schema": "3-class"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert r.status_code == 403


def test_predict_accepts_valid_token(auth_headers):
    r = client.post(
        "/predict",
        json={"text": "Great product!", "schema": "3-class"},
        headers=auth_headers,
    )
    assert r.status_code == 200


def test_predict_rejects_oversized_payload(auth_headers):
    r = client.post(
        "/predict",
        json={"text": "a" * 6000, "schema": "3-class"},
        headers=auth_headers,
    )
    assert r.status_code == 422


def test_health_stays_unauthenticated():
    assert client.get("/health").status_code == 200


def test_rate_limit_keys_on_forwarded_for(monkeypatch, auth_headers):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    headers_a = {**auth_headers, "X-Forwarded-For": "10.0.0.1"}
    headers_b = {**auth_headers, "X-Forwarded-For": "10.0.0.2"}

    # Exhaust client A's bucket (2 allowed, 3rd blocked).
    codes_a = [
        client.post(
            "/predict", json={"text": "hi", "schema": "3-class"}, headers=headers_a
        ).status_code
        for _ in range(3)
    ]
    assert codes_a == [200, 200, 429]

    # Client B has never made a request — its bucket must still be fresh,
    # proving A and B are not sharing state.
    r_b = client.post("/predict", json={"text": "hi", "schema": "3-class"}, headers=headers_b)
    assert r_b.status_code == 200


def test_rate_limit_falls_back_to_client_host_without_forwarded_for(monkeypatch, auth_headers):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    codes = [
        client.post(
            "/predict", json={"text": "hi", "schema": "3-class"}, headers=auth_headers
        ).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]


def test_rate_limit_returns_429(monkeypatch, auth_headers):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    codes = []
    for _ in range(5):
        r = client.post(
            "/predict",
            json={"text": "hi", "schema": "3-class"},
            headers=auth_headers,
        )
        codes.append(r.status_code)
    assert 429 in codes
