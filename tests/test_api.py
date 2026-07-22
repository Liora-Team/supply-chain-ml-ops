"""Unit tests for the FastAPI inference service."""

from fastapi.testclient import TestClient

from api.main import app, production_model
from src import registry

client = TestClient(app)


def test_api_falls_back_without_tracking_uri(monkeypatch):
    """Without MLflow configuration, /predict serves the local joblib model."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    production_model.cache_clear()

    response = client.post(
        "/predict",
        json={
            "text": "Fantastic, the best I ever bought!",
            "schema": "3-class",
        },
    )

    assert response.status_code == 200
    _, source = production_model("3-class")
    assert source == "local_joblib"


def test_api_uses_registry_when_available(monkeypatch):
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
    )

    assert response.status_code == 200
    _, source = production_model("3-class")
    assert source == "registry"
    assert loaded_uris == ["models:/reviews-classifier-3class@production"]


def test_api_falls_back_when_registry_fails(monkeypatch):
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
    )

    assert response.status_code == 200
    _, source = production_model("3-class")
    assert source == "local_joblib"
    assert attempted_uris == ["models:/reviews-classifier-3class@production"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_models_lists_a_default():
    r = client.get("/models", params={"schema": "3-class"})
    assert r.status_code == 200
    models = r.json()["models"]
    assert models, "expected at least one loadable model"
    assert sum(m["default"] for m in models) == 1


def test_models_reports_local_joblib_source(monkeypatch):
    """Without MLflow configuration, /models reports the local serving path."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    production_model.cache_clear()

    response = client.get("/models", params={"schema": "3-class"})

    assert response.status_code == 200
    models = response.json()["models"]
    assert models, "expected at least one loadable model"
    assert all(model["source"] == "local_joblib" for model in models)


def test_models_reports_registry_source(monkeypatch):
    """When registry loading succeeds, /models reports the registry path."""
    local_pipeline, _ = registry._load_local_pipeline("3-class")
    monkeypatch.setattr(
        registry,
        "load_production",
        lambda schema: (local_pipeline, "registry"),
    )
    production_model.cache_clear()

    try:
        response = client.get("/models", params={"schema": "3-class"})

        assert response.status_code == 200
        models = response.json()["models"]
        assert models, "expected at least one loadable model"
        assert all(model["source"] == "registry" for model in models)
    finally:
        production_model.cache_clear()


def test_predict_positive_review():
    payload = {"text": "Fantastic, the best I ever bought!", "schema": "3-class"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["label_display"] == "Positive"
    assert abs(sum(body["probs"].values()) - 1.0) < 1e-6


def test_predict_rejects_bad_schema():
    r = client.post("/predict", json={"text": "hi", "schema": "7-class"})
    assert r.status_code == 422


def test_predict_unknown_model_id():
    r = client.post("/predict", json={"text": "hi", "schema": "3-class", "model_id": "nope"})
    assert r.status_code == 404
