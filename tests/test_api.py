"""Unit tests for the FastAPI inference service."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


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
