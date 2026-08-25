"""Unit tests for the Prometheus /metrics endpoint (Card 4.3)."""

import os

from fastapi.testclient import TestClient

from api.main import app

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")

client = TestClient(app)


def test_metrics_serves_prometheus_format_unauthenticated():
    """/metrics needs no token and exposes the default HTTP metric families."""
    # Generate at least one observation so the histogram families exist.
    assert client.get("/health").status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    # Family names only — values vary with test order.
    assert "http_requests_total" in response.text
    assert "http_request_duration_seconds" in response.text
