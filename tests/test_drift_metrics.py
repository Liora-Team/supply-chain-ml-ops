"""Unit tests for monitoring/drift_metrics.py (Card 4.1 -> 4.2 alerting handoff)."""

from __future__ import annotations

import json

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from monitoring.drift_metrics import DriftStatusCollector


@pytest.fixture
def registry_with_status(tmp_path, monkeypatch):
    status_path = tmp_path / "drift_status.json"
    monkeypatch.setattr("monitoring.drift_metrics.STATUS_PATH", status_path)

    registry = CollectorRegistry()
    registry.register(DriftStatusCollector())
    return registry, status_path


def _scrape(registry) -> str:
    return generate_latest(registry).decode()


def test_no_status_file_yields_no_series(registry_with_status):
    registry, _ = registry_with_status

    output = _scrape(registry)

    assert "drift_detected{" not in output


def test_ok_schema_exposed_as_gauge(registry_with_status):
    registry, status_path = registry_with_status
    status_path.write_text(json.dumps({"3-class": {"status": "ok", "drift_detected": True}}))

    output = _scrape(registry)

    assert 'drift_detected{schema="3-class"} 1.0' in output


def test_no_drift_exposed_as_zero(registry_with_status):
    registry, status_path = registry_with_status
    status_path.write_text(json.dumps({"3-class": {"status": "ok", "drift_detected": False}}))

    output = _scrape(registry)

    assert 'drift_detected{schema="3-class"} 0.0' in output


def test_insufficient_data_schema_is_absent_not_zero(registry_with_status):
    registry, status_path = registry_with_status
    status_path.write_text(json.dumps({"3-class": {"status": "insufficient_data"}}))

    output = _scrape(registry)

    assert "drift_detected{" not in output


def test_multiple_schemas_scraped_independently(registry_with_status):
    registry, status_path = registry_with_status
    status_path.write_text(
        json.dumps(
            {
                "3-class": {"status": "ok", "drift_detected": True},
                "5-class": {"status": "ok", "drift_detected": False},
            }
        )
    )

    output = _scrape(registry)

    assert 'drift_detected{schema="3-class"} 1.0' in output
    assert 'drift_detected{schema="5-class"} 0.0' in output


def test_corrupt_status_file_does_not_crash_scrape(registry_with_status):
    registry, status_path = registry_with_status
    status_path.write_text("{not valid json")

    output = _scrape(registry)  # must not raise

    assert "drift_detected{" not in output


def test_register_helper_is_idempotent():
    from monitoring.drift_metrics import register

    registry = CollectorRegistry()
    register(registry)
    register(registry)  # second call must not raise or double-register

    collectors = [c for c in registry._collector_to_names if isinstance(c, DriftStatusCollector)]
    assert len(collectors) == 1
