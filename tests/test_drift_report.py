"""Unit tests for monitoring/drift_report.py (Card 4.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monitoring import drift_report


def _same_distribution(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"good product number {i}" for i in range(n)],
            "text_length": [30 + (i % 5) for i in range(n)],
            "word_count": [6 + (i % 2) for i in range(n)],
        }
    )


def _shifted_distribution(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"totally different long complaint {i}" for i in range(n)],
            "text_length": [250 + (i % 5) for i in range(n)],
            "word_count": [30 + (i % 2) for i in range(n)],
        }
    )


@pytest.fixture
def paths(tmp_path):
    return {
        "status_path": tmp_path / "drift_status.json",
        "reports_dir": tmp_path / "reports",
    }


def test_insufficient_data_short_circuits_before_loading_reference(monkeypatch, paths):
    monkeypatch.setattr(drift_report, "load_current", lambda **kw: pd.DataFrame())
    monkeypatch.setattr(
        drift_report,
        "load_reference",
        lambda **kw: pytest.fail("load_reference should not be called on insufficient data"),
    )

    result = drift_report.run_drift_report("3-class", min_rows=10, **paths)

    assert result["status"] == "insufficient_data"
    assert result["drift_detected"] is False
    assert result["report_path"] is None
    assert json.loads(paths["status_path"].read_text())["3-class"]["status"] == "insufficient_data"


def test_no_drift_when_distributions_match(monkeypatch, paths):
    monkeypatch.setattr(drift_report, "load_current", lambda **kw: _same_distribution())
    monkeypatch.setattr(drift_report, "load_reference", lambda **kw: _same_distribution())

    result = drift_report.run_drift_report("3-class", min_rows=10, **paths)

    assert result["status"] == "ok"
    assert result["drift_detected"] is False
    assert result["report_path"] is not None
    assert Path(result["report_path"]).exists()


def test_drift_detected_when_distributions_shift(monkeypatch, paths):
    monkeypatch.setattr(drift_report, "load_current", lambda **kw: _shifted_distribution())
    monkeypatch.setattr(drift_report, "load_reference", lambda **kw: _same_distribution())

    result = drift_report.run_drift_report("3-class", min_rows=10, **paths)

    assert result["status"] == "ok"
    assert result["drift_detected"] is True
    assert result["drift_share"] > 0


def test_status_file_overwrites_same_schema_but_keeps_others(monkeypatch, paths):
    monkeypatch.setattr(drift_report, "load_current", lambda **kw: _shifted_distribution())
    monkeypatch.setattr(drift_report, "load_reference", lambda **kw: _same_distribution())
    drift_report.run_drift_report("3-class", min_rows=10, **paths)

    # Drift clears on the next run — the flag must flip back to false, not stay stuck.
    monkeypatch.setattr(drift_report, "load_current", lambda **kw: _same_distribution())
    drift_report.run_drift_report("3-class", min_rows=10, **paths)

    # A different schema's entry must survive untouched.
    monkeypatch.setattr(drift_report, "load_current", lambda **kw: pd.DataFrame())
    drift_report.run_drift_report("5-class", min_rows=10, **paths)

    status = json.loads(paths["status_path"].read_text())
    assert status["3-class"]["drift_detected"] is False
    assert status["5-class"]["status"] == "insufficient_data"
