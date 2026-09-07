"""Unit tests for monitoring/datasets.py (Card 4.1)."""

from __future__ import annotations

import importlib

import pandas as pd
import pytest


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point both the reference dir and the request store at throwaway paths."""
    monkeypatch.setenv("REFERENCE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MONITORING_DB_PATH", str(tmp_path / "request_store.db"))

    import monitoring.datasets as datasets
    import monitoring.request_store as request_store

    importlib.reload(request_store)
    importlib.reload(datasets)
    return datasets, request_store, tmp_path


def test_load_reference_missing_file_raises_actionable_error(env):
    datasets, _, _ = env

    with pytest.raises(FileNotFoundError, match="dvc pull"):
        datasets.load_reference()


def test_load_reference_reads_train_csv(env):
    datasets, _, tmp_path = env
    pd.DataFrame({"review": ["great product", "terrible service"], "stars": [5, 1]}).to_csv(
        tmp_path / "train.csv", index=False
    )

    df = datasets.load_reference()

    assert list(df.columns) == ["text", "text_length", "word_count"]
    assert len(df) == 2
    assert df.loc[0, "word_count"] == 2


def test_load_reference_missing_review_column_raises(env):
    datasets, _, tmp_path = env
    pd.DataFrame({"body": ["oops"], "stars": [3]}).to_csv(tmp_path / "train.csv", index=False)

    with pytest.raises(ValueError, match="review"):
        datasets.load_reference()


def test_load_reference_sample_caps_row_count(env):
    datasets, _, tmp_path = env
    pd.DataFrame({"review": [f"review {i}" for i in range(50)], "stars": [3] * 50}).to_csv(
        tmp_path / "train.csv", index=False
    )

    df = datasets.load_reference(sample=10)

    assert len(df) == 10


def test_load_current_empty_store_returns_empty_frame_with_columns(env):
    datasets, _, _ = env

    df = datasets.load_current(schema="3-class", window_hours=24, max_rows=500)

    assert list(df.columns) == ["text", "text_length", "word_count"]
    assert len(df) == 0


def test_load_current_reflects_logged_requests(env):
    datasets, request_store, _ = env
    request_store.log_prediction("great product", "positive", 0.9, "m1", schema="3-class")
    request_store.log_prediction("bad experience overall", "negative", 0.7, "m1", schema="3-class")

    df = datasets.load_current(schema="3-class", window_hours=24, max_rows=500)

    assert len(df) == 2
    assert set(df["text"]) == {"great product", "bad experience overall"}


def test_load_current_filters_by_schema(env):
    datasets, request_store, _ = env
    request_store.log_prediction("3-class review", "positive", 0.9, "m1", schema="3-class")
    request_store.log_prediction("5-class review", "5", 0.9, "m1", schema="5-class")

    df = datasets.load_current(schema="5-class", window_hours=24, max_rows=500)

    assert list(df["text"]) == ["5-class review"]
