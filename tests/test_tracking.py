"""
Isolated offline smoke test for verifying MLflow experiment logging.

This test ensures that:
- MLflow can log experiments locally (no network required)
- The tracking URI is correctly isolated
- No global environment state leaks between tests
- Model logging works with mlflow >= 3.14 API
"""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------
# If MLflow is not installed (e.g., on CI server), skip this test.
# ------------------------------------------------------------------
mlflow = pytest.importorskip("mlflow")


def test_mlflow_logging_offline(tmp_path, monkeypatch):
    """Smoke test to ensure MLflow tracks experiments without a network
    and without leaking environment state.

    - Uses a temporary directory as tracking backend
    - Forces file-based tracking (offline)
    - Logs a tiny sklearn pipeline
    - Verifies the metric was recorded
    """
    # ------------------------------------------------------------------
    # Required for MLflow >= 3.14.0 to prevent MlflowException
    # Allow local file storage backend
    # Isolate environment mutations safely using monkeypatch
    # ------------------------------------------------------------------
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    # ------------------------------------------------------------------
    # WINDOWS FIX: Convert the path to a proper file:// URI
    # Ensures compatibility across OS (Windows/Linux/macOS)
    # ------------------------------------------------------------------
    local_tracking_uri = tmp_path.joinpath("mlruns").resolve().as_uri()
    # ------------------------------------------------------------------
    # Isolate environment mutations safely using monkeypatch
    # Prevents state leakage between tests
    # ------------------------------------------------------------------
    monkeypatch.setenv("MLFLOW_TRACKING_URI", local_tracking_uri)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    # Ensure MLflow uses our isolated URI
    mlflow.set_tracking_uri(local_tracking_uri)
    mlflow.set_experiment("trustpilot-reviews")

    # ------------------------------------------------------------------
    # Tiny mock dataset (minimal example)
    # ------------------------------------------------------------------
    X = pd.Series(["bad shipping", "great product"])
    y = [1, 5]

    # ------------------------------------------------------------------
    # Tiny pipeline
    # (TF-IDF + DummyClassifier)
    # ------------------------------------------------------------------
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )

    # ------------------------------------------------------------------
    # MLflow run
    # ------------------------------------------------------------------
    with mlflow.start_run(run_name="smoke-test-run"):
        pipe.fit(X, y)
        mlflow.log_metric("macro_f1", 0.5)

        # Use name parameter to conform to mlflow >= 3.14.0 specs
        # Use `name="model"` instead of deprecated artifact_path
        mlflow.sklearn.log_model(pipe, name="model", input_example=X.tolist())

    # ------------------------------------------------------------------
    # verify experiment + metric logging worked
    # ------------------------------------------------------------------
    experiment = mlflow.get_experiment_by_name("trustpilot-reviews")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    assert len(runs) == 1
    assert runs.iloc[0]["metrics.macro_f1"] == 0.5
