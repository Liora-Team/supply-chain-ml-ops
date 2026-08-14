"""
Isolated offline smoke test for verifying MLflow experiment logging.

This test ensures that:
- MLflow can log experiments locally (no network required).
- The tracking URI is correctly isolated.
- No global environment state leaks between tests.
- Model logging works with mlflow >= 3.14 API.
"""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Guarded import for CI environments lacking MLflow dependencies.
mlflow = pytest.importorskip("mlflow")


def test_mlflow_logging_offline(tmp_path, monkeypatch):
    """Smoke test to ensure MLflow tracks experiments without a network
    and without leaking environment state.
    """
    # IMPORT ALIAS: Safely load the sklearn submodule without shadowing
    # the global 'mlflow' variable (Fixes Ruff F823, E402, and F811).
    import mlflow.sklearn as mlflow_sklearn

    # Required for MLflow >= 3.14.0 to prevent MlflowException.
    # Allow local file storage backend safely during tests.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    # WINDOWS FIX: Convert the path to a proper file:// URI
    # Ensures compatibility across OS (Windows/Linux/macOS)
    local_tracking_uri = tmp_path.joinpath("mlruns").resolve().as_uri()

    # Isolate environment mutations safely using monkeypatch (Fixes Should-fix 3).
    monkeypatch.setenv("MLFLOW_TRACKING_URI", local_tracking_uri)

    # Ensure MLflow uses our isolated URI explicitly.
    mlflow.set_tracking_uri(local_tracking_uri)
    mlflow.set_experiment("trustpilot-reviews")

    # Tiny mock dataset (minimal example)
    X = pd.Series(["bad shipping", "great product"])
    y = [1, 5]

    # Tiny pipeline (TF-IDF + DummyClassifier)
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )

    # MLflow run
    with mlflow.start_run(run_name="smoke-test-run"):
        pipe.fit(X, y)
        mlflow.log_metric("macro_f1", 0.5)

        # Use `name="model"` instead of deprecated artifact_path
        mlflow_sklearn.log_model(pipe, name="model", input_example=X.tolist())

    # Verify experiment + metric logging worked
    experiment = mlflow.get_experiment_by_name("trustpilot-reviews")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    assert len(runs) == 1
    assert runs.iloc[0]["metrics.macro_f1"] == 0.5

    # Verify model artefact exists (Card 2.1 requirement / Should-fix 4d)
    run_id = runs.iloc[0]["run_id"]
    try:
        loaded_model = mlflow_sklearn.load_model(f"runs:/{run_id}/model")
        assert loaded_model is not None, "Model artefact missing in MLflow tracking store"
    except Exception as e:
        pytest.fail(f"Failed to load model artefact from MLflow: {e}")
