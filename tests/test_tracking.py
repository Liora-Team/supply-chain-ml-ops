import os

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# PROFESSIONAL FIX: If mlflow is not installed (like on the CI server), skip this test file!
mlflow = pytest.importorskip("mlflow")


def test_mlflow_logging_offline(tmp_path):
    """Smoke test to ensure MLflow tracks experiments without a network."""
    # NEW MLFLOW FIX: Allow local file storage
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    # WINDOWS FIX: Convert the path to a proper file:// URI
    local_tracking_uri = tmp_path.joinpath("mlruns").resolve().as_uri()

    os.environ["MLFLOW_TRACKING_URI"] = local_tracking_uri
    mlflow.set_tracking_uri(local_tracking_uri)
    mlflow.set_experiment("trustpilot-reviews")

    # Tiny mock data
    X = pd.Series(["bad shipping", "great product"])
    y = [1, 5]

    # Tiny pipeline
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )

    with mlflow.start_run(run_name="smoke-test-run"):
        pipe.fit(X, y)
        mlflow.log_metric("macro_f1", 0.5)
        mlflow.sklearn.log_model(pipe, "model", input_example=X.tolist())

    # Assertions to prove it worked
    experiment = mlflow.get_experiment_by_name("trustpilot-reviews")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    assert len(runs) == 1
    assert runs.iloc[0]["metrics.macro_f1"] == 0.5
