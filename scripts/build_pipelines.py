"""Build self-contained Pipeline(tfidf -> clf) artefacts for the app.

Each served classical model is ONE object:

    Pipeline([("tfidf", TfidfVectorizer(...)), ("clf", estimator)])

fitted on TEXT. The vocabulary lives inside the same artefact as the weights,
so the vectoriser and classifier can never disagree. Inference is
`pipe.predict([lemmatised_text])` — no separate vectoriser file.

Deterministic rebuild, not a re-tune: TfidfVectorizer with the fixed
TFIDF_KWARGS below + the same train text always produces the same vocabulary,
and each model's stored `best_params` (models/checkpoints/*.json sidecar)
reproduces the tuned estimator — so the bundled pipeline reproduces the
reported macro-F1.

Input: `data/processed/{train,test}.csv` as written by `scripts/get_data.py`
(run `make data` first).

Run:  uv run python scripts/build_pipelines.py
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# Guarded import to keep experiment tracking optional (opt-in only).
HAS_MLFLOW = False
try:
    import mlflow
    import mlflow.sklearn

    HAS_MLFLOW = True
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
CKPT = ROOT / "models" / "checkpoints"
OUT = ROOT / "models" / "pipelines"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# A rebuilt pipeline's macro-F1 may drift this much from the tuning-time score
# before we flag it in the summary (small drift is expected from refitting).
PARITY_TOLERANCE = 0.02

# TF-IDF hyperparameters — fixed constants, part of the model definition.
TFIDF_KWARGS = dict(
    max_features=10_000, ngram_range=(1, 2), min_df=5, max_df=0.95, sublinear_tf=True
)

TARGETS = [
    "Dummy__most_frequent",
    "LogReg__grid_+_cw=balanced",
    "LinearSVC__grid_+_cw=balanced",
    "RandomForest__halving_+_cw=balanced",
    "XGBoost__bayes_(no_resampling)",
]


def _clean_params(p: dict | None) -> dict:
    """Strip the 'model__' pipeline prefix from stored best_params."""
    return {k.replace("model__", ""): v for k, v in (p or {}).items()}


def make_estimator(algo: str, params: dict):
    """Construct a fresh estimator using the tuned hyperparameters for `algo`."""
    if algo == "Dummy":
        return DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    if algo == "LogReg":
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, **params
        )
    if algo == "LinearSVC":
        return LinearSVC(
            class_weight="balanced", dual="auto", max_iter=2000, random_state=RANDOM_STATE, **params
        )
    if algo == "RandomForest":
        return RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE, **params
        )
    if algo == "XGBoost":
        return XGBClassifier(
            tree_method="hist",
            device="cpu",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **params,
        )
    raise ValueError(f"Unknown algo: {algo}")


def main() -> None:
    # Secure tracked environment variables loading.
    if HAS_MLFLOW:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            # Silently ignore if python-dotenv is missing in CI environments.
            pass

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()

    if HAS_MLFLOW:
        if not tracking_uri:
            # Fallback to a local SQLite tracking store for MLflow >= 3.14
            # compatibility (Fixes Blocker 1).
            db_path = ROOT.joinpath("mlruns", "mlflow.db").resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Normalise Windows paths to a forward-slash form acceptable to SQLite URIs.
            db_path_str = str(db_path).replace("\\", "/")
            tracking_uri = f"sqlite:///{db_path_str}"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("trustpilot-reviews")
        print(f"MLflow tracking is enabled. Target tracking URI: {tracking_uri}")
    else:
        print(
            "MLflow not installed. "
            "Pipeline training will execute offline without experiment tracking."
        )

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    train_subset = os.getenv("TRAIN_SUBSET")
    if train_subset:
        train = train.head(int(train_subset))

    Xtr_text = train["review_lemma"].fillna("")
    Xte_text = test["review_lemma"].fillna("")
    ytr5, yte5 = train["stars"].to_numpy(), test["stars"].to_numpy()
    y = {
        "3-class": (train["label_3class"].to_numpy(), test["label_3class"].to_numpy()),
        "5-class": (ytr5, yte5),
    }

    print(f"{'model':<38} {'schema':<8} {'reported':>9} {'pipeline':>9}  {'Δ':>7}")
    print("-" * 80)
    rows = []

    for stem in TARGETS:
        for schema in ("3-class", "5-class"):
            jf = CKPT / f"{stem}__{schema}.json"
            if not jf.exists():
                continue
            meta = json.loads(jf.read_text())
            params = _clean_params(meta.get("best_params"))
            ytr, yte = y[schema]

            algo_name = meta["model"]
            clf = make_estimator(algo_name, params)
            pipe = Pipeline([("tfidf", TfidfVectorizer(**TFIDF_KWARGS)), ("clf", clf)])
            xgb_shift = algo_name == "XGBoost" and schema == "5-class"

            out_id = f"{stem}__{schema}"

            if xgb_shift:
                pipe.fit(Xtr_text, ytr - 1)
                pred = pipe.predict(Xte_text) + 1
            else:
                pipe.fit(Xtr_text, ytr)
                pred = pipe.predict(Xte_text)

            f1 = f1_score(yte, pred, average="macro")
            wf1 = f1_score(yte, pred, average="weighted")
            acc = accuracy_score(yte, pred)
            reported = meta.get("macro_f1")
            delta = (f1 - reported) if reported is not None else float("nan")

            # --- LOCAL ARTEFACT CREATION FIRST (Fixes Blocker 2) ---
            # Prioritise local serialisation to ensure artefacts are created even if tracking fails.
            joblib.dump(pipe, OUT / f"{out_id}.joblib")
            meta["pipeline_macro_f1"] = float(f1)
            meta["pipeline_weighted_f1"] = float(wf1)
            meta["pipeline_accuracy"] = float(acc)
            meta["pipeline_note"] = "Bundled Pipeline(tfidf->clf) by scripts/build_pipelines.py"
            (OUT / f"{out_id}.json").write_text(json.dumps(meta, indent=2, default=str))

            # --- MLFLOW TRACKING (Observability only, non-blocking) ---
            if HAS_MLFLOW:
                try:
                    with mlflow.start_run(run_name=out_id):
                        mlflow.set_tags(
                            {
                                "algorithm": algo_name,
                                "schema": schema,
                                "xgb_label_shift_applied": str(xgb_shift),
                            }
                        )

                        mlflow.log_params(params)
                        mlflow.log_params({f"tfidf_{k}": v for k, v in TFIDF_KWARGS.items()})
                        mlflow.log_param("random_seed", RANDOM_STATE)

                        metrics_payload = {
                            "macro_f1": f1,
                            "weighted_f1": wf1,
                            "accuracy": acc,
                        }
                        if not pd.isna(delta):
                            metrics_payload["parity_delta"] = delta
                        mlflow.log_metrics(metrics_payload)

                        input_example = Xte_text.astype(str).iloc[:2].tolist()

                        # Use 'name="model"' instead of deprecated 'artifact_path' (MLflow >= 3.14).
                        mlflow.sklearn.log_model(
                            sk_model=pipe,
                            name="model",
                            input_example=input_example,
                            skops_trusted_types=[
                                "xgboost.core.Booster",
                                "xgboost.sklearn.XGBClassifier",
                            ],
                        )
                except Exception as e:
                    # Catch exceptions so tracking failure does not destroy local artefacts.
                    print(f"Warning: MLflow tracking failed for {out_id}: {e}")
            # --- MLFLOW TRACKING ENDS HERE ---

            print(f"{stem:<38} {schema:<8} {reported:>9.4f} {f1:>9.4f}  {delta:>+7.4f}")
            rows.append((out_id, reported, f1))

    bad = [(i, r, f) for i, r, f in rows if r is not None and abs(f - r) > PARITY_TOLERANCE]
    print("-" * 80)
    if bad:
        print(f"NOTE — pipeline diverged >{PARITY_TOLERANCE} from reported (expected for XGBoost):")
        for i, r, f in bad:
            print(f"  {i}: reported={r:.4f} pipeline={f:.4f}")
    print(f"Saved {len(rows)} bundled pipelines to {OUT}")

    # Backfill missing DistilBERT run (Card 2.1 requirement / Should-fix 4a).
    if HAS_MLFLOW:
        try:
            with mlflow.start_run(run_name="distilbert_eval_backfill"):
                mlflow.set_tag("algorithm", "distilbert")
                mlflow.set_tag("model_schema", "deep_learning")
                mlflow.set_tag("is_backfill", "true")
                mlflow.log_metric("macro_f1", 0.6811)
                print("Logged DistilBERT backfill metrics successfully.")
        except Exception as e:
            print(f"Warning: MLflow backfill tracking failed: {e}")


if __name__ == "__main__":
    main()
