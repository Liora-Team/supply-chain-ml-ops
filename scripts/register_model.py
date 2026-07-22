"""Register an MLflow model and assign the production alias.

Use a configured team registry:

    uv run python scripts/register_model.py \
        --schema 3-class \
        --run-id <approved-run-id> \
        --tracking-uri <tracking-uri>

If --tracking-uri is omitted, MLFLOW_TRACKING_URI is used. If neither
is provided, the project's local mlruns directory is used.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make src/ importable when this file is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry import REGISTERED_MODEL_NAMES  # noqa: E402, I001

EXPERIMENT_NAME = "trustpilot-reviews"
PRODUCTION_ALIAS = "production"


def parse_args() -> argparse.Namespace:
    """Read the schema and optional run ID."""
    parser = argparse.ArgumentParser(description="Register and promote an MLflow model.")

    parser.add_argument(
        "--schema",
        required=True,
        choices=REGISTERED_MODEL_NAMES.keys(),
        help="Model schema to promote.",
    )

    parser.add_argument(
        "--run-id",
        help=(
            "Approved MLflow run ID. If omitted, the latest finished "
            "LogReg run for the schema is selected."
        ),
    )
    parser.add_argument(
        "--tracking-uri",
        help=(
            "MLflow tracking URI. Overrides MLFLOW_TRACKING_URI. "
            "If neither is provided, the project's local mlruns directory is used."
        ),
    )
    return parser.parse_args()


def select_run_id(client, schema: str, run_id: str | None) -> str:
    """Use an explicit run or find the latest suitable LogReg run."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(f"Experiment {EXPERIMENT_NAME!r} does not exist.")

    # Final promotion: use the exact run approved by the team.
    if run_id:
        run = client.get_run(run_id)

        if str(run.info.experiment_id) != str(experiment.experiment_id):
            raise RuntimeError(f"Run {run_id!r} is not in {EXPERIMENT_NAME!r}.")

        if run.info.status != "FINISHED":
            raise RuntimeError(f"Run {run_id!r} is not finished.")

        if run.data.tags.get("schema") != schema:
            raise RuntimeError(f"Run {run_id!r} does not use schema {schema!r}.")

        return run_id

    # Local/default flow: newest finished LogReg run first.
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )

    for run in runs:
        if run.data.tags.get("schema") == schema and run.data.tags.get("algorithm") == "LogReg":
            return run.info.run_id

    raise RuntimeError(f"No finished LogReg run for {schema!r} was found in {EXPERIMENT_NAME!r}.")


def register_and_promote(
    client,
    schema: str,
    run_id: str,
) -> tuple[str, str, str | None]:
    """Register the run's model and move the production alias."""
    import mlflow

    model_name = REGISTERED_MODEL_NAMES[schema]

    # Remember the current version so it can be restored if needed.
    try:
        current = client.get_model_version_by_alias(
            name=model_name,
            alias=PRODUCTION_ALIAS,
        )
        previous_version = str(current.version)
    except mlflow.exceptions.MlflowException:
        previous_version = None

    # Card 2.1 logs every fitted pipeline under the artifact path "model".
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name,
    )

    # Use aliases, not deprecated model stages.
    client.set_registered_model_alias(
        name=model_name,
        alias=PRODUCTION_ALIAS,
        version=registered.version,
    )

    return model_name, str(registered.version), previous_version


def print_rollback(
    model_name: str,
    previous_version: str | None,
    tracking_uri: str,
) -> None:
    """Print the command for restoring the previous production version."""
    if previous_version is None:
        print("No previous production version exists.")
        return

    print("\nTo roll back, run this Python code:")
    print("import mlflow")
    print("from mlflow import MlflowClient")
    print(f"mlflow.set_tracking_uri({tracking_uri!r})")
    print("client = MlflowClient()")
    print("client.set_registered_model_alias(")
    print(f"    name={model_name!r},")
    print(f"    alias={PRODUCTION_ALIAS!r},")
    print(f"    version={previous_version!r},")
    print(")")


def main() -> None:
    """Configure MLflow, choose a run, and promote it."""
    args = parse_args()

    import mlflow

    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri is None:
        tracking_uri = str(PROJECT_ROOT / "mlruns")
        print(
            f"WARNING: MLFLOW_TRACKING_URI was not set. Using local MLflow store: {tracking_uri}",
            file=sys.stderr,
        )
        print(
            "Do not use this local registry for the final team promotion.",
            file=sys.stderr,
        )

    mlflow.set_tracking_uri(tracking_uri)
    print(f"Tracking URI: {tracking_uri}")
    client = mlflow.MlflowClient()

    run_id = select_run_id(
        client=client,
        schema=args.schema,
        run_id=args.run_id,
    )

    model_name, new_version, previous_version = register_and_promote(
        client=client,
        schema=args.schema,
        run_id=run_id,
    )

    print(f"Selected run: {run_id}")
    print(f"Registered model: {model_name}")
    print(f"New version: {new_version}")
    print(f"Alias: {PRODUCTION_ALIAS}")

    print_rollback(model_name, previous_version, tracking_uri)


if __name__ == "__main__":
    main()
