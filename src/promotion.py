from __future__ import annotations

import logging
from dataclasses import dataclass

from src.registry import REGISTERED_MODEL_NAMES

logger = logging.getLogger(__name__)


def should_promote(
    candidate_macro_f1: float,
    production_macro_f1: float | None,
) -> bool:
    """Return True only when the candidate strictly beats Production."""
    if production_macro_f1 is None:
        return True

    return candidate_macro_f1 > production_macro_f1


@dataclass(frozen=True)
class PromotionResult:
    """Summary of one candidate model promotion decision."""

    run_id: str
    candidate_macro_f1: float
    production_macro_f1: float | None
    promoted: bool
    new_version: str | None
    previous_version: str | None


def get_run_macro_f1(client, run_id: str) -> float:
    """Return the macro-F1 metric for one MLflow run."""
    run = client.get_run(run_id)

    try:
        return float(run.data.metrics["macro_f1"])
    except KeyError as exc:
        raise RuntimeError(
            f"MLflow run {run_id!r} does not contain the 'macro_f1' metric."
        ) from exc


def get_production_macro_f1(
    client,
    schema: str,
) -> float | None:
    """Return the current Production model's macro-F1, or None if none exists."""
    import mlflow

    model_name = REGISTERED_MODEL_NAMES[schema]

    try:
        model_version = client.get_model_version_by_alias(
            name=model_name,
            alias="production",
        )
    except mlflow.exceptions.MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None

        raise

    return get_run_macro_f1(
        client=client,
        run_id=model_version.run_id,
    )


def record_promotion_verdict(
    client,
    result: PromotionResult,
) -> None:
    """Write the promotion decision and scores to the candidate MLflow run."""
    tags = {
        "promotion.verdict": "promoted" if result.promoted else "rejected",
        "promotion.candidate_macro_f1": str(result.candidate_macro_f1),
        "promotion.production_macro_f1": (
            "none" if result.production_macro_f1 is None else str(result.production_macro_f1)
        ),
    }

    if result.new_version is not None:
        tags["promotion.new_version"] = result.new_version

    if result.previous_version is not None:
        tags["promotion.previous_version"] = result.previous_version

    for key, value in tags.items():
        client.set_tag(result.run_id, key, value)


def evaluate_and_maybe_promote(
    client,
    schema: str,
    candidate_run_id: str,
) -> PromotionResult:
    """Compare a candidate with Production and promote only if it is better."""
    from scripts.register_model import register_and_promote

    candidate_score = get_run_macro_f1(
        client=client,
        run_id=candidate_run_id,
    )
    production_score = get_production_macro_f1(
        client=client,
        schema=schema,
    )

    if should_promote(candidate_score, production_score):
        _, new_version, previous_version = register_and_promote(
            client=client,
            schema=schema,
            run_id=candidate_run_id,
        )

        result = PromotionResult(
            run_id=candidate_run_id,
            candidate_macro_f1=candidate_score,
            production_macro_f1=production_score,
            promoted=True,
            new_version=new_version,
            previous_version=previous_version,
        )
    else:
        result = PromotionResult(
            run_id=candidate_run_id,
            candidate_macro_f1=candidate_score,
            production_macro_f1=production_score,
            promoted=False,
            new_version=None,
            previous_version=None,
        )
    logger.info(
        (
            "Promotion decision: candidate_run_id=%s "
            "candidate_macro_f1=%s production_macro_f1=%s "
            "verdict=%s new_version=%s previous_version=%s"
        ),
        result.run_id,
        result.candidate_macro_f1,
        result.production_macro_f1,
        "promoted" if result.promoted else "rejected",
        result.new_version,
        result.previous_version,
    )

    record_promotion_verdict(
        client=client,
        result=result,
    )

    return result
