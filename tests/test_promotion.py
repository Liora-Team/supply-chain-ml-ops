"""Tests for model promotion logic."""

from types import SimpleNamespace
from unittest.mock import Mock

from src.promotion import (
    PromotionResult,
    evaluate_and_maybe_promote,
    get_production_macro_f1,
    get_run_macro_f1,
    record_promotion_verdict,
    should_promote,
)


def test_better_candidate_is_promoted() -> None:
    """A candidate with a higher score should be promoted."""
    assert should_promote(0.72, 0.70) is True


def test_worse_candidate_is_rejected() -> None:
    """A candidate with a lower score should be rejected."""
    assert should_promote(0.68, 0.70) is False


def test_equal_candidate_is_rejected() -> None:
    """An equal score is not a strict improvement."""
    assert should_promote(0.70, 0.70) is False


def test_first_candidate_is_promoted() -> None:
    """The first candidate should be promoted when Production is empty."""
    assert should_promote(0.40, None) is True


def test_get_run_macro_f1_returns_metric() -> None:
    """The metric should be read from the requested MLflow run."""
    client = Mock()
    client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.73}))

    result = get_run_macro_f1(client, "candidate-run-123")

    assert result == 0.73
    client.get_run.assert_called_once_with("candidate-run-123")


def test_get_run_macro_f1_raises_when_metric_is_missing() -> None:
    """A missing macro-F1 metric should raise a clear error."""
    client = Mock()
    client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={}))

    try:
        get_run_macro_f1(client, "candidate-run-123")
    except RuntimeError as exc:
        assert "macro_f1" in str(exc)
        assert "candidate-run-123" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError was not raised.")


def test_get_production_macro_f1_returns_current_score() -> None:
    """The production alias should lead to its run's macro-F1."""
    client = Mock()

    client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="production-run-456")
    client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.71}))

    result = get_production_macro_f1(client, "3-class")

    assert result == 0.71
    client.get_model_version_by_alias.assert_called_once_with(
        name="reviews-classifier-3class",
        alias="production",
    )
    client.get_run.assert_called_once_with("production-run-456")


def test_production_missing_macro_f1_raises_error() -> None:
    """A Production run without macro-F1 should not be treated as zero."""
    client = Mock()

    client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="production-run-456")
    client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={}))

    try:
        get_production_macro_f1(client, "3-class")
    except RuntimeError as exc:
        assert "macro_f1" in str(exc)
        assert "production-run-456" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError was not raised.")


def test_production_lookup_does_not_hide_unexpected_mlflow_errors() -> None:
    """Connection or registry failures must not look like an empty registry."""
    import mlflow

    client = Mock()
    client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException(
        "MLflow connection failed"
    )

    try:
        get_production_macro_f1(client, "3-class")
    except mlflow.exceptions.MlflowException as exc:
        assert "connection failed" in str(exc)
    else:
        raise AssertionError("Expected MLflow exception was not raised.")


def test_record_promotion_verdict_writes_expected_tags() -> None:
    """The verdict, scores, and versions should be written to the candidate run."""
    client = Mock()
    result = PromotionResult(
        run_id="candidate-run-123",
        candidate_macro_f1=0.74,
        production_macro_f1=0.71,
        promoted=True,
        new_version="8",
        previous_version="7",
    )

    record_promotion_verdict(client, result)

    expected_calls = [
        (("candidate-run-123", "promotion.verdict", "promoted"),),
        (("candidate-run-123", "promotion.candidate_macro_f1", "0.74"),),
        (("candidate-run-123", "promotion.production_macro_f1", "0.71"),),
        (("candidate-run-123", "promotion.new_version", "8"),),
        (("candidate-run-123", "promotion.previous_version", "7"),),
    ]

    actual_calls = [call.args for call in client.set_tag.call_args_list]

    assert actual_calls == [item[0] for item in expected_calls]


def test_better_candidate_calls_register_and_promote(monkeypatch) -> None:
    """A better candidate should be registered and promoted."""
    client = Mock()

    client.get_run.side_effect = [
        SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.74})),
        SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.71})),
    ]
    client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="production-run-456")

    promote_mock = Mock(return_value=("reviews-classifier-3class", "8", "7"))

    monkeypatch.setattr(
        "scripts.register_model.register_and_promote",
        promote_mock,
    )

    result = evaluate_and_maybe_promote(
        client=client,
        schema="3-class",
        candidate_run_id="candidate-run-123",
    )

    assert result.promoted is True
    assert result.new_version == "8"
    assert result.previous_version == "7"

    promote_mock.assert_called_once_with(
        client=client,
        schema="3-class",
        run_id="candidate-run-123",
    )


def test_worse_candidate_is_rejected_without_promotion(monkeypatch) -> None:
    """A worse candidate must not call the alias-changing promotion function."""
    client = Mock()

    client.get_run.side_effect = [
        SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.68})),
        SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.71})),
    ]
    client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="production-run-456")

    promote_mock = Mock()

    monkeypatch.setattr(
        "scripts.register_model.register_and_promote",
        promote_mock,
    )

    result = evaluate_and_maybe_promote(
        client=client,
        schema="3-class",
        candidate_run_id="candidate-run-123",
    )

    assert result.promoted is False
    assert result.new_version is None
    assert result.previous_version is None

    promote_mock.assert_not_called()


def test_equal_candidate_is_rejected_without_promotion(monkeypatch) -> None:
    """An equal candidate is not a strict improvement and must be rejected."""
    client = Mock()

    client.get_run.side_effect = [
        SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.71})),
        SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.71})),
    ]
    client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="production-run-456")

    promote_mock = Mock()

    monkeypatch.setattr(
        "scripts.register_model.register_and_promote",
        promote_mock,
    )

    result = evaluate_and_maybe_promote(
        client=client,
        schema="3-class",
        candidate_run_id="candidate-run-123",
    )

    assert result.promoted is False
    assert result.new_version is None
    assert result.previous_version is None

    promote_mock.assert_not_called()


def test_first_candidate_is_promoted_when_no_production_exists(
    monkeypatch,
) -> None:
    """The first candidate should be promoted when no Production alias exists."""
    import mlflow
    from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

    client = Mock()

    client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.60}))
    client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException(
        "Production alias does not exist",
        error_code=INVALID_PARAMETER_VALUE,
    )

    promote_mock = Mock(return_value=("reviews-classifier-3class", "1", None))

    monkeypatch.setattr(
        "scripts.register_model.register_and_promote",
        promote_mock,
    )

    result = evaluate_and_maybe_promote(
        client=client,
        schema="3-class",
        candidate_run_id="candidate-run-123",
    )

    assert result.promoted is True
    assert result.production_macro_f1 is None
    assert result.new_version == "1"
    assert result.previous_version is None

    promote_mock.assert_called_once_with(
        client=client,
        schema="3-class",
        run_id="candidate-run-123",
    )


def test_promotion_decision_is_logged(monkeypatch, caplog) -> None:
    """The promotion decision should be visible in orchestrator logs."""
    client = Mock()

    client.get_run.side_effect = [
        SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.74})),
        SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.71})),
    ]
    client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="production-run-456")

    promote_mock = Mock(return_value=("reviews-classifier-3class", "8", "7"))
    monkeypatch.setattr(
        "scripts.register_model.register_and_promote",
        promote_mock,
    )

    caplog.set_level("INFO", logger="src.promotion")

    evaluate_and_maybe_promote(
        client=client,
        schema="3-class",
        candidate_run_id="candidate-run-123",
    )

    assert "candidate_run_id=candidate-run-123" in caplog.text
    assert "candidate_macro_f1=0.74" in caplog.text
    assert "production_macro_f1=0.71" in caplog.text
    assert "verdict=promoted" in caplog.text
    assert "new_version=8" in caplog.text
    assert "previous_version=7" in caplog.text


def test_candidate_already_in_production_is_idempotent(monkeypatch) -> None:
    """A retry must not register a candidate already behind Production."""
    client = Mock()

    client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={"macro_f1": 0.74}))
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        run_id="candidate-run-123",
        version="8",
    )

    promote_mock = Mock()
    monkeypatch.setattr(
        "scripts.register_model.register_and_promote",
        promote_mock,
    )

    result = evaluate_and_maybe_promote(
        client=client,
        schema="3-class",
        candidate_run_id="candidate-run-123",
    )

    assert result.promoted is True
    assert result.candidate_macro_f1 == 0.74
    assert result.production_macro_f1 == 0.74
    assert result.new_version == "8"
    assert result.previous_version is None

    promote_mock.assert_not_called()
    client.get_model_version_by_alias.assert_called_once_with(
        name="reviews-classifier-3class",
        alias="production",
    )
