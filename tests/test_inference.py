"""Unit tests for classical inference on the served model pipelines."""

import pytest

from src import registry
from src.inference import Prediction, _star_shift, predict_classical


@pytest.fixture(scope="module")
def best_3class_id() -> str:
    loadable = [e for e in registry.classical("3-class") if e.loadable]
    if not loadable:
        pytest.skip("no loadable 3-class model pipeline is present")
    return loadable[0].id


def test_predict_returns_valid_prediction(best_3class_id):
    p = predict_classical(best_3class_id, "Absolutely wonderful, exceeded my expectations!")
    assert isinstance(p, Prediction)
    assert p.label_display in {"Negative", "Neutral", "Positive"}
    assert set(p.probs) == {"Negative", "Neutral", "Positive"}
    assert abs(sum(p.probs.values()) - 1.0) < 1e-6


def test_sentiment_direction(best_3class_id):
    pos = predict_classical(best_3class_id, "Best purchase ever, amazing quality!")
    neg = predict_classical(best_3class_id, "Terrible, broke in a day. Waste of money.")
    assert pos.label_display == "Positive"
    assert neg.label_display == "Negative"


def test_unknown_model_raises_keyerror():
    with pytest.raises(KeyError):
        predict_classical("this-model-does-not-exist", "hello")


def test_star_shift_only_for_zero_indexed_5class():
    # XGBoost 5-class is fit on 0-indexed stars (0-4) and must be shifted back;
    # every other combination predicts its labels directly. Mirror of the fit
    # side in scripts/build_pipelines.py.
    assert _star_shift([0, 1, 2, 3, 4], "5-class") == 1  # XGBoost-style, shift
    assert _star_shift([1, 2, 3, 4, 5], "5-class") == 0  # stars predicted directly
    assert _star_shift([0, 1, 2], "3-class") == 0  # 3-class is always 0-indexed
