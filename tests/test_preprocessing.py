"""Unit tests for the preprocessing main-step functions (Phase 1 requirement)."""

import numpy as np

from src.preprocessing import (
    basic_clean,
    collapse_to_3class,
    label_name,
    preprocess_text,
    remove_stopwords,
)


def test_basic_clean_lowercases_and_removes_punct_and_digits():
    assert basic_clean("Great PRODUCT!! Bought 100 of them.") == "great product bought of them"


def test_remove_stopwords_drops_common_words_keeps_content():
    out = remove_stopwords("this is a great product").split()
    assert "great" in out and "product" in out
    assert "this" not in out and "is" not in out and "a" not in out


def test_preprocess_text_full_chain_and_lemmatisation():
    out = preprocess_text("The cats were running terribly!!")
    assert out == out.lower()
    assert not any(ch.isdigit() for ch in out)
    assert "cat" in out.split()  # 'cats' -> 'cat'


def test_collapse_to_3class_maps_stars_to_sentiment():
    stars = np.array([1, 2, 3, 4, 5])
    assert collapse_to_3class(stars).tolist() == [0, 0, 1, 2, 2]


def test_label_name_per_schema():
    assert label_name(0, "3-class") == "Negative"
    assert label_name(1, "3-class") == "Neutral"
    assert label_name(2, "3-class") == "Positive"
    assert label_name(5, "5-class") == "5★"
