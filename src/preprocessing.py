"""Preprocessing — the SINGLE source of truth for preparing review text.

Turns raw review text into the exact representation the trained models expect:
clean -> stopword removal -> lemmatisation, plus the 3-class label collapse.
The app, the API, and scripts/get_data.py all import these same functions, so
no consumer can drift from the representation the models were trained on.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np


# Lazy NLTK resource download — only fetch what is missing
def _ensure_nltk() -> None:
    """Download NLTK stopwords + wordnet if not already cached."""
    import nltk

    for pkg, path in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


# Stateless text cleaning
def basic_clean(text: str) -> str:
    """Lowercase, remove punctuation and digits, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # keep letters and spaces only
    text = re.sub(r"\s+", " ", text).strip()  # collapse multiple spaces
    return text


# Stopword set + lemmatiser are expensive to build — cache them once
@lru_cache(maxsize=1)
def _stopwords() -> frozenset:
    _ensure_nltk()
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatiser():
    _ensure_nltk()
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


def remove_stopwords(text: str) -> str:
    """Drop NLTK English stopwords."""
    sw = _stopwords()
    return " ".join(w for w in text.split() if w not in sw)


def lemmatise(text: str) -> str:
    """Map each token to its WordNet lemma."""
    lemma = _lemmatiser()
    return " ".join(lemma.lemmatize(w) for w in text.split())


# Full text pipeline — raw user input -> the `review_lemma` model input
def preprocess_text(text: str) -> str:
    """Run the full clean -> stopword -> lemmatise chain on one review.

    This is exactly how the `review_lemma` model-input column is produced
    (basic_clean -> remove_stopwords -> lemmatise); the app feeds the user's
    whole text box through the same chain.
    """
    return lemmatise(remove_stopwords(basic_clean(text)))


# 3-class collapse — the operational negative/neutral/positive view
def collapse_to_3class(y: np.ndarray) -> np.ndarray:
    """Map raw 1-5 stars to {0=neg, 1=neu, 2=pos}."""
    y = np.asarray(y)
    out = np.empty_like(y)
    out[(y == 1) | (y == 2)] = 0  # negative
    out[y == 3] = 1  # neutral
    out[(y == 4) | (y == 5)] = 2  # positive
    return out


# Human-readable label names, indexed by the integer label each schema uses.
# 5-class models predict raw stars 1-5; 3-class models predict 0/1/2.
LABELS_5CLASS = {1: "1★", 2: "2★", 3: "3★", 4: "4★", 5: "5★"}
LABELS_3CLASS = {0: "Negative", 1: "Neutral", 2: "Positive"}


def label_name(label: int, schema: str) -> str:
    """Return the display name for an integer label under a given schema."""
    if schema == "3-class":
        return LABELS_3CLASS.get(int(label), str(label))
    return LABELS_5CLASS.get(int(label), str(label))
