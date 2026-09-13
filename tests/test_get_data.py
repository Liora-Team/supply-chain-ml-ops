# tests/test_get_data.py
"""Unit tests for the get_data step functions (refactored for the Card 3.1 DAG).

Covers two test surfaces:
1. Original Card 3.1 tests (already merged by the team):
   `preprocess` and `split_and_write` on a small hand-made frame.
   `load_raw` needs the network, so it is not tested here.

2. Card 4.4 / PR #40 blocker-fix tests (Marco's review):
   `append_to_train_only` leakage guard — test.csv must never be modified
   by a category replay; incoming rows whose review text appears in test.csv
   must not enter train.csv; the function must fail closed if the review
   column is absent or if the baseline split is missing.
"""

import pandas as pd
import pytest

from scripts.get_data import main, preprocess, split_and_write
from src.preprocessing import collapse_to_3class, preprocess_text

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review": [
                "Great product, fast delivery!",
                "Terrible support. Never again.",
                "It was okay, nothing special.",
                "Absolutely love it, works perfectly.",
                "Broken on arrival and no refund.",
                "Average experience overall.",
            ],
            # two rows per star value so a stratified split can put one in each
            # side
            "stars": [5, 1, 3, 5, 1, 3],
        }
    )


# ---------------------------------------------------------------------------
# Card 3.1 — original team tests (already merged)
# ---------------------------------------------------------------------------


def test_preprocess_adds_lemma_and_3class_columns(raw_df):
    out = preprocess(raw_df)
    assert list(out["review_lemma"]) == [preprocess_text(t) for t in raw_df["review"]]
    assert list(out["label_3class"]) == list(collapse_to_3class(raw_df["stars"].to_numpy()))
    # input frame is not mutated
    assert "review_lemma" not in raw_df.columns


def test_split_and_write_writes_both_csvs(tmp_path, raw_df):
    df = preprocess(raw_df)
    train_path, test_path = split_and_write(df, out_dir=tmp_path, test_size=0.5, seed=0)

    assert train_path == tmp_path / "train.csv"
    assert test_path == tmp_path / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    assert len(train) == len(test) == 3
    assert list(train.columns) == list(df.columns)
    # stratified on stars: each star value lands in both splits
    assert set(train["stars"]) == set(test["stars"]) == {1, 3, 5}


def test_main_cli_category_triggers_append_not_split(monkeypatch, tmp_path, raw_df):
    seen: dict = {}

    # Card 4.4: fake_load_raw accepts the new optional `category` flag.
    def fake_load_raw(sample=0, category=None):
        seen["sample"] = sample
        seen["category"] = category
        return raw_df

    def fake_append_to_train_only(df, *, out_dir=None, **kwargs):
        seen["append_called"] = True
        return tmp_path / "train.csv"

    def split_should_not_be_called(*args, **kwargs):
        raise AssertionError("split_and_write should not be called when --category is set")

    monkeypatch.setattr("scripts.get_data.load_raw", fake_load_raw)
    monkeypatch.setattr("scripts.get_data.append_to_train_only", fake_append_to_train_only)
    monkeypatch.setattr("scripts.get_data.split_and_write", split_should_not_be_called)

    monkeypatch.setattr(
        "sys.argv",
        [
            "get_data.py",
            "--sample",
            "500",
            "--category",
            "Sports",
            "--test-size",
            "0.3",
            "--seed",
            "7",
        ],
    )

    main()

    assert seen == {"sample": 500, "category": "Sports", "append_called": True}


def test_main_cli_no_category_reaches_split_and_write(monkeypatch, tmp_path, raw_df):
    seen: dict = {}

    def fake_load_raw(sample=0, category=None):
        seen["sample"] = sample
        seen["category"] = category
        return raw_df

    def fake_split_and_write(df, *, test_size=0.2, seed=42, out_dir=None):
        seen["test_size"], seen["seed"] = test_size, seed
        return tmp_path / "train.csv", tmp_path / "test.csv"

    def append_should_not_be_called(*args, **kwargs):
        raise AssertionError("append_to_train_only should not be called when --category is not set")

    monkeypatch.setattr("scripts.get_data.load_raw", fake_load_raw)
    monkeypatch.setattr("scripts.get_data.split_and_write", fake_split_and_write)
    monkeypatch.setattr("scripts.get_data.append_to_train_only", append_should_not_be_called)

    monkeypatch.setattr(
        "sys.argv",
        [
            "get_data.py",
            "--sample",
            "500",
            "--test-size",
            "0.3",
            "--seed",
            "7",
        ],
    )

    main()

    assert seen == {
        "sample": 500,
        "category": None,
        "test_size": 0.3,
        "seed": 7,
    }


# ---------------------------------------------------------------------------
# Card 4.4 / PR #40 blocker-fix — append_to_train_only leakage guard
# ---------------------------------------------------------------------------


def _make_processed_dir(tmp_path):
    """Create a tmp data/processed directory with baseline train.csv and test.csv."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    train_df = pd.DataFrame(
        {
            "review": ["train review A", "train review B", "train review C"],
            "stars": [5, 4, 3],
            "review_lemma": ["train review a", "train review b", "train review c"],
            "label_3class": [2, 2, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "review": ["test review X", "test review Y"],
            "stars": [1, 2],
            "review_lemma": ["test review x", "test review y"],
            "label_3class": [0, 0],
        }
    )

    train_path = processed / "train.csv"
    test_path = processed / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return processed, train_path, test_path


def test_test_csv_is_never_modified(tmp_path):
    """Baseline: test.csv bytes must be identical before and after a category replay."""
    from scripts.get_data import append_to_train_only

    processed, train_path, test_path = _make_processed_dir(tmp_path)
    test_before = test_path.read_bytes()

    incoming = pd.DataFrame(
        {
            "review": ["brand new review Z", "train review A"],
            "stars": [5, 5],
            "review_lemma": ["brand new review z", "train review a"],
            "label_3class": [2, 2],
        }
    )

    append_to_train_only(incoming, out_dir=processed)

    test_after = test_path.read_bytes()
    assert test_before == test_after, "test.csv must not be modified by category replay"


def test_test_reviews_never_enter_train(tmp_path):
    """BLOCKER FIX: rows from test.csv must not enter train.csv after category replay.

    The incoming slice contains:
    - "train review A"     -> already in train; should be deduplicated.
    - "test review X"      -> in test.csv; must be excluded (leakage guard).
    - "brand new review Z" -> genuinely new; should be appended.
    """
    from scripts.get_data import append_to_train_only

    processed, train_path, test_path = _make_processed_dir(tmp_path)

    incoming = pd.DataFrame(
        {
            "review": ["train review A", "test review X", "brand new review Z"],
            "stars": [5, 1, 4],
            "review_lemma": [
                "train review a",
                "test review x",
                "brand new review z",
            ],
            "label_3class": [2, 0, 2],
        }
    )

    append_to_train_only(incoming, out_dir=processed)

    after_train = pd.read_csv(train_path)
    train_reviews = set(after_train["review"])

    # test reviews must not be in train
    assert (
        "test review X" not in train_reviews
    ), "LEAKAGE DETECTED: 'test review X' from test.csv entered train.csv"
    assert (
        "test review Y" not in train_reviews
    ), "LEAKAGE DETECTED: 'test review Y' from test.csv entered train.csv"

    # genuinely new row must be present
    assert (
        "brand new review Z" in train_reviews
    ), "Genuinely new row 'brand new review Z' was unexpectedly excluded"

    # existing train rows preserved
    assert "train review A" in train_reviews
    assert "train review B" in train_reviews


def test_column_mismatch_raises(tmp_path):
    """Column mismatch between baseline train.csv and incoming slice must raise SystemExit."""
    from scripts.get_data import append_to_train_only

    processed, _, _ = _make_processed_dir(tmp_path)

    bad_incoming = pd.DataFrame({"review": ["x"], "unknown_col": [1]})

    with pytest.raises(SystemExit, match="Column mismatch"):
        append_to_train_only(bad_incoming, out_dir=processed)


def test_missing_review_column_raises(tmp_path):
    """Missing 'review' column must raise SystemExit (fail-closed leakage guard)."""
    from scripts.get_data import append_to_train_only

    processed, train_path, test_path = _make_processed_dir(tmp_path)

    # Write a train.csv without the 'review' column to simulate the edge case.
    no_review_df = pd.DataFrame(
        {"stars": [5, 4], "review_lemma": ["a", "b"], "label_3class": [2, 2]}
    )
    no_review_df.to_csv(train_path, index=False)

    incoming = pd.DataFrame({"stars": [3], "review_lemma": ["c"], "label_3class": [1]})

    with pytest.raises(SystemExit, match="Cannot protect"):
        append_to_train_only(incoming, out_dir=processed)


def test_missing_baseline_raises(tmp_path):
    """No baseline train.csv must raise SystemExit with a clear DVC pull suggestion."""
    from scripts.get_data import append_to_train_only

    empty_dir = tmp_path / "data" / "processed"
    empty_dir.mkdir(parents=True)

    incoming = pd.DataFrame(
        {
            "review": ["x"],
            "stars": [1],
            "review_lemma": ["x"],
            "label_3class": [0],
        }
    )

    with pytest.raises(SystemExit, match="Incremental category replay requires"):
        append_to_train_only(incoming, out_dir=empty_dir)


def test_no_duplication_of_existing_train_rows(tmp_path):
    """Replaying a slice entirely in train.csv must not add duplicate rows."""
    from scripts.get_data import append_to_train_only

    processed, train_path, _ = _make_processed_dir(tmp_path)

    train_before = pd.read_csv(train_path)
    n_before = len(train_before)

    # Incoming is entirely a subset of the existing train.
    duplicate_slice = train_before.copy()

    append_to_train_only(duplicate_slice, out_dir=processed)

    train_after = pd.read_csv(train_path)
    assert len(train_after) == n_before, (
        f"Row count changed after replaying all-duplicate slice: "
        f"{n_before} -> {len(train_after)}"
    )
