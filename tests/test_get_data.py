"""Unit tests for the get_data step functions (refactored for the Card 3.1 DAG).

`load_raw` needs the network, so it is not tested here; `preprocess` and
`split_and_write` are exercised on a small hand-made frame.
"""

import pandas as pd
import pytest

from scripts.get_data import main, preprocess, split_and_write
from src.preprocessing import collapse_to_3class, preprocess_text


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
            # two rows per star value so a stratified split can put one in each side
            "stars": [5, 1, 3, 5, 1, 3],
        }
    )


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


def test_main_cli_flags_reach_the_step_functions(monkeypatch, tmp_path, raw_df):
    seen: dict = {}

    # Card 4.4: fake_load_raw now accepts the new optional `category` flag.
    def fake_load_raw(sample=0, category=None):
        seen["sample"] = sample
        seen["category"] = category
        return raw_df

    def fake_split_and_write(df, *, test_size=0.2, seed=42):
        seen["test_size"], seen["seed"] = test_size, seed
        return tmp_path / "train.csv", tmp_path / "test.csv"

    monkeypatch.setattr("scripts.get_data.load_raw", fake_load_raw)
    monkeypatch.setattr("scripts.get_data.split_and_write", fake_split_and_write)
    monkeypatch.setattr(
        "sys.argv",
        [
            "get_data.py",
            "--sample",
            "500",
            "--category",
            "Sports",  # Card 4.4: verify the new CLI flag reaches load_raw
            "--test-size",
            "0.3",
            "--seed",
            "7",
        ],
    )

    main()

    # Card 4.4: `category` is now part of the captured flags.
    assert seen == {"sample": 500, "category": "Sports", "test_size": 0.3, "seed": 7}
