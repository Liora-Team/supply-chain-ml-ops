"""Unit tests for scripts/simulate_drift.py (Card 4.1 proof-of-work step).

`_load_dataset_df` needs the network (downloads from the HF hub), so it is mocked
here — same precedent as `tests/test_get_data.py` not testing `load_raw`. Everything
downstream of it (slicing, replay, and a real drift check) is exercised for real.
"""

from __future__ import annotations

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

import scripts.simulate_drift as simulate_drift


@pytest.fixture
def fake_dataset(monkeypatch):
    """A tiny 2-category frame standing in for the real HF download."""
    df = pd.DataFrame(
        {
            "category": ["Electronics"] * 5 + ["Pets"] * 3,
            "review": [f"electronics review {i}" for i in range(5)]
            + [f"pets review {i}" for i in range(3)],
        }
    )
    monkeypatch.setattr(simulate_drift, "_load_dataset_df", lambda: df)
    return df


def test_load_category_slice_filters_and_caps(fake_dataset):
    texts = simulate_drift.load_category_slice("Electronics", n=3)

    assert len(texts) == 3
    assert all("electronics" in t for t in texts)


def test_load_category_slice_unknown_category_raises(fake_dataset):
    with pytest.raises(SystemExit, match="Unknown category"):
        simulate_drift.load_category_slice("Nonexistent", n=10)


def test_list_categories_prints_counts(fake_dataset, capsys):
    simulate_drift.list_categories()

    out = capsys.readouterr().out
    assert "Electronics" in out
    assert "Pets" in out


class _RecordingPredictHandler(BaseHTTPRequestHandler):
    """Minimal stand-in for the real /predict endpoint: records what it received."""

    received: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server's required method name
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.__class__.received.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *args):  # silence default request logging in test output
        pass


@pytest.fixture
def stub_predict_server():
    _RecordingPredictHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _RecordingPredictHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", _RecordingPredictHandler
    finally:
        server.shutdown()


def test_replay_posts_each_text_with_auth_header(stub_predict_server):
    base_url, handler = stub_predict_server

    ok, failed = simulate_drift.replay(
        ["review one", "review two"],
        base_url=base_url,
        token="fake-token",
        schema="3-class",
        sleep=0,
    )

    assert (ok, failed) == (2, 0)
    assert len(handler.received) == 2
    assert handler.received[0]["path"] == "/predict"
    assert handler.received[0]["auth"] == "Bearer fake-token"
    assert handler.received[0]["body"] == {"text": "review one", "schema": "3-class"}


def test_replay_counts_failures_without_raising(stub_predict_server):
    base_url, _ = stub_predict_server
    dead_url = "http://127.0.0.1:1"  # nothing listens here

    ok, failed = simulate_drift.replay(
        ["a text"], base_url=dead_url, token="t", schema="3-class", sleep=0
    )

    assert (ok, failed) == (0, 1)


class _RateLimitedHandler(BaseHTTPRequestHandler):
    """Returns 429 for the first `fail_count` requests, then 200."""

    fail_count = 0
    seen = 0

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        self.__class__.seen += 1
        if self.__class__.seen <= self.__class__.fail_count:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *args):
        pass


@pytest.fixture
def rate_limited_server():
    _RateLimitedHandler.seen = 0
    _RateLimitedHandler.fail_count = 2
    server = HTTPServer(("127.0.0.1", 0), _RateLimitedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


def test_replay_retries_on_429_and_eventually_succeeds(rate_limited_server):
    ok, failed = simulate_drift.replay(
        ["one text"], base_url=rate_limited_server, token="t", schema="3-class", sleep=0
    )

    assert (ok, failed) == (1, 0)
    assert _RateLimitedHandler.seen == 3  # two 429s, then the successful attempt


def test_replay_gives_up_after_max_retries_on_persistent_429(rate_limited_server):
    _RateLimitedHandler.fail_count = 999  # never succeeds

    ok, failed = simulate_drift.replay(
        ["one text"],
        base_url=rate_limited_server,
        token="t",
        schema="3-class",
        sleep=0,
        max_retries=2,
    )

    assert (ok, failed) == (0, 1)
    assert _RateLimitedHandler.seen == 3  # 1 initial attempt + 2 retries


def test_skewed_slice_actually_flips_drift_detected(monkeypatch, tmp_path):
    """End-to-end proof: a genuinely skewed category slice trips the real drift job."""
    monkeypatch.setenv("MONITORING_DB_PATH", str(tmp_path / "store.db"))
    monkeypatch.setenv("REFERENCE_DATA_DIR", str(tmp_path))

    import monitoring.datasets as datasets
    import monitoring.drift_report as drift_report
    import monitoring.request_store as request_store

    importlib.reload(request_store)
    importlib.reload(datasets)
    importlib.reload(drift_report)

    # Reference: short, uniform "normal" reviews (stands in for data/processed/train.csv).
    pd.DataFrame(
        {"review": [f"fine product, works well {i % 3}" for i in range(300)], "stars": [4] * 300}
    ).to_csv(tmp_path / "train.csv", index=False)

    # Current: a skewed slice — much longer, unrelated-vocabulary "reviews".
    skewed_texts = [
        f"the shipment arrived extremely late and the packaging was completely "
        f"destroyed during transit which was very disappointing overall case {i}"
        for i in range(80)
    ]
    for text in skewed_texts:
        request_store.log_prediction(text, "negative", 0.8, "sim", schema="3-class")

    result = drift_report.run_drift_report("3-class", min_rows=10)

    assert result["status"] == "ok"
    assert result["drift_detected"] is True