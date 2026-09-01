"""Shared pytest fixtures for the API test suite."""

import pytest

from api.auth import create_access_token, reset_rate_limits


@pytest.fixture
def auth_headers() -> dict:
    token = create_access_token("tester")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def _clear_rate_limits():
    """Reset the in-memory rate-limit bucket before/after every test.

    TestClient uses a fixed client host, so without this the bucket accumulates
    hits across unrelated tests and test_rate_limit_returns_429 stops actually
    testing burst behaviour.
    """
    reset_rate_limits()
    yield
    reset_rate_limits()
