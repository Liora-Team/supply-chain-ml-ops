"""JWT authentication dependency for the inference API (Card 3.3)."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

ALGORITHM = "HS256"
DEFAULT_EXPIRE_MINUTES = 60

_WINDOW_S = 60
_hits: dict[str, list[float]] = defaultdict(list)

security = HTTPBearer(auto_error=False)


def _load_env_once() -> None:
    """Lazily load .env for host runs (`make api`); no-op inside containers
    where .env is excluded via .dockerignore. Mirrors the guarded-import
    pattern in scripts/build_pipelines.py."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _secret() -> str:
    """Return the JWT signing secret from the environment."""
    _load_env_once()
    secret = os.environ.get("JWT_SECRET_KEY")

    if not secret:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")

    return secret


def create_access_token(
    subject: str,
    expires_minutes: int = DEFAULT_EXPIRE_MINUTES,
) -> str:
    """Issue a signed JWT for a client (used by scripts/tests, not by end users)."""
    expire = datetime.now(UTC) + timedelta(minutes=expires_minutes)

    payload = {
        "sub": subject,
        "exp": expire,
    }

    return jwt.encode(
        payload,
        _secret(),
        algorithm=ALGORITHM,
    )


def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """FastAPI dependency: validates the Bearer JWT, returns the subject claim."""

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            _secret(),
            algorithms=[ALGORITHM],
        )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired token.",
        ) from None

    return payload.get("sub", "unknown")


def _rate_limit() -> int:
    return int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))


def _client_id(request: Request) -> str:
    """Resolve the caller's IP, preferring the first X-Forwarded-For hop
    set by the nginx proxy. Falls back to the direct connection's host
    when called without a proxy in front (e.g. `make api` on :8000)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _prune_idle_clients() -> None:
    """Drop buckets with no hits inside the current window (called periodically,
    not on every request, to avoid extra work per call)."""
    now = time.monotonic()
    for client_id in list(_hits):
        _hits[client_id][:] = [t for t in _hits[client_id] if now - t < _WINDOW_S]
        if not _hits[client_id]:
            del _hits[client_id]


def enforce_rate_limit(request: Request) -> None:
    client_id = _client_id(request)
    now = time.monotonic()
    hits = _hits[client_id]
    hits[:] = [t for t in hits if now - t < _WINDOW_S]
    if len(hits) >= _rate_limit():
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    hits.append(now)
    if len(_hits) % 100 == 0:  # amortised cleanup, avoids per-request cost
        _prune_idle_clients()


def reset_rate_limits() -> None:
    """Test helper — clears in-memory rate-limit state."""
    _hits.clear()
