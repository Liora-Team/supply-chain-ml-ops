"""JWT authentication dependency for the inference API (Card 3.3)."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

load_dotenv()


ALGORITHM = "HS256"
DEFAULT_EXPIRE_MINUTES = 60

_WINDOW_S = 60
_hits: dict[str, list[float]] = defaultdict(list)

security = HTTPBearer(auto_error=False)


def _secret() -> str:
    """Return the JWT signing secret from the environment."""
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


def enforce_rate_limit(request: Request) -> None:
    client_id = request.client.host if request.client else "unknown"
    now = time.monotonic()
    hits = _hits[client_id]
    hits[:] = [t for t in hits if now - t < _WINDOW_S]
    if len(hits) >= _rate_limit():
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    hits.append(now)


def reset_rate_limits() -> None:
    """Test helper — clears in-memory rate-limit state."""
    _hits.clear()
