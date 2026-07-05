from __future__ import annotations

import logging
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


class AuthenticatedUser:
    """Lightweight container for the verified JWT claims."""

    def __init__(self, *, user_id: str, email: str) -> None:
        self.user_id = user_id
        self.email = email

    def __repr__(self) -> str:  # pragma: no cover
        return f"AuthenticatedUser(user_id={self.user_id!r}, email={self.email!r})"


def _decode_token(token: str) -> dict[str, Any]:
    """
    Decode and verify a Supabase JWT.

    Supabase signs JWTs with HS256 using the project's JWT secret.
    The 'audience' claim is always 'authenticated' for logged-in users.

    Raises jwt.InvalidTokenError on any verification failure.
    """
    return jwt.decode(
        token,
        settings.supabase_jwt_secret,
        algorithms=["HS256"],
        audience="authenticated",
        options={"verify_exp": True},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> AuthenticatedUser:
    """
    FastAPI dependency — requires a valid Supabase JWT in the Authorization header.

    Usage:
        @router.get("/protected")
        def my_endpoint(user: AuthenticatedUser = Depends(get_current_user)):
            ...

    Raises:
        HTTP 401 — missing header, expired token, bad signature, missing sub claim.
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Provide: Authorization: Bearer <token>",
        )

    token = credentials.credentials

    try:
        payload = _decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please sign in again.")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Token audience mismatch.")
    except jwt.InvalidTokenError as exc:
        logger.debug("JWT verification failed: %s", exc)
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token is missing the 'sub' (user ID) claim.")

    email: str = payload.get("email", "")
    return AuthenticatedUser(user_id=user_id, email=email)


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> AuthenticatedUser | None:
    """
    FastAPI dependency — like get_current_user but returns None instead of
    raising for unauthenticated requests.

    Use this on endpoints that support BOTH authenticated and anonymous access
    (e.g. POST /simulate/start — v0.1 anonymous flow must remain intact).

    Usage:
        @router.post("/simulate/start")
        def start(user: AuthenticatedUser | None = Depends(get_optional_user)):
            user_id = user.user_id if user else None
    """
    if credentials is None:
        return None
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None
