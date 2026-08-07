"""Security helpers: optional bearer-token authentication for the API."""
from __future__ import annotations

import secrets

from fastapi import HTTPException, Request

from ..config import get_settings

SCHEME = "Bearer"


def is_auth_enabled() -> bool:
    return bool(get_settings().app_api_key)


def check_auth(request: Request) -> None:
    """Raise 401/403 unless the request carries a valid API key."""
    if not is_auth_enabled():
        return
    header = request.headers.get("authorization", "")
    if not header.startswith(SCHEME + " "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = header[len(SCHEME) + 1 :].strip()
    if not secrets.compare_digest(token, get_settings().app_api_key or ""):
        raise HTTPException(status_code=403, detail="Invalid API key")


def new_api_key() -> str:
    """Generate a random key for `python -m app.core.security`."""
    return "ada_" + secrets.token_urlsafe(32)
