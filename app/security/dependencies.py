import os
from functools import lru_cache

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from app.security.api_keys import (
    APIKeyAuthenticationError,
    APIKeyAuthenticator,
    APIKeyPrincipal,
)
from app.security.config import load_api_key_authenticator


_api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


@lru_cache(maxsize=1)
def get_api_key_authenticator() -> APIKeyAuthenticator:
    """
    Load server-provisioned credentials once per process.

    Missing or invalid configuration must not create an
    unauthenticated fallback.
    """
    return load_api_key_authenticator(os.environ)


def require_api_key_principal(
    api_key: str | None = Depends(_api_key_header),
    authenticator: APIKeyAuthenticator = Depends(
        get_api_key_authenticator
    ),
) -> APIKeyPrincipal:
    try:
        return authenticator.authenticate(api_key)
    except APIKeyAuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        ) from None