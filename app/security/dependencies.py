import os
from functools import lru_cache

from fastapi import Depends, HTTPException
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

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

_bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def get_api_key_authenticator() -> APIKeyAuthenticator:
    """
    Load server-provisioned credentials once per process.

    Missing or invalid configuration must not create an
    unauthenticated fallback.
    """
    return load_api_key_authenticator(os.environ)


def _credential_from_request(
    api_key: str | None,
    bearer: HTTPAuthorizationCredentials | None,
) -> str | None:
    """Prefer X-API-Key; fall back to Authorization Bearer (Open WebUI)."""
    if isinstance(api_key, str) and api_key:
        return api_key
    if bearer is not None and bearer.scheme.lower() == "bearer":
        return bearer.credentials
    return None


def require_api_key_principal(
    api_key: str | None = Depends(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer),
    authenticator: APIKeyAuthenticator = Depends(
        get_api_key_authenticator
    ),
) -> APIKeyPrincipal:
    try:
        return authenticator.authenticate(
            _credential_from_request(api_key, bearer)
        )
    except APIKeyAuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey, Bearer"},
        ) from None
