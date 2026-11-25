from fastapi.testclient import TestClient

from app.api import app
from app.security.dependencies import (
    get_api_key_authenticator,
)
from app.security.api_keys import APIKeyAuthenticator


def test_privileged_routes_require_authentication():
    """
    No privileged route may be accessed without credentials.

    Authentication must reject the request before the
    handler performs document or tool operations.
    """
    app.dependency_overrides.clear()
    
    app.dependency_overrides[ get_api_key_authenticator ] = lambda: APIKeyAuthenticator({})

    with TestClient(app) as client:
        routes = [
            ("POST", "/search"),
            ("POST", "/ask"),
            ("POST", "/v1/chat/completions"),
            ("GET", "/v1/models"),
            ("POST", "/documents"),
            ("GET", "/documents"),
            ("GET", "/documents/test-document"),
            ("DELETE", "/documents/test-document"),
        ]

        for method, path in routes:
            response = client.request(
                method,
                path,
            )

            assert response.status_code == 401, (
                method,
                path,
                response.status_code,
            )