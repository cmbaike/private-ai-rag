from hashlib import sha256
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app
from app.security.api_keys import APIKeyAuthenticator, APIKeyPrincipal
from app.security.dependencies import get_api_key_authenticator


RAW_KEY = "open-webui-test-key"
DIGEST = sha256(RAW_KEY.encode("utf-8")).hexdigest()


def setup_function():
    app.dependency_overrides.clear()
    app.dependency_overrides[get_api_key_authenticator] = lambda: (
        APIKeyAuthenticator(
            {
                DIGEST: APIKeyPrincipal(
                    principal_id="webui-user",
                    allowed_namespaces=frozenset({"ai-ops-demo"}),
                )
            }
        )
    )


def teardown_function():
    app.dependency_overrides.clear()


@patch("app.openai_compat.run_grounded_ask")
def test_chat_completions_accepts_bearer_token(mock_ask):
    mock_ask.return_value = {
        "request_id": "req-bearer",
        "status": "answered",
        "question": "hi",
        "answer": "hello",
        "sources": [],
        "sufficiency": {"sufficient": True},
        "timings": {
            "retrieval_ms": 1.0,
            "sufficiency_ms": 1.0,
            "generation_ms": 1.0,
            "total_ms": 3.0,
        },
    }

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {RAW_KEY}"},
            json={
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"


def test_chat_completions_rejects_bad_bearer():
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-key"},
            json={
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 401
