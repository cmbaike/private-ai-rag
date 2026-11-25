from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app
from app.ask_pipeline import INSUFFICIENT_EVIDENCE_MESSAGE
from app.security.api_keys import APIKeyPrincipal
from app.security.dependencies import require_api_key_principal

client = TestClient(app)


SOURCE = {
    "filename": "runbook.pdf",
    "chunk_index": 12,
    "page_number": 6,
    "content": (
        "Red means at least one primary shard is unassigned."
    ),
    "similarity": 0.91,
    "label": "runbook.pdf#page-6-chunk-12",
    "excerpt": (
        "Red means at least one primary shard is unassigned."
    ),
}


def _auth_override():
    return APIKeyPrincipal(
        principal_id="api-test",
        allowed_namespaces=frozenset({"ai-ops-demo"}),
    )


def setup_function():
    app.dependency_overrides[require_api_key_principal] = _auth_override


def teardown_function():
    app.dependency_overrides.pop(require_api_key_principal, None)


@patch("app.openai_compat.run_grounded_ask")
def test_chat_completions_answered(mock_ask):
    mock_ask.return_value = {
        "request_id": "req-1",
        "status": "answered",
        "question": "What does red mean?",
        "answer": "At least one primary shard is unassigned.",
        "sources": [SOURCE],
        "sufficiency": {"sufficient": True},
        "timings": {
            "retrieval_ms": 1.0,
            "sufficiency_ms": 1.0,
            "generation_ms": 1.0,
            "total_ms": 3.0,
        },
    }

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-4B",
            "messages": [
                {"role": "user", "content": "What does red mean?"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    content = body["choices"][0]["message"]["content"]
    assert "At least one primary shard is unassigned." in content
    assert "Sources:" in content
    assert "runbook.pdf#page-6-chunk-12" in content
    mock_ask.assert_called_once()


@patch("app.openai_compat.run_grounded_ask")
def test_chat_completions_insufficient(mock_ask):
    mock_ask.return_value = {
        "request_id": "req-2",
        "status": "insufficient_evidence",
        "question": "Who invented Kubernetes?",
        "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
        "sources": [],
        "sufficiency": {"sufficient": False},
        "timings": {
            "retrieval_ms": 1.0,
            "sufficiency_ms": 1.0,
            "generation_ms": None,
            "total_ms": 2.0,
        },
    }

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Who invented Kubernetes?"},
            ],
        },
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content == INSUFFICIENT_EVIDENCE_MESSAGE


@patch("app.openai_compat.run_grounded_ask")
def test_chat_completions_stream(mock_ask):
    mock_ask.return_value = {
        "request_id": "req-3",
        "status": "answered",
        "question": "q",
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

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "q"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text = response.text
    assert "data:" in text
    assert "[DONE]" in text
    assert "hello" in text


def test_models_lists_configured_model():
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert ids


def test_chat_completions_requires_user_message():
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "You are helpful."},
            ],
        },
    )
    assert response.status_code == 400
