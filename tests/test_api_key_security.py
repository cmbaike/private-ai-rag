from hashlib import sha256
from uuid import uuid4

import pytest

from app.security.api_keys import (
    APIKeyAuthenticator,
    APIKeyAuthenticationError,
    APIKeyPrincipal,
    NamespaceAuthorizationError,
    authorize_namespace,
)
from app.tools.orchestration import run_read_only_tool_request


@pytest.fixture
def authenticator():
    digest = sha256(b"private-beta-test-key").hexdigest()

    return APIKeyAuthenticator(
        {
            digest: APIKeyPrincipal(
                principal_id="beta-user-1",
                allowed_namespaces=frozenset(
                    {"ai-ops-demo", "staging"}
                ),
            )
        }
    )


def test_valid_api_key_authenticates(authenticator):
    principal = authenticator.authenticate(
        "private-beta-test-key"
    )

    assert principal.principal_id == "beta-user-1"


@pytest.mark.parametrize(
    "api_key",
    [None, "", "incorrect-key"],
)
def test_invalid_api_key_is_rejected(authenticator, api_key):
    with pytest.raises(APIKeyAuthenticationError):
        authenticator.authenticate(api_key)


def test_authorized_namespace_is_allowed(authenticator):
    principal = authenticator.authenticate(
        "private-beta-test-key"
    )

    authorize_namespace(principal, "ai-ops-demo")


def test_unauthorized_namespace_is_rejected(authenticator):
    principal = authenticator.authenticate(
        "private-beta-test-key"
    )

    with pytest.raises(NamespaceAuthorizationError):
        authorize_namespace(principal, "production")


def test_namespace_matching_is_exact(authenticator):
    principal = authenticator.authenticate(
        "private-beta-test-key"
    )

    with pytest.raises(NamespaceAuthorizationError):
        authorize_namespace(principal, "ai-ops-demo-extra")
            
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.security.api_keys import (
    APIKeyPrincipal,
    NamespaceAuthorizationError,
)
from app.tools.models import ToolCall
from app.tools.orchestration import run_read_only_tool_request


def test_unauthorized_namespace_never_reaches_execution(monkeypatch):
    principal = APIKeyPrincipal(
        principal_id="beta-user-1",
        allowed_namespaces=frozenset({"ai-ops-demo"}),
    )

    call = ToolCall(
        name="k8s.get_pods",
        arguments={"namespace": "production"},
    )

    monkeypatch.setattr(
        "app.tools.orchestration.propose_tool",
        lambda user_text: object(),
    )
    monkeypatch.setattr(
        "app.tools.orchestration.gate_tool_proposal",
        lambda proposal, user_text: SimpleNamespace(
            status="bound",
            call=call,
        ),
    )

    def forbidden_execution(*args, **kwargs):
        raise AssertionError(
            "Unauthorized call reached audited execution."
        )

    monkeypatch.setattr(
        "app.tools.orchestration.execute_audited_tool_call",
        forbidden_execution,
    )

    with pytest.raises(NamespaceAuthorizationError):
        run_read_only_tool_request(
            request_id=uuid4(),
            user_text="Inspect pods in namespace production.",
            principal=principal,
        )

       