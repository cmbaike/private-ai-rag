import json
from hashlib import sha256

import pytest

from app.security.api_keys import APIKeyAuthenticationError
from app.security.config import (
    APIKeyConfigurationError,
    load_api_key_authenticator,
)


TEST_KEY = "private-beta-test-key"


def valid_environment():
    return {
        "PRIVATE_AI_API_KEYS_JSON": json.dumps([
            {
                "principal_id": "beta-user-1",
                "key_sha256": sha256(
                    TEST_KEY.encode("utf-8")
                ).hexdigest(),
                "allowed_namespaces": [
                    "ai-ops-demo",
                    "staging",
                ],
            }
        ])
    }


def test_valid_configuration_authenticates():
    authenticator = load_api_key_authenticator(
        valid_environment()
    )

    principal = authenticator.authenticate(TEST_KEY)

    assert principal.principal_id == "beta-user-1"
    assert principal.allowed_namespaces == frozenset({
        "ai-ops-demo",
        "staging",
    })


def test_missing_configuration_fails_closed():
    with pytest.raises(APIKeyConfigurationError):
        load_api_key_authenticator({})


def test_malformed_json_fails_closed():
    with pytest.raises(APIKeyConfigurationError):
        load_api_key_authenticator({
            "PRIVATE_AI_API_KEYS_JSON": "{invalid"
        })


@pytest.mark.parametrize(
    "namespaces",
    [
        [],
        ["production", "production"],
        ["Production"],
        ["invalid_namespace"],
        ["a" * 64],
        ["*"],
    ],
)
def test_invalid_namespace_configuration_rejected(namespaces):
    environment = valid_environment()
    entries = json.loads(
        environment["PRIVATE_AI_API_KEYS_JSON"]
    )

    entries[0]["allowed_namespaces"] = namespaces

    environment["PRIVATE_AI_API_KEYS_JSON"] = json.dumps(
        entries
    )

    with pytest.raises(APIKeyConfigurationError):
        load_api_key_authenticator(environment)


def test_duplicate_digest_rejected():
    environment = valid_environment()
    entries = json.loads(
        environment["PRIVATE_AI_API_KEYS_JSON"]
    )

    entries.append({
        **entries[0],
        "principal_id": "beta-user-2",
    })

    environment["PRIVATE_AI_API_KEYS_JSON"] = json.dumps(
        entries
    )

    with pytest.raises(APIKeyConfigurationError):
        load_api_key_authenticator(environment)


def test_duplicate_principal_rejected():
    environment = valid_environment()
    entries = json.loads(
        environment["PRIVATE_AI_API_KEYS_JSON"]
    )

    entries.append({
        **entries[0],
        "key_sha256": sha256(b"another-key").hexdigest(),
    })

    environment["PRIVATE_AI_API_KEYS_JSON"] = json.dumps(
        entries
    )

    with pytest.raises(APIKeyConfigurationError):
        load_api_key_authenticator(environment)


def test_unknown_api_key_rejected():
    authenticator = load_api_key_authenticator(
        valid_environment()
    )

    with pytest.raises(APIKeyAuthenticationError):
        authenticator.authenticate("wrong-key")