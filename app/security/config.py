import json
import re
from hashlib import sha256
from collections.abc import Mapping

from app.security.api_keys import (
    APIKeyAuthenticator,
    APIKeyPrincipal,
)


class APIKeyConfigurationError(ValueError):
    """Server-side API-key configuration is invalid."""


_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_NAMESPACE_PATTERN = re.compile(
    r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$"
)


def load_api_key_authenticator(
    environment: Mapping[str, str],
) -> APIKeyAuthenticator:
    """
    Load server-managed API-key digests and namespace allowlists.

    Fail closed on missing or malformed configuration.
    Never include configuration values in error messages.
    """

    raw = environment.get("PRIVATE_AI_API_KEYS_JSON")

    if not raw:
        raise APIKeyConfigurationError(
            "API-key configuration is missing."
        )

    try:
        entries = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise APIKeyConfigurationError(
            "API-key configuration is invalid."
        ) from exc

    if not isinstance(entries, list) or not entries:
        raise APIKeyConfigurationError(
            "API-key configuration must contain entries."
        )

    principals_by_digest = {}
    principal_ids = set()

    for entry in entries:
        if not isinstance(entry, dict):
            raise APIKeyConfigurationError(
                "API-key configuration contains an invalid entry."
            )

        if set(entry) != {
            "principal_id",
            "key_sha256",
            "allowed_namespaces",
        }:
            raise APIKeyConfigurationError(
                "API-key configuration contains invalid fields."
            )

        principal_id = entry["principal_id"]
        digest = entry["key_sha256"]
        namespaces = entry["allowed_namespaces"]

        if (
            not isinstance(principal_id, str)
            or not principal_id
            or len(principal_id) > 128
            or principal_id in principal_ids
        ):
            raise APIKeyConfigurationError(
                "API-key configuration contains an invalid principal."
            )

        if (
            not isinstance(digest, str)
            or not _DIGEST_PATTERN.fullmatch(digest)
            or digest in principals_by_digest
        ):
            raise APIKeyConfigurationError(
                "API-key configuration contains an invalid digest."
            )

        if (
            not isinstance(namespaces, list)
            or not namespaces
            or len(namespaces) != len(set(
                value for value in namespaces
                if isinstance(value, str)
            ))
        ):
            raise APIKeyConfigurationError(
                "API-key configuration contains invalid namespaces."
            )

        if not all(
            isinstance(namespace, str)
            and len(namespace) <= 63
            and _NAMESPACE_PATTERN.fullmatch(namespace)
            for namespace in namespaces
        ):
            raise APIKeyConfigurationError(
                "API-key configuration contains invalid namespaces."
            )

        principals_by_digest[digest] = APIKeyPrincipal(
            principal_id=principal_id,
            allowed_namespaces=frozenset(namespaces),
        )

        principal_ids.add(principal_id)

    return APIKeyAuthenticator(principals_by_digest)