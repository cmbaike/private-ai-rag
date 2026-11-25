from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest


class APIKeyAuthenticationError(Exception):
    """API key is missing or invalid."""


class NamespaceAuthorizationError(Exception):
    """Authenticated principal lacks namespace access."""


@dataclass(frozen=True)
class APIKeyPrincipal:
    principal_id: str
    allowed_namespaces: frozenset[str]


class APIKeyAuthenticator:
    """
    Authenticate API keys against server-provisioned SHA-256 digests.

    Raw API keys must never be logged or stored in audit records.
    """

    def __init__(
        self,
        principals_by_digest: dict[str, APIKeyPrincipal],
    ) -> None:
        self._principals = dict(principals_by_digest)

    def authenticate(self, api_key: str | None) -> APIKeyPrincipal:
        if not isinstance(api_key, str) or not api_key:
            raise APIKeyAuthenticationError("Invalid API key.")

        digest = sha256(api_key.encode("utf-8")).hexdigest()

        # Compare against every configured digest rather than using
        # a direct dictionary lookup.
        principal = None

        for stored_digest, candidate in self._principals.items():
            if compare_digest(digest, stored_digest):
                principal = candidate

        if principal is None:
            raise APIKeyAuthenticationError("Invalid API key.")

        return principal


def authorize_namespace(
    principal: APIKeyPrincipal,
    namespace: str,
) -> None:
    if namespace not in principal.allowed_namespaces:
        raise NamespaceAuthorizationError(
            "Namespace access denied."
        )
