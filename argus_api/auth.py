"""Minimal token-to-role RBAC primitives for the ARGUS operator surface."""
from __future__ import annotations

import hmac
from dataclasses import dataclass

from ops.security import OperatorRole, role_allows


@dataclass(frozen=True)
class Principal:
    subject: str
    role: OperatorRole


class StaticTokenAuthenticator:
    """Small deployable auth boundary; production secrets come from env/secret manager."""

    def __init__(self, tokens: dict[str, tuple[str, OperatorRole | str]]) -> None:
        self._tokens = {token: Principal(subject, OperatorRole(role)) for token, (subject, role) in tokens.items()}

    def authenticate(self, token: str) -> Principal | None:
        for candidate, principal in self._tokens.items():
            if hmac.compare_digest(candidate, token):
                return principal
        return None

    def authorize(self, token: str, permission: str) -> Principal:
        principal = self.authenticate(token)
        if principal is None:
            raise PermissionError("invalid operator token")
        if not role_allows(principal.role, permission):
            raise PermissionError(f"role {principal.role.value} lacks {permission}")
        return principal
