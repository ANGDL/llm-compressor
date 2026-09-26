"""Role-based authorization for the model quality control plane."""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

ROLES = ("viewer", "operator", "publisher", "admin")

# A role grants the union of its permissions. ``viewer`` is observational only,
# ``operator`` may drive planning/execution, ``publisher`` additionally owns
# publication and backup, and ``admin`` is the only role that may restore a
# backup into a new run.
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset({"read"}),
    "operator": frozenset(
        {"read", "ops.read", "plan", "start", "retry", "cancel", "evaluate"}
    ),
    "publisher": frozenset(
        {
            "read",
            "ops.read",
            "plan",
            "start",
            "retry",
            "cancel",
            "evaluate",
            "publish",
            "backup",
        }
    ),
    "admin": frozenset(
        {
            "read",
            "ops.read",
            "plan",
            "start",
            "retry",
            "cancel",
            "evaluate",
            "publish",
            "backup",
            "restore",
            "schedule",
        }
    ),
}

PERMISSIONS = frozenset(
    permission for role in ROLES for permission in ROLE_PERMISSIONS[role]
)


class AuthenticationError(PermissionError):
    """Raised when the request carries no usable identity."""


class PermissionDenied(PermissionError):
    """Raised when an authenticated actor lacks the required permission."""


@dataclass(frozen=True)
class Principal:
    """An authenticated actor and the roles granted to it."""

    actor: str
    roles: frozenset[str] = field(default_factory=lambda: frozenset({"viewer"}))
    authenticated: bool = False

    @property
    def permissions(self) -> frozenset[str]:
        granted: set[str] = set()
        for role in self.roles:
            granted |= ROLE_PERMISSIONS.get(role, frozenset())
        return frozenset(granted)

    def can(self, permission: str) -> bool:
        if permission not in PERMISSIONS:
            raise ValueError(f"unknown permission: {permission!r}")
        return permission in self.permissions

    def require(self, permission: str) -> None:
        if not self.can(permission):
            raise PermissionDenied(
                f"actor {self.actor!r} lacks {permission!r} permission"
            )

    @property
    def is_authenticated(self) -> bool:
        return self.authenticated


ANONYMOUS = Principal(actor="anonymous", roles=frozenset({"viewer"}))


def parse_roles(values: Iterable[str] | None) -> frozenset[str]:
    """Normalize a header value or CLI argument into known roles."""

    if values is None:
        return frozenset({"viewer"})
    if isinstance(values, str):
        values = [part for part in values.replace(" ", ",").split(",") if part]
    roles = {value.strip().lower() for value in values if value and value.strip()}
    unknown = roles - set(ROLES)
    if unknown:
        raise ValueError(f"unknown roles: {', '.join(sorted(unknown))}")
    return frozenset(roles) if roles else frozenset({"viewer"})


def principal_from_headers(
    headers: dict[str, str],
    *,
    required_token: str | None = None,
    allow_anonymous: bool = True,
) -> Principal:
    """Build a principal from HTTP headers.

    Identity is header-driven until an OIDC/SSO integration replaces it; the
    shared token only gates *writes*, so a deployment can expose read-only
    dashboards without issuing per-user credentials.
    """

    lowered = {key.lower(): value for key, value in headers.items()}
    actor = (lowered.get("x-model-quality-actor") or "").strip()
    roles = (
        lowered.get("x-model-quality-roles")
        or lowered.get("x-model-quality-role")
        or ""
    )
    authorization = lowered.get("authorization", "")
    token = required_token or os.getenv("MODEL_QUALITY_WEB_TOKEN") or ""

    if token and not hmac.compare_digest(
        authorization.removeprefix("Bearer ").strip(), token
    ):
        raise AuthenticationError("a valid bearer token is required")

    if not actor:
        if allow_anonymous:
            return ANONYMOUS
        raise AuthenticationError("X-Model-Quality-Actor is required")
    return Principal(
        actor=actor,
        roles=parse_roles(roles or None),
        authenticated=bool(token),
    )


def as_dict(principal: Principal) -> dict[str, Any]:
    return {
        "actor": principal.actor,
        "roles": sorted(principal.roles),
        "permissions": sorted(principal.permissions),
    }
