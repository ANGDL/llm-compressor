"""Resolve only the placeholders owned by model quality CI."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")


class PlaceholderError(ValueError):
    """Raised when a CI-owned placeholder has no value."""


def resolve_argument(
    argument: str, values: Mapping[str, str], *, allowed: Iterable[str] | None = None
) -> str:
    """Replace known CI placeholders while preserving arbitrary braces verbatim."""

    allowed_names = set(values if allowed is None else allowed)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in allowed_names:
            return match.group(0)
        try:
            return values[name]
        except KeyError as error:
            raise PlaceholderError(
                f"missing value for placeholder {{{name}}}"
            ) from error

    return _PLACEHOLDER.sub(replace, argument)


def resolve_argv(
    argv: list[str], values: Mapping[str, str], *, allowed: Iterable[str] | None = None
) -> list[str]:
    """Resolve a complete argv list without invoking shell or str.format."""

    return [resolve_argument(value, values, allowed=allowed) for value in argv]
