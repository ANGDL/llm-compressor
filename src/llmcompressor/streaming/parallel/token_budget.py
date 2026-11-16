"""Deterministic token capacity planning for expert owner devices."""

from __future__ import annotations

from collections.abc import Callable

from ..admission import DeviceMemoryError

TOKEN_BUDGET_POLICY = "ep-tokens-v2"


def validate_token_budget(value: str | int) -> None:
    if value == "auto":
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "expert_parallel_token_budget must be 'auto' or a positive integer"
        )


def select_token_capacity(
    *,
    owner: str,
    usable_bytes: int | None,
    required_bytes: Callable[[int], int],
    maximum_tokens: int,
    minimum_tokens: int = 1,
) -> int:
    """Find the largest admitted capacity for a monotone memory estimate."""
    if minimum_tokens < 1 or maximum_tokens < minimum_tokens:
        raise DeviceMemoryError(
            f"{owner} cannot fit the minimum token window: "
            f"minimum={minimum_tokens}, maximum={maximum_tokens}"
        )
    if usable_bytes is None:
        return minimum_tokens
    if required_bytes(minimum_tokens) > usable_bytes:
        raise DeviceMemoryError(
            f"Insufficient memory for {owner}: requested "
            f"{required_bytes(minimum_tokens)} bytes for {minimum_tokens} tokens, "
            f"available {usable_bytes} bytes after safety reserve"
        )
    lower = minimum_tokens
    upper = maximum_tokens
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        if required_bytes(candidate) <= usable_bytes:
            lower = candidate
        else:
            upper = candidate - 1
    return lower
