"""Stable routed-token segments that remain visible across FX boundaries."""

from itertools import accumulate

import torch


def group_expert_routes(
    weights: torch.Tensor, indices: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Group routes once while preserving the original token and slot order."""
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    if indices.ndim != 2 or weights.shape != indices.shape:
        raise ValueError("Route weights and indices must have matching 2D shapes")
    if indices.is_meta:
        return (
            indices.new_empty((indices.numel(),)),
            weights.new_empty((weights.numel(), 1)),
            (0,) * (num_experts + 1),
        )
    flattened = indices.flatten()
    order = flattened.argsort(stable=True)
    rows = order.div(indices.shape[-1], rounding_mode="floor")
    grouped_weights = weights.flatten().index_select(0, order).unsqueeze(-1)
    counts = torch.bincount(flattened, minlength=num_experts).tolist()
    return rows, grouped_weights, tuple(accumulate(counts, initial=0))


def select_expert_routes(
    grouped: tuple[torch.Tensor, torch.Tensor, tuple[int, ...]], expert_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one expert using host offsets without a device scalar read."""
    rows, weights, offsets = grouped
    start, end = offsets[expert_id : expert_id + 2]
    return rows[start:end], weights[start:end]
