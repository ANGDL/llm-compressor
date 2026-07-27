"""Reusable execution plans for sequential model calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from compressed_tensors.utils import match_named_modules
from loguru import logger
from torch import nn

from llmcompressor.utils.pytorch.module import infer_sequential_targets

from .helpers import Subgraph, trace_subgraphs

__all__ = ["SequentialExecutionPlan", "trace_sequential_plan"]


@dataclass(frozen=True)
class SequentialExecutionPlan:
    """The ordered subgraphs and targets used by sequential calibration."""

    subgraphs: tuple[Subgraph, ...]
    target_names: tuple[str, ...]
    target_subgraph_indices: tuple[int, ...]

    def target_name(self, subgraph_index: int) -> str | None:
        try:
            target_index = self.target_subgraph_indices.index(subgraph_index)
        except ValueError:
            return None
        return self.target_names[target_index]


def trace_sequential_plan(
    model: nn.Module,
    sample_input: dict[str, Any],
    sequential_targets: str | Sequence[str] | None = None,
    tracing_ignore: Sequence[str] = (),
    targets_per_subgraph: int = 1,
) -> SequentialExecutionPlan:
    """Trace the same execution graph used by the oneshot sequential pipeline."""

    patterns = infer_sequential_targets(model, sequential_targets)
    if isinstance(patterns, str):
        patterns = [patterns]
    subgraphs = trace_subgraphs(
        model,
        sample_input,
        list(patterns),
        list(tracing_ignore),
        targets_per_subgraph,
    )
    matched_names = {}
    for name, module in match_named_modules(model, patterns):
        matched_names.setdefault(module, name)
    matched = set(matched_names)
    target_names = []
    target_indices = []
    covered_modules = set()
    for index, subgraph in enumerate(subgraphs):
        covered_modules.update(subgraph.submodules(model))
        targets = [
            module
            for module in subgraph.submodules(model, recurse=False)
            if module in matched
        ]
        if not targets:
            continue
        for target in targets:
            target_names.append(matched_names[target])
            target_indices.append(index)

    uncovered_target_names = [
        name for module, name in matched_names.items() if module not in covered_modules
    ]
    if uncovered_target_names:
        preview_size = 10
        preview = ", ".join(uncovered_target_names[:preview_size])
        if len(uncovered_target_names) > preview_size:
            preview += f", ... (+{len(uncovered_target_names) - preview_size} more)"
        logger.warning(
            "Sequential tracing left "
            f"{len(uncovered_target_names)}/{len(matched_names)} matched target "
            "modules "
            "outside all subgraphs: "
            f"{preview}. Calls hidden inside autowrapped or data-dependent control "
            "flow are not represented as call_module nodes. Uncovered modules will "
            "not receive sequential boundary updates. "
            "Use a trace-visible parent target or pipeline='independent'."
        )

    if not target_names:
        raise ValueError("Sequential tracing did not produce any target subgraphs")
    return SequentialExecutionPlan(
        subgraphs=tuple(subgraphs),
        target_names=tuple(target_names),
        target_subgraph_indices=tuple(target_indices),
    )
