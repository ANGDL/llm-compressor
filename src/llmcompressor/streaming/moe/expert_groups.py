"""Bounded-residency execution for linearized MoE experts.

The executor in this module owns expert grouping and weight residency only. It
does not implement a quantization algorithm. A caller can attach existing
modifier collectors to the loaded expert modules and run the same group
multiple times for calibration before evicting it.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.utils.helpers import DisableQuantization

from ..activations import BoundaryActivationStore
from ..artifacts import fingerprint_json
from ..loading import LoadedSubgraph, SubgraphWeightSession
from ..parallel import PlacementEntry, PlacementPlan

__all__ = [
    "ExpertGroup",
    "ExpertGroupExecutor",
    "ExpertGroupPlan",
    "RoutedExpertBatch",
    "execute_expert_group",
]


@dataclass(frozen=True)
class ExpertGroup:
    """One ordered set of experts sharing a residency window."""

    experts_name: str
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.experts_name:
            raise ValueError("experts_name must not be empty")
        if not self.indices:
            raise ValueError("an expert group must contain at least one expert")
        if tuple(sorted(set(self.indices))) != self.indices:
            raise ValueError("expert indices must be unique and sorted")
        if self.indices[0] < 0:
            raise ValueError("expert indices must be non-negative")

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.experts_name}.{expert_index}" for expert_index in self.indices
        )


ExpertGroupCallback = Callable[
    [ExpertGroup, LoadedSubgraph, nn.Module, torch.Tensor, torch.Tensor, torch.Tensor],
    None,
]
ExpertGroupAfterCallback = Callable[
    [
        ExpertGroup,
        LoadedSubgraph,
        nn.Module,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    None,
]


@dataclass(frozen=True)
class ExpertGroupPlan:
    """Deterministic partition of one experts module."""

    experts_name: str
    num_experts: int
    group_size: int
    groups: tuple[ExpertGroup, ...]

    def __post_init__(self) -> None:
        if not self.experts_name:
            raise ValueError("experts_name must not be empty")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        expected = tuple(range(self.num_experts))
        actual = tuple(index for group in self.groups for index in group.indices)
        if actual != expected:
            raise ValueError(
                "expert groups must cover every expert exactly once in index order"
            )
        if any(len(group.indices) > self.group_size for group in self.groups):
            raise ValueError("an expert group exceeds group_size")
        if any(group.experts_name != self.experts_name for group in self.groups):
            raise ValueError("all groups must belong to experts_name")

    @classmethod
    def from_num_experts(
        cls,
        experts_name: str,
        num_experts: int,
        group_size: int,
    ) -> "ExpertGroupPlan":
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        groups = tuple(
            ExpertGroup(
                experts_name,
                tuple(range(start, min(start + group_size, num_experts))),
            )
            for start in range(0, num_experts, group_size)
        )
        return cls(experts_name, num_experts, group_size, groups)

    @classmethod
    def from_module(
        cls,
        experts_name: str,
        experts: nn.Module,
        group_size: int,
    ) -> "ExpertGroupPlan":
        num_experts = getattr(experts, "num_experts", None)
        if num_experts is None:
            try:
                num_experts = len(experts)
            except TypeError as error:
                raise TypeError(
                    f"Cannot infer expert count from {type(experts).__name__}"
                ) from error
        return cls.from_num_experts(experts_name, int(num_experts), group_size)

    @property
    def placement(self) -> PlacementPlan:
        return PlacementPlan(
            entries=tuple(
                PlacementEntry(
                    logical_name=name, owner_rank=0, expert_id=expert_index
                )
                for group in self.groups
                for expert_index, name in zip(group.indices, group.module_names)
            )
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(
            {
                "version": 1,
                "placement": self.placement.to_dict(),
                "groups": [list(group.indices) for group in self.groups],
                "propagation": "modified-weights-without-fake-quant",
            }
        )


@dataclass(frozen=True)
class RoutedExpertBatch:
    """Frozen router results with token rows and top-k slots kept in order."""

    batch_index: int
    hidden_states: torch.Tensor
    top_k_index: torch.Tensor
    top_k_weights: torch.Tensor
    original_shape: tuple[int, ...] | None = None


class ExpertGroupModifier(Protocol):
    """Algorithm-owned calibration lifecycle for one resident model view."""

    def calibration(self, model: nn.Module) -> AbstractContextManager[None]: ...


def _group_model(loaded: LoadedSubgraph) -> nn.Module:
    view = nn.Module()
    for module_name in loaded.module_names:
        owner = view
        components = module_name.split(".")
        for component in components[:-1]:
            if component not in owner._modules:
                owner.add_module(component, nn.Module())
            owner = owner.get_submodule(component)
        owner.add_module(components[-1], loaded.model.get_submodule(module_name))
    view.eval()
    return view


def _validate_routing(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
) -> None:
    if hidden_states.ndim != 2:
        raise ValueError(
            "hidden_states must have shape [num_tokens, hidden_size]; "
            f"got {tuple(hidden_states.shape)}"
        )
    if top_k_index.ndim != 2 or top_k_weights.shape != top_k_index.shape:
        raise ValueError(
            "top_k_index and top_k_weights must have matching shape [tokens, top_k]"
        )
    if top_k_index.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing token count must match hidden_states")
    if top_k_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("top_k_index must use an integer dtype")
    if any(
        tensor.device != hidden_states.device
        for tensor in (top_k_index, top_k_weights)
    ):
        raise ValueError("routing tensors must share the hidden_states device")
    if not top_k_weights.is_floating_point():
        raise ValueError("top_k_weights must use a floating point dtype")
    if top_k_index.numel() and (
        top_k_index.min().item() < 0 or top_k_index.max().item() >= num_experts
    ):
        raise ValueError(
            "routing contains an expert index outside the routed expert range"
        )


def _expert_count(experts: nn.Module) -> int:
    num_experts = getattr(experts, "num_experts", None)
    if num_experts is not None:
        return int(num_experts)
    try:
        return len(experts)
    except TypeError as error:
        raise TypeError(
            f"{type(experts).__name__} is not an expert container"
        ) from error


def execute_expert_group(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    group: ExpertGroup,
    *,
    output: torch.Tensor | None = None,
    calibrate_all_experts: bool = False,
) -> torch.Tensor:
    """Execute one group and add its routed results into ``output``.

    The expert loop and ``torch.where`` layout match ``LinearExperts2D.forward``.
    Groups must be executed in ascending expert order when reproducible
    accumulation order is required. ``calibrate_all_experts`` preserves the
    existing calibration behavior by running every expert on all hidden states
    while retaining only its routed output.
    """
    num_experts = _expert_count(experts)
    _validate_routing(hidden_states, top_k_index, top_k_weights, num_experts)
    if any(index >= num_experts for index in group.indices):
        raise ValueError("expert group contains an out-of-range expert")
    if getattr(experts, "novices", None) is not None:
        raise ValueError(
            "expert group execution does not support novice experts; "
            "handle novice outputs in a model-specific adapter"
        )

    result = torch.zeros_like(hidden_states) if output is None else output
    if (
        result.shape != hidden_states.shape
        or result.device != hidden_states.device
        or result.dtype != hidden_states.dtype
    ):
        raise ValueError("output must match hidden_states shape, dtype and device")

    for expert_index in group.indices:
        expert_mask = top_k_index == expert_index
        top_k_pos, token_indices = torch.where(expert_mask.transpose(0, 1))
        expert = experts[expert_index]
        if calibrate_all_experts:
            expert_output = expert(hidden_states)[token_indices]
        else:
            expert_output = expert(hidden_states[token_indices])
        expert_weights = top_k_weights[token_indices, top_k_pos, None]
        weighted_output = expert_output * expert_weights
        result.index_add_(
            0,
            token_indices,
            weighted_output.to(result.dtype),
        )
    return result


class ExpertGroupExecutor:
    """Load, execute, and evict one bounded group of experts at a time."""

    def __init__(
        self,
        model: nn.Module,
        weight_session: SubgraphWeightSession,
        plan: ExpertGroupPlan,
    ) -> None:
        self.model = model
        self.weight_session = weight_session
        self.plan = plan
        if weight_session.model is not model:
            raise ValueError("weight_session must own the executor model")
        try:
            experts = model.get_submodule(plan.experts_name)
        except AttributeError as error:
            raise ValueError(
                f"Unknown experts module {plan.experts_name!r}"
            ) from error
        actual_plan = ExpertGroupPlan.from_module(
            plan.experts_name, experts, plan.group_size
        )
        if actual_plan != plan:
            raise ValueError(
                f"Expert group plan does not match {plan.experts_name!r}"
            )

    @contextmanager
    def loaded_group(
        self,
        group: ExpertGroup,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Generator[LoadedSubgraph, None, None]:
        """Install exactly one expert group and restore it on exit."""
        if group not in self.plan.groups:
            raise ValueError("group does not belong to this executor plan")
        load_plan = self.weight_session.plan_modules(group.module_names)
        with self.weight_session.loaded_plan(
            load_plan,
            device=device,
            dtype=dtype,
        ) as loaded:
            yield loaded

    @torch.no_grad()
    def execute(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        calibrate_all_experts: bool = False,
        on_group: ExpertGroupCallback | None = None,
        after_group: ExpertGroupAfterCallback | None = None,
    ) -> torch.Tensor:
        """Execute all groups while keeping only one group resident.

        ``on_group`` runs after the group is loaded and before its routed
        forward. It can invoke existing algorithm code against the resident
        modules, for example attaching a GPTQ statistics collector or
        quantizing the group after a caller-owned calibration phase. The
        callback must not retain module or tensor references after the group
        exits. ``after_group`` runs after the routed forward, while the group
        is still resident, and can detach collectors and export statistics.
        """
        resolved_device = torch.device(device)
        if hidden_states.device != resolved_device:
            raise ValueError(
                "hidden_states must already be on the executor device; "
                f"got {hidden_states.device}, expected {resolved_device}"
            )
        result = torch.zeros_like(hidden_states)
        experts = self.model.get_submodule(self.plan.experts_name)
        for group in self.plan.groups:
            with self.loaded_group(
                group, device=resolved_device, dtype=dtype
            ) as loaded:
                if on_group is not None:
                    on_group(
                        group,
                        loaded,
                        experts,
                        hidden_states,
                        top_k_index,
                        top_k_weights,
                    )
                accumulated_output = execute_expert_group(
                    experts,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    group,
                    output=result,
                    calibrate_all_experts=calibrate_all_experts,
                )
                if after_group is not None:
                    after_group(
                        group,
                        loaded,
                        experts,
                        hidden_states,
                        top_k_index,
                        top_k_weights,
                        accumulated_output,
                    )
        return result

    @contextmanager
    def loaded_router(
        self,
        adapter,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Generator[LoadedSubgraph, None, None]:
        """Keep an adapter's router resident while expert groups are rotated."""
        adapter.validate(self.model)
        load_plan = self.weight_session.plan_modules(
            adapter.router_module_names()
        )
        with self.weight_session.loaded_plan(
            load_plan,
            device=device,
            dtype=dtype,
        ) as loaded:
            yield loaded

    @torch.no_grad()
    def execute_with_adapter(
        self,
        adapter,
        hidden_states: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        batch_index: int = 0,
        calibrate_all_experts: bool = False,
    ) -> torch.Tensor:
        """Route with a resident model adapter, then stream expert groups."""
        resolved_device = torch.device(device)
        if hidden_states.device != resolved_device:
            raise ValueError(
                "hidden_states must already be on the executor device; "
                f"got {hidden_states.device}, expected {resolved_device}"
            )
        with self.loaded_router(
            adapter, device=resolved_device, dtype=dtype
        ):
            routed = adapter.route(
                self.model,
                hidden_states,
                batch_index=batch_index,
            )
            output = self.execute(
                routed.hidden_states,
                routed.top_k_index,
                routed.top_k_weights,
                device=resolved_device,
                dtype=dtype,
                calibrate_all_experts=calibrate_all_experts,
            )
        return adapter.restore_output(routed, output)

    @torch.no_grad()
    def calibrate(
        self,
        *,
        inputs: BoundaryActivationStore,
        outputs: BoundaryActivationStore,
        input_boundary: int,
        output_boundary: int,
        modifier: ExpertGroupModifier,
        save_group: Callable[[ExpertGroup, LoadedSubgraph], None],
        device: torch.device | str,
        dtype: torch.dtype,
        calibrate_all_experts: bool,
    ) -> None:
        """Replay all batches for each group, compress, propagate, and evict.

        Input boundaries contain ``RoutedExpertBatch`` snapshots supplied by a
        model adapter. The caller owns the decoder transaction: ``save_group``
        must consume resident state synchronously, and commit only after every
        group and the rest of the decoder succeed. On failure retry the whole
        decoder. Output accumulators are recreated from the first group, so
        partial outputs from an earlier attempt are never accepted as complete.
        """
        if inputs is outputs and input_boundary == output_boundary:
            raise ValueError("routed inputs and outputs need separate boundaries")
        batch_indices = inputs.batch_indices(input_boundary)
        if not batch_indices:
            raise ValueError("routed calibration inputs must not be empty")
        if tuple(batch_indices) != tuple(range(len(batch_indices))):
            raise ValueError("routed calibration batch indices must be contiguous")
        experts = self.model.get_submodule(self.plan.experts_name)
        if getattr(experts, "novices", None) is not None:
            raise ValueError("expert group calibration does not support novices")
        outputs.delete(output_boundary)
        try:
            for group_index, group in enumerate(self.plan.groups):
                with self.loaded_group(group, device=device, dtype=dtype) as loaded:
                    view = _group_model(loaded)
                    with modifier.calibration(view), DisableQuantization(view):
                        for batch_index in batch_indices:
                            batch = self._routed_batch(
                                inputs, input_boundary, batch_index, device
                            )
                            execute_expert_group(
                                experts,
                                batch.hidden_states,
                                batch.top_k_index,
                                batch.top_k_weights,
                                group,
                                calibrate_all_experts=calibrate_all_experts,
                            )
                            del batch
                    with HooksMixin.disable_hooks(), DisableQuantization(view):
                        for batch_index in batch_indices:
                            batch = self._routed_batch(
                                inputs, input_boundary, batch_index, device
                            )
                            accumulated = (
                                outputs.get(
                                    output_boundary, batch_index, device=device
                                )
                                if group_index
                                else torch.zeros_like(batch.hidden_states)
                            )
                            execute_expert_group(
                                experts,
                                batch.hidden_states,
                                batch.top_k_index,
                                batch.top_k_weights,
                                group,
                                output=accumulated,
                                calibrate_all_experts=calibrate_all_experts,
                            )
                            outputs.put(output_boundary, batch_index, accumulated)
                            del batch, accumulated
                    save_group(group, loaded)
                    del view
        except BaseException:
            outputs.delete(output_boundary)
            raise

    def _routed_batch(
        self,
        inputs: BoundaryActivationStore,
        boundary: int,
        batch_index: int,
        device: torch.device | str,
    ) -> RoutedExpertBatch:
        batch = inputs.get(boundary, batch_index, device=device)
        if not isinstance(batch, RoutedExpertBatch):
            raise TypeError("routed inputs must contain RoutedExpertBatch values")
        if batch.batch_index != batch_index:
            raise ValueError("routed batch identity does not match boundary index")
        _validate_routing(
            batch.hidden_states,
            batch.top_k_index,
            batch.top_k_weights,
            self.plan.num_experts,
        )
        return batch
