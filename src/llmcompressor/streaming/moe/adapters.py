"""Model adapters for streaming MoE execution.

The executor owns grouping and residency. Adapters own model-facing details such
as locating the router, flattening hidden states, and decoding standard router
outputs. Model-specific implementations can register alongside this generic
adapter without changing the executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from .expert_groups import ExpertGroupPlan, RoutedExpertBatch

__all__ = [
    "MoEModelAdapter",
    "StandardTransformersMoEAdapter",
    "discover_standard_moe_adapters",
]


class MoEModelAdapter(Protocol):
    """Stable model boundary used by :class:`ExpertGroupExecutor`."""

    adapter_id: str

    def validate(self, model: nn.Module) -> None: ...

    def expert_module_name(self) -> str: ...

    def router_module_names(self) -> tuple[str, ...]: ...

    def build_plan(self, model: nn.Module, group_size: int) -> ExpertGroupPlan: ...

    def route(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        *,
        batch_index: int = 0,
    ) -> RoutedExpertBatch: ...

    def restore_output(
        self, routed: RoutedExpertBatch, output: torch.Tensor
    ) -> torch.Tensor: ...


def _join_name(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _expert_count(experts: nn.Module) -> int:
    count = getattr(experts, "num_experts", None)
    if count is not None:
        return int(count)
    try:
        return len(experts)
    except TypeError as error:
        raise TypeError(
            f"{type(experts).__name__} does not expose a standard expert collection"
        ) from error


@dataclass(frozen=True)
class StandardTransformersMoEAdapter:
    """Adapter for the common Transformers router + linearized experts layout.

    The expected module shape is equivalent to:

    .. code-block:: text

        block.experts     # ModuleList or LinearExperts2D
        block.gate/router # returns (logits, top_k_weights, top_k_indices)

    The experts must already be linearized into individually addressable
    ``experts.<index>`` modules. Fused 3D expert tensors and model-specific
    router signatures belong to separate adapters.
    """

    moe_name: str
    experts_attr: str = "experts"
    router_attr: str | None = None
    adapter_id: str = "transformers-standard-v1"

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        moe_name: str,
        *,
        experts_attr: str = "experts",
        router_attr: str | None = None,
    ) -> "StandardTransformersMoEAdapter":
        adapter = cls(moe_name, experts_attr, router_attr)
        adapter.validate(model)
        return adapter

    def _moe(self, model: nn.Module) -> nn.Module:
        try:
            return model.get_submodule(self.moe_name)
        except AttributeError as error:
            raise ValueError(
                f"Unknown standard Transformers MoE module {self.moe_name!r}"
            ) from error

    def _router_attr(self, model: nn.Module) -> str:
        moe = self._moe(model)
        if self.router_attr is not None:
            if not hasattr(moe, self.router_attr):
                raise ValueError(
                    f"MoE {self.moe_name!r} has no router attribute "
                    f"{self.router_attr!r}"
                )
            return self.router_attr
        candidates = ("gate", "router")
        matches = [name for name in candidates if hasattr(moe, name)]
        if len(matches) != 1:
            raise ValueError(
                f"MoE {self.moe_name!r} must expose exactly one standard "
                f"router attribute from {candidates}; found {matches}"
            )
        return matches[0]

    def expert_module_name(self) -> str:
        return _join_name(self.moe_name, self.experts_attr)

    def router_module_names(self) -> tuple[str, ...]:
        # This method is intentionally resolved against the model in validate;
        # the common attribute names keep the plan serializable.
        if self.router_attr is None:
            raise ValueError(
                "Call from_model() or provide router_attr before requesting "
                "router module names"
            )
        return (_join_name(self.moe_name, self.router_attr),)

    def _experts(self, model: nn.Module) -> nn.Module:
        try:
            experts = model.get_submodule(self.expert_module_name())
        except AttributeError as error:
            raise ValueError(
                f"MoE {self.moe_name!r} has no experts module "
                f"{self.expert_module_name()!r}"
            ) from error
        return experts

    def validate(self, model: nn.Module) -> None:
        self._experts(model)
        router_attr = self._router_attr(model)
        router = getattr(self._moe(model), router_attr)
        if not callable(router):
            raise TypeError(
                f"MoE router {self.moe_name}.{router_attr} must be callable"
            )
        if self.router_attr is None:
            object.__setattr__(self, "router_attr", router_attr)

    def build_plan(self, model: nn.Module, group_size: int) -> ExpertGroupPlan:
        self.validate(model)
        return ExpertGroupPlan.from_module(
            self.expert_module_name(),
            self._experts(model),
            group_size,
        )

    def route(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        *,
        batch_index: int = 0,
    ) -> RoutedExpertBatch:
        self.validate(model)
        if hidden_states.ndim < 2:
            raise ValueError("standard MoE hidden_states must have at least 2 dims")
        original_shape = tuple(hidden_states.shape)
        flattened = hidden_states.reshape(-1, original_shape[-1])
        router = getattr(self._moe(model), self.router_attr)
        routed = router(flattened)
        if not isinstance(routed, (tuple, list)) or len(routed) < 3:
            raise ValueError(
                "standard Transformers MoE routers must return "
                "(logits, top_k_weights, top_k_indices)"
            )
        top_k_weights = routed[1]
        top_k_index = routed[2]
        if not isinstance(top_k_weights, torch.Tensor) or not isinstance(
            top_k_index, torch.Tensor
        ):
            raise TypeError("standard router outputs must be tensors")
        return RoutedExpertBatch(
            batch_index=batch_index,
            hidden_states=flattened,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            original_shape=original_shape,
        )

    def restore_output(
        self, routed: RoutedExpertBatch, output: torch.Tensor
    ) -> torch.Tensor:
        if routed.original_shape is None:
            return output
        expected_tokens = 1
        for size in routed.original_shape[:-1]:
            expected_tokens *= size
        if output.shape[0] != expected_tokens:
            raise ValueError(
                "expert output token count does not match routed input: "
                f"{output.shape[0]} != {expected_tokens}"
            )
        return output.reshape(routed.original_shape)


def discover_standard_moe_adapters(
    model: nn.Module,
) -> tuple[StandardTransformersMoEAdapter, ...]:
    """Discover standard router + linearized expert blocks in module order."""
    adapters = []
    for name, module in model.named_modules():
        if not name:
            continue
        if not hasattr(module, "experts"):
            continue
        if not any(hasattr(module, candidate) for candidate in ("gate", "router")):
            continue
        adapter = StandardTransformersMoEAdapter.from_model(model, name)
        if _expert_count(model.get_submodule(adapter.expert_module_name())) <= 0:
            continue
        adapters.append(adapter)
    return tuple(adapters)
