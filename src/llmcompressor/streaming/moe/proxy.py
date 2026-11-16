"""External proxies for local expert execution.

The proxy keeps EP policy outside model classes. Standard expert containers can
be replaced temporarily without changing their checkpoint names or requiring a
model-specific ``configure_streaming_ep`` method.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn

from .expert_groups import ExpertGroup, execute_expert_group

__all__ = ["LocalExpertProxy", "install_local_expert_proxy"]


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


class LocalExpertProxy(nn.Module):
    """Temporarily expose only one EP rank's experts through a standard call."""

    def __init__(
        self,
        experts: nn.Module,
        local_expert_ids: Sequence[int],
        *,
        process_group: dist.ProcessGroup | None = None,
        calibrate_all_experts: bool = False,
    ) -> None:
        super().__init__()
        num_experts = _expert_count(experts)
        local_ids = tuple(sorted(local_expert_ids))
        if len(set(local_ids)) != len(local_ids):
            raise ValueError("local_expert_ids must be unique")
        if any(expert_id < 0 or expert_id >= num_experts for expert_id in local_ids):
            raise ValueError(
                f"local_expert_ids must be in [0, {num_experts})"
            )
        if process_group is not None and not dist.is_initialized():
            raise RuntimeError(
                "process_group requires an initialized torch.distributed group"
            )
        if process_group is None and len(local_ids) != num_experts:
            raise ValueError(
                "A single-process local expert proxy must own every expert"
            )

        self.num_experts = num_experts
        self.local_expert_ids = local_ids
        self.process_group = process_group
        self.calibrate_all_experts = calibrate_all_experts
        self._source_ref = weakref.ref(experts)
        self._source_modules = dict(experts._modules)
        if len(self._source_modules) < num_experts:
            raise ValueError(
                "The expert container does not expose all experts as child modules"
            )
        for name, module in self._source_modules.items():
            self.add_module(name, module)

    def __getitem__(self, expert_id: int) -> nn.Module:
        names = tuple(self._source_modules)
        return self._modules[names[expert_id]]

    def __len__(self) -> int:
        return self.num_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = execute_expert_group(
            self,
            hidden_states,
            top_k_index,
            top_k_weights,
            ExpertGroup("experts", self.local_expert_ids),
            output=torch.zeros_like(hidden_states),
            calibrate_all_experts=self.calibrate_all_experts,
        )
        if self.process_group is not None:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.process_group)
        return output

    def restore_source(self) -> nn.Module:
        source = self._source_ref()
        if source is None:
            raise RuntimeError("The proxied expert container no longer exists")
        for name in tuple(self._modules):
            self._modules.pop(name)
        source._modules.clear()
        source._modules.update(self._source_modules)
        return source


@contextmanager
def install_local_expert_proxy(
    model: nn.Module,
    module_name: str,
    local_expert_ids: Sequence[int],
    *,
    process_group: dist.ProcessGroup | None = None,
    calibrate_all_experts: bool = False,
) -> Iterator[LocalExpertProxy]:
    """Install and restore a local expert proxy without changing model code."""

    original = model.get_submodule(module_name)
    proxy = LocalExpertProxy(
        original,
        local_expert_ids,
        process_group=process_group,
        calibrate_all_experts=calibrate_all_experts,
    )
    original._modules.clear()
    model.set_submodule(module_name, proxy)
    try:
        yield proxy
    finally:
        restored = proxy.restore_source()
        model.set_submodule(module_name, restored)
