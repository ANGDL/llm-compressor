"""Collect sufficient statistics without quantizing or modifying weights."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from llmcompressor.modifiers.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_gptq_statistics,
)
from llmcompressor.observers.imatrix import (
    accumulate_imatrix_statistics,
    make_empty_imatrix_statistics,
)


def _canonical_input(args: tuple[Any, ...]) -> torch.Tensor | None:
    if not args:
        return None
    value = args[0]
    if isinstance(value, tuple):
        value = value[0] if value else None
    return value if isinstance(value, torch.Tensor) else None


class _StatisticsCollector(AbstractContextManager):
    """Own hook handles and detached statistics for named modules."""

    def __init__(
        self,
        modules: Mapping[str, nn.Module],
        *,
        storage_device: torch.device | str = "cpu",
    ):
        if not modules:
            raise ValueError("At least one module is required for collection")
        if any(not name for name in modules):
            raise ValueError("Statistic module names must be non-empty")
        if len({id(module) for module in modules.values()}) != len(modules):
            raise ValueError("A module may only have one statistic name")
        self.modules = dict(modules)
        self.storage_device = torch.device(storage_device)
        if self.storage_device.type == "meta":
            raise ValueError("Statistics cannot be stored on the meta device")
        self._handles: list[RemovableHandle] = []

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    def attach(self) -> None:
        if self.attached:
            raise RuntimeError("Statistics collector is already attached")
        try:
            for name, module in self.modules.items():
                handle = module.register_forward_pre_hook(
                    self._make_hook(name, module)
                )
                self._handles.append(handle)
        except Exception:
            self.detach()
            raise

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.detach()

    def export(self) -> dict[str, torch.Tensor]:
        statistics = {}
        for name in self.modules:
            self._require_initialized(name)
            for statistic_name, tensor in self._export_module(name).items():
                statistics[f"{name}.{statistic_name}"] = (
                    tensor.detach().to("cpu").contiguous().clone()
                )
        return statistics

    @abstractmethod
    def _require_initialized(self, name: str) -> None: ...

    @abstractmethod
    def _make_hook(self, name: str, module: nn.Module): ...

    @abstractmethod
    def _export_module(self, name: str) -> dict[str, torch.Tensor]: ...


class GPTQStatisticsCollector(_StatisticsCollector):
    """Collect the exact raw Hessian and sample count used by GPTQ."""

    def __init__(self, modules: Mapping[str, nn.Module], **kwargs):
        super().__init__(modules, **kwargs)
        self._hessians: dict[str, torch.Tensor] = {}
        self._num_samples: dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str, module: nn.Module):
        if not hasattr(module, "weight"):
            raise TypeError(f"GPTQ module {name!r} does not have a weight")

        def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            inputs = _canonical_input(args)
            if inputs is None:
                return
            if name not in self._hessians:
                self._hessians[name], self._num_samples[name] = (
                    make_empty_gptq_statistics(
                        module, device=self.storage_device
                    )
                )
            hessian, count = accumulate_hessian(
                inputs.detach(),
                module,
                self._hessians[name],
                self._num_samples[name],
            )
            self._hessians[name] = hessian
            self._num_samples[name] = count

        return hook

    def _require_initialized(self, name: str) -> None:
        if name not in self._hessians:
            raise RuntimeError(f"No GPTQ inputs were collected for module {name!r}")

    def _export_module(self, name: str) -> dict[str, torch.Tensor]:
        return {
            "gptq_hessian": self._hessians[name],
            "gptq_num_samples": self._num_samples[name],
        }


class IMatrixStatisticsCollector(_StatisticsCollector):
    """Collect per-input-channel sum(x squared) and token count."""

    def __init__(self, modules: Mapping[str, nn.Module], **kwargs):
        super().__init__(modules, **kwargs)
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str, module: nn.Module):
        in_features = getattr(module, "in_features", None)
        if not isinstance(in_features, int) or in_features <= 0:
            raise TypeError(
                f"iMatrix module {name!r} must define positive in_features"
            )

        def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            inputs = _canonical_input(args)
            if inputs is None:
                return
            if inputs.ndim == 0 or inputs.shape[-1] != module.in_features:
                raise ValueError(
                    f"Input to {name!r} has final dimension "
                    f"{inputs.shape[-1] if inputs.ndim else 0}; expected "
                    f"{module.in_features}"
                )
            if name not in self._sums:
                self._sums[name], self._counts[name] = (
                    make_empty_imatrix_statistics(
                        in_features, device=self.storage_device
                    )
                )
            self._sums[name], self._counts[name] = (
                accumulate_imatrix_statistics(
                    inputs, self._sums[name], self._counts[name]
                )
            )

        return hook

    def _require_initialized(self, name: str) -> None:
        if name not in self._sums:
            raise RuntimeError(
                f"No iMatrix inputs were collected for module {name!r}"
            )

    def _export_module(self, name: str) -> dict[str, torch.Tensor]:
        return {
            "imatrix_sum": self._sums[name],
            "imatrix_count": self._counts[name],
        }


class StatisticsCollectorGroup(AbstractContextManager):
    """Attach multiple algorithms together and export one artifact mapping."""

    def __init__(self, collectors: Sequence[_StatisticsCollector]):
        if not collectors:
            raise ValueError("At least one statistics collector is required")
        self.collectors = tuple(collectors)
        self._attached: list[_StatisticsCollector] = []

    @property
    def attached(self) -> bool:
        return bool(self._attached)

    def attach(self) -> None:
        if self.attached:
            raise RuntimeError("Statistics collector group is already attached")
        try:
            for collector in self.collectors:
                collector.attach()
                self._attached.append(collector)
        except Exception:
            self.detach()
            raise

    def detach(self) -> None:
        for collector in reversed(self._attached):
            collector.detach()
        self._attached.clear()

    def export(self) -> dict[str, torch.Tensor]:
        statistics = {}
        for collector in self.collectors:
            exported = collector.export()
            overlap = statistics.keys() & exported.keys()
            if overlap:
                raise ValueError(
                    f"Duplicate exported statistic names: {sorted(overlap)}"
                )
            statistics.update(exported)
        return statistics

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.detach()
