"""Owner-local calibration windows and RTN updates for streaming."""

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter

import torch
from compressed_tensors.quantization.utils import is_module_quantized
from loguru import logger
from torch.utils.hooks import RemovableHandle

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.calibration import (
    freeze_module_quantization,
    observe,
    update_qparams,
)
from llmcompressor.modifiers.quantization.quantization.mixin import QuantizationMixin
from llmcompressor.observers import ACTIVATION_OBS

from .modifier_contract import ModifierContractError

_WINDOW_INITIALIZATION_LOCK = RLock()
_OBSERVER_NAMES = ("input", "weight", "output", "q", "k", "v")


@dataclass
class QuantizationWindowAdapter:
    """Manage observer ownership independently of the quantization algorithm."""

    modifier: QuantizationMixin
    owner_update_scope = "owner_local"
    owner_update_state = "independent"
    _window_handles: dict[int, set[RemovableHandle]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def supports_independent_expert_windows(self) -> bool:
        """Allow token coalescing only for row-independent activation schemes."""
        return all(
            args is None or (args.dynamic is True and args.strategy == "token")
            for scheme in self.modifier.resolved_config.config_groups.values()
            for args in (scheme.input_activations, scheme.output_activations)
        )

    def validate_owner_groups(self, groups) -> None:
        """Reject observer and qparam objects shared by different owners."""
        state_owners: dict[int, torch.device] = {}
        for group in groups:
            for module in group.modules:
                for base_name in _OBSERVER_NAMES:
                    for suffix in ("observer", "scale", "zero_point", "global_scale"):
                        name = f"{base_name}_{suffix}"
                        value = getattr(module, name, None)
                        if value is None:
                            continue
                        previous = state_owners.setdefault(id(value), group.owner)
                        if previous != group.owner:
                            raise ModifierContractError(
                                f"Quantization state {name!r} is shared between "
                                f"owners {previous} and {group.owner}"
                            )

    def _remove_window_hooks(self, handles: set[RemovableHandle]) -> None:
        self.modifier.remove_hooks(handles)
        with _WINDOW_INITIALIZATION_LOCK:
            self.modifier._calibration_hooks.difference_update(handles)

    @contextmanager
    def expert_window(self, model, modules):
        """Initialize and freeze observers for only this resident window."""
        handles: set[RemovableHandle] = set()
        try:
            self.modifier.start_layerwise_calibration(
                model, list(modules), hook_registry=handles
            )
            with _WINDOW_INITIALIZATION_LOCK:
                self.modifier._calibration_hooks.update(handles)
            self._window_handles[id(modules)] = handles
            yield
        finally:
            self._window_handles.pop(id(modules), None)
            self._remove_window_hooks(handles)
            for module in modules:
                freeze_module_quantization(module)

    def after_update(self, modules: Sequence[torch.nn.Module]) -> None:
        """Detach this owner's observers before activation propagation."""
        self._remove_window_hooks(self._window_handles.pop(id(modules), set()))
        for module in modules:
            for base_name in _OBSERVER_NAMES:
                observer = getattr(module, f"{base_name}_observer", None)
                if observer is not None:
                    observer.detach(module)


@dataclass
class QuantizationStreamingMemoryAdapter(QuantizationWindowAdapter):
    """Run RTN observer and qparam updates independently on each EP owner."""

    modifier: QuantizationModifier

    def estimate_update_memory(self, modules: Sequence[torch.nn.Module]) -> int:
        """Keep the modifier's bound for serial observation within each owner."""
        return max(
            (
                self.modifier.estimate_sequential_memory(module, 0)[1]
                for module in modules
            ),
            default=0,
        )

    def estimate_update_work(self, modules: Sequence[torch.nn.Module]) -> float:
        """Balance owner updates by the number of weight elements observed."""
        return float(
            sum(
                module.weight.numel()
                for module in modules
                if is_module_quantized(module)
                and module.quantization_scheme.weights is not None
            )
        )

    def estimate_batch_memory(
        self, module: torch.nn.Module, dispatch_tokens: int, phase: str
    ) -> int:
        """Bound the FP32 conversion and square used by iMatrix calibration."""
        scheme = getattr(module, "quantization_scheme", None)
        if (
            phase != "calibration"
            or scheme is None
            or scheme.weights is None
            or scheme.weights.observer != "imatrix_mse"
        ):
            return 0
        return 2 * dispatch_tokens * module.weight.shape[-1] * torch.float32.itemsize

    def update_owner(
        self,
        owner: torch.device,
        modules: Sequence[torch.nn.Module],
        *,
        available_bytes: int | None = None,
    ) -> None:
        """Preserve RTN's single-participant sequential update order."""
        modules = [module for module in modules if is_module_quantized(module)]
        if not modules:
            return
        started = perf_counter()
        self.modifier.sync_obs_act_stats(modules)
        synced = perf_counter()
        update_qparams(modules, ACTIVATION_OBS)
        activation_qparams = perf_counter()
        observe(modules, "weight")
        observed = perf_counter()
        update_qparams(modules, "weight")
        finished = perf_counter()
        logger.info(
            f"RTN | update timing | device={owner} | modules={len(modules)} | "
            f"sync_s={synced - started:.2f} | "
            f"act_qparams_s={activation_qparams - synced:.2f} | "
            f"observe_s={observed - activation_qparams:.2f} | "
            f"weight_qparams_s={finished - observed:.2f} | "
            f"total_s={finished - started:.2f}"
        )
