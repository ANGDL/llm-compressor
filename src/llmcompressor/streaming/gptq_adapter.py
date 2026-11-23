"""GPTQ resource and coalesced-call compatibility owned by streaming."""

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from time import perf_counter

import torch
from compressed_tensors.offload.dist_utils import is_source_process as is_src
from loguru import logger
from torch.utils.hooks import RemovableHandle

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.gptq.helpers import (
    GPTQ_PRECISION,
    available_batch_bytes,
    batch_key,
    predict_batch_size,
    quantization_batch_peak_bytes,
    quantization_peak_bytes,
)
from llmcompressor.modifiers.quantization.calibration import (
    freeze_module_quantization,
    observe,
    update_qparams,
)
from llmcompressor.observers import ACTIVATION_OBS

from .calibration import calibration_sample_count
from .modifier_contract import ModifierContractError
from .numerics import native_gptq_factorization

_WINDOW_INITIALIZATION_LOCK = RLock()


@dataclass
class GPTQStreamingMemoryAdapter:
    modifier: GPTQModifier
    owner_update_scope = "owner_local"
    owner_update_state = "independent"
    _owner_update_baseline: int | None = field(default=None, init=False, repr=False)
    _owner_update_count: int = field(default=0, init=False, repr=False)
    _window_handles: dict[int, set[RemovableHandle]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def supports_independent_expert_windows(self) -> bool:
        """Coalesce only schemes whose activation semantics preserve token rows."""
        return all(
            args is None or (args.dynamic is True and args.strategy == "token")
            for scheme in self.modifier.resolved_config.config_groups.values()
            for args in (scheme.input_activations, scheme.output_activations)
        )

    def validate_owner_groups(self, groups) -> None:
        """Reject observer or qparam state shared across owner groups."""

        state_owners: dict[int, torch.device] = {}
        state_names = (
            "weight_observer",
            "input_observer",
            "output_observer",
            "q_observer",
            "k_observer",
            "v_observer",
            "weight_scale",
            "weight_zero_point",
            "weight_global_scale",
        )
        for group in groups:
            for module in group.modules:
                for name in state_names:
                    value = getattr(module, name, None)
                    if value is None:
                        continue
                    identity = id(value)
                    previous = state_owners.setdefault(identity, group.owner)
                    if previous != group.owner:
                        raise ModifierContractError(
                            f"GPTQ state {name!r} is shared between owner "
                            f"{previous} and {group.owner}"
                        )

    def estimate_update_memory(self, modules: Sequence[torch.nn.Module]) -> int:
        """Bound compression scratch before admission.

        ``"auto"`` is sized from live device memory at update time, so the
        admission estimate only requires one module to fit. Explicit integer
        caps retain their configured bound.
        """
        batches = {}
        for module in modules:
            if not self.modifier._is_calibrated(module):
                continue
            key = batch_key(module)
            batches.setdefault(key if key is not None else id(module), []).append(
                module
            )
        scratch = 0
        for members in batches.values():
            batch_size = (
                1
                if self.modifier.batched_quantization == "auto"
                else min(
                    len(members),
                    self.modifier.batched_quantization or 1,
                )
            )
            scratch = max(
                scratch,
                quantization_batch_peak_bytes(
                    members[0],
                    batch_size,
                    self.modifier.block_size,
                ),
            )
        return scratch

    def estimate_update_work(self, modules: Sequence[torch.nn.Module]) -> float:
        """Estimate owner placement work for GPTQ compression."""
        return float(
            sum(
                quantization_peak_bytes(module, self.modifier.block_size)
                for module in modules
                if self.modifier._is_calibrated(module)
            )
        )

    def estimate_batch_memory(
        self, module: torch.nn.Module, dispatch_tokens: int, phase: str
    ) -> int:
        """Bound FP32 input conversion and the temporary Hessian product."""
        if phase != "calibration" or not self.modifier._is_calibrated(module):
            return 0
        columns = module.weight.shape[1]
        return (2 * dispatch_tokens * columns + columns * columns) * (
            GPTQ_PRECISION.itemsize
        )

    def update_owner(
        self,
        owner: torch.device,
        modules: Sequence[torch.nn.Module],
        *,
        available_bytes: int | None = None,
    ) -> None:
        """Observe and compress only the Hessians owned by one EP device."""

        modules = [module for module in modules if self.modifier._is_calibrated(module)]
        if not modules:
            return
        started = perf_counter()
        observe(modules, base_name="weight")
        observed = perf_counter()
        self.modifier.sync_obs_act_stats(modules)
        synced = perf_counter()
        update_qparams(
            modules,
            ACTIVATION_OBS,
            only_update_onload=not is_src(),
        )
        qparams = perf_counter()
        batch_size_cap = None
        if self.modifier.batched_quantization == "auto":
            device_batch_bytes = (
                available_batch_bytes(owner)
                if available_bytes is None
                else available_bytes
            )
            batch_size_cap = predict_batch_size(
                list(modules),
                self.modifier.batched_quantization,
                self.modifier.block_size,
                device_batch_bytes,
            )
            available_bytes = device_batch_bytes
        predicted = perf_counter()
        self.modifier.compress_module_list(
            modules,
            batch_size_cap=batch_size_cap,
        )
        compressed = perf_counter()
        max_weight_bytes = max(
            module.weight.numel() * module.weight.element_size() for module in modules
        )
        max_module_peak_bytes = max(
            quantization_peak_bytes(module, self.modifier.block_size)
            for module in modules
        )
        batch_groups = len(
            {key for module in modules if (key := batch_key(module)) is not None}
        )
        available_text = (
            f"{available_bytes / 2**30:.2f}" if available_bytes is not None else "n/a"
        )
        logger.info(
            f"GPTQ | update timing | device={owner} | modules={len(modules)} | "
            f"groups={batch_groups} | "
            f"batch_cap={batch_size_cap or self.modifier.batched_quantization} | "
            f"batch_budget_gib={available_text} | "
            f"weight_max_gib={max_weight_bytes / 2**30:.3f} | "
            f"module_peak_gib={max_module_peak_bytes / 2**30:.3f} | "
            f"predict_s={predicted - qparams:.2f} | "
            f"observe_s={observed - started:.2f} | sync_s={synced - observed:.2f} | "
            f"qparams_s={qparams - synced:.2f} | "
            f"compress_s={compressed - predicted:.2f} | "
            f"total_s={compressed - started:.2f}"
        )

    def after_update(self, modules: Sequence[torch.nn.Module]) -> None:
        """Stop this owner's calibration hooks before propagation starts.

        GPTQ consumes and deletes weight-observer statistics during compression.
        The iMatrix observer hook must therefore be detached before the same
        expert is forwarded again for activation propagation. Hooks belonging to
        other owner windows remain active.
        """
        with _WINDOW_INITIALIZATION_LOCK:
            handles = self._window_handles.pop(id(modules), set())
            self.modifier.remove_hooks(handles)
            self.modifier._calibration_hooks.difference_update(handles)
        for module in modules:
            for base_name in ("input", "weight", "output", "q", "k", "v"):
                observer = getattr(module, f"{base_name}_observer", None)
                if observer is not None:
                    observer.detach(module)

    def begin_owner_update(self, modules: Sequence[torch.nn.Module]) -> None:
        """Snapshot shared bookkeeping before concurrent compression starts."""

        self._owner_update_baseline = self.modifier._num_compressed_modules
        self._owner_update_count = sum(
            self.modifier._is_calibrated(module) for module in modules
        )

    def finalize_owner_update(self) -> None:
        """Restore deterministic shared compression bookkeeping after the join."""

        if self._owner_update_baseline is None:
            return
        self.modifier._num_compressed_modules = (
            self._owner_update_baseline + self._owner_update_count
        )
        self._owner_update_baseline = None
        self._owner_update_count = 0

    @contextmanager
    def expert_window(self, model, modules):
        """Initialize and release only this window's observer hooks."""
        modifier = self.modifier
        handles = set()
        try:
            with _WINDOW_INITIALIZATION_LOCK:
                previous = set(modifier._calibration_hooks)
                try:
                    modifier.start_layerwise_calibration(model, list(modules))
                finally:
                    handles = modifier._calibration_hooks - previous
                self._window_handles[id(modules)] = handles
            yield
        finally:
            with _WINDOW_INITIALIZATION_LOCK:
                self._window_handles.pop(id(modules), None)
                modifier.remove_hooks(handles)
                modifier._calibration_hooks.difference_update(handles)
            for module in modules:
                freeze_module_quantization(module)
                modifier._hessians.pop(module, None)
                modifier._num_samples.pop(module, None)

    @contextmanager
    def calibration_context(self):
        """Preserve original call counts while retaining GPTQ's original hook."""
        modifier = self.modifier
        original = modifier.calibrate_module
        had_override = "calibrate_module" in vars(modifier)
        previous_override = vars(modifier).get("calibrate_module")

        @wraps(original)
        def calibrate(module, args, output):
            result = original(module, args, output)
            expected = calibration_sample_count(args[0].ndim)
            if expected is not None:
                observed = 1 if args[0].ndim == 2 else args[0].shape[0]
                modifier._num_samples[module].add_(expected - observed)
            return result

        object.__setattr__(modifier, "calibrate_module", calibrate)
        try:
            with native_gptq_factorization():
                yield
        finally:
            if had_override:
                object.__setattr__(modifier, "calibrate_module", previous_override)
            else:
                object.__delattr__(modifier, "calibrate_module")
