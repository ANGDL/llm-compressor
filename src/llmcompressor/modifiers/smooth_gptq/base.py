from contextlib import contextmanager
import re
from typing import Any, Callable, cast

import torch
from compressed_tensors.utils import getattr_chain, match_named_modules

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.quantization.calibration import update_weight_global_scale
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales


class SmoothGPTQModifier(GPTQModifier, AutoSmoothModifier):
    """
    Combined AutoSmooth + GPTQ modifier.

    Lifecycle ordering is intentional:
    1) Run AutoSmooth setup and smoothing first.
    2) Run GPTQ calibration/compression on the smoothed model.

    This class avoids duplicated calibration init/finalize logic from parent
    classes to prevent hook conflicts and repeated observer setup.
    """

    # Keep defaults aligned with expected SmoothGPTQ behavior.
    norm_func: str = "adaptive"
    activation_scale_type: str = "mean"

    def on_initialize(self, state: State, **kwargs) -> bool:
        # AutoSmooth initializes quantization config (if provided), validates
        # smoothing constraints, and resolves smoothing mappings.
        with _backup_ignore(self):
            AutoSmoothModifier.on_initialize(self, state, **kwargs)

        self._resolved_config = None
        GPTQModifier.on_initialize(self, state, **kwargs)
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        # AutoSmooth performs a single calibration setup pass and installs
        # activation caches required for smoothing.
        # Keep quantization disabled between sequential passes so later
        # calibration activations remain fp16 and do not pick up partially
        # quantized intermediate states.
        AutoSmoothModifier.on_start(self, state, event, **kwargs)

        # Register GPTQ hessian hooks without re-running start_calibration.
        # Repeating start_calibration would duplicate observers/hooks.
        added_hook = False
        named_modules = list(
            match_named_modules(state.model, self.resolved_targets, self.ignore)
        )

        for _, module in named_modules:
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                # Keep GPTQ behavior: skip embeddings for now.
                if not isinstance(module, torch.nn.Embedding):
                    self.register_hook(
                        module,
                        cast(Callable[[Any], Any], self.calibrate_module),
                        "forward",
                    )
                    added_hook = True

        # Keep global-scale updates aligned with GPTQModifier behavior.
        for _, module in named_modules:
            update_weight_global_scale(module)

        for module in state.model.modules():
            update_fused_layer_weight_global_scales(module)

        if not added_hook:
            raise ValueError(
                "SmoothGPTQModifier requires a weight quantization config be "
                "specified by this modifier or a preceding modifier"
            )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, event)

        if event.type_ in (EventType.SEQUENTIAL_EPOCH_END, EventType.CALIBRATION_EPOCH_END):
            # Always smooth before GPTQ compression so the hessian-calibrated
            # quantization sees the final smoothed weights.
            self._apply_smoothing(state.model)
            self.compress_modules()

            if event.type_ == EventType.CALIBRATION_EPOCH_END and not self.ended_:
                self.on_end(state, event)

    def on_end(self, state: State, event: Event, **kwargs):
        # Ensure all AutoSmooth caches are consumed and then finalize
        # quantization calibration exactly once.
        self._assert_all_activations_consumed()
        self.ended_ = True
        self.end_calibration(state.model)
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.ended_:
            self.on_end(state, Event(type_=EventType.FINALIZE))

        # AutoSmooth cleanup
        self._log_error_metrics()
        self._parent_args_cache.clear()
        self._smooth_activation_scales.clear()
        self._resolved_mappings.clear()
        self._error_metrics.clear()
        self._autosmooth_target_modules.clear()
        self._autosmooth_weight_qargs.clear()

        # GPTQ cleanup and validation
        if len(self._num_samples) > 0:
            raise ValueError(f"Failed to compress {len(self._num_samples)} modules")

        self._hessians = dict()
        self._num_samples = dict()
        self._module_names = dict()

        return True

@contextmanager
def _backup_ignore(modifier: SmoothGPTQModifier):
    """
    Context manager to temporarily modify the `ignore` attribute of a `SmoothGPTQModifier` instance.

    This function ensures that patterns covered by the `mappings` attribute are excluded from the `ignore` list.
    It modifies the `ignore` list to exclude patterns that are already handled by `mappings`, ensuring no redundant
    exclusions. The original `ignore` list is restored after the context manager exits.

    Args:
        modifier (SmoothGPTQModifier): The modifier instance whose `ignore` attribute will be temporarily modified.

    Yields:
        None: The context manager does not return any value but ensures the `ignore` list is modified within the context.

    Example:
        >>> with _backup_ignore(modifier):
        ...     # Perform operations with the modified `ignore` list
        ...     pass
    """
    original_ignore = list(modifier.ignore)
    mappings = modifier.mappings or []

    mapping_patterns: list[str] = []
    for mapping_item in mappings:
        smooth_layer = getattr(mapping_item, "smooth_layer", None)
        balance_layers = getattr(mapping_item, "balance_layers", None)

        if isinstance(mapping_item, dict):
            smooth_layer = mapping_item.get("smooth_layer", smooth_layer)
            balance_layers = mapping_item.get("balance_layers", balance_layers)

        if isinstance(smooth_layer, str) and smooth_layer:
            mapping_patterns.append(smooth_layer)

        if isinstance(balance_layers, (list, tuple)):
            for balance_layer in balance_layers:
                if isinstance(balance_layer, str) and balance_layer:
                    mapping_patterns.append(balance_layer)

    mapping_patterns = list(dict.fromkeys(mapping_patterns))
    regex_mapping_patterns = [
        pattern[3:] for pattern in mapping_patterns if pattern.startswith("re:")
    ]
    literal_mapping_patterns = {
        pattern for pattern in mapping_patterns if not pattern.startswith("re:")
    }

    union_patterns = [f"(?:{pattern})" for pattern in regex_mapping_patterns]
    union_patterns.extend(
        f"(?:{re.escape(pattern)}$)" for pattern in literal_mapping_patterns
    )
    mapping_union = "|".join(union_patterns)

    def _is_covered_by_mapping(name: str) -> bool:
        if name in literal_mapping_patterns:
            return True
        return any(re.match(pattern, name) for pattern in regex_mapping_patterns)

    new_ignore = []
    for ignore_pattern in original_ignore:
        if ignore_pattern.startswith("re:") and mapping_union:
            raw_pattern = ignore_pattern[3:]
            new_ignore.append(f"re:(?!(?:{mapping_union}))(?:{raw_pattern})")
            continue

        if not ignore_pattern.startswith("re:") and _is_covered_by_mapping(ignore_pattern):
            continue

        new_ignore.append(ignore_pattern)

    try:
        modifier.ignore = new_ignore
        yield
    finally:
        modifier.ignore = original_ignore
