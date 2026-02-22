import torch
from compressed_tensors.utils import getattr_chain, match_named_modules
from typing import Any, Callable, cast

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
        AutoSmoothModifier.on_initialize(self, state, **kwargs)

        # GPTQ-specific initialization: collect target module names for logs
        # and later compression bookkeeping. Avoid calling GPTQ.on_initialize
        # directly to prevent duplicated quantization initialization.
        self._module_names = {
            m: name
            for name, m in match_named_modules(
                state.model, self.resolved_targets, self.ignore
            )
        }

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        # AutoSmooth performs a single calibration setup pass and installs
        # activation caches required for smoothing.
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

        # GPTQ cleanup and validation
        if len(self._num_samples) > 0:
            raise ValueError(f"Failed to compress {len(self._num_samples)} modules")

        self._hessians = dict()
        self._num_samples = dict()
        self._module_names = dict()

        return True
