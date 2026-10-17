"""Native Transformers inference with checkpoint-backed module offload."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from threading import RLock
from typing import Any

import torch
from accelerate import cpu_offload
from accelerate.hooks import (
    AlignDevicesHook,
    ModelHook,
    SequentialHook,
    add_hook_to_module,
    remove_hook_from_module,
)
from accelerate.utils import send_to_device, set_module_tensor_to_device
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from llmcompressor.streaming.artifacts import fingerprint_json
from llmcompressor.streaming.checkpoint import CheckpointWeightSource
from llmcompressor.streaming.loading import HostMemoryBudget, build_meta_model
from llmcompressor.streaming.materialization import (
    CastWeightMaterializer,
    WeightMaterializer,
    apply_transformers_weight_conversions,
)

from .quantization import install_activation_qdq, resolve_schemes
from .weights import StreamingWeights


def _execution_hooks(hook):
    if isinstance(hook, SequentialHook):
        for child in hook.hooks:
            yield from _execution_hooks(child)
    elif isinstance(hook, AlignDevicesHook):
        yield hook


class _ModuleWeightGroup(ModelHook):
    """Compose Accelerate's direct-module hooks into a safe preload group.

    Some Accelerate versions enumerate descendant nonpersistent buffers with
    bare names in recursive preload. Reusing nonrecursive hooks avoids that
    ambiguity and keeps derived-buffer movement at the owning child.
    """

    def __init__(self, hooks, device):
        self.hooks = hooks
        self.device = device

    def pre_forward(self, _module, *args, **kwargs):
        for child, hook in self.hooks:
            hook.pre_forward(child)
        return send_to_device(args, self.device), send_to_device(kwargs, self.device)

    def post_forward(self, _module, output):
        for child, hook in reversed(self.hooks):
            hook.post_forward(child, None)
        return output


def _group_module_hooks(model, class_names, device):
    def visit(module):
        if type(module).__name__ in class_names:
            hooks = []
            for child in module.modules():
                if child is module:
                    continue
                hook = getattr(child, "_hf_hook", None)
                if hook is not None:
                    remove_hook_from_module(child)
                    hooks.append((child, hook))
            add_hook_to_module(module, _ModuleWeightGroup(hooks, device), append=True)
            return
        for child in module.children():
            visit(child)

    visit(model)


def _buffer_hooks(model: nn.Module, device: torch.device) -> list:
    """Keep derived buffers on CPU outside the owning module's actual call."""
    handles = []
    for module in model.modules():
        names = tuple(
            name
            for name in module._non_persistent_buffers_set
            if module._buffers.get(name) is not None
            and name not in getattr(module, "_streaming_cpu_buffers", ())
        )
        if not names:
            continue
        if any(module._buffers[name].is_meta for name in names):
            raise ValueError("Nonpersistent buffers must be initialized before offload")

        def onload(owner, _args, names=names):
            for name in names:
                owner._buffers[name] = owner._buffers[name].to(device)

        def offload(owner, _args, _output, names=names):
            for name in names:
                owner._buffers[name] = owner._buffers[name].to("cpu")

        handles.append(module.register_forward_pre_hook(onload))
        handles.append(module.register_forward_hook(offload, always_call=True))
    return handles


def _check_weight_budget(model, weights, preload_classes, limit):
    """Bound the weights along one active module call stack, using headers only."""
    if limit is not None and limit <= 0:
        raise ValueError("max_device_weight_bytes must be positive")
    direct_sizes = {}
    for item in weights.describe():
        owner = item["name"].rpartition(".")[0]
        direct_sizes[owner] = direct_sizes.get(owner, 0) + item["nbytes"]
    subtree_sizes = {}

    def total(module, prefix):
        size = direct_sizes.get(prefix, 0)
        for name, child in module.named_children():
            size += total(child, f"{prefix}.{name}" if prefix else name)
        subtree_sizes[prefix] = size
        return size

    total(model, "")
    peak, peak_name = 0, ""

    def visit(module, prefix, ancestors):
        nonlocal peak, peak_name
        preload = type(module).__name__ in preload_classes
        current = subtree_sizes[prefix] if preload else direct_sizes.get(prefix, 0)
        live = ancestors + current
        if live > peak:
            peak, peak_name = live, prefix
        if not preload:
            for name, child in module.named_children():
                visit(child, f"{prefix}.{name}" if prefix else name, live)

    visit(model, "", 0)
    if limit is not None and peak > limit:
        raise MemoryError(
            f"Inference weight working set at {peak_name!r} needs {peak} bytes, "
            f"limit is {limit}; use finer preload_module_classes or a larger budget"
        )
    return peak, peak_name


class StreamingInference(AbstractContextManager):
    """Thin, inference-only access to a native model and its lazy weight source.

    Cache objects belong to the caller/Transformers. This class does not
    implement a generation loop, masks, positions or a second KV store.
    Calls are serialized because weight installation mutates the shared model.
    """

    def __init__(self, model, weights, device, handles, *, activation_qdq, weight_peak):
        self.model = model
        self.weights = weights
        self.device = device
        self.activation_qdq = activation_qdq
        self.weight_peak = weight_peak
        self._handles = handles
        self._closed = False
        self._lock = RLock()

    def _release_weights(self):
        for name in self.weights:
            owner_name, _, local = name.rpartition(".")
            owner = self.model.get_submodule(owner_name) if owner_name else self.model
            tensor = getattr(owner, local)
            if tensor is not None and not tensor.is_meta:
                set_module_tensor_to_device(owner, local, "meta")

    def _run(self, function, *args, **kwargs):
        with self._lock, torch.inference_mode():
            if self._closed:
                raise RuntimeError("Streaming inference is closed")
            cache = kwargs.get("past_key_values")
            if getattr(cache, "_streaming_failed", False):
                raise RuntimeError(
                    "Cache was partially updated by a failed forward; discard it "
                    "and prefill a new request"
                )
            try:
                return function(
                    *send_to_device(args, self.device),
                    **send_to_device(kwargs, self.device),
                )
            except BaseException:
                if cache is not None:
                    cache._streaming_failed = True
                raise
            finally:
                # Accelerate's forward wrapper does not run post_forward when
                # the native forward raises. Clean up partially installed groups.
                self._release_weights()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._run(self.model, *args, **kwargs)

    def call(self, method: str, *args, **kwargs):
        """Run a native auxiliary method, such as encode_image, under offload."""
        return self._run(getattr(self.model, method), *args, **kwargs)

    def generate(self, *args, **kwargs):
        # All checkpoint-backed parameters are deliberately meta between calls.
        # HF's comparison with model.device otherwise emits a misleading warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are calling \.generate\(\).*device type different.*",
                category=UserWarning,
            )
            return self._run(self.model.generate, *args, **kwargs)

    def report(self) -> dict:
        """Return serializable execution choices and loading counters."""
        return {
            "device": str(self.device),
            "model_class": type(self.model).__qualname__,
            "config_fingerprint": fingerprint_json(self.model.config.to_dict()),
            "software": {
                name: version(name)
                for name in (
                    "torch",
                    "transformers",
                    "accelerate",
                    "compressed-tensors",
                )
            },
            "activation_mode": (
                "quantized_reference" if self.activation_qdq else "dequantized_weights"
            ),
            "estimated_peak_weight_bytes": self.weight_peak[0],
            "peak_weight_module": self.weight_peak[1],
            "tensor_reads": self.weights.tensor_reads,
            "materializations": self.weights.materializations,
            "peak_estimated_preparation_bytes": self.weights.peak_preparation_bytes,
            "materializer": self.weights.materializer.identifier,
            "weights": self.weights.describe(),
        }

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._release_weights()
            remove_hook_from_module(self.model, recurse=True)
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._closed = True

    def __exit__(self, *_exc):
        self.close()


def load_streaming_model(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    config: Any = None,
    model_factory: Callable[..., nn.Module] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    materializer: WeightMaterializer | None = None,
    source: CheckpointWeightSource | None = None,
    activation_qdq: bool = True,
    preload_module_classes: list[str] | None = None,
    max_device_weight_bytes: int | None = None,
    max_preparation_bytes: int | None = None,
    host_memory_budget: HostMemoryBudget | None = None,
) -> StreamingInference:
    """Build a local model on meta and load only the currently called modules.

    INT4/INT8 PTQ weights are decoded on CPU at loading time, defaulting to BF16.
    Dynamic token A8 Q/DQ is retained by default. ``model_factory`` and
    ``materializer`` support registered/custom architectures and source layouts
    without model-name branches in this entry point.

    The device budget covers weights, not activations or KV cache. Native HF KV
    offload can be requested through ``generate(cache_implementation="offloaded")``.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(
            "Streaming model loading requires a local checkpoint directory"
        )
    device = torch.device(device)
    if device.type == "meta":
        raise ValueError("The execution device cannot be meta")
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Inference dtype must be BF16, FP16 or FP32")
    config = deepcopy(
        config
        if config is not None
        else AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    )
    factory = model_factory or AutoModelForCausalLM.from_config
    kwargs = dict(model_kwargs or {})
    if model_factory is None:
        kwargs["dtype"] = dtype
    # Explicit FP32 control parameters in custom factories remain FP32.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device("cpu"):
            model = build_meta_model(
                factory, config, keep_nonpersistent_buffers=True, **kwargs
            ).eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    for module_name, module in model.named_modules():
        for name in module._non_persistent_buffers_set:
            buffer = module._buffers.get(name)
            if buffer is not None and buffer.is_meta:
                raise ValueError(
                    f"Uninitialized nonpersistent buffer {module_name}.{name}; "
                    "the model inference adapter must reconstruct constants "
                    "and provide request-owned cache state"
                )
    schemes, format_name = resolve_schemes(
        model, getattr(config, "quantization_config", None)
    )
    materializer = apply_transformers_weight_conversions(
        materializer or CastWeightMaterializer(), model
    )
    source = (
        source if source is not None else materializer.create_source(str(checkpoint))
    )
    weights = StreamingWeights(
        model,
        source,
        materializer,
        schemes,
        format_name=format_name,
        max_preparation_bytes=max_preparation_bytes,
        host_memory_budget=host_memory_budget,
    )
    preload_classes = list(preload_module_classes or [])
    weight_peak = _check_weight_budget(
        model, weights, preload_classes, max_device_weight_bytes
    )
    handles = install_activation_qdq(model, schemes, enabled=activation_qdq)

    def validate_output(_module, _args, kwargs, output):
        if kwargs.get("use_cache") and getattr(output, "past_key_values", None) is None:
            raise RuntimeError(
                "Model did not return past_key_values with use_cache=True; "
                "its incremental inference adapter must be implemented first"
            )
        if getattr(getattr(output, "logits", None), "is_meta", False):
            raise RuntimeError(
                "Model returned meta logits; check hidden parameter access and "
                "preload_module_classes"
            )

    handles.append(model.register_forward_hook(validate_output, with_kwargs=True))
    try:
        # Initializing hooks on CPU keeps derived buffers from every layer from
        # being copied to the accelerator at setup. Only their owning call will
        # move them, while Accelerate retains ownership of parameter offload.
        cpu_offload(
            model,
            execution_device=torch.device("cpu"),
            state_dict=weights,
            offload_buffers=True,
        )
        for module in model.modules():
            for hook in _execution_hooks(getattr(module, "_hf_hook", None)):
                hook.execution_device = device
        _group_module_hooks(model, preload_classes, device)
        handles.extend(_buffer_hooks(model, device))
        generation_path = checkpoint / "generation_config.json"
        if generation_path.is_file():
            model.generation_config = GenerationConfig.from_pretrained(
                checkpoint, local_files_only=True
            )
    except BaseException:
        remove_hook_from_module(model, recurse=True)
        for handle in handles:
            handle.remove()
        raise
    return StreamingInference(
        model,
        weights,
        device,
        handles,
        activation_qdq=activation_qdq,
        weight_peak=weight_peak,
    )
