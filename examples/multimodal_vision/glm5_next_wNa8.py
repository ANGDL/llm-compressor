"""Calibrate GLM-5.3-Flash-BF16 to W4A8 routed experts and W8A8 elsewhere.

Routed MoE experts (``mlp.experts.<i>.{gate,up,down}_proj``) are quantized to
W4A8: int4 channel-wise weights with int8 per-token dynamic activations.
Every other quantizable ``Linear`` module keeps W8A8: int8 channel-wise
weights with int8 per-token dynamic activations. Both groups use explicit
schemes matching the GLM-5 MoE WNA8 recipe.
Pass ``--use-float32-scale-dtype`` to serialize weight scales as float32;
the default keeps the library-selected scale dtype.

``GLM5_NEXT_W8A8_IGNORES`` summarizes the
``quantization_config.modules_to_not_convert`` entries from the released
GLM-5.3-Flash config into one regex. This keeps KDA projections, mHC
parameters, indexer projections, norms, embeddings, the language-model head,
and the multimodal vision branch in BF16 without reading a reference checkpoint
at runtime.

The W8A8 targets are not ``["Linear"]``: after the permanent MoE replacement
unpacks the routed experts into plain ``Linear`` modules, the script enumerates
the model, drops the expert projections, drops everything matching the ignore
list, and emits the remaining names as one anchored, namespace-portable regex.
That difference set therefore never overlaps the W4A8 experts, and it still
matches ``model.language_model.*`` as well as the ``model`` /
``language_model.model`` spellings used by runtime backends. Once the
checkpoint is saved, ``--update-ignore-from-index`` rewrites the saved ignore
list from the real ``model.safetensors.index.json``.

Example::

    python examples/multimodal_vision/glm5_next_wNa8.py \
        --model_id /ssd4/models/GLM-5.3-Flash-BF16 \
        --save_dir /ssd4/models \
        --modifier RTN --observer imatrix_mse \
        --num_calibration_samples 256 --max_sequence_length 8192 \
        --dataset_id lmms-lab/flickr30k --dataset_split test \
        --text_dataset_id HuggingFaceH4/ultrachat_200k \
        --text_dataset_split train_sft --text_calibration_samples 256 \
        --pipeline sequential
    
"""

import argparse
import base64
import json
import os
import re
from collections import defaultdict
from contextlib import contextmanager
from io import BytesIO

import torch
from compressed_tensors.offload import get_device_map
from compressed_tensors.offload.dispatch import dispatch_model as _dispatch_model
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.utils.match import match_name
from loguru import logger
from transformers import AutoProcessor
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
)

# don't move this import to the top of the file
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.core import active_session
from llmcompressor.logger import LoggerConfig, configure_logger
from llmcompressor.modeling.glm5_next import (
    CalibrationGlm5NextTextMoE,  # noqa: F401 - registers the calibration adapter
    attach_mtp_layer,
)
from llmcompressor.modeling.moe_context import (
    moe_calibration_context as replace_moe_calibration_modules,
)
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.utils import ImatrixFallbackStats

# Routed experts are matched separately with W4A8. The W8A8 targets are built
# from the actual model modules below, so they are the difference between all
# remaining Linear modules, the routed expert projections, and the ignore list.
GLM5_NEXT_W4A8_EXPERT_TARGETS = [
    r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
]
# Runtime backends expose the text stack either as the HF checkpoint namespace
# ``model.language_model`` or as ``model`` / ``language_model.model`` wrappers.
# Targets and ignores share this root so both spellings keep matching.
GLM5_NEXT_MODEL_ROOT = r"(?:.*\.)?model(?:\.language_model)?"
GLM5_NEXT_HF_TEXT_ROOT = "model.language_model."
GLM5_NEXT_VLLM_FUSED_IGNORES = [
    r"re:^(?:model\.language_model|(?:language_model\.|mtp\.)?model)"
    r"\.layers\.\d+(?:\.mtp_block)?\.self_attn\.in_proj_qkvgfab$"
]


def _config_value(config, key):
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def build_glm5_next_w8a8_targets(model, ignores: list[str]) -> list[str]:
    """Return the non-expert Linear difference set as one compact regex.

    GLM-5.3 repeats the same projection suffixes across a small set of layer
    types. Grouping those names by suffix and layer index keeps the target
    exact while avoiding one alternative per module (which made saved
    ``group_0.targets`` needlessly large).
    """
    expert_suffixes = {"gate_proj", "up_proj", "down_proj"}
    linear_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and not (
            ".mlp.experts." in name
            and name.rsplit(".", 1)[-1] in expert_suffixes
        )
    ]
    linear_names = [
        name
        for name in linear_names
        if not any(match_name(name, ignore) for ignore in ignores)
    ]
    if not linear_names:
        raise ValueError("No non-expert Linear modules remain after applying ignores")

    # The HF model has a stable ``model.language_model.layers.<idx>`` prefix.
    # Collect the varying layer index for each remaining suffix, then emit one
    # alternative per suffix. Any unexpected non-layer module falls back to
    # the original escaped-name handling so this remains safe for model
    # variants with additional Linear modules.
    grouped: dict[str, set[int]] = {}
    non_layer_names = []
    layer_pattern = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
    for name in linear_names:
        match = layer_pattern.fullmatch(name)
        if match is None:
            non_layer_names.append(name)
            continue
        grouped.setdefault(match.group(2), set()).add(int(match.group(1)))

    # Merge suffixes that occur on exactly the same layer set.
    by_layers: dict[frozenset[int], list[str]] = {}
    for suffix, layer_set in grouped.items():
        by_layers.setdefault(frozenset(layer_set), []).append(suffix)

    alternatives = []
    for layer_indices_set, suffixes in sorted(
        by_layers.items(), key=lambda item: (min(item[0]), item[1])
    ):
        suffixes = tuple(sorted(suffixes))
        layer_indices = "(?:" + "|".join(map(str, sorted(layer_indices_set))) + ")"
        suffix_pattern = (
            re.escape(suffixes[0])
            if len(suffixes) == 1
            else "(?:" + "|".join(map(re.escape, suffixes)) + ")"
        )
        alternatives.append(
            GLM5_NEXT_MODEL_ROOT
            + r"\.layers\."
            + layer_indices
            + r"\."
            + suffix_pattern
        )
    alternatives.extend(_runtime_portable_pattern(name) for name in non_layer_names)
    return [r"re:^(?:" + "|".join(alternatives) + r")$"]


def _runtime_portable_pattern(name: str) -> str:
    """Rewrite an HF module name so runtime namespace variants also match."""
    if name.startswith(GLM5_NEXT_HF_TEXT_ROOT):
        rest = name.removeprefix(GLM5_NEXT_HF_TEXT_ROOT)
        return GLM5_NEXT_MODEL_ROOT + r"\." + re.escape(rest)
    return re.escape(name)


def build_glm5_next_w8a8_ignores(config) -> list[str]:
    """Build runtime-portable ignores from the configured KDA layer layout."""
    text_config = _config_value(config, "text_config")
    linear_attn_config = _config_value(text_config, "linear_attn_config")
    kda_layers = _config_value(linear_attn_config, "kda_layers")
    num_hidden_layers = _config_value(text_config, "num_hidden_layers")
    if not isinstance(kda_layers, (list, tuple)) or not kda_layers:
        raise ValueError(
            "GLM-5.3 text config must define linear_attn_config.kda_layers"
        )
    if not isinstance(num_hidden_layers, int) or any(
        not isinstance(layer, int) or layer < 0 or layer >= num_hidden_layers
        for layer in kda_layers
    ):
        raise ValueError("GLM-5.3 text config contains invalid KDA layer indices")

    kda_layer_pattern = "(?:" + "|".join(map(str, sorted(set(kda_layers)))) + ")"
    # Optional leading components cover vLLM wrappers such as
    # ``language_model.model`` and a separately loaded ``mtp.model``. The HF
    # model's checkpoint-facing namespace is the ``model.language_model``
    # alternative.
    model_root = GLM5_NEXT_MODEL_ROOT
    layer_root = model_root + r"\.layers"
    decoder_prefix = layer_root + r"\.\d+\.(?:mtp_block\.)?"
    ignore_alternatives = (
        r"(?:.*\.)?lm_head",
        r"(?:.*\.)?visual(?:\..*)?",
        r"(?:.*\.)?(?:attn_mha|attn_mqa|dt_bias|hyper_connection|"
        r"mapping_proj|router|weights_proj)",
        model_root + r"\.(?:embed_tokens|norm)",
        decoder_prefix + r"(?:hc_(?:attn|ffn)_(?:base|fn|scale)|input_layernorm|"
        r"post_attention_layernorm)",
        layer_root + rf"\.{kda_layer_pattern}\.self_attn(?:\..*)?",
        decoder_prefix + r"mlp\.gate(?:\..*)?",
        decoder_prefix + r"self_attn\.(?:indexer(?:\..*)?|kv_a_layernorm|kv_b_proj|"
        r"q_a_layernorm)",
        layer_root + r"\.\d+\.(?:eh_proj|enorm|hnorm|shared_head\.norm)",
    )
    ignore_pattern = r"re:^(?:" + "|".join(ignore_alternatives) + r")$"
    return [ignore_pattern]


def _glm5_next_runtime_name_aliases(name: str, config) -> set[str]:
    """Add aliases for runtime structures not covered by vLLM's name mapper."""
    aliases = {name}
    text_config = _config_value(config, "text_config")
    num_hidden_layers = _config_value(text_config, "num_hidden_layers")
    num_mtp_layers = _config_value(text_config, "num_nextn_predict_layers") or 0

    if not name.startswith("model.language_model."):
        return aliases

    relative_name = name.removeprefix("model.language_model.")
    if isinstance(num_hidden_layers, int):
        layer_parts = relative_name.split(".", 2)
        try:
            layer_idx = int(layer_parts[1])
        except (IndexError, ValueError):
            layer_idx = -1
        if (
            len(layer_parts) == 3
            and layer_parts[0] == "layers"
            and num_hidden_layers <= layer_idx < num_hidden_layers + num_mtp_layers
        ):
            mtp_relative_name = layer_parts[2]
            shared_prefixes = ("enorm", "hnorm", "eh_proj", "shared_head")
            if mtp_relative_name.split(".", 1)[0] not in shared_prefixes:
                mtp_relative_name = "mtp_block." + mtp_relative_name
            mtp_name = f"model.layers.{layer_idx}.{mtp_relative_name}"
            aliases.add(mtp_name)
            aliases.add("mtp." + mtp_name)

    # Transformers nests these checkpoint projections under ``forget_gate``.
    for alias in tuple(aliases):
        prefix, separator, suffix = alias.rpartition(".self_attn.")
        if separator and suffix in {"f_a_proj", "f_b_proj"}:
            aliases.add(f"{prefix}.self_attn.forget_gate.{suffix}")
    return aliases


def collect_glm5_next_ignored_checkpoint_names(
    tensor_names: list[str],
    config,
) -> list[str]:
    """Infer exhaustive ignores from the checkpoint's actual quantized tensors."""
    quantized_modules = {
        name.removesuffix(suffix)
        for name in tensor_names
        for suffix in (".weight_scale", ".weight_scale_inv")
        if name.endswith(suffix)
    }
    quantization_suffixes = (
        ".weight_scale",
        ".weight_scale_inv",
        ".weight_zero_point",
        ".weight_packed",
        ".weight_shape",
        ".weight_g_idx",
        ".input_scale",
        ".input_zero_point",
        ".output_scale",
        ".output_zero_point",
        ".input_global_scale",
    )
    ignored_names = set(GLM5_NEXT_VLLM_FUSED_IGNORES)
    for tensor_name in tensor_names:
        if tensor_name.endswith(quantization_suffixes):
            continue
        parent_name, _, parameter_name = tensor_name.rpartition(".")
        if parameter_name in {"weight", "bias"}:
            module_name = parent_name
        else:
            module_name = tensor_name
        # Packed and naive quantized modules keep auxiliary tensors such as
        # ``weight_packed``/``weight_shape`` next to the quantized weight.
        if module_name in quantized_modules or parent_name in quantized_modules:
            continue
        ignored_names.update(_glm5_next_runtime_name_aliases(module_name, config))
        if parameter_name != "weight":
            ignored_names.update(_glm5_next_runtime_name_aliases(tensor_name, config))
    return sorted(ignored_names)


def replace_glm5_next_saved_ignores(
    config_data: dict,
    checkpoint_ignores: list[str],
) -> dict:
    """Replace saved ignores with names derived from the checkpoint index."""
    quantization_config = config_data.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise ValueError("Saved config is missing quantization_config")

    quantization_config["ignore"] = list(dict.fromkeys(checkpoint_ignores))
    return config_data


def update_glm5_next_saved_ignores(save_path: str) -> None:
    config_path = os.path.join(save_path, "config.json")
    index_path = os.path.join(save_path, "model.safetensors.index.json")
    with open(config_path, encoding="utf-8") as file:
        config_data = json.load(file)
    with open(index_path, encoding="utf-8") as file:
        tensor_names = list(json.load(file)["weight_map"])
    checkpoint_ignores = collect_glm5_next_ignored_checkpoint_names(
        tensor_names,
        config_data,
    )
    replace_glm5_next_saved_ignores(
        config_data,
        checkpoint_ignores,
    )
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config_data, file, indent=2, sort_keys=True)
        file.write("\n")


configure_logger(LoggerConfig(console_log_level="DEBUG"))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _iter_debug_tensors(value, path="value"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_debug_tensors(item, f"{path}.{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_debug_tensors(item, f"{path}.{key}")


def _activation_debug_stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach()
    result = {
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "numel": value.numel(),
    }
    if value.numel() == 0:
        return {**result, "finite": True, "nan": 0, "inf": 0, "max_abs": 0.0}

    if value.is_floating_point() or value.is_complex():
        finite = torch.isfinite(value)
        all_finite = bool(finite.all().item())
        result.update(
            finite=all_finite,
            nan=int(torch.isnan(value).sum().item()) if not all_finite else 0,
            inf=int(torch.isinf(value).sum().item()) if not all_finite else 0,
        )
        if all_finite:
            result["max_abs"] = float(value.abs().max().item())
        elif bool(finite.any().item()):
            result["max_abs"] = float(value.abs().masked_fill(~finite, 0).max().item())
        else:
            result["max_abs"] = None
    else:
        result.update(
            finite=True,
            nan=0,
            inf=0,
            max_abs=float(value.abs().max().item()),
        )
    return result


@contextmanager
def activation_debug_logging(model, layer_indices: list[int] | None):
    """Log GLM decoder boundaries and the first non-finite leaf outputs."""
    if not layer_indices:
        yield
        return

    layers = model.model.language_model.layers
    invalid = sorted(
        index for index in set(layer_indices) if not 0 <= index < len(layers)
    )
    if invalid:
        raise ValueError(
            f"--debug-activation-layers contains invalid indices {invalid}; "
            f"model has {len(layers)} language layers"
        )

    handles = []
    call_counts = defaultdict(int)
    reported_nonfinite = set()

    def batch_index():
        try:
            return active_session().state.current_batch_idx
        except (AttributeError, RuntimeError):
            return -1

    def emit(name, direction, value, always):
        call_counts[(name, direction)] += 1
        call = call_counts[(name, direction)]
        for tensor_path, tensor in _iter_debug_tensors(value, direction):
            stats = _activation_debug_stats(tensor)
            key = (name, direction, tensor_path)
            if not always and stats["finite"]:
                continue
            if not stats["finite"] and key in reported_nonfinite:
                continue
            if not stats["finite"]:
                reported_nonfinite.add(key)
            log = logger.debug if stats["finite"] else logger.warning
            log(
                "[activation-debug] batch={} call={} module={} tensor={} "
                "shape={} dtype={} finite={} nan={} inf={} max_abs_finite={}",
                batch_index(),
                call,
                name,
                tensor_path,
                stats["shape"],
                stats["dtype"],
                stats["finite"],
                stats["nan"],
                stats["inf"],
                stats["max_abs"],
            )

    def make_pre_hook(name):
        def hook(_module, inputs):
            emit(name, "input", inputs, always=True)

        return hook

    def make_output_hook(name, always=False):
        def hook(_module, _inputs, output):
            emit(name, "output", output, always=always)
            if name.endswith(".mlp.gate") and isinstance(output, tuple):
                topk_indices = output[2]
                if isinstance(topk_indices, torch.Tensor):
                    experts = torch.unique(topk_indices.detach()).cpu().tolist()
                    logger.debug(
                        "[activation-debug] batch={} module={} routed_experts={}",
                        batch_index(),
                        name,
                        experts,
                    )

        return hook

    for layer_index in sorted(set(layer_indices)):
        layer = layers[layer_index]
        prefix = f"model.language_model.layers.{layer_index}"
        stage_modules = {
            prefix: layer,
            f"{prefix}.attn_hc": layer.attn_hc,
            f"{prefix}.input_layernorm": layer.input_layernorm,
            f"{prefix}.self_attn": layer.self_attn,
            f"{prefix}.ffn_hc": layer.ffn_hc,
            f"{prefix}.post_attention_layernorm": layer.post_attention_layernorm,
            f"{prefix}.mlp": layer.mlp,
            f"{prefix}.mlp.gate": getattr(layer.mlp, "gate", None),
        }
        stage_modules = {
            name: module
            for name, module in stage_modules.items()
            if module is not None
        }
        handles.append(layer.register_forward_pre_hook(make_pre_hook(prefix)))
        for name, module in stage_modules.items():
            handles.append(
                module.register_forward_hook(make_output_hook(name, always=True))
            )

        stage_ids = {id(module) for module in stage_modules.values()}
        for relative_name, module in layer.named_modules():
            if (
                not relative_name
                or id(module) in stage_ids
                or any(module.children())
            ):
                continue
            name = f"{prefix}.{relative_name}"
            handles.append(module.register_forward_hook(make_output_hook(name)))

    logger.info(
        "[activation-debug] installed {} hooks for language layers {}",
        len(handles),
        sorted(set(layer_indices)),
    )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        logger.info("[activation-debug] removed activation hooks")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id", type=str, default="/path/to/models/GLM-5.3-Flash-BF16"
)
parser.add_argument("--save_dir", type=str, default="/path/to/models")
parser.add_argument("--observer", choices=["", "mse", "imatrix_mse"], default="")
parser.add_argument("--modifier", choices=["GPTQ", "RTN"], default="GPTQ")
parser.add_argument(
    "--use-float32-scale-dtype",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use float32 for weight scale tensors instead of the default dtype.",
)
parser.add_argument("--dataset_id", type=str, default="lmms-lab/flickr30k")
parser.add_argument("--dataset_split", type=str, default="test")
parser.add_argument("--num_calibration_samples", type=positive_int, default=32)
parser.add_argument(
    "--text_dataset_id", type=str, default="HuggingFaceH4/ultrachat_200k"
)
parser.add_argument("--text_dataset_split", type=str, default="train_sft")
parser.add_argument(
    "--text_calibration_samples",
    type=int,
    default=128,
    help=(
        "Number of pure-text samples to append to multimodal calibration "
        "samples; 0 disables them."
    ),
)
parser.add_argument("--max_sequence_length", type=positive_int, default=8192)
parser.add_argument(
    "--pipeline",
    choices=["independent", "sequential", "basic", "data_free"],
    default="independent",
)
parser.add_argument("--dispatch_extra_memory_gb", type=float, default=20.0)
parser.add_argument("--max-memory-cpu-gb", type=float, default=1500.0)
parser.add_argument(
    "--skip_restore_from_accelerate",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--moe-calibrate-all-experts",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--debug-activation-layers",
    type=int,
    nargs="+",
    default=None,
    metavar="LAYER",
    help=(
        "Log activation finiteness at selected GLM language-layer boundaries and "
        "report the first non-finite output from each leaf module."
    ),
)
parser.add_argument(
    "--update-ignore-from-index",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Replace the saved quantization ignore list using the actual "
        "model.safetensors.index.json contents."
    ),
)
args = parser.parse_args()
if args.text_calibration_samples < 0:
    parser.error("--text_calibration_samples must be non-negative")


def patch_pipeline_dispatch(extra_memory_gb: float):
    extra_memory_bytes = int(extra_memory_gb * 1024**3)

    def has_disk_offload(model) -> bool:
        try:
            device_map = get_device_map(model)
        except Exception:
            return False
        return any(offload == "disk" for _onload, offload in device_map.values())

    def dispatch(model):
        if has_disk_offload(model):
            logger.info("Preserving the existing disk offload map")
            return model
        return _dispatch_model(model, extra_memory=extra_memory_bytes)

    from llmcompressor.pipelines.basic import pipeline as basic_pipeline
    from llmcompressor.pipelines.data_free import pipeline as data_free_pipeline

    basic_pipeline.dispatch_model = dispatch
    data_free_pipeline.dispatch_model = dispatch


@contextmanager
def maybe_skip_from_accelerate(skip_restore: bool):
    if not skip_restore:
        yield
        return
    import llmcompressor.transformers.compression.compressed_tensors_utils as ct_utils

    original = ct_utils.from_accelerate
    ct_utils.from_accelerate = lambda model: ({}, None)
    try:
        yield
    finally:
        ct_utils.from_accelerate = original


patch_pipeline_dispatch(args.dispatch_extra_memory_gb)

dataset = load_dataset(
    args.dataset_id,
    split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
).shuffle(seed=42)
text_dataset = (
    load_dataset(
        args.text_dataset_id,
        split=f"{args.text_dataset_split}[:{args.text_calibration_samples}]",
    ).shuffle(seed=42)
    if args.text_calibration_samples > 0
    else None
)
text_examples = list(text_dataset) if text_dataset is not None else []

model = Glm5NextForConditionalGeneration.from_pretrained(
    args.model_id,
    dtype="auto",
    device_map=None,
)
GLM5_NEXT_W8A8_IGNORES = build_glm5_next_w8a8_ignores(model.config)
processor = AutoProcessor.from_pretrained(args.model_id)
tokenizer = processor.tokenizer


def _caption_text(example) -> str:
    caption = example.get("caption", "")
    if isinstance(caption, (list, tuple)):
        caption = caption[0] if caption else ""
    return str(caption)


def _image_as_data_url(image) -> str:
    buffered = BytesIO()
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"


def _multimodal_messages(example):
    image_url = _image_as_data_url(example["image"])
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": "What does the image show?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": _caption_text(example)}],
        },
    ]


def _normalize_text_content(content):
    """Normalize common chat-dataset content formats for the GLM processor."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        content = [content]

    normalized = []
    if not isinstance(content, (list, tuple)):
        return normalized
    for item in content:
        if isinstance(item, str):
            normalized.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            if item.get("type") == "text":
                normalized.append({"type": "text", "text": str(item.get("text", ""))})
            elif "text" in item:
                normalized.append({"type": "text", "text": str(item["text"])})
    return normalized


def _format_text_example(example):
    messages = example.get("messages")
    if messages is None:
        text = example.get("text")
        if text is None:
            return []
        return [{"role": "user", "content": [{"type": "text", "text": str(text)}]}]

    formatted = []
    for message in messages:
        if not isinstance(message, dict) or "role" not in message:
            continue
        formatted.append(
            {
                "role": message["role"],
                "content": _normalize_text_content(message.get("content", "")),
            }
        )
    return formatted


def _build_calibration_messages(multimodal_example, text_example=None):
    messages = _multimodal_messages(multimodal_example)
    if text_example is not None:
        messages.extend(_format_text_example(text_example))
    return messages


def preprocess(example, idx):
    if "image" in example and example["image"] is not None:
        # tokenize=True is required here: GLM-5.3 expands <|image|> to the
        # processor-computed number of visual patch tokens.
        encoded = processor.apply_chat_template(
            _build_calibration_messages(
                example,
                text_examples[idx % len(text_examples)] if text_examples else None,
            ),
            tokenize=True,
            return_dict=True,
            padding=False,
            max_length=args.max_sequence_length,
            truncation=True,
        )
        return dict(encoded)
    if "messages" in example:
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return dict(
            processor(
                text=text,
                padding=False,
                max_length=args.max_sequence_length,
                truncation=True,
                add_special_tokens=False,
            )
        )
    if "text" in example:
        return dict(
            processor(
                text=example["text"],
                padding=False,
                max_length=args.max_sequence_length,
                truncation=True,
                add_special_tokens=False,
            )
        )
    raise ValueError("Calibration examples must contain 'image', 'messages' or 'text'")


dataset = dataset.map(
    preprocess,
    with_indices=True,
    remove_columns=dataset.column_names,
)


def data_collator(batch):
    """Collate one variable-size multimodal calibration sample."""
    if len(batch) != 1:
        raise ValueError("GLM-5.3 multimodal calibration requires batch_size=1")

    collated = {}
    for key, value in batch[0].items():
        if value is None:
            continue
        dtype = torch.bfloat16 if key == "pixel_values" else None
        collated[key] = torch.as_tensor(value, dtype=dtype)
    return collated


# HF intentionally ignores the flat MTP continuation. Attach it at the
# canonical checkpoint path so save_pretrained remains consistent with
# ``model.language_model.layers.<num_hidden_layers>.*``.
attach_mtp_layer(model, args.model_id)

# The generic oneshot entrypoint linearizes any packed expert implementation before
# calibration. Replace GLM-5.3's MoE blocks first so its registered
# ``MoECalibrationModule`` adapter owns expert unpacking and routing instead.
with replace_moe_calibration_modules(
    model, calibrate_all_experts=args.moe_calibrate_all_experts
):
    # CalibrationGlm5NextTextMoE is permanent, so the replacements remain on
    # ``model`` after this block and are seen by the subsequent oneshot call.
    # The W8A8 difference set is computed afterwards from the unpacked modules.
    logger.info(
        "Targeting routed GLM-5.3 experts with W4A8 and the remaining "
        "quantizable Linear modules with W8A8",
    )

GLM5_NEXT_W8A8_TARGETS = build_glm5_next_w8a8_targets(
    model, GLM5_NEXT_W8A8_IGNORES
)
# Match the GLM-5 MoE WNA8 recipe: channel-wise static weights and dynamic
# per-token activations.  The preset W4A8 scheme uses group-wise weights, which
# is a different quantization contract from this model's per-channel recipe.
weight_scale_dtype = (
    torch.float32 if args.use_float32_scale_dtype else None
)
weights_args_4 = QuantizationArgs(
    num_bits=4,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.CHANNEL,
    symmetric=True,
    dynamic=False,
    scale_dtype=weight_scale_dtype,
)
weights_args_8 = QuantizationArgs(
    num_bits=8,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.CHANNEL,
    symmetric=True,
    dynamic=False,
    scale_dtype=weight_scale_dtype,
)
activations_args = QuantizationArgs(
    num_bits=8,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.TOKEN,
    symmetric=True,
    dynamic=True,
    observer=None,
    scale_dtype=torch.float32,
)
experts_w4_scheme = QuantizationScheme(
    targets=GLM5_NEXT_W4A8_EXPERT_TARGETS,
    weights=weights_args_4,
    input_activations=activations_args,
)
other_linear_w8_scheme = QuantizationScheme(
    targets=GLM5_NEXT_W8A8_TARGETS,
    weights=weights_args_8,
    input_activations=activations_args,
)
if args.observer:
    experts_w4_scheme.weights.observer = args.observer
    other_linear_w8_scheme.weights.observer = args.observer

tail_name = "-W4A8Experts-W8A8Other"
recipes = []
if args.observer == "imatrix_mse":
    fallback_stats = ImatrixFallbackStats()
    fallback_stats.install_hooks()
    recipes.append(
        IMatrixGatherer(
            targets=GLM5_NEXT_W8A8_TARGETS + GLM5_NEXT_W4A8_EXPERT_TARGETS,
            ignore=GLM5_NEXT_W8A8_IGNORES,
            attach_by_initialize=args.pipeline != "sequential",
        )
    )
    tail_name += "-IMatrix"

modifier_kwargs = {
    "config_groups": {
        "experts_w4a8": experts_w4_scheme,
        "other_linear_w8a8": other_linear_w8_scheme,
    },
    "targets": GLM5_NEXT_W8A8_TARGETS + GLM5_NEXT_W4A8_EXPERT_TARGETS,
    "ignore": GLM5_NEXT_W8A8_IGNORES,
}
if args.modifier == "GPTQ":
    recipes.append(GPTQModifier(**modifier_kwargs))
    tail_name += "-GPTQ"
else:
    recipes.append(QuantizationModifier(**modifier_kwargs))
    tail_name += "-RTN"

oneshot_kwargs = dict(
    model=model,
    dataset=dataset,
    recipe=recipes,
    max_seq_length=args.max_sequence_length,
    num_calibration_samples=args.num_calibration_samples,
    batch_size=1,
    data_collator=data_collator,
    processor=processor,
    moe_calibrate_all_experts=args.moe_calibrate_all_experts,
    pipeline=args.pipeline,
    shuffle_calibration_samples=False
)
with activation_debug_logging(model, args.debug_activation_layers):
    if args.observer == "imatrix_mse":
        with fallback_stats:
            oneshot(**oneshot_kwargs)
    else:
        oneshot(**oneshot_kwargs)

save_name = os.path.basename(args.model_id.rstrip("/")) + tail_name
save_path = os.path.join(args.save_dir, save_name)
with maybe_skip_from_accelerate(args.skip_restore_from_accelerate):
    model.save_pretrained(save_path, save_compressed=True)
processor.save_pretrained(save_path)
if args.update_ignore_from_index:
    update_glm5_next_saved_ignores(save_path)
logger.info("Saved compressed GLM-5.3 checkpoint to {}", save_path)
