"""Calibrate GLM-5.3-Flash-BF16 to W8A8.

The fixed target set in ``GLM5_NEXT_W8A8_TARGETS`` covers dense/shared/routed
MLP projections and the four DSA projections. KDA projections, mHC
parameters, indexer projections, norms, embeddings and the language-model head
remain BF16. ``GLM5_NEXT_W8A8_IGNORES`` also excludes the multimodal vision
branch under ``model.visual``. No separate FP8 checkpoint is required to
determine this scope.

Example::

    python /data/llm-compressor/examples/quantizing_moe/glm5_next_w8a8.py \
        --model_id /ssd4/models/GLM-5.3-Flash-BF16 \
        --save_dir /ssd4/models \
        --modifier RTN --observer imatrix_mse \
        --num_calibration_samples 256 --max_sequence_length 8192 \
        --dataset_id lmms-lab/flickr30k --dataset_split test \
        --text_dataset_id HuggingFaceH4/ultrachat_200k \
        --text_dataset_split train_sft --text_calibration_samples 128 \
        --check-imatrix-nonfinite \
        --pipeline sequential
"""

import argparse
import base64
import os
from contextlib import contextmanager
from io import BytesIO

import torch
from compressed_tensors.offload import get_device_map
from compressed_tensors.offload.dispatch import dispatch_model as _dispatch_model
from compressed_tensors.quantization import preset_name_to_scheme
from loguru import logger
from transformers import AutoProcessor
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
)

# don't move this import to the top of the file
from datasets import load_dataset
from llmcompressor import oneshot
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

# These are the GLM-5.3 modules that use the W8A8 INT8 scheme. Keep this list
# local to the quantization entrypoint so the model adapter remains model-only.
GLM5_NEXT_W8A8_TARGETS = [
    r"re:^model\.language_model\.layers\.\d+\.mlp\."
    r"(gate_proj|up_proj|down_proj)$",
    r"re:^model\.language_model\.layers\.\d+\.mlp\.shared_experts\."
    r"(gate_proj|up_proj|down_proj)$",
    r"re:^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(gate_proj|up_proj|down_proj)$",
    r"re:^model\.language_model\.layers\.\d+\.self_attn\."
    r"(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)$",
]
GLM5_NEXT_W8A8_IGNORES = [
    "lm_head",
    r"re:^model\.visual\..*",
]

configure_logger(LoggerConfig(console_log_level="DEBUG"))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id", type=str, default="/Users/ang/models/GLM-5.3-Flash-BF16"
)
parser.add_argument("--save_dir", type=str, default="/Users/ang/models")
parser.add_argument("--observer", choices=["", "mse", "imatrix_mse"], default="")
parser.add_argument("--modifier", choices=["GPTQ", "RTN"], default="GPTQ")
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
    "--check-imatrix-nonfinite",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Check iMatrix x/x^2/FP32 channel reductions during calibration and "
        "raise on the first NaN or Inf."
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
# ``model.language_model.layers.45.*``.
attach_mtp_layer(model, args.model_id)

# The generic oneshot entrypoint linearizes any packed expert implementation before
# calibration. Replace GLM-5.3's MoE blocks first so its registered
# ``MoECalibrationModule`` adapter owns expert unpacking and routing instead.
with replace_moe_calibration_modules(
    model, calibrate_all_experts=args.moe_calibrate_all_experts
):
    # CalibrationGlm5NextTextMoE is permanent, so the replacements remain on
    # ``model`` after this block and are seen by the subsequent oneshot call.
    logger.info(
        "Using fixed GLM-5.3 W8A8 target patterns for {} module groups",
        len(GLM5_NEXT_W8A8_TARGETS),
    )

scheme = preset_name_to_scheme("W8A8", ["Linear"])
if scheme.weights is None:
    raise RuntimeError("W8A8 preset did not provide weight quantization arguments")
if args.observer:
    scheme.weights.observer = args.observer

tail_name = "-W8A8"
recipes = []
if args.observer == "imatrix_mse":
    fallback_stats = ImatrixFallbackStats(
        check_nonfinite=args.check_imatrix_nonfinite
    )
    fallback_stats.install_hooks()
    recipes.append(
        IMatrixGatherer(
            targets=GLM5_NEXT_W8A8_TARGETS,
            ignore=GLM5_NEXT_W8A8_IGNORES,
            attach_by_initialize=args.pipeline != "sequential",
        )
    )
    tail_name += "-IMatrix"

modifier_kwargs = {
    "config_groups": {"group_0": scheme},
    "targets": GLM5_NEXT_W8A8_TARGETS,
    "ignore": GLM5_NEXT_W8A8_IGNORES,
}
if args.modifier == "GPTQ":
    recipes.append(GPTQModifier(**modifier_kwargs, offload_hessians=True))
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
)
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
logger.info("Saved compressed GLM-5.3 checkpoint to {}", save_path)
