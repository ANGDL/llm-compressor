"""Calibrate GLM-5.3-Flash-BF16 to W8A8.

The target set is intentionally derived from the GLM-5.3-Flash FP8 checkpoint
layout: dense/shared/routed MLP projections and the four DSA projections only.
KDA projections, mHC parameters, indexer projections, norms, embeddings and
the language-model head remain BF16, matching the FP8 reference.

Example::

    python examples/quantizing_moe/glm5_next_w8a8.py \
        --model_id /Users/ang/models/GLM-5.3-Flash-BF16 \
        --fp8_reference_model /Users/ang/models/GLM-5.3-Flash \
        --save_dir /Users/ang/models \
        --modifier RTN --observer imatrix_mse \
        --num_calibration_samples 256 --max_sequence_length 8192 \
        --dataset_id ./ultrachat_200k --dataset_split train_sft \
        --pipeline sequential
"""

import argparse
import os
from contextlib import contextmanager

from compressed_tensors.offload import get_device_map, load_offloaded_model
from compressed_tensors.offload.dispatch import dispatch_model as _dispatch_model
from compressed_tensors.quantization import preset_name_to_scheme
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.logger import LoggerConfig, configure_logger
from llmcompressor.modeling.glm5_next import (
    GLM5_NEXT_W8A8_TARGETS,
    CalibrationGlm5NextTextMoE,  # noqa: F401 - registers the calibration adapter
    attach_mtp_layer,
    validate_fp8_target_alignment,
)
from llmcompressor.modeling.moe_context import (
    moe_calibration_context as replace_moe_calibration_modules,
)
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.utils import ImatrixFallbackStats

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
parser.add_argument(
    "--fp8_reference_model", type=str, default="/Users/ang/models/GLM-5.3-Flash"
)
parser.add_argument("--save_dir", type=str, default="/Users/ang/models")
parser.add_argument("--observer", choices=["", "mse", "imatrix_mse"], default="")
parser.add_argument("--modifier", choices=["GPTQ", "RTN"], default="GPTQ")
parser.add_argument("--dataset_id", type=str, default="HuggingFaceH4/ultrachat_200k")
parser.add_argument("--dataset_split", type=str, default="train_sft")
parser.add_argument("--num_calibration_samples", type=positive_int, default=32)
parser.add_argument("--max_sequence_length", type=positive_int, default=8192)
parser.add_argument(
    "--pipeline",
    choices=["independent", "sequential", "basic", "data_free"],
    default="independent",
)
parser.add_argument("--offload_folder", type=str, default="./offload_folder")
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
args = parser.parse_args()


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

with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype="auto",
        device_map="auto_offload",
        offload_folder=args.offload_folder,
        max_memory={"cpu": int(args.max_memory_cpu_gb * 1e9)},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)


def preprocess(example):
    if "messages" in example:
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }
    if "text" in example:
        return {"text": example["text"]}
    raise ValueError("Calibration examples must contain 'messages' or 'text'")


dataset = dataset.map(preprocess)
dataset = dataset.map(
    lambda sample: tokenizer(
        sample["text"],
        padding=False,
        max_length=args.max_sequence_length,
        truncation=True,
        add_special_tokens=False,
    ),
    remove_columns=dataset.column_names,
)

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
    quantized_names = validate_fp8_target_alignment(model, args.fp8_reference_model)
    logger.info(
        "Validated {} W8A8 modules against the FP8 reference", len(quantized_names)
    )

scheme = preset_name_to_scheme("W8A8", ["Linear"])
if scheme.weights is None:
    raise RuntimeError("W8A8 preset did not provide weight quantization arguments")
if args.observer:
    scheme.weights.observer = args.observer

recipes = []
if args.observer == "imatrix_mse":
    fallback_stats = ImatrixFallbackStats()
    fallback_stats.install_hooks()
    recipes.append(
        IMatrixGatherer(
            targets=GLM5_NEXT_W8A8_TARGETS,
            ignore=["lm_head"],
            attach_by_initialize=args.pipeline != "sequential",
        )
    )

modifier_kwargs = {
    "config_groups": {"group_0": scheme},
    "targets": GLM5_NEXT_W8A8_TARGETS,
    "ignore": ["lm_head"],
}
if args.modifier == "GPTQ":
    recipes.append(GPTQModifier(**modifier_kwargs, offload_hessians=True))
else:
    recipes.append(QuantizationModifier(**modifier_kwargs))

oneshot_kwargs = dict(
    model=model,
    dataset=dataset,
    recipe=recipes,
    max_seq_length=args.max_sequence_length,
    num_calibration_samples=args.num_calibration_samples,
    batch_size=1,
    moe_calibrate_all_experts=args.moe_calibrate_all_experts,
    pipeline=args.pipeline,
    # Transformers' default no-split list only contains the main text decoder
    # (and vision blocks).  Keep the attached MTP decoder as its own sequential
    # boundary so its Linear observers/GPTQ hooks are finalized independently.
    sequential_targets=["Glm5NextTextDecoderLayer", "Glm5NextMTPLayer"],
)
if args.observer == "imatrix_mse":
    with fallback_stats:
        oneshot(**oneshot_kwargs)
else:
    oneshot(**oneshot_kwargs)

save_name = os.path.basename(args.model_id.rstrip("/")) + "-W8A8-GLM5.3"
save_path = os.path.join(args.save_dir, save_name)
with maybe_skip_from_accelerate(args.skip_restore_from_accelerate):
    model.save_pretrained(save_path, save_compressed=True)
tokenizer.save_pretrained(save_path)
logger.info("Saved compressed GLM-5.3 checkpoint to {}", save_path)
