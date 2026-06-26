"""
Qwen3 dense quantization example with KV-cache quantization support.

Configurable via CLI flags so the same script can run W8A8 / W4A8, with or
without AutoSmooth, RTN vs GPTQ, optional IMatrix MSE observer, and optional
KV-cache quantization. The control flow mirrors examples/quantizing_moe/glm5_w8a8.py.

Examples:
    # Default: W8A8 RTN + IMatrix observer, no KV-cache quantization
    python qwen3_dense_int8_chanel_kv_example.py

    # W4A8 GPTQ with KV-cache int8 channelwise
    python qwen3_dense_int8_chanel_kv_example.py \\
        --scheme W4A8 --modifier GPTQ --observer imatrix_mse --kv-cache

    # Mixed: experts W4 + attention W8 (the previous hard-coded behaviour)
    python qwen3_dense_int8_chanel_kv_example.py --scheme mixed-w4-attn-w8
"""

import argparse
import os
from typing import Any, cast

from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationScheme, preset_name_to_scheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    type=str,
    default="/Users/ang/models/Qwen3-0.6B",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="/Users/ang/models/",
)
parser.add_argument(
    "--scheme",
    type=str,
    default="W8A8",
    choices=["W8A8", "W4A8", "mixed-w4-attn-w8"],
    help=(
        "Quantization scheme. 'W8A8' / 'W4A8' apply the preset to every Linear; "
        "'mixed-w4-attn-w8' uses W4 for {gate,up,down}_proj and W8 for self_attn "
        "projections."
    ),
)
parser.add_argument(
    "--modifier",
    type=str,
    default="RTN",
    choices=["RTN", "GPTQ"],
    help="Underlying quantizer: RTN (QuantizationModifier) or GPTQ.",
)
parser.add_argument(
    "--transform",
    type=str,
    default="",
    choices=["", "AutoSmooth"],
    help="Optional pre-quantization transform: empty (default) or AutoSmooth.",
)
parser.add_argument(
    "--observer",
    type=str,
    default="imatrix_mse",
    choices=["", "mse", "imatrix_mse"],
    help=(
        "Observer for weight quantization. 'imatrix_mse' inserts an IMatrixGatherer "
        "before the quantizer; 'mse' just sets the observer on the scheme; '' keeps "
        "the preset default."
    ),
)
parser.add_argument(
    "--kv-cache",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Quantize the KV cache (int8 channelwise, static, symmetric).",
)
parser.add_argument(
    "--dataset_id",
    type=str,
    default="/Users/ang/Downloads/llm-demo/datasets/ultrachat_200k",
)
parser.add_argument("--dataset_split", type=str, default="train_sft")
parser.add_argument(
    "--num_calibration_samples",
    type=positive_int,
    default=16,
    help="Number of calibration samples to load from the dataset.",
)
parser.add_argument(
    "--max_sequence_length",
    type=positive_int,
    default=2048,
    help="Maximum sequence length used for tokenization and calibration.",
)
args = parser.parse_args()


# ---- Load model ---------------------------------------------------------
model_id = args.model_id
# Force eager attention for MPS/GQA stability during quantized calibration.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# ---- Calibration dataset ------------------------------------------------
DATASET_ID = args.dataset_id
DATASET_SPLIT = args.dataset_split
NUM_CALIBRATION_SAMPLES = args.num_calibration_samples
MAX_SEQUENCE_LENGTH = args.max_sequence_length

ds = cast(
    Dataset,
    load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"),
)
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=list(ds.column_names))


# ---- KV-cache quantization config (optional) ----------------------------
kv_cache_args = (
    QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        dynamic=False,
        symmetric=True,
    )
    if args.kv_cache
    else None
)


# ---- Build quantization config groups -----------------------------------
def _build_config_groups(scheme_name: str) -> dict[str, QuantizationScheme]:
    """Resolve --scheme into a config_groups dict consumable by both
    QuantizationModifier and GPTQModifier."""

    if scheme_name in ("W8A8", "W4A8"):
        scheme = preset_name_to_scheme(scheme_name, ["Linear"])
        return {"group_0": scheme}

    if scheme_name == "mixed-w4-attn-w8":
        # Activations: 8-bit, per-token, asymmetric, dynamic
        activations_args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=False,
            dynamic=True,
            observer=None,
        )
        # Experts (gate / up / down): 4-bit channelwise weights
        experts_scheme = QuantizationScheme(
            targets=[r"re:.*(gate_proj|up_proj|down_proj)$"],
            weights=QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.CHANNEL,
                symmetric=True,
                dynamic=False,
            ),
            input_activations=activations_args,
        )
        # Self-attention projections: 8-bit channelwise weights
        attn_scheme = QuantizationScheme(
            targets=[r"re:.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$"],
            weights=QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.CHANNEL,
                symmetric=True,
                dynamic=False,
            ),
            input_activations=activations_args,
        )
        return {"experts_w4": experts_scheme, "other_linear_w8": attn_scheme}

    raise ValueError(f"Unknown scheme: {scheme_name}")


config_groups = _build_config_groups(args.scheme)


# Apply observer to every group's weights (in-place on the QuantizationArgs).
if args.observer in ("mse", "imatrix_mse"):
    for group_name, group_scheme in config_groups.items():
        if group_scheme.weights is None:
            raise RuntimeError(f"Scheme group {group_name!r} missing weights args")
        group_scheme.weights.observer = args.observer


# ---- Build recipe -------------------------------------------------------
ignores = ["lm_head"]
recipe: list[Any] = []
tail_name = f"-{args.scheme}"

if args.transform == "AutoSmooth":
    # AutoSmoothModifier only consumes its own smoothing knobs; config_groups /
    # ignore live on the downstream quantizer (RTN / GPTQ).
    recipe.append(
        AutoSmoothModifier(
            n_grid=5,
            duo_scaling="both",
        )
    )
    tail_name += "-AutoSmooth"

if args.observer == "imatrix_mse":
    recipe.append(IMatrixGatherer(ignore=ignores))
    tail_name += "-IMatrix"
elif args.observer == "mse":
    tail_name += "-MSE"

if args.modifier == "GPTQ":
    recipe.append(
        cast(Any, GPTQModifier)(
            config_groups=config_groups,
            ignore=ignores,
            kv_cache_scheme=kv_cache_args,
        )
    )
    tail_name += "-GPTQ"
elif args.modifier == "RTN":
    recipe.append(
        QuantizationModifier(
            config_groups=config_groups,
            ignore=ignores,
            kv_cache_scheme=kv_cache_args,
        )
    )
    tail_name += "-RTN"

if args.kv_cache:
    tail_name += "-KV"


# ---- Run oneshot --------------------------------------------------------
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed before dispatching for generation.
SAVE_NAME = model_id.rstrip("/").split("/")[-1] + tail_name
SAVE_DIR = os.path.join(args.save_dir, SAVE_NAME)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")
