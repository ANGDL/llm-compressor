"""Stream DeepSeek-V4 calibration and W8A8/WNA8/W4A8 RTN quantization.

The source checkpoint may be the native FP8+FP4 format or an ordinary BF16
checkpoint. ``DeepSeekV4WeightMaterializer`` decodes the former on demand and
casts quantized execution weights to BF16 while one traced subgraph is resident;
ordinary checkpoint state keeps its source dtype. Calibration boundaries remain
in memory by default; ``--checkpoint-progress`` enables the optional durable
recovery path. Output uses the raw DeepSeek checkpoint naming expected by
SGLang and vLLM unless ``--no-save-raw-checkpoint-format`` is set.

Example::

    python examples/streaming_oneshot/deepseek_v4_wNa8.py \
        --model-id /Users/ang/models/DeepSeek-V4-Pro-Tiny-bf16 \
        --dataset-id /Users/ang/Downloads/llm-demo/datasets/ultrachat_200k \
        --quant-mode w4a8 \
        --reference-kernel \
        /Users/ang/models/DeepSeek-V4-Flash-0731/inference/kernel.py \
        --pack-to-int8 \
        --output-dir /Users/ang/models/DeepSeek-V4-Pro-Tiny-w4a8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from transformers import AutoTokenizer

# Importing the native package registers DeepSeek-V4 with Transformers.
import llmcompressor.modeling.deepseekv4  # noqa: F401
from datasets import Dataset, concatenate_datasets, load_dataset
from llmcompressor.modeling.deepseekv4.config import ModelConfig
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import DeepSeekV4WeightMaterializer, streaming_oneshot
from llmcompressor.utils import ImatrixFallbackStats


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _configure_reference_kernel(path: Path | None) -> None:
    """Override the reference-kernel environment with the CLI-selected path."""
    if path is None:
        return
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference TileLang kernel does not exist: {path}")
    os.environ["DEEPSEEK_V4_KERNEL_BACKEND"] = "reference"
    os.environ["DEEPSEEK_V4_REFERENCE_KERNEL_PATH"] = str(path)
    # The package is imported before argument parsing; clear the lazy loader so
    # a stale path from the parent environment cannot override this argument.
    from llmcompressor.modeling.deepseekv4.kernels import reset_kernel_backend_cache

    reset_kernel_backend_cache()


def _is_local_json(path: str) -> bool:
    candidate = Path(path).expanduser()
    return candidate.is_file() or candidate.suffix.lower() in {".json", ".jsonl"}


def _encode_example(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize chat/text examples to the schema expected by the tokenizer."""
    if "messages" in example:
        from llmcompressor.modeling.deepseekv4.encoding.encoding_dsv4 import (
            encode_messages,
        )

        return {
            "text": encode_messages(example["messages"], thinking_mode="thinking")
        }
    if "text" in example:
        return {"text": example["text"]}
    if "input_ids" in example:
        return example
    raise ValueError(
        "Calibration examples must contain 'messages', 'text', or 'input_ids'"
    )


def _load_source(source_id: str, split: str, samples: int) -> Dataset:
    if _is_local_json(source_id):
        return load_dataset(
            "json", data_files=source_id, split=f"train[:{samples}]"
        )
    return load_dataset(source_id, split=f"{split}[:{samples}]")


def load_calibration_dataset(
    source_ids: list[str], split: str, samples: int, seed: int = 42
) -> Dataset:
    """Load sources, allocate the requested total, and normalize their schemas."""
    if samples < len(source_ids):
        raise ValueError(
            "num-calibration-samples must be at least the number of dataset "
            f"sources ({samples} < {len(source_ids)})"
        )
    base, remainder = divmod(samples, len(source_ids))
    parts = []
    for index, source_id in enumerate(source_ids):
        count = base + (index < remainder)
        part = _load_source(source_id, split, count).shuffle(seed=seed)
        columns = set(part.column_names)
        if "input_ids" not in columns:
            part = part.map(_encode_example, remove_columns=part.column_names)
        parts.append(part)
    if len(parts) == 1:
        return parts[0]
    schemas = [tuple(part.column_names) for part in parts]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError(
            "All calibration sources must produce the same columns; "
            f"got {schemas}"
        )
    return concatenate_datasets(parts)


_WO_A_PATTERN = r"re:.*attn\.wo_a$"


def _ignores(*, quantize_wo_a: bool = False) -> list[str]:
    ignores = [
        "lm_head",
        r"re:.*embed$",
        r"re:.*ffn\.gate$",
        r"re:.*attn\.compressor\.wgate$",
        r"re:.*attn\.compressor\.wkv$",
        r"re:.*attn\.indexer\.compressor\.wgate$",
        r"re:.*attn\.indexer\.compressor\.wkv$",
        r"re:.*attn\.indexer\.weights_proj$",
        r"re:.*confidence_head\.proj$",
    ]
    if not quantize_wo_a:
        ignores.append(_WO_A_PATTERN)
    return ignores


def _int_scheme(
    num_bits: int,
    scale_dtype: torch.dtype | None = None,
) -> QuantizationScheme:
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=num_bits,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
            observer="imatrix_mse",
            scale_dtype=scale_dtype,
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=False,
            dynamic=True,
            observer=None,
        ),
    )


def _schemes(
    quant_mode: str,
    *,
    dspark: bool = False,
    quantize_wo_a: bool = False,
    scale_dtype: torch.dtype | None = None,
):
    ignores = _ignores(quantize_wo_a=quantize_wo_a)
    if quant_mode == "w8a8":
        scheme = preset_name_to_scheme("W8A8", ["Linear"])
        if scheme.weights is None:
            raise RuntimeError("W8A8 preset is missing weight settings")
        scheme.weights.observer = "imatrix_mse"
        scheme.weights.scale_dtype = scale_dtype
        return {"group_0": scheme}, ignores
    if quant_mode == "w4a8":
        return {"group_0": _int_scheme(4, scale_dtype)}, ignores
    if quant_mode == "wna8":
        experts = _int_scheme(4, scale_dtype)
        experts.targets = [r"re:.*ffn\.experts\.\d+\.(w1|w2|w3)$"]
        other = _int_scheme(8, scale_dtype)
        other_targets = [
            r"re:.*attn\.(wq_a|wq_b|wkv|wo_b)$",
            r"re:.*attn\.indexer\.wq_b$",
            r"re:.*ffn\.shared_experts\.(w1|w2|w3)$",
        ]
        if quantize_wo_a:
            other_targets.append(_WO_A_PATTERN)
        other_targets.append(
            r"re:.*mtp\.0\.main_proj$"
            if dspark
            else r"re:.*mtp\.\d+\.(e_proj|h_proj)$"
        )
        other.targets = other_targets
        return {"experts_w4a8": experts, "other_w8a8": other}, ignores
    raise ValueError(f"Unknown quantization mode: {quant_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument(
        "--reference-kernel",
        type=Path,
        default=None,
        help=(
            "Reference TileLang kernel.py. Selecting it enables the reference "
            "backend and overrides DEEPSEEK_V4_REFERENCE_KERNEL_PATH."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        nargs="+",
        default=["HuggingFaceH4/ultrachat_200k"],
        help="Hugging Face dataset ID(s) or local JSON/JSONL path(s).",
    )
    parser.add_argument("--dataset-split", default="train_sft")
    parser.add_argument("--num-calibration-samples", type=positive_int, default=32)
    parser.add_argument("--max-sequence-length", type=positive_int, default=2048)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument(
        "--quant-mode",
        choices=("w8a8", "wna8", "w4a8"),
        default="w8a8",
        help="Uniform W8A8, mixed W4A8 experts/W8A8 other, or uniform W4A8.",
    )
    parser.add_argument(
        "--quantize-wo-a",
        "--quantize_wo_a",
        dest="quantize_wo_a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quantize attn.wo_a; wna8 always uses the INT8 scheme for it.",
    )
    parser.add_argument(
        "--use-float32-scale-dtype",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use float32 for weight scale tensors instead of bfloat16.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist boundaries and transactions for crash recovery.",
    )
    parser.add_argument(
        "--async-save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write validated final safetensors shards on a bounded CPU worker.",
    )
    parser.add_argument(
        "--moe-calibrate-all-experts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Send all tokens through every main-model expert. DSpark draft "
            "layers always cover every expert because their short fixed block "
            "cannot provide complete routing coverage."
        ),
    )
    parser.add_argument(
        "--pack-to-int8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pack INT4 weights into INT8 storage in the output checkpoint.",
    )
    parser.add_argument(
        "--save-raw-checkpoint-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use raw DeepSeek tensor names required by SGLang and vLLM.",
    )
    args = parser.parse_args()
    _configure_reference_kernel(args.reference_kernel)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    config = ModelConfig.from_pretrained(args.model_id)
    config.max_batch_size = args.batch_size
    is_dspark = config.is_dspark
    config.max_seq_len = args.max_sequence_length + (
        config.dspark_block_size if is_dspark else 0
    )
    model_factory = None
    if is_dspark:
        from llmcompressor.modeling.deepseekv4.model_dspark import (
            DeepseekV4DSparkForCausalLM,
        )

        model_factory = DeepseekV4DSparkForCausalLM
    dataset = load_calibration_dataset(
        args.dataset_id,
        args.dataset_split,
        args.num_calibration_samples,
    )
    config_groups, ignores = _schemes(
        args.quant_mode,
        dspark=is_dspark,
        quantize_wo_a=args.quantize_wo_a,
        scale_dtype=(torch.float32 if args.use_float32_scale_dtype else None),
    )
    recipe = [
        IMatrixGatherer(ignore=ignores),
        QuantizationModifier(
            config_groups=config_groups,
            ignore=ignores,
        ),
    ]
    imatrix_fallback_stats = ImatrixFallbackStats()
    imatrix_fallback_stats.install_hooks()
    with imatrix_fallback_stats:
        result = streaming_oneshot(
            model=args.model_id,
            model_config=config,
            model_factory=model_factory,
            dataset=dataset,
            tokenizer=tokenizer,
            recipe=recipe,
            output_dir=args.output_dir,
            work_dir=args.work_dir,
            num_calibration_samples=args.num_calibration_samples,
            max_seq_length=args.max_sequence_length,
            batch_size=args.batch_size,
            moe_calibrate_all_experts=args.moe_calibrate_all_experts,
            materializer=DeepSeekV4WeightMaterializer(
                save_raw_checkpoint_format=args.save_raw_checkpoint_format
            ),
            checkpoint_progress=args.checkpoint_progress,
            async_save=args.async_save,
            pack_to_int8=args.pack_to_int8,
            overwrite_output=True,
        )
    tokenizer.save_pretrained(result)
    print(
        f"Saved streaming DeepSeek-V4 {args.quant_mode.upper()} checkpoint to "
        f"{result}"
    )


if __name__ == "__main__":
    main()
