"""Quantize original V4.1 checkpoints with native CSA2, Engram, DSpark and vision.

Example:
    python examples/streaming_oneshot/deepseek_v41_wNa8.py \
        --model-id /path/to/DeepSeek-V4.1-Flash \
        --dataset-id lmms-lab/flickr30k \
        --text-dataset-id HuggingFaceH4/ultrachat_200k \
        --text-calibration-samples 128 \
        --quant-mode wna8 --output-dir /path/to/DeepSeek-V4.1-Flash-WNa8

The primary calibration dataset accepts image rows or text conversations.
Text sources are normalized before mixing datasets with different metadata.
Each image row is converted
using the DeepSeek-V4.1 image preprocessing contract into ViT patches and
image-span metadata. Optional text rows are appended to the same prompt so
every calibration batch exercises the vision tower, aligner, text backbone and
DSpark. Official FP8 Engram lookup tables are preserved byte-for-byte. BF16
tables are exported in bounded chunks as FP8 E4M3 + UE8M0 row32 scales for
runtime compatibility; q_weight/k_weight keep their original floating precision.
Pass ``--quantize-engram-wkv`` to quantize Engram projections to W8A8
with FP32 weight scales in every quantization mode.
Attention ``wo_a`` projections are quantized with the other attention
projections by default.
Pass ``--no-quantize-wo-a`` to keep them in execution precision instead.
Pass ``--use-float32-scale-dtype`` to use FP32 weight scales for all PTQ groups.
Output tensor names and quantization selectors use the original checkpoint
namespace unless ``--no-save-raw-checkpoint-format`` is set.
"""

import argparse
import base64
import importlib.util
import io
import json
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import QuantizationArgs
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llmcompressor.datasets import (
    build_calibration_messages,
    load_calibration_datasets,
)
from llmcompressor.modeling.deepseekv41 import (
    DeepseekV41NativeForCausalLM,
    ModelConfig,
)
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.streaming import DeepSeekV41WeightMaterializer, streaming_oneshot

_WO_A_PATTERN = r"re:.*\.attn\.wo_a$"


def _schemes(
    mode, *, quantize_wo_a=True, quantize_engram_wkv=False, scale_dtype=None
):
    routed = r"re:model\.(layers|mtp)\.\d+\.ffn\.experts\.\d+\.(w1|w2|w3)$"
    other = [
        r"re:model\.(layers|mtp)\.\d+\.attn\.(wq_a|wq_b|wkv|wo_b)$",
        r"re:model\.layers\.\d+\.attn\.indexer\.wq_b$",
        r"re:model\.(layers|mtp)\.\d+\.ffn\.shared_experts\.(w1|w2|w3)$",
        r"re:model\.mtp\.0\.main_proj$",
    ]
    if quantize_wo_a:
        other.append(_WO_A_PATTERN)

    def scheme(bits, targets, *, scale_dtype=None):
        return QuantizationScheme(
            targets=targets,
            weights=QuantizationArgs(
                num_bits=bits,
                type="int",
                strategy="channel",
                symmetric=True,
                observer="imatrix_mse",
                scale_dtype=scale_dtype,
            ),
            input_activations=QuantizationArgs(
                num_bits=8,
                type="int",
                strategy="token",
                symmetric=True,
                dynamic=True,
            ),
        )

    if mode == "wna8":
        groups = {
            "routed_w4a8": scheme(4, [routed], scale_dtype=scale_dtype),
            "other_w8a8": scheme(8, other, scale_dtype=scale_dtype),
        }
    else:
        groups = {
            "all": scheme(
                8 if mode == "w8a8" else 4, [routed, *other], scale_dtype=scale_dtype
            )
        }
    if quantize_engram_wkv:
        groups["engram_w8a8"] = scheme(
            8,
            [r"re:model\.layers\.\d+\.engram\.wkv$"],
            scale_dtype=torch.float32,
        )
    ignores = [
        "model.lm_head",
        r"re:.*\.embed$",
        r"re:model\.(vision|aligner)(\..*)?$",
        (
            r"re:.*\.engram\.embed(\..*)?$"
            if quantize_engram_wkv
            else r"re:.*\.engram(\..*)?$"
        ),
        r"re:.*\.ffn\.gate$",
        r"re:.*\.attn\.compressor\.(wkv|wgate)$",
        r"re:.*\.attn\.indexer\.(wk|weights_proj)$",
        r"re:.*\.(markov_head|confidence_head)(\..*)?$",
    ]
    if not quantize_wo_a:
        ignores.append(_WO_A_PATTERN)
    return groups, ignores


def _load_encoding(model_id: Path):
    encoding_path = model_id / "encoding" / "encoding.py"
    if not encoding_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("dsv41_encoding", encoding_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_image_processor(model_id: Path):
    """Load the checkpoint's official V4.1 image processor when available."""
    processor_path = model_id / "inference" / "image_processor.py"
    if not processor_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "dsv41_official_image_processor",
        processor_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_calibration_messages(
    example: dict[str, Any],
    text_example: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], bool]:
    if "input_ids" in example:
        return [example], False
    return (
        build_calibration_messages(example, text_example, image_format="image_url"),
        example.get("image") is not None,
    )


def _image_processor_args(
    config: ModelConfig,
    args,
    model_id: Path | None = None,
) -> SimpleNamespace:
    vision = config.vision_config
    inference_values = {}
    if model_id is not None:
        inference_config = model_id / "inference" / "config.json"
        if inference_config.is_file():
            inference_values = json.loads(inference_config.read_text())
    default_min_pixels = 544 * 544 if vision.patch_size >= 14 else 0
    return SimpleNamespace(
        vision_patch_size=vision.patch_size,
        vision_downsample_ratio=vision.downsample_ratio,
        vision_max_n_token=args.vision_max_n_token
        if args.vision_max_n_token is not None
        else getattr(
            config,
            "vision_max_n_token",
            inference_values.get("vision_max_n_token", 1024),
        ),
        vision_min_pixels=args.vision_min_pixels
        if args.vision_min_pixels is not None
        else getattr(
            config,
            "vision_min_pixels",
            inference_values.get("vision_min_pixels", default_min_pixels),
        ),
        vision_max_wh_ratio=getattr(
            config,
            "vision_max_wh_ratio",
            inference_values.get("vision_max_wh_ratio"),
        ),
    )


def _llm_grid(best_height, best_width, patch_size, downsample_ratio):
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def _num_image_tokens(n_llm_h, n_llm_w):
    return n_llm_h * (n_llm_w + 1) + 2


def _solve_resize_ratio(height, width, args):
    ratio = height / width
    max_w_float = math.sqrt((args.vision_max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    cell = args.vision_patch_size * args.vision_downsample_ratio
    if max_w_float < 1.0:
        return (args.vision_max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (args.vision_max_n_token - 3) * cell
    beta = min(
        math.floor(max_w_float) * cell / width,
        math.floor(max_h_float) * cell / height,
    )
    return (
        math.floor(height * beta / args.vision_patch_size) * args.vision_patch_size,
        math.floor(width * beta / args.vision_patch_size) * args.vision_patch_size,
    )


def _image_bytes(record: dict[str, Any]) -> bytes:
    url = record.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        header, _, payload = url.partition(",")
        if ";base64" not in header:
            raise ValueError(f"Unsupported image data URL: {header}")
        return base64.b64decode(payload)
    raise ValueError("DeepSeek V4.1 calibration images must use data URLs")


def _load_image_fallback(image_record: dict[str, Any], args):
    with Image.open(io.BytesIO(_image_bytes(image_record))) as source:
        image = source.convert("RGB")
    patch_size = args.vision_patch_size
    width, height = image.width, image.height
    if (
        args.vision_max_wh_ratio is not None
        and width > height * args.vision_max_wh_ratio
    ):
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    n_llm_h, n_llm_w = _llm_grid(
        best_height,
        best_width,
        patch_size,
        args.vision_downsample_ratio,
    )
    if _num_image_tokens(n_llm_h, n_llm_w) > args.vision_max_n_token:
        best_height, best_width = _solve_resize_ratio(height, width, args)
        n_llm_h, n_llm_w = _llm_grid(
            best_height,
            best_width,
            patch_size,
            args.vision_downsample_ratio,
        )
    n_vit_h, n_vit_w = best_height // patch_size, best_width // patch_size
    if (
        args.vision_max_wh_ratio is not None
        and image.width >= args.vision_max_wh_ratio * image.height
    ):
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32))
    pixels = pixels.permute(2, 0, 1) / 255
    pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        pixels.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def _load_image(image_record: dict[str, Any], args, image_processor=None):
    if image_processor is not None:
        return image_processor.load_image(image_record, args)
    # Tiny fixtures created by tools/create_tiny_dsv41.py intentionally omit
    # the original inference/ directory. Keep a local compatibility path for
    # those fixtures; full checkpoints use the official implementation above.
    return _load_image_fallback(image_record, args)


def _encode_multimodal(
    messages,
    *,
    tokenizer,
    encoding,
    image_processor,
    config,
    image_args,
    max_sequence_length,
    reasoning_effort,
):
    if encoding is None:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            reasoning_effort=reasoning_effort,
        )
        image_records = [
            {
                "type": "image",
                "url": block["image_url"]["url"],
            }
            for message in messages
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "image_url"
        ]
    else:
        prompt, media = encoding.encode_messages(
            messages,
            thinking_mode="thinking",
            reasoning_effort=reasoning_effort,
            return_multi_modal_data=True,
        )
        image_records = media["images"]
    if len(image_records) != 1:
        raise ValueError(
            "DeepSeek V4.1 streaming calibration currently supports exactly "
            "one image per row"
        )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    image_token_id = int(config.image_token_id)
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = _load_image(
        image_records[0],
        image_args,
        image_processor,
    )
    if image_processor is not None:
        image_types = image_processor.image_token_types(n_llm_h, n_llm_w).tolist()
    else:
        image_types = [0]
        image_types.extend(([1] * n_llm_w + [2]) * n_llm_h)
        image_types.append(3)
    expanded_ids, token_types = [], []
    for token_id in token_ids:
        if token_id == image_token_id:
            expanded_ids.extend([image_token_id] * len(image_types))
            token_types.extend(image_types)
        else:
            expanded_ids.append(token_id)
            token_types.append(-1)
    if len(expanded_ids) > max_sequence_length:
        image_end = max(index for index, value in enumerate(token_types) if value >= 0)
        if image_end >= max_sequence_length:
            raise ValueError(
                "max-sequence-length truncates the image span; increase it to "
                f"at least {image_end + 1}"
            )
        expanded_ids = expanded_ids[:max_sequence_length]
        token_types = token_types[:max_sequence_length]
    if not any(value >= 0 for value in token_types):
        raise ValueError("Image prompt lost its image span during truncation")
    token_types = torch.tensor(token_types, dtype=torch.int64)
    return {
        "input_ids": torch.tensor(expanded_ids, dtype=torch.int64),
        "attention_mask": torch.ones(len(expanded_ids), dtype=torch.bool),
        "image_mask": token_types.ge(0),
        "image_token_types": token_types,
        "image_patches": patches,
        "image_height": n_vit_h,
        "image_width": n_vit_w,
    }


def _encode_text(
    messages,
    *,
    example,
    tokenizer,
    encoding,
    max_sequence_length,
    reasoning_effort,
):
    if "input_ids" in example:
        input_ids = example["input_ids"]
    elif encoding is not None:
        input_ids = tokenizer.encode(
            encoding.encode_messages(
                messages,
                thinking_mode="thinking",
                reasoning_effort=reasoning_effort,
            ),
            add_special_tokens=False,
        )
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            reasoning_effort=reasoning_effort,
        )
        input_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = list(input_ids)[:max_sequence_length]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.bool),
    }


def preprocess_calibration(
    example: dict[str, Any],
    index: int,
    *,
    tokenizer,
    encoding,
    image_processor,
    config,
    image_args,
    text_examples,
    max_sequence_length,
    reasoning_effort,
):
    text_example = text_examples[index % len(text_examples)] if text_examples else None
    messages, has_image = _build_calibration_messages(example, text_example)
    if has_image:
        return _encode_multimodal(
            messages,
            tokenizer=tokenizer,
            encoding=encoding,
            image_processor=image_processor,
            config=config,
            image_args=image_args,
            max_sequence_length=max_sequence_length,
            reasoning_effort=reasoning_effort,
        )
    return _encode_text(
        messages,
        example=example,
        tokenizer=tokenizer,
        encoding=encoding,
        max_sequence_length=max_sequence_length,
        reasoning_effort=reasoning_effort,
    )


def data_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("DeepSeek V4.1 multimodal calibration requires batch_size=1")
    result = {}
    for key, value in batch[0].items():
        if value is None:
            continue
        if key in {"image_height", "image_width"}:
            result[key] = int(value)
            continue
        if key in {"attention_mask", "image_mask"}:
            tensor = torch.as_tensor(value, dtype=torch.bool)
        elif key in {"input_ids", "image_token_types"}:
            tensor = torch.as_tensor(value, dtype=torch.int64)
        elif key == "image_patches":
            tensor = torch.as_tensor(value, dtype=torch.bfloat16)
        else:
            tensor = torch.as_tensor(value)
        result[key] = tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
    return result


def _load_dataset(args, tokenizer, config):
    encoding = _load_encoding(args.model_id)
    image_processor = _load_image_processor(args.model_id)
    image_args = _image_processor_args(config, args, args.model_id)
    multimodal, text_examples = load_calibration_datasets(
        args.dataset_id,
        args.dataset_split,
        args.text_dataset_id,
        args.text_dataset_split,
        args.num_calibration_samples,
        args.text_calibration_samples,
        args.seed,
    )
    processed = multimodal.map(
        preprocess_calibration,
        with_indices=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "encoding": encoding,
            "image_processor": image_processor,
            "config": config,
            "image_args": image_args,
            "text_examples": text_examples,
            "max_sequence_length": args.max_sequence_length,
            "reasoning_effort": args.reasoning_effort,
        },
        remove_columns=multimodal.column_names,
    )
    return DataLoader(
        processed,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--dataset-id", nargs="+", required=True)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--text-dataset-id", nargs="+", default=[])
    parser.add_argument("--text-dataset-split", default="train_sft")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--quant-mode", choices=["w8a8", "w4a8", "wna8"], default="wna8"
    )
    parser.add_argument(
        "--quantize-engram-wkv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quantize Engram wkv to W8A8 with FP32 weight scales in every mode.",
    )
    parser.add_argument(
        "--quantize-wo-a",
        "--quantize_wo_a",
        dest="quantize_wo_a",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize attn.wo_a with the other attention projections "
        "(default: enabled).",
    )
    parser.add_argument(
        "--use-float32-scale-dtype",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use float32 for all PTQ weight scale tensors instead of bfloat16.",
    )
    parser.add_argument("--num-calibration-samples", type=int, default=128)
    parser.add_argument("--text-calibration-samples", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", help="Execution device, e.g. cpu or cuda:0")
    parser.add_argument("--reasoning-effort", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vision-min-pixels",
        type=int,
        default=None,
        help="Override the official minimum image area before patching.",
    )
    parser.add_argument(
        "--vision-max-n-token",
        type=int,
        default=None,
        help="Override the maximum number of image tokens.",
    )
    parser.add_argument("--moe-calibrate-all-experts", action="store_true")
    parser.add_argument("--checkpoint-progress", action="store_true")
    parser.add_argument(
        "--async-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlap checkpoint writes with the next layer (default: enabled).",
    )
    parser.add_argument(
        "--save-raw-checkpoint-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use raw DeepSeek tensor names and quantization targets in the output.",
    )
    args = parser.parse_args()
    if (
        min(args.num_calibration_samples, args.max_sequence_length, args.batch_size)
        <= 0
    ):
        parser.error("Sample count, sequence length and batch size must be positive")
    if args.text_calibration_samples < 0:
        parser.error("--text-calibration-samples must be non-negative")
    if args.batch_size != 1:
        parser.error("DeepSeek V4.1 multimodal calibration requires --batch-size 1")
    if not 1 <= args.reasoning_effort <= 100:
        parser.error("reasoning-effort must be within 1..100")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    config = ModelConfig.from_pretrained(args.model_id, local_files_only=True)
    text = config.text_config
    names = json.loads((args.model_id / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    has_mtp = any(name.startswith(("mtp.", "model.mtp.")) for name in names)
    if not has_mtp:
        text.num_nextn_predict_layers = 0
        text.dspark_block_size = 0
        text.dspark_target_layer_ids = []
        text.compress_ratios = text.compress_ratios[: text.n_layers]
    text.max_seq_len = max(
        args.max_sequence_length,
        text.window_size + text.dspark_block_size,
    )
    text.max_batch_size = args.batch_size
    groups, ignores = _schemes(
        args.quant_mode,
        quantize_wo_a=args.quantize_wo_a,
        quantize_engram_wkv=args.quantize_engram_wkv,
        scale_dtype=torch.float32 if args.use_float32_scale_dtype else None,
    )
    factory = partial(
        DeepseekV41NativeForCausalLM,
        tokenizer=tokenizer,
        engram_checkpoint=args.model_id,
    )
    result = streaming_oneshot(
        model=args.model_id,
        model_config=config,
        model_factory=factory,
        dataset=_load_dataset(args, tokenizer, config),
        tokenizer=tokenizer,
        recipe=[
            QuantizationModifier(config_groups=groups, ignore=ignores),
        ],
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_sequence_length,
        batch_size=args.batch_size,
        shuffle_calibration_samples=False,
        device=args.device,
        moe_calibrate_all_experts=args.moe_calibrate_all_experts,
        materializer=DeepSeekV41WeightMaterializer(
            save_raw_checkpoint_format=args.save_raw_checkpoint_format,
            source_has_mtp=has_mtp,
        ),
        checkpoint_progress=args.checkpoint_progress,
        async_save=args.async_save,
        pack_to_int8=True,
    )
    tokenizer.save_pretrained(result)
    print(f"Saved V4.1 {args.quant_mode.upper()} checkpoint to {result}")


if __name__ == "__main__":
    main()
