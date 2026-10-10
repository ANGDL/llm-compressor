"""Stream GLM-5.3-Flash calibration and mixed W4A8/W8A8 RTN quantization.

This example uses the out-of-core :func:`streaming_oneshot` frontend. Packed
routed experts are linearized by the frontend before recipe targets are resolved.
Routed expert projections use W4A8, while selected attention and dense/shared MLP
projections use W8A8.
KDA/indexer parameters, norms, embeddings, heads, and the visual branch remain
in their source dtype.

Example::

    python examples/streaming_oneshot/glm5_next_wNa8.py \
        --model-id /models/GLM-5.3-Flash-BF16 \
        --dataset-id lmms-lab/flickr30k \
        --text-dataset-id HuggingFaceH4/ultrachat_200k \
        --text-calibration-samples 256 \
        --num-calibration-samples 256 \
        --output-dir /models/GLM-5.3-Flash-WNA8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
)

from datasets import load_dataset
from llmcompressor.modeling.glm5_next import attach_mtp_layer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import streaming_oneshot


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _is_local_json(value: str) -> bool:
    path = Path(value).expanduser()
    return path.is_file() or path.suffix.lower() in {".json", ".jsonl"}


def _load_dataset(source: str, split: str, samples: int):
    if _is_local_json(source):
        return load_dataset("json", data_files=source, split=f"train[:{samples}]")
    return load_dataset(source, split=f"{split}[:{samples}]")


def load_calibration_dataset(
    sources: list[str], split: str, samples: int, seed: int
):
    if samples < len(sources):
        raise ValueError(
            "num-calibration-samples must be at least the number of dataset sources"
        )
    base, remainder = divmod(samples, len(sources))
    parts = []
    for index, source in enumerate(sources):
        count = base + (index < remainder)
        part = _load_dataset(source, split, count).shuffle(seed=seed)
        parts.append(part)
    if len(parts) == 1:
        return parts[0]
    from datasets import concatenate_datasets

    columns = [tuple(part.column_names) for part in parts]
    if any(value != columns[0] for value in columns[1:]):
        raise ValueError(f"Calibration sources have different columns: {columns}")
    return concatenate_datasets(parts)


def _caption_text(example: dict[str, Any]) -> str:
    caption = example.get("caption", "")
    if isinstance(caption, (list, tuple)):
        caption = caption[0] if caption else ""
    return str(caption)


def _image_as_data_url(image: Any) -> str:
    import base64
    from io import BytesIO

    buffered = BytesIO()
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"


def _multimodal_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": _image_as_data_url(example["image"])},
                {"type": "text", "text": "What does the image show?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": _caption_text(example)}],
        },
    ]


def _normalize_text_content(content: Any) -> list[dict[str, str]]:
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


def _format_text_example(example: dict[str, Any]) -> list[dict[str, Any]]:
    messages = example.get("messages")
    if messages is None:
        text = example.get("text")
        if text is None:
            raise ValueError("Text calibration records must contain messages or text")
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


def _build_calibration_messages(
    example: dict[str, Any], text_example: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if "image" in example and example["image"] is not None:
        messages = _multimodal_messages(example)
    elif "messages" in example:
        messages = _format_text_example(example)
    elif "text" in example:
        messages = _format_text_example(example)
    elif "input_ids" in example:
        return example
    else:
        raise ValueError(
            "Calibration examples must contain image, messages, text, or input_ids"
        )
    if text_example is not None and "image" in example:
        messages.extend(_format_text_example(text_example))
    return messages


def preprocess_calibration(
    example: dict[str, Any],
    index: int,
    *,
    processor: Any,
    text_examples: list[dict[str, Any]],
    max_sequence_length: int,
) -> dict[str, Any]:
    if "input_ids" in example:
        return example
    messages = _build_calibration_messages(
        example,
        text_examples[index % len(text_examples)] if text_examples else None,
    )
    if "image" in example and example["image"] is not None:
        return dict(
            processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                padding=False,
                max_length=max_sequence_length,
                truncation=True,
            )
        )
    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return dict(
        processor(
            text=text,
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )
    )


def data_collator(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("GLM-5.3 multimodal calibration requires batch_size=1")
    return {
        key: torch.as_tensor(
            value, dtype=torch.bfloat16 if key == "pixel_values" else None
        )
        for key, value in batch[0].items()
        if value is not None
    }


def build_ignores(config) -> list[str]:
    """Modules excluded from mixed W4A8/W8A8 quantization."""
    text_config = getattr(config, "text_config", config)
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    kda_layers = getattr(linear_attn_config, "kda_layers", ())
    kda = (
        [
            r"re:.*\.layers\.(?:"
            + "|".join(str(index) for index in sorted(set(kda_layers)))
            + r")\.self_attn(?:\..*)?$"
        ]
        if kda_layers
        else []
    )
    return kda + [
        "lm_head",
        r"re:.*(?:^|\.)embed_tokens$",
        r"re:.*(?:^|\.)visual(?:\..*)?$",
        r"re:.*(?:^|\.)norm$",
        r"re:.*(?:^|\.)(?:input_layernorm|post_attention_layernorm)$",
        r"re:.*(?:^|\.)(?:attn_hc|ffn_hc|hyper_connection|mapping_proj|router)(?:\..*)?$",
        r"re:.*\.self_attn\.(?:indexer|kv_a_layernorm|kv_b_proj|q_a_layernorm)(?:\..*)?$",
        r"re:.*\.self_attn\.(?:f_a_proj|f_b_proj)$",
        r"re:.*\.mlp\.gate(?:\..*)?$",
        r"re:.*\.(?:enorm|hnorm|eh_proj|shared_head\.norm)$",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--dataset-id", nargs="+", default=["lmms-lab/flickr30k"])
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument(
        "--text-dataset-id", default="HuggingFaceH4/ultrachat_200k"
    )
    parser.add_argument("--text-dataset-split", default="train_sft")
    parser.add_argument("--text-calibration-samples", type=int, default=0)
    parser.add_argument("--num-calibration-samples", type=positive_int, default=32)
    parser.add_argument("--max-sequence-length", type=positive_int, default=4096)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--use-float32-scale-dtype", action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--shared-experts-bits",
        type=int,
        choices=(4, 8),
        default=8,
        help="Weight bits for mlp.shared_experts projections (default: 8).",
    )
    parser.add_argument(
        "--moe-calibrate-all-experts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--checkpoint-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--overwrite-output",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("GLM-5.3 multimodal calibration requires --batch-size 1")
    if args.text_calibration_samples < 0:
        parser.error("--text-calibration-samples must be non-negative")

    config = AutoConfig.from_pretrained(args.model_id, local_files_only=True)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer
    dataset = load_calibration_dataset(
        args.dataset_id,
        args.dataset_split,
        args.num_calibration_samples,
        args.seed,
    )
    text_examples = []
    if args.text_dataset_id and args.text_calibration_samples:
        text_dataset = load_calibration_dataset(
            [args.text_dataset_id],
            args.text_dataset_split,
            args.text_calibration_samples,
            args.seed,
        )
        text_examples = list(text_dataset)
    dataset = dataset.map(
        preprocess_calibration,
        with_indices=True,
        fn_kwargs={
            "processor": processor,
            "text_examples": text_examples,
            "max_sequence_length": args.max_sequence_length,
        },
        remove_columns=dataset.column_names,
    )
    calibration_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )
    scale_dtype = torch.float32 if args.use_float32_scale_dtype else None
    weights_args_4 = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
        observer="imatrix_mse",
        scale_dtype=scale_dtype,
    )
    weights_args_8 = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
        observer="imatrix_mse",
        scale_dtype=scale_dtype,
    )
    activation_args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
        observer=None,
        scale_dtype=torch.float32,
    )
    routed_expert_targets = [
        r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
    ]
    shared_expert_target = (
        r"re:.*\.mlp\.shared_experts(?:\..+)?\.(gate_proj|up_proj|down_proj)$"
    )
    if args.shared_experts_bits == 4:
        routed_expert_targets.append(shared_expert_target)
    other_targets = [
        r"re:.*\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$",
        r"re:.*\.mlp\.(gate_proj|up_proj|down_proj)$",
    ]
    if args.shared_experts_bits == 8:
        other_targets.append(shared_expert_target)
    schemes = {
        "experts_w4a8": QuantizationScheme(
            targets=routed_expert_targets,
            weights=weights_args_4,
            input_activations=activation_args,
        ),
        "other_w8a8": QuantizationScheme(
            targets=other_targets,
            weights=weights_args_8,
            input_activations=activation_args,
        ),
    }

    ignores = build_ignores(config)
    recipe = [
        IMatrixGatherer(ignore=ignores),
        QuantizationModifier(config_groups=schemes, ignore=ignores),
    ]

    def model_factory(model_config):
        # GLM5 Next is registered as a vision-language model in Transformers;
        # AutoModelForCausalLM does not know this configuration class.
        model = AutoModelForImageTextToText.from_config(model_config)
        text_config = getattr(model_config, "text_config", model_config)
        mtp_count = getattr(text_config, "num_nextn_predict_layers", 0)
        index_path = args.model_id / "model.safetensors.index.json"
        if mtp_count and index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            prefix = f"model.language_model.layers.{text_config.num_hidden_layers}."
            if any(name.startswith(prefix) for name in index.get("weight_map", {})):
                attach_mtp_layer(model, str(args.model_id))
        return model

    streaming_oneshot(
        model=args.model_id,
        model_config=config,
        model_factory=model_factory,
        dataset=calibration_dataloader,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_sequence_length,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        moe_calibrate_all_experts=args.moe_calibrate_all_experts,
        checkpoint_progress=args.checkpoint_progress,
        overwrite_output=args.overwrite_output,
        pack_to_int8=True,
    )


if __name__ == "__main__":
    main()
