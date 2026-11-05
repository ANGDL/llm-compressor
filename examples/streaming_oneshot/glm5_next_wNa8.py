"""Stream GLM-5.3-Flash calibration and mixed W4A8/W8A8 RTN quantization.

This example uses the out-of-core :func:`streaming_oneshot` frontend. Packed
routed experts are linearized by the frontend before recipe targets are resolved.
Routed expert projections use W4A8, while selected attention and dense/shared MLP
projections use W8A8.
KDA/indexer parameters, norms, embeddings, heads, and the visual branch remain
in their source dtype. The text config must provide a non-empty
``linear_attn_config.kda_layers`` layout with valid layer indices; missing or
invalid layouts fail before calibration data is loaded.

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

from llmcompressor.datasets import (
    build_calibration_messages,
    load_calibration_datasets,
)
from llmcompressor.modeling.glm5_next import attach_mtp_layer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import streaming_oneshot


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


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
    messages = build_calibration_messages(
        example,
        text_examples[index % len(text_examples)] if text_examples else None,
        image_format="url",
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


def _config_value(config, key):
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def build_ignores(config) -> list[str]:
    """Modules excluded from mixed W4A8/W8A8 quantization."""
    text_config = _config_value(config, "text_config")
    if text_config is None:
        text_config = config
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
    return [
        rf"re:.*\.layers\.{kda_layer_pattern}\.self_attn(?:\..*)?$",
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
    ignores = build_ignores(config)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer
    dataset, text_examples = load_calibration_datasets(
        args.dataset_id,
        args.dataset_split,
        args.text_dataset_id,
        args.text_dataset_split,
        args.num_calibration_samples,
        args.text_calibration_samples if args.text_dataset_id else 0,
        args.seed,
    )
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
