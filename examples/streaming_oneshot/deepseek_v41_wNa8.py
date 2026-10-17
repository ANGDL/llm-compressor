"""Quantize original V4.1 checkpoints with native CSA2, Engram and DSpark.

Example:
    python examples/streaming_oneshot/deepseek_v41_wNa8.py \
        --model-id /path/to/DeepSeek-V4.1-Flash \
        --dataset-id /path/to/calibration.jsonl \
        --quant-mode wna8 --output-dir /path/to/DeepSeek-V4.1-Flash-WNa8

This first workflow calibrates text inputs. Vision/aligner receive no extra
PTQ; Engram lookup tables are preserved byte-for-byte. DSpark is
teacher forced explicitly, including every draft expert and both draft heads.
"""

import argparse
import importlib.util
import json
from functools import partial
from pathlib import Path

from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import QuantizationArgs
from transformers import AutoTokenizer

from datasets import concatenate_datasets, load_dataset
from llmcompressor.modeling.deepseekv41 import (
    DeepseekV41NativeForCausalLM,
    ModelConfig,
)
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import DeepSeekV41WeightMaterializer, streaming_oneshot


def _schemes(mode):
    routed = r"re:model\.(layers|mtp)\.\d+\.ffn\.experts\.\d+\.(w1|w2|w3)$"
    other = [
        r"re:model\.(layers|mtp)\.\d+\.attn\.(wq_a|wq_b|wkv|wo_b)$",
        r"re:model\.layers\.\d+\.attn\.indexer\.wq_b$",
        r"re:model\.(layers|mtp)\.\d+\.ffn\.shared_experts\.(w1|w2|w3)$",
        r"re:model\.mtp\.0\.main_proj$",
    ]

    def scheme(bits, targets):
        return QuantizationScheme(
            targets=targets,
            weights=QuantizationArgs(
                num_bits=bits,
                type="int",
                strategy="channel",
                symmetric=True,
                observer="imatrix_mse",
            ),
            input_activations=QuantizationArgs(
                num_bits=8,
                type="int",
                strategy="token",
                symmetric=False,
                dynamic=True,
            ),
        )

    if mode == "wna8":
        groups = {
            "routed_w4a8": scheme(4, [routed]),
            "other_w8a8": scheme(8, other),
        }
    else:
        groups = {"all": scheme(8 if mode == "w8a8" else 4, [routed, *other])}
    ignores = [
        "model.lm_head",
        r"re:.*\.embed$",
        r"re:model\.(vision|aligner)(\..*)?$",
        r"re:.*\.engram(\..*)?$",
        r"re:.*\.ffn\.gate$",
        r"re:.*\.attn\.wo_a$",
        r"re:.*\.attn\.compressor\.(wkv|wgate)$",
        r"re:.*\.attn\.indexer\.(wk|weights_proj)$",
        r"re:.*\.(markov_head|confidence_head)(\..*)?$",
    ]
    return groups, ignores


def _load_dataset(args, tokenizer):
    encoding_path = args.model_id / "encoding" / "encoding.py"
    encode = None
    if encoding_path.is_file():
        spec = importlib.util.spec_from_file_location("dsv41_encoding", encoding_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        encode = module.encode_messages

    def prepare(example):
        if "input_ids" in example:
            return example
        if "messages" in example:
            if encode is not None:
                text = encode(
                    example["messages"],
                    thinking_mode="thinking",
                    reasoning_effort=args.reasoning_effort,
                )
            else:
                text = tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    reasoning_effort=args.reasoning_effort,
                )
            return {"text": text}
        if "text" not in example:
            raise ValueError("Calibration rows need messages, text or input_ids")
        return {"text": example["text"]}

    if args.num_calibration_samples < len(args.dataset_id):
        raise ValueError("Need at least one calibration sample per dataset")
    parts = []
    count, extra = divmod(args.num_calibration_samples, len(args.dataset_id))
    for index, source in enumerate(args.dataset_id):
        samples = count + (index < extra)
        if Path(source).is_file():
            dataset = load_dataset(
                "json", data_files=source, split=f"train[:{samples}]"
            )
        else:
            dataset = load_dataset(source, split=f"{args.dataset_split}[:{samples}]")
        parts.append(dataset.map(prepare, remove_columns=dataset.column_names))
    return concatenate_datasets(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--dataset-id", nargs="+", required=True)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--quant-mode", choices=["w8a8", "w4a8", "wna8"], default="wna8"
    )
    parser.add_argument("--num-calibration-samples", type=int, default=128)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", help="Execution device, e.g. cpu or cuda:0")
    parser.add_argument("--reasoning-effort", type=int, default=75)
    parser.add_argument("--moe-calibrate-all-experts", action="store_true")
    parser.add_argument("--checkpoint-progress", action="store_true")
    parser.add_argument("--async-save", action="store_true")
    args = parser.parse_args()
    if (
        min(args.num_calibration_samples, args.max_sequence_length, args.batch_size)
        <= 0
    ):
        parser.error("Sample count, sequence length and batch size must be positive")
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
    groups, ignores = _schemes(args.quant_mode)
    factory = partial(
        DeepseekV41NativeForCausalLM,
        tokenizer=tokenizer,
        engram_checkpoint=args.model_id,
    )
    result = streaming_oneshot(
        model=args.model_id,
        model_config=config,
        model_factory=factory,
        dataset=_load_dataset(args, tokenizer),
        tokenizer=tokenizer,
        recipe=[
            IMatrixGatherer(ignore=ignores),
            QuantizationModifier(config_groups=groups, ignore=ignores),
        ],
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_sequence_length,
        batch_size=args.batch_size,
        device=args.device,
        moe_calibrate_all_experts=args.moe_calibrate_all_experts,
        materializer=DeepSeekV41WeightMaterializer(source_has_mtp=has_mtp),
        checkpoint_progress=args.checkpoint_progress,
        async_save=args.async_save,
        pack_to_int8=True,
    )
    tokenizer.save_pretrained(result)
    print(f"Saved V4.1 {args.quant_mode.upper()} checkpoint to {result}")


if __name__ == "__main__":
    main()
