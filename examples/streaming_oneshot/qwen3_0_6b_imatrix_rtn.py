"""Quantize Qwen3-0.6B with out-of-core W8A8 iMatrix RTN.

The model checkpoint must be local and use safetensors. The calibration dataset
may be a local Hugging Face dataset directory or a Hub dataset name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.quantization import preset_name_to_scheme
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from llmcompressor.streaming import build_meta_model, streaming_oneshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset-split", default="train_sft")
    parser.add_argument("--output-dir", default="Qwen3-0.6B-W8A8-IMatrix-RTN")
    parser.add_argument("--work-dir", default="streaming-work-qwen3-0.6b")
    parser.add_argument("--num-calibration-samples", type=int, default=16)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps:0"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
    )
    return parser.parse_args()


def _load_calibration_batches(args, tokenizer, config, device):
    dataset = load_dataset(
        args.dataset,
        split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
    ).shuffle(seed=42)
    tokenized_batches = []
    digest = hashlib.sha256()
    for sample in dataset:
        text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            max_length=args.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        tokenized_batches.append(encoded)
        digest.update(json.dumps(input_ids.cpu().tolist()).encode())

    # Bootstrap the boundary before decoder layer zero without loading the full
    # model. Only the embedding table is read from the source checkpoint.
    checkpoint = Path(args.model) / "model.safetensors"
    with safe_open(checkpoint, framework="pt", device=str(device)) as file:
        embedding_weight = file.get_tensor("model.embed_tokens.weight").to(
            torch.bfloat16
        )
    rotary = Qwen3RotaryEmbedding(config, device=device)
    batches = []
    with torch.no_grad():
        for encoded in tokenized_batches:
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            position_ids = torch.arange(
                input_ids.shape[1], device=device
            ).unsqueeze(0)
            hidden_states = torch.nn.functional.embedding(
                input_ids, embedding_weight
            )
            position_embeddings = rotary(hidden_states, position_ids)
            causal_mask = create_causal_mask(
                config=config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            batches.append(
                {
                    "hidden_states": hidden_states.cpu(),
                    "attention_mask": (
                        causal_mask.cpu() if causal_mask is not None else None
                    ),
                    "position_ids": position_ids.cpu(),
                    "position_embeddings": tuple(
                        tensor.cpu() for tensor in position_embeddings
                    ),
                    "use_cache": False,
                }
            )
    del embedding_weight, rotary
    return batches, digest.hexdigest()


def _forward_layer(layer: torch.nn.Module, value: dict[str, Any]):
    hidden_states = layer(**value)
    if not isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states[0]
    return {**value, "hidden_states": hidden_states}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    calibration_batches, fingerprint = _load_calibration_batches(
        args, tokenizer, config, device
    )

    meta_model = build_meta_model(
        AutoModelForCausalLM.from_config, config, attn_implementation="eager"
    )
    targets = tuple(
        f"model.layers.{index}" for index in range(config.num_hidden_layers)
    )
    module_names = tuple(
        name
        for name, module in meta_model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.startswith("model.layers.")
    )
    base_scheme = preset_name_to_scheme("W8A8", ["Linear"])
    assert base_scheme.weights is not None
    base_scheme.weights.observer = "imatrix_mse"
    schemes = {}
    for name in module_names:
        scheme = deepcopy(base_scheme)
        scheme.targets = [name]
        schemes[name] = scheme
    del meta_model

    output = streaming_oneshot(
        model_factory=AutoModelForCausalLM.from_config,
        model_args=(config,),
        model_kwargs={"attn_implementation": "eager"},
        checkpoint=args.model,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        calibration_batches=calibration_batches,
        targets=targets,
        recipe={
            "IMatrixGatherer": {"ignore": ["lm_head"]},
            "QuantizationModifier": {"scheme": "W8A8"},
        },
        dataset_fingerprint=fingerprint,
        schemes=schemes,
        algorithms=("imatrix",),
        use_gptq=False,
        device=device,
        target_dtype=torch.bfloat16,
        forward_target=_forward_layer,
        num_samples=args.num_calibration_samples,
        max_seq_length=args.max_sequence_length,
        seed=42,
    )
    tokenizer.save_pretrained(output)
    print(f"Published compressed checkpoint to {Path(output).resolve()}")


if __name__ == "__main__":
    main()
