import argparse
import os
import shutil
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.awq import AWQMapping
from llmcompressor.modifiers.quantization import GPTQModifier


MODEL_ARTIFACT_FILES = {
    "config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}
MODEL_ARTIFACT_SUFFIXES = (".safetensors", ".bin")


def copy_original_non_model_files(source_dir, save_dir):
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        if (
            filename in MODEL_ARTIFACT_FILES
            or filename.endswith(MODEL_ARTIFACT_SUFFIXES)
            or not os.path.isfile(source_path)
        ):
            continue
        shutil.copy2(source_path, os.path.join(save_dir, filename))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-id", default="/ssd4/models/Qwen3-235B-A22B-Instruct-2507"
)
parser.add_argument("--output-dir")
parser.add_argument("--dataset-id", default="HuggingFaceH4/ultrachat_200k")
parser.add_argument("--dataset-split", default="train_sft")
parser.add_argument("--num-calibration-samples", type=int, default=128)
parser.add_argument("--max-sequence-length", type=int, default=3072)
args = parser.parse_args()
if args.num_calibration_samples < 1 or args.max_sequence_length < 1:
    parser.error("calibration samples and sequence length must be positive")

model_id = args.model_id
num_calibration_samples = args.num_calibration_samples
max_sequence_length = args.max_sequence_length
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    trust_remote_code=True,
    device_map=None,
    local_files_only=True,
)

dataset_path = Path(args.dataset_id)
if dataset_path.is_dir():
    arrow_files = sorted(dataset_path.rglob("*.arrow"))
    parquet_files = sorted(dataset_path.rglob("*.parquet"))
    if arrow_files:
        dataset = Dataset.from_file(str(arrow_files[0]))
    elif parquet_files:
        dataset = load_dataset(
            "parquet",
            data_files={args.dataset_split: [str(path) for path in parquet_files if args.dataset_split in path.name]},
            split=args.dataset_split,
        )
    else:
        raise FileNotFoundError(
            f"no .arrow or .parquet files found under {dataset_path}"
        )
    dataset = dataset.select(range(min(num_calibration_samples, len(dataset))))
else:
    dataset = load_dataset(
        args.dataset_id,
        split=f"{args.dataset_split}[:{num_calibration_samples}]",
    )
dataset = dataset.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
    }


dataset = dataset.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=False,
    )


dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

mapping = [
    AWQMapping(
        "re:.*input_layernorm$",
        ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
    AWQMapping(
        "re:.*post_attention_layernorm$",
        [
            "re:.*mlp.experts.*.gate_proj$",
            "re:.*mlp.experts.*.up_proj$",
        ],
    ),
]

recipe = [
    AutoSmoothModifier(
        activation_scale_type="max",
        norm_func="adaptive",
        mappings=mapping,
    ),
    GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=["re:.*lm_head", "re:.*mlp.gate$"],
        actorder=None,
        batched_quantization=1,
    ),
]

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=max_sequence_length,
    num_calibration_samples=num_calibration_samples,
    trust_remote_code_model=True,
    batch_size=64,
    concatenate_data=True,
    moe_calibrate_all_experts=False,
    sequential_targets=["Qwen3MoeDecoderLayer"],
    sequential_targets_per_subgraph=1,
)

save_dir = args.output_dir or os.path.join(
    "/data/models", model_id.rstrip("/").split("/")[-1] + "-w8a8-gptq"
)
model.save_pretrained(save_dir, save_compressed=True)
copy_original_non_model_files(model_id, save_dir)
