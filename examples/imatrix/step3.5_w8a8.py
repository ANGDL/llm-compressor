from llmcompressor.modeling.step3p5 import CalibrationStep3p5MoEMLP  # noqa: F401

from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer

from datasets import load_dataset

import argparse
import os
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--quantize-only-moe", action="store_true")
parser.add_argument("--num-calibration-samples", type=int, default=512)
parser.add_argument("--max-sequence-length", type=int, default=2048)
parser.add_argument("--moe-calibrate-all-experts", action="store_true")
args = parser.parse_args()


# Select model and load it.
model_id = "/ssd2/models/Step-3.5-Flash"

model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "./ultrachat_200k"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = args.num_calibration_samples
MAX_SEQUENCE_LENGTH = args.max_sequence_length
DATASET_SPLIT = "train_sft"


# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
#   * trigger a calibration pass with IMatrixGatherer so the observer can collect E[x²]
#   * quantize the weights to 4 bit with group size 128
#   * use imatrix_mse observer to weight quantization error by channel importance
scheme = preset_name_to_scheme("W8A8", ["Linear"])
if scheme.weights is None:
    raise ValueError("W8A8 scheme is missing weight quantization settings")
scheme.weights.observer = "imatrix_mse"

if args.quantize_only_moe:
    ignores = [
        # Global exclusions
        "lm_head",
        "model.embed_tokens",
        "model.norm",
        # Keep all attention projections in BF16
        "re:.*self_attn\\.(g_proj|q_proj|k_proj|v_proj|o_proj)$",
        # Keep all dense MLP projections in BF16 by default
        "re:.*mlp\\.(gate_proj|up_proj|down_proj)$",
        # Keep MoE router gate in BF16; only expert projections are quantized
        "re:.*moe\\.gate$",
        # Keep shared expert projections in BF16
        "re:.*share_expert\\.(gate_proj|up_proj|down_proj)$",
    ]
else:
    ignores = [
        "lm_head",
        "re:.*moe.gate$",
        "re:.*transformer\\.shared_head\\.output$",
        "re:.*eh_proj$",
    ]

recipe = [
    IMatrixGatherer(ignore=ignores),
    QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=ignores,
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    moe_calibrate_all_experts=args.moe_calibrate_all_experts,
)

# Confirm generations of the quantized model look sane.
# print("\n\n")
# print("========== SAMPLE GENERATION ==============")
# dispatch_model(model)
# sample = tokenizer("Hello my name is", return_tensors="pt")
# sample = {key: value.to(model.device) for key, value in sample.items()}
# output = model.generate(**sample, max_new_tokens=100)
# print(tokenizer.decode(output[0]))
# print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W8A8"
SAVE_DIR = os.path.join("/ssd2/models", SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
try:
    model.save_pretrained(SAVE_DIR, save_compressed=True, max_shard_size='10GB')
except torch.OutOfMemoryError as e:
    print(f"This error is only for P800, and can be ignored if it happens during saving: {e}")

for i, mtp_id in enumerate(["45", "46", "47"]):
    mtp_prefix = f"model.layers.{mtp_id}"
    save_mtp_tensors_to_checkpoint(
        source_model=model_id,
        dest_dir=SAVE_DIR,
        mtp_prefix=mtp_prefix,
        shard_name=f"model_mtp_{i}.safetensors",
    )
