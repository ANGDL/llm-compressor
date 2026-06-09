from llmcompressor.modeling.step3p5 import CalibrationStep3p5MoEMLP  # noqa: F401

from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import preset_name_to_scheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from datasets import load_dataset

import os
import torch


# Select model and load it.
model_id = "/ssd2/models/Step-3.5-Flash"

model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "./ultrachat_200k"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
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
    moe_calibrate_all_experts=False,
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
