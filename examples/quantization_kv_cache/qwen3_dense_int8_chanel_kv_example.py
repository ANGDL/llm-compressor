from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization.quant_args import (
  QuantizationArgs,
  QuantizationStrategy,
  QuantizationType,
)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

import os

# Select model and load it.
model_id = "/Users/ang/models/Qwen3-0.6B-Base"
# Force eager attention for MPS/GQA stability during quantized calibration.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
DATASET_ID = "/Users/ang/Downloads/llm-demo/datasets/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 16
MAX_SEQUENCE_LENGTH = 2048

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

kv_cache_args = QuantizationArgs(
  num_bits=8,
  type=QuantizationType.INT,
  strategy=QuantizationStrategy.CHANNEL,
  dynamic=False,
  symmetric=True,
)

# Configure the quantization algorithm to run.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A8",
    ignore=["lm_head"],
    kv_cache_scheme=kv_cache_args,
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed before dispatching for generation.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-int8-kv-channel"
SAVE_DIR = os.path.join("/Users/ang/models/", SAVE_DIR)
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
