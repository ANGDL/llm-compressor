import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.awq import AWQMapping
import os
import shutil

# select a Mixture of Experts model for quantization

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

MODEL_ID = "/ssd1/models/Qwen3-235B-A22B-Instruct-2507/"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype="auto", trust_remote_code=True, device_map=None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 3072


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
# since the MoE gate layers are sensitive to quantization, we add them to the ignore
# list so they remain at full precision
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
    AWQMapping(
        "re:.*up_proj$",
        ["re:.*down_proj$"],
    ),
]

# Recipe
recipe = [
    AutoSmoothModifier(activation_scale_type="max", norm_func='adaptive', mappings=mapping),
    GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=["lm_head", "re:.*mlp.gate$"],
        offload_hessians=True,
    ),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    batch_size=64,
    concatenate_data=True,
    moe_calibrate_all_experts=False,
)

# print("========== SAMPLE GENERATION ==============")
# try:
#     dispatch_for_generation(model)
#     sample = tokenizer("Hello my name is", return_tensors="pt")
#     sample = {key: value.to(model.device) for key, value in sample.items()}
#     output = model.generate(**sample, max_new_tokens=100)
#     print(tokenizer.decode(output[0]))
#     print("==========================================")
# except Exception as e:
#     print(f"Failed to generate sample: {e}")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-w8a8-smooth-gptq"
SAVE_DIR = os.path.join("/ssd1/models", SAVE_DIR)

try:
    model.save_pretrained(SAVE_DIR, save_compressed=True)
except torch.OutOfMemoryError as e:
    print(f"This error is just for accelerator dispatch, and can be ignored if it happens during saving: {e}")

copy_original_non_model_files(MODEL_ID, SAVE_DIR)
