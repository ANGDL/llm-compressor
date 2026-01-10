import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.awq import AWQMapping
import os

# select a Mixture of Experts model for quantization
MODEL_ID = "/ssd1/model/Qwen3-Next-80B-A3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype="auto", trust_remote_code=True, device_map=None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# Select calibration dataset.
DATASET_ID = "/data/minmax/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
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
            "re:.*block_sparse_moe.experts.*.w1$",
            "re:.*block_sparse_moe.experts.*.w3$",
            # Balance the router to avoid changing expert routing probabilities when smoothing.
            "re:.*block_sparse_moe.gate$",
        ],
    ),
    AWQMapping(
        "re:.*w3$",
        ["re:.*w2$"],
    ),
]

# Recipe
recipe = [
    # AutoSmoothModifier(activation_scale_type="mean", norm_func='adaptive', mappings=mapping),
    GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=[
            "lm_head",
             "re:.*mlp.gate$",
             "re:.*mlp.shared_expert_gate$",
             "re:.*linear_attn.*",
            ],
        #offload_hessians=True,
    ),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    batch_size=32,
)

print("========== SAMPLE GENERATION ==============")
try:
    dispatch_for_generation(model)
    sample = tokenizer("Hello my name is", return_tensors="pt")
    sample = {key: value.to(model.device) for key, value in sample.items()}
    output = model.generate(**sample, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================")
except Exception as e:
    print(f"Failed to generate sample: {e}")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-w8a8"
SAVE_DIR = os.path.join("/ssd1/model", SAVE_DIR)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)