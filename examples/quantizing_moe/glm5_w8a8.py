import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.glm_moe_dsa import CalibrationGlmMoeDsaMoE  # noqa: F401
from llmcompressor.modifiers.gptq import GPTQModifier

# Load the model
model_id = "/ssd4/models/GLM-5"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map=None)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# MoE calibration is now handled automatically by the pipeline.
# The `CalibrationGlmMoeDsaMoE` modules (from `llmcompressor.modeling.glm_moe_dsa`)
# will be applied during calibration to enable proper expert calibration.
# These permanently unpack the fused 3D expert weights into individual nn.Linear
# layers for quantization target matching and vLLM compatibility.

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 32
MAX_SEQUENCE_LENGTH = 8192

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

ignores = [
    # Ignore the output head
    "re:.*mlp.gate$",
    "re:.*embed_tokens$",
    "re:.*indexer.*",
    "lm_head",
]

# Configure the quantization algorithm to run.
#   * quantize the weights to 8 bit with GPTQ per-channel quantization 
recipe = GPTQModifier(
    targets="Linear", 
    scheme="W8A8", 
    ignore=ignores,
#    muti_gpu_compression=True,
    offload_hessians=True,
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    batch_size=1,
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W8A8-GPTQ"
SAVE_DIR = os.path.join("/ssd3/models", SAVE_DIR)

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
