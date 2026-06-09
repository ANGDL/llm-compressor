import os
from llmcompressor import model_free_ptq

MODEL_ID = "/ssd2/models/Qwen2.5-32B-Instruct"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W8A8-INT8"
SAVE_DIR = os.path.join("/ssd2/models", SAVE_DIR)

# Apply W8A8 to the model
# Once quantized, the model is saved
# using compressed-tensors to the SAVE_DIR.
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="W8A8",
    ignore=[
        "lm_head",
    ],
    max_workers=15,
    device="cuda:0",
)
