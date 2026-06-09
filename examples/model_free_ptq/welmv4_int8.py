import os
from llmcompressor import model_free_ptq

MODEL_ID = "/ssd4/models/welmv4"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W8A8-INT8"
SAVE_DIR = os.path.join("/ssd4/models", SAVE_DIR)

# Apply W8A8 to the model
# Once quantized, the model is saved
# using compressed-tensors to the SAVE_DIR.
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="W8A8",
    ignore=[
        "lm_head",
        "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate.*",
        "re:.*norm.*",
        "re:.*embed_tokens.*",
        "re:.*self_attn\\.gate_proj$",
        "re:.*linear_attn.*",
        "re:.*embed",
        "re:.*oe_up_proj$",
        "score",
    ],
    max_workers=15,
    device="cuda:0",
)
