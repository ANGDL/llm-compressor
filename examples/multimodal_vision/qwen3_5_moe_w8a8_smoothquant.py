import base64
import argparse
from io import BytesIO
import os

os.environ["DISABLE_OFFSET_NORM_CALIBRATION"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
from qwen_vl_utils import process_vision_info

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.awq import AWQMapping
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load model.
model_id = "/data/models/Qwen3.5-35B-A3B"
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(model_id, device_map=None, dtype="auto",local_files_only=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot argume32
NUM_CALIBRATION_SAMPLES = 64
MAX_SEQUENCE_LENGTH = 8192

DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = f"test[:{NUM_CALIBRATION_SAMPLES}]"
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-smoothquant"
SAVE_DIR = os.path.join("/data/models", SAVE_DIR)

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)


# Apply chat template and tokenize inputs.
def preprocess_and_tokenize(example):
    # preprocess
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base64_qwen},
                {"type": "text", "text": "What does the image show?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                "text": f"{example['caption'][0]}",
                "type": "text"
                }
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # tokenize
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )


ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }

def build_layer_input_ln_regex(layer_indices):
    if not layer_indices:
        return None
    layer_pattern = "|".join(str(idx) for idx in layer_indices)
    return rf"re:.*layers\.({layer_pattern})\.input_layernorm$"


num_layers = len(model.model.language_model.layers)
layers = model.model.language_model.layers

full_attn_layer_indices = []
linear_attn_layer_indices = []
linear_attn_balance_layers = None

for idx, layer in enumerate(layers):
    self_attn = getattr(layer, "self_attn", None)
    linear_attn = getattr(layer, "linear_attn", None)

    has_full_attn = self_attn is not None
    has_linear_attn = linear_attn is not None

    if has_full_attn and has_linear_attn:
        raise ValueError(
            f"Layer {idx} has both self_attn and linear_attn, expected mutual exclusivity"
        )

    if has_full_attn:
        full_attn_layer_indices.append(idx)
        continue

    if has_linear_attn:
        linear_attn_layer_indices.append(idx)

        if linear_attn_balance_layers is None:
            if all(
                hasattr(linear_attn, name)
                for name in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z")
            ):
                linear_attn_balance_layers = [
                    "re:.*linear_attn.in_proj_qkv$",
                    "re:.*linear_attn.in_proj_a$",
                    "re:.*linear_attn.in_proj_b$",
                    "re:.*linear_attn.in_proj_z$",
                ]
            else:
                raise ValueError(
                    "Unsupported linear_attn projection structure for AWQ mapping: "
                    "expected in_proj_qkv/in_proj_a/in_proj_b/in_proj_z"
                )

# Fallback for models where submodule layout is opaque.
if not full_attn_layer_indices and not linear_attn_layer_indices:
    full_attention_interval = getattr(model.config, "full_attention_layer_interval", None)
    if isinstance(full_attention_interval, int) and full_attention_interval > 0:
        full_attn_layer_indices = list(
            range(full_attention_interval - 1, num_layers, full_attention_interval)
        )
        full_attn_layer_index_set = set(full_attn_layer_indices)
        linear_attn_layer_indices = [
            idx for idx in range(num_layers) if idx not in full_attn_layer_index_set
        ]

if not linear_attn_balance_layers:
    # Default to the 4-way split used by this Qwen3.5 MoE variant.
    linear_attn_balance_layers = [
        "re:.*linear_attn.in_proj_qkv$",
        "re:.*linear_attn.in_proj_a$",
        "re:.*linear_attn.in_proj_b$",
        "re:.*linear_attn.in_proj_z$",
    ]

overlap = set(full_attn_layer_indices) & set(linear_attn_layer_indices)
if overlap:
    raise ValueError(f"Found overlap between full_attn and linear_attn indices: {sorted(overlap)}")

self_attn_input_ln_regex = build_layer_input_ln_regex(full_attn_layer_indices)
linear_attn_input_ln_regex = build_layer_input_ln_regex(linear_attn_layer_indices)

mapping = []
if self_attn_input_ln_regex:
    mapping.append(
        AWQMapping(
            self_attn_input_ln_regex,
            [
                "re:.*self_attn.q_proj$",
                "re:.*self_attn.k_proj$",
                "re:.*self_attn.v_proj$",
            ],
            norm_bias=1.0
        )
    )
    mapping.append(AWQMapping("re:.*self_attn.v_proj$", ["re:.*self_attn.o_proj$"]))

if linear_attn_input_ln_regex:
    mapping.append(
        AWQMapping(
            linear_attn_input_ln_regex,
            linear_attn_balance_layers,
            norm_bias=1.0
        )
    )

mapping.extend(
    [
        AWQMapping(
            "re:.*post_attention_layernorm$",
            [
                "re:.*mlp.experts.*.gate_proj$",
                "re:.*mlp.experts.*.up_proj$",
                "re:.*mlp.shared_expert.gate_proj$",
                "re:.*mlp.shared_expert.up_proj$",
                "re:.*mlp.shared_expert_gate$",
            ],
            activation_hook_target="shared_expert_gate",
            norm_bias=1.0
        ),
        AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
    ]
)

recipe = [
    AutoSmoothModifier(
        activation_scale_type="mean", 
        norm_func='adaptive',
        mappings=mapping,
        scheme="W8A8",
        offload_device=torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cpu"),
        ignore=[
            "re:.*lm_head",
            "re:visual.*",
            "re:model.visual.*",
            "re:.*mlp.gate$",
            "re:.*embed_tokens$",
        ],
    ),
    QuantizationModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=[
            "re:.*lm_head",
            "re:visual.*",
            "re:model.visual.*",
            "re:.*mlp.gate$",
            "re:.*embed_tokens$",
            "re:.*shared_expert_gate$",
 #           "re:.*linear_attn.*",
        ],
    ),
]

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
    output_dir=SAVE_DIR,
    moe_calibrate_all_experts=False,
)