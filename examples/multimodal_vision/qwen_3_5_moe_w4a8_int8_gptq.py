import base64
import argparse
from io import BytesIO
import os

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
from qwen_vl_utils import process_vision_info

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier 
from llmcompressor.utils import dispatch_for_generation

from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint



# Load model.
model_id = "/ssd4/models/Qwen3.5-397B-A17B"
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(model_id, device_map=None, dtype="auto",local_files_only=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


# Oneshot arguments
NUM_CALIBRATION_SAMPLES = 64
MAX_SEQUENCE_LENGTH = 8192

DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = f"test[:{NUM_CALIBRATION_SAMPLES}]"
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-w4a8-gptq-unpacked"
SAVE_DIR = os.path.join("/ssd3/models", SAVE_DIR)

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

# ---- Quantization config: W4A8 (int4 weights, int8 activations) ----
# Weights: 4-bit, channelwise, symmetric, static
weights_args = QuantizationArgs(
    num_bits=4,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.CHANNEL,
    symmetric=True,
    dynamic=False,
)

# Activations: 8-bit, per-token, asymmetric, dynamic
activations_args = QuantizationArgs(
    num_bits=8,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.TOKEN,
    symmetric=False,
    dynamic=True,
    observer=None,
)

# Apply to all Linear layers, excluding lm_head
scheme = QuantizationScheme(
    targets=["Linear"],
    weights=weights_args,
    input_activations=activations_args,
)


recipe = [
    GPTQModifier(
        targets="Linear",
        offload_hessians=True,
        config_groups={"group_0": scheme}, 
        ignore=[
            "re:.*lm_head",
            "re:visual.*",
            "re:model.visual.*",
            "re:.*mlp.gate$",
            "re:.*embed_tokens$",
            "re:.*shared_expert_gate$",
            "re:.*linear_attn.*",
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
    sequential_targets=["Qwen3_5MoeDecoderLayer"],
    preprocessing_num_workers=2,
    dataloader_num_workers=2,
    sequential_prefetch=True,
)

# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
save_mtp_tensors_to_checkpoint(source_model=model_id, dest_dir=SAVE_DIR)
