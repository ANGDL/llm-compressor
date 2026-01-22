import os
import base64
from io import BytesIO


import torch
from qwen_vl_utils import process_vision_info

from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modeling.gpt_oss import convert_model_for_quantization_gptoss
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.awq import AWQMapping
from datasets import load_dataset


MODEL_ID = "/data/models/Qwen2.5-VL-72B-Instruct"
BASE_NAME = MODEL_ID.rstrip("/").split("/")[-1]

OUTPUT_DIR = f"{BASE_NAME}-w4a8-smooth-gptq-max"
OUTPUT_DIR = os.path.join("/data/models", OUTPUT_DIR)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto", device_map=None)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

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
    return {key: torch.tensor(value) for key, value in batch[0].items()}

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

# Scope mappings to the language model decoder blocks so each pattern resolves to
# exactly one module per block. This avoids AutoSmooth matching the entire stack
# of layer norms at once (which triggers the "Smooth needs to match a single
# smoothlayer" error).
mapping = [
    AWQMapping(
        r"re:model\.language_model\.layers\.[0-9]+\.input_layernorm$",
        [
            r"re:model\.language_model\.layers\.[0-9]+\.self_attn\.q_proj$",
            r"re:model\.language_model\.layers\.[0-9]+\.self_attn\.k_proj$",
            r"re:model\.language_model\.layers\.[0-9]+\.self_attn\.v_proj$",
        ],
    ),
    AWQMapping(
        r"re:model\.language_model\.layers\.[0-9]+\.self_attn\.v_proj$",
        [r"re:model\.language_model\.layers\.[0-9]+\.self_attn\.o_proj$"],
    ),
    AWQMapping(
        r"re:model\.language_model\.layers\.[0-9]+\.post_attention_layernorm$",
        [
            r"re:model\.language_model\.layers\.[0-9]+\.mlp\.gate_proj$",
            r"re:model\.language_model\.layers\.[0-9]+\.mlp\.up_proj$",
        ],
    ),
    AWQMapping(
        r"re:model\.language_model\.layers\.[0-9]+\.mlp\.up_proj$",
        [r"re:model\.language_model\.layers\.[0-9]+\.mlp\.down_proj$"],
    ),
]

recipe = [
    AutoSmoothModifier(
        activation_scale_type="max", 
        norm_func='adaptive', 
        mappings=mapping,
    ),
    GPTQModifier(
        config_groups={"group_0": scheme}, 
        ignore=["lm_head", "re:visual.*", "re:model.visual.*"],
    )
    #QuantizationModifier(config_groups={"group_0": scheme}, ignore=["lm_head", "re:visual.*", "re:model.visual.*"],)

]

print(f"Starting oneshot quantization → {OUTPUT_DIR}")
# Perform oneshot
oneshot(
    model=model,
    tokenizer=MODEL_ID,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
    sequential_targets=["Qwen2_5_VLDecoderLayer"],
)

try:
    # Confirm generations of the quantized model look sane.
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Please describe this image\n"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)
    print(processor.decode(output[0], skip_special_tokens=True))
    print("==========================================")
except Exception as e:
    print(f"Error during generation: {e}")

# Save to disk compressed.
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
processor.save_pretrained(OUTPUT_DIR)

print(f"Quantization finished. Quantized model written to: {OUTPUT_DIR}")