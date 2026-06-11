import argparse
import base64
from io import BytesIO
import os
import shutil

import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.awq import AWQMapping


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="/data/models/Qwen3-VL-8B-Instruct")
parser.add_argument("--save_dir", type=str, default="/data/models/")
parser.add_argument("--num_calibration_samples", type=int, default=128)
parser.add_argument("--max_sequence_length", type=int, default=2048)
parser.add_argument("--dataset_id", type=str, default="lmms-lab/flickr30k")
parser.add_argument("--dataset_split", type=str, default="test")
parser.add_argument(
    "--text_dataset_id", type=str, default="HuggingFaceH4/ultrachat_200k"
)
parser.add_argument("--text_dataset_split", type=str, default="train_sft")
parser.add_argument(
    "--text_calibration_samples",
    type=int,
    default=128,
    help="Number of text samples to append to calibration. Set 0 to disable.",
)
parser.add_argument(
    "--quantizer",
    type=str,
    default="GPTQ",
    choices=["GPTQ", "RTN", "none"],
    help="Quantization modifier: GPTQ, RTN, or none (skip quantization).",
)
parser.add_argument(
    "--disable_autosmooth",
    action="store_true",
    default=False,
    help="Disable AutoSmoothModifier.",
)
parser.add_argument(
    "--activation_scale_type",
    type=str,
    default="mean",
    choices=["mean", "max", "min"],
    help="Activation scale type for AutoSmoothModifier.",
)
parser.add_argument(
    "--norm_func",
    type=str,
    default="adaptive",
    choices=["adaptive", "l1", "l2", "linf"],
    help="Norm function for AutoSmoothModifier.",
)
args = parser.parse_args()

use_autosmooth = not args.disable_autosmooth


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

# Load model.
model_id = args.model_id
model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
NUM_CALIBRATION_SAMPLES = args.num_calibration_samples
MAX_SEQUENCE_LENGTH = args.max_sequence_length
NUM_TEXT_CALIBRATION_SAMPLES = args.text_calibration_samples

DATASET_ID = args.dataset_id
DATASET_SPLIT = args.dataset_split

# Load multimodal dataset.
multimodal_ds = load_dataset(
    DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"
)
multimodal_ds = multimodal_ds.shuffle(seed=42)

# Load text dataset.
text_ds = (
    load_dataset(
        args.text_dataset_id,
        split=f"{args.text_dataset_split}[:{NUM_TEXT_CALIBRATION_SAMPLES}]",
    )
    if NUM_TEXT_CALIBRATION_SAMPLES > 0
    else None
)
if text_ds is not None:
    text_ds = text_ds.shuffle(seed=42)
    text_examples = list(text_ds)
else:
    text_examples = []


def _normalize_text_content(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    normalized = []
    for item in content:
        if isinstance(item, str):
            normalized.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            if item.get("type") == "text":
                normalized.append({"type": "text", "text": item.get("text", "")})
            elif "text" in item:
                normalized.append({"type": "text", "text": item["text"]})
    return normalized


def format_text_example(example):
    return [
        {
            "role": message["role"],
            "content": _normalize_text_content(message["content"]),
        }
        for message in example["messages"]
    ]


def build_multimodal_messages(example):
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return [
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
                    "type": "text",
                }
            ],
        },
    ]


def preprocess_and_tokenize(example, idx):
    messages = build_multimodal_messages(example)

    if text_examples:
        text_example = text_examples[idx % len(text_examples)]
        messages.extend(format_text_example(text_example))

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )


multimodal_ds = multimodal_ds.map(
    preprocess_and_tokenize,
    with_indices=True,
    remove_columns=multimodal_ds.column_names,
)


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

# Build recipe based on flags
recipe = []
tail_name = ""

if use_autosmooth:
    recipe.append(
        AutoSmoothModifier(
            activation_scale_type=args.activation_scale_type,
            norm_func=args.norm_func,
        )
    )
    tail_name += f"-AutoSmooth-{args.norm_func}"

if args.quantizer == "GPTQ":
    recipe.append(
        GPTQModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=["lm_head", "re:visual.*", "re:model.visual.*"],
            offload_hessians=True,
        ),
    )
    tail_name += "-W8A8-GPTQ"
elif args.quantizer == "RTN":
    recipe.append(
        QuantizationModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=["lm_head", "re:visual.*", "re:model.visual.*"],
        ),
    )
    tail_name += "-W8A8-RTN"

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=multimodal_ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "http://images.cocodataset.org/train2017/000000231895.jpg",
            },
            {"type": "text", "text": "Please describe the animal in this image\n"},
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
    max_length=MAX_SEQUENCE_LENGTH,
    truncation=True,
    return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")


# Save to disk compressed.
SAVE_NAME = model_id.rstrip("/").split("/")[-1] + tail_name
SAVE_DIR = os.path.join(args.save_dir, SAVE_NAME)
model.save_pretrained(SAVE_DIR, save_compressed=True)
copy_original_non_model_files(model_id, SAVE_DIR)
