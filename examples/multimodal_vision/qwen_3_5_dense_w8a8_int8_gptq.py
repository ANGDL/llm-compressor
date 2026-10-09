import base64
import argparse
from io import BytesIO
import os
from pathlib import Path
import shutil

import torch
from datasets import Dataset, load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from qwen_vl_utils import process_vision_info

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from compressed_tensors.quantization.quant_args import (
  QuantizationArgs,
  QuantizationStrategy,
  QuantizationType,
)


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

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default="/data/models/Qwen3.5-27B")
parser.add_argument("--output-dir")
parser.add_argument("--dataset-id", default="lmms-lab/flickr30k")
parser.add_argument("--num-calibration-samples", type=int, default=128)
parser.add_argument("--max-sequence-length", type=int, default=8192)
args = parser.parse_args()
if args.num_calibration_samples < 1 or args.max_sequence_length < 1:
    parser.error("calibration samples and sequence length must be positive")

model_id = args.model_id
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


# Oneshot arguments
NUM_CALIBRATION_SAMPLES = args.num_calibration_samples
MAX_SEQUENCE_LENGTH = args.max_sequence_length

DATASET_ID = args.dataset_id

# Load dataset and preprocess.
dataset_path = Path(DATASET_ID)
if dataset_path.is_dir():
    shards = sorted(dataset_path.glob("**/flickr30k-test-*.arrow"))
    if not shards:
        parser.error(f"no Flickr30k test Arrow shards in {DATASET_ID}")
    ds = Dataset.from_file(str(shards[0]))
    if len(ds) < NUM_CALIBRATION_SAMPLES:
        parser.error(f"first Arrow shard has fewer than {NUM_CALIBRATION_SAMPLES} samples")
    ds = ds.select(range(NUM_CALIBRATION_SAMPLES))
else:
    ds = load_dataset(DATASET_ID, split=f"test[:{NUM_CALIBRATION_SAMPLES}]")
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
model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, device_map=None, dtype="auto")


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


recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=[
            "re:.*lm_head",
            "re:visual.*",
            "re:model.visual.*",
            "re:.*embed_tokens$",
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
    preprocessing_num_workers=2,
    dataloader_num_workers=2,
    sequential_prefetch=True,
)

# Save to disk compressed.
SAVE_DIR = args.output_dir or os.path.join(
    "/data/models", model_id.rstrip("/").split("/")[-1] + "-W8A8K-gptq"
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
copy_original_non_model_files(model_id, SAVE_DIR)
save_mtp_tensors_to_checkpoint(source_model=model_id, dest_dir=SAVE_DIR)
