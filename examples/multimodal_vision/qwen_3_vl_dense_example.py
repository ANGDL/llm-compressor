import argparse
import base64
import os
import shutil
from io import BytesIO
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


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


parser = argparse.ArgumentParser(description="Quantize a Qwen3-VL dense checkpoint")
parser.add_argument("--model-id", default="/ssd4/models/Qwen3-VL-2B-Instruct")
parser.add_argument("--output-dir")
parser.add_argument("--dataset-id", default="lmms-lab/flickr30k")
parser.add_argument("--num-calibration-samples", type=int, default=128)
parser.add_argument("--max-sequence-length", type=int, default=4096)
args = parser.parse_args()
if args.num_calibration_samples < 1 or args.max_sequence_length < 1:
    parser.error("calibration samples and sequence length must be positive")

model_id = args.model_id
num_calibration_samples = args.num_calibration_samples
max_sequence_length = args.max_sequence_length
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

dataset_path = Path(args.dataset_id)
if dataset_path.is_dir():
    shards = sorted(dataset_path.glob("**/flickr30k-test-*.arrow"))
    if not shards:
        parser.error(f"no Flickr30k test Arrow shards in {args.dataset_id}")
    dataset = Dataset.from_file(str(shards[0]))
    if len(dataset) < num_calibration_samples:
        parser.error(
            f"first Arrow shard has fewer than {num_calibration_samples} samples"
        )
    dataset = dataset.select(range(num_calibration_samples))
else:
    dataset = load_dataset(
        args.dataset_id, split=f"test[:{num_calibration_samples}]"
    )
dataset = dataset.shuffle(seed=42)


def preprocess_and_tokenize(example):
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{encoded_image}"},
                {"type": "text", "text": "What does the image show?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=max_sequence_length,
        truncation=True,
    )


dataset = dataset.map(preprocess_and_tokenize, remove_columns=dataset.column_names)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, device_map=None, dtype="auto", local_files_only=True
)


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=["re:.*lm_head", "re:visual.*", "re:model.visual.*"],
    ),
]

oneshot(
    model=model,
    tokenizer=model_id,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=max_sequence_length,
    num_calibration_samples=num_calibration_samples,
    trust_remote_code_model=True,
    data_collator=data_collator,
    preprocessing_num_workers=2,
    dataloader_num_workers=2,
    sequential_prefetch=True,
    sequential_targets=["Qwen3VLTextDecoderLayer"],
)

save_dir = args.output_dir or os.path.join(
    "/data/models", model_id.rstrip("/").split("/")[-1] + "-W8A8-gptq"
)
model.save_pretrained(save_dir, save_compressed=True)
copy_original_non_model_files(model_id, save_dir)
