import argparse
import os

from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    ProcessorMixin,
    Qwen3OmniMoeForConditionalGeneration,
    default_data_collator,
)

from datasets import concatenate_datasets, load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.streaming import streaming_oneshot
from llmcompressor.utils import ImatrixFallbackStats


class Qwen3OmniCalibrationModel(Qwen3OmniMoeForConditionalGeneration):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
    ):
        return self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        )


parser = argparse.ArgumentParser(description="Quantize Qwen3-Omni with iMatrix + RTN")
parser.add_argument(
    "--model-id",
    required=True,
    help="local directory containing the full-precision Qwen3-Omni checkpoint",
)
parser.add_argument("--output-dir", required=True)
parser.add_argument(
    "--dataset-id",
    type=str,
    nargs="+",
    default=["lmms-lab/flickr30k"],
    help=(
        "HuggingFace dataset ID(s) or path(s) to local JSON/JSONL files. "
        "Sources with image columns calibrate the vision tower, text-only sources "
        "calibrate the text tower. Multiple sources are concatenated."
    ),
)
parser.add_argument(
    "--dataset-split",
    type=str,
    nargs="+",
    default=["test"],
    help=(
        "split(s) for `--dataset-id`: one value applies to every source, or pass "
        "one per source (e.g. `test train_sft` for an image source and a text "
        "source). JSON/JSONL sources always use their `train` split."
    ),
)
parser.add_argument("--num-calibration-samples", type=int, default=512)
parser.add_argument("--max-sequence-length", type=int, default=2048)
parser.add_argument(
    "--moe-calibrate-all-experts",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="calibrate all MoE experts instead of only routed experts",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="image samples require 1; text-only sources can use larger batches",
)
args = parser.parse_args()
if (
    args.batch_size < 1
    or args.num_calibration_samples < 1
    or args.max_sequence_length < 1
):
    parser.error("batch size, calibration samples and sequence length must be positive")

# Select calibration dataset.
# Support multiple sources (HuggingFace Hub and/or local Arrow/JSON files).
DATASET_IDS = args.dataset_id
DATASET_SPLITS = args.dataset_split
NUM_CALIBRATION_SAMPLES = args.num_calibration_samples
MAX_SEQUENCE_LENGTH = args.max_sequence_length

if NUM_CALIBRATION_SAMPLES < len(DATASET_IDS):
    raise ValueError(
        f"num_calibration_samples ({NUM_CALIBRATION_SAMPLES}) must be >= "
        f"number of dataset sources ({len(DATASET_IDS)}): {DATASET_IDS}"
    )

# text and image sources usually live in different splits, so allow one split per
# source: `--dataset-split test train_sft`
if len(DATASET_SPLITS) == 1:
    DATASET_SPLITS = DATASET_SPLITS * len(DATASET_IDS)
elif len(DATASET_SPLITS) != len(DATASET_IDS):
    parser.error(
        f"--dataset-split expects 1 or {len(DATASET_IDS)} values, "
        f"got {len(DATASET_SPLITS)}"
    )

model_id = args.model_id

# Load dataset (sources, splits and sample budget).
per_source_samples = NUM_CALIBRATION_SAMPLES // len(DATASET_IDS)
remainder = NUM_CALIBRATION_SAMPLES % len(DATASET_IDS)

all_datasets = []
for i, dataset_id in enumerate(DATASET_IDS):
    n_samples = per_source_samples + (1 if i < remainder else 0)
    if os.path.isfile(dataset_id) or dataset_id.endswith((".json", ".jsonl")):
        part = load_dataset(
            "json",
            data_files=dataset_id,
            split=f"train[:{n_samples}]",
        )
    else:
        part = load_dataset(dataset_id, split=f"{DATASET_SPLITS[i]}[:{n_samples}]")
    if len(part) == 0:
        parser.error(f"no samples loaded from {dataset_id!r}")
    all_datasets.append(part.shuffle(seed=42))

# Load processor.
processor = AutoProcessor.from_pretrained(model_id)


# Preprocess each source to a unified {"text", "images"} schema before
# concatenation, since different sources may have different column schemas.
# Sources without images keep `images=None`, i.e. they only calibrate the text tower.
def preprocess(example):
    images = example.get("image", example.get("images"))

    if "caption" in example:
        captions = example["caption"]
        caption = captions if isinstance(captions, str) else " ".join(captions)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What does this image show?"},
                ],
            },
            {"role": "assistant", "content": caption},
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
    elif "messages" in example:
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False
        )
    elif "text" in example:
        text = example["text"]
    else:
        raise ValueError(
            "calibration rows must contain `caption`, `messages` or `text` columns, "
            f"got {list(example.keys())}"
        )

    return {"text": text, "images": images}


all_datasets = [
    part.map(preprocess, remove_columns=part.column_names) for part in all_datasets
]


# Tokenize inputs
def tokenize(sample):
    inputs = processor(
        text=sample["text"],
        images=sample["images"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )
    # mirror `TextGenerationDataset.tokenize`: strip the batch dim multimodal
    # processors add, but leave already-unbatched tensors (e.g. `image_grid_thw`)
    if isinstance(processor, ProcessorMixin):
        for key in inputs:
            if isinstance(inputs[key], list) and len(inputs[key]) == 1:
                inputs[key] = inputs[key][0]
    return inputs


# Tokenize each source separately: text-only sources produce no vision tensors, and a
# single map over mixed sources would drop or misalign the vision columns
all_datasets = [
    part.map(tokenize, remove_columns=["text", "images"]) for part in all_datasets
]

# Pad the columns a source does not produce (e.g. `pixel_values` for text-only
# sources) with nulls so that the sources can be concatenated
features = {}
for part in all_datasets:
    for name, feature in part.features.items():
        features.setdefault(name, feature)

for i, part in enumerate(all_datasets):
    for name, feature in features.items():
        if name not in part.column_names:
            part = part.add_column(name, [None] * len(part), feature=feature)
    all_datasets[i] = part

if len(all_datasets) == 1:
    ds = all_datasets[0]
else:
    ds = concatenate_datasets(all_datasets)

ignores = [
    r"re:.*lm_head$",
    r"re:.*visual.*",
    r"re:.*code2wav.*",
    r"re:.*audio_tower.*",
]

# Recipe
recipe = [
    QuantizationModifier(
        scheme="W8A8",
        targets="Linear",
        ignore=ignores,
        weight_observer="imatrix_mse",
    ),
]


def data_collator(features):
    # text-only samples carry no vision tensors
    features = [
        {key: value for key, value in sample.items() if value is not None}
        for sample in features
    ]

    if len(features) > 1:
        if any("pixel_values" in sample for sample in features):
            raise ValueError(
                "image samples require --batch-size 1: vision tensors are "
                "collated per sample"
            )
        # pad text inputs to the longest sample in the batch
        pad_id = processor.tokenizer.pad_token_id
        pad_id = processor.tokenizer.eos_token_id if pad_id is None else pad_id
        max_len = max(len(sample["input_ids"]) for sample in features)
        for sample in features:
            num_pads = max_len - len(sample["input_ids"])
            sample["input_ids"] += [pad_id] * num_pads
            sample["attention_mask"] += [0] * num_pads

    batch = default_data_collator(features)
    if "image_grid_thw" in batch:
        # (batch, num_images, 3) -> (num_images, 3) for the thinker forward pass
        batch["image_grid_thw"] = batch["image_grid_thw"].reshape(-1, 3)
    return batch


calibration_dataloader = DataLoader(
    ds,
    batch_size=args.batch_size,
    collate_fn=data_collator,
)

# Perform oneshot
with ImatrixFallbackStats():
    result = streaming_oneshot(
        model=model_id,
        model_factory=Qwen3OmniCalibrationModel,
        tokenizer=processor,
        dataset=calibration_dataloader,
        recipe=recipe,
        batch_size=args.batch_size,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=args.num_calibration_samples,
        shuffle_calibration_samples=False,
        moe_calibrate_all_experts=args.moe_calibrate_all_experts,
        overwrite_output=True,
        output_dir=args.output_dir,
    )
    processor.save_pretrained(result)
    print(f"save qwen3 omni w8a8 model and processor saved to: {result}")
