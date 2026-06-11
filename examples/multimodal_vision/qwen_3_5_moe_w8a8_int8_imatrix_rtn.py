import base64
import argparse
from collections import Counter
from contextvars import ContextVar
from io import BytesIO
import os
import shutil

import torch
from datasets import load_dataset
from loguru import logger
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
from qwen_vl_utils import process_vision_info

from llmcompressor import oneshot
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.modifiers.quantization import QuantizationModifier


from compressed_tensors.quantization import QuantizationScheme, preset_name_to_scheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.utils import match_named_modules, save_mtp_tensors_to_checkpoint
from llmcompressor.observers.imatrix import IMatrixMSEObserver


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="/data/models/Qwen3.5-122B-A10B")
parser.add_argument("--save_dir", type=str, default="/data/models")
parser.add_argument("--dataset_id", type=str, default="lmms-lab/flickr30k")
parser.add_argument(
    "--text_dataset_id",
    type=str,
    default="HuggingFaceH4/ultrachat_200k",
)
parser.add_argument("--text_dataset_split", type=str, default="train_sft")
parser.add_argument(
    "--enable_non_multimodal_calibration",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable appending non-multimodal text calibration data.",
)
args = parser.parse_args()


IMATRIX_ALL_ZERO_COUNTER = Counter()
IMATRIX_MODULE_NAME_BY_ID = {}
IMATRIX_CURRENT_MODULE: ContextVar[str | None] = ContextVar(
    "imatrix_current_module", default=None
)
IMATRIX_WARNING_SINK_ID = None


def _imatrix_warning_sink(message):
    text = message.record["message"]
    if text.startswith("imatrix_mse: all zeros. Falling back to uniform MSE."):
        module_name = IMATRIX_CURRENT_MODULE.get() or "<unknown>"
        IMATRIX_ALL_ZERO_COUNTER[module_name] += 1


def _install_imatrix_all_zero_stats_hooks():
    global IMATRIX_WARNING_SINK_ID

    if getattr(IMatrixMSEObserver, "_script_all_zero_stats_installed", False):
        return

    if IMATRIX_WARNING_SINK_ID is None:
        IMATRIX_WARNING_SINK_ID = logger.add(
            _imatrix_warning_sink,
            level="WARNING",
            format="{message}",
            enqueue=False,
        )

    original_attach = IMatrixMSEObserver.attach
    original_validate = IMatrixMSEObserver._get_validated_importance
    original_gatherer_init = IMatrixGatherer.on_initialize

    def _gatherer_init_with_module_names(self, state, **kwargs):
        resolved_targets = self.targets if isinstance(self.targets, list) else [self.targets]
        for module_name, module in match_named_modules(
            state.model,
            resolved_targets,
            self.ignore,
        ):
            IMATRIX_MODULE_NAME_BY_ID[id(module)] = module_name

        return original_gatherer_init(self, state, **kwargs)

    def _attach_with_module_name(self, module):
        self._imatrix_script_module_name = IMATRIX_MODULE_NAME_BY_ID.get(
            id(module),
            module.__class__.__name__,
        )
        return original_attach(self, module)

    def _validate_with_stats(self, observed):
        module_name = getattr(
            self,
            "_imatrix_script_module_name",
            "<unknown>",
        )
        token = IMATRIX_CURRENT_MODULE.set(module_name)
        try:
            return original_validate(self, observed)
        finally:
            IMATRIX_CURRENT_MODULE.reset(token)

    IMatrixGatherer.on_initialize = _gatherer_init_with_module_names
    IMatrixMSEObserver.attach = _attach_with_module_name
    IMatrixMSEObserver._get_validated_importance = _validate_with_stats
    IMatrixMSEObserver._script_all_zero_stats_installed = True


def _print_imatrix_all_zero_summary():
    print("\n[imatrix_mse] all-zero fallback summary")
    if not IMATRIX_ALL_ZERO_COUNTER:
        print("  no modules triggered all-zero fallback")
        return

    total_hits = sum(IMATRIX_ALL_ZERO_COUNTER.values())
    print(
        f"  modules_triggered={len(IMATRIX_ALL_ZERO_COUNTER)} total_hits={total_hits}"
    )

    for module_name, hit_count in sorted(
        IMATRIX_ALL_ZERO_COUNTER.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {hit_count:6d}  {module_name}")


_install_imatrix_all_zero_stats_hooks()



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
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(model_id, device_map=None, dtype="auto",local_files_only=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


# Oneshot arguments
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 8192

DATASET_ID = args.dataset_id
DATASET_SPLIT = f"test[:{NUM_CALIBRATION_SAMPLES}]"
TEXT_DATASET_ID = args.text_dataset_id
TEXT_DATASET_SPLIT = args.text_dataset_split

def _load_text_dataset(dataset_id, dataset_split):
    return load_dataset(
        dataset_id,
        split=f"{dataset_split}[:{NUM_CALIBRATION_SAMPLES}]",
    )


# Load dataset and preprocess.
multimodal_ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
multimodal_ds = multimodal_ds.shuffle(seed=42)

text_examples = []
if args.enable_non_multimodal_calibration:
    text_ds = _load_text_dataset(TEXT_DATASET_ID, TEXT_DATASET_SPLIT)
    text_ds = text_ds.shuffle(seed=42)
    text_examples = list(text_ds)


def _normalize_text_content(content):
    if isinstance(content, str):
        stripped = content.strip()
        return [{"type": "text", "text": stripped}] if stripped else []

    if isinstance(content, dict):
        content = [content]

    normalized = []
    for item in content if isinstance(content, list) else []:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                normalized.append({"type": "text", "text": stripped})
        elif isinstance(item, dict):
            text_value = None
            if item.get("type") == "text":
                text_value = item.get("text")
            elif item.get("type") == "reasoning":
                text_value = item.get("reasoning")
            else:
                text_value = item.get("text") or item.get("reasoning")

            if isinstance(text_value, str):
                stripped = text_value.strip()
                if stripped:
                    normalized.append({"type": "text", "text": stripped})
    return normalized


def _format_messages(messages):
    formatted = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        if role not in {"user", "assistant", "system"}:
            role = "user"

        normalized_content = _normalize_text_content(message.get("content", ""))
        if normalized_content:
            formatted.append({"role": role, "content": normalized_content})

    return formatted


def format_text_example(example):
    messages = _format_messages(example.get("messages", []))
    if not messages:
        raise ValueError("Could not extract text messages from calibration sample")

    return messages


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


def build_calibration_messages(
    multimodal_example,
    text_example=None,
):
    messages = build_multimodal_messages(multimodal_example)

    if text_example is not None:
        messages.extend(format_text_example(text_example))

    return messages


def _select_calibration_example(examples, idx):
    if not examples:
        return None
    return examples[idx % len(examples)]


# Apply chat template and tokenize inputs.
def preprocess_and_tokenize(multimodal_example, idx):
    text_example = _select_calibration_example(text_examples, idx)

    messages = build_calibration_messages(
        multimodal_example,
        text_example=text_example,
    )

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


ds = multimodal_ds.map(
    preprocess_and_tokenize,
    with_indices=True,
    remove_columns=multimodal_ds.column_names,
)


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

# Apply to all Linear layers, excluding lm_head
scheme = preset_name_to_scheme("W8A8", ["Linear"])
scheme.weights.observer = "imatrix_mse"

ignore=[
    "re:.*lm_head",
    "re:visual.*",
    "re:model.visual.*",
    "re:.*mlp.gate$",
    "re:.*embed_tokens$",
    "re:.*shared_expert_gate$",
]

recipe = [
    IMatrixGatherer(ignore=ignore),
    QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=ignore,
    ),
]


# Perform oneshot
try:
    oneshot(
        model=model,
        tokenizer=model_id,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        data_collator=data_collator,
    )
finally:
    _print_imatrix_all_zero_summary()

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W8A8-iRTN"
SAVE_DIR = os.path.join(args.save_dir, SAVE_DIR)
model.save_pretrained(SAVE_DIR, save_compressed=True)
copy_original_non_model_files(model_id, SAVE_DIR)
save_mtp_tensors_to_checkpoint(source_model=model_id, dest_dir=SAVE_DIR)