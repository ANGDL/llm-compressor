import argparse
import base64
from io import BytesIO
import os
from contextlib import contextmanager

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.pipelines.basic import pipeline as basic_pipeline
from llmcompressor.pipelines.data_free import pipeline as data_free_pipeline
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from compressed_tensors.offload import get_device_map
from compressed_tensors.offload.dispatch import dispatch_model as _dispatch_model
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization import QuantizationScheme


"""
python kimi2_5_wNa8.py --model_id /ssd3/models/Kimi-K2.6-bf16/ --dataset_id /home/llm-compressor/lmms-lab___flickr30k --modifier=RTN --observer imatrix_mse > k26.log 2>&1 &
python src/llmcompressor/utils/pack_int4_to_int8.py -i /ssd2/models/Kimi-K2.6-bf16-W4A8Experts-W8A8Other-IMatrix-RTN-unpacked/ -o /ssd3/models/Kimi-K2.6-w4a8-v2
"""


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="/ssd3/models/Kimi-K2.5_bf16")
parser.add_argument("--save_dir", type=str, default="/ssd2/models/")
parser.add_argument("--offload_folder", type=str, default="./offload_folder")
parser.add_argument("--observer", type=str, default=None, choices=[None, "imatrix_mse"])
parser.add_argument("--modifier", type=str, default="RTN", choices=["GPTQ", "RTN"])
parser.add_argument("--dataset_id", type=str, default="lmms-lab/flickr30k")
parser.add_argument("--dataset_split", type=str, default="test")
parser.add_argument(
    "--skip_restore_from_accelerate",
    action=argparse.BooleanOptionalAction,
    help=(
        "Skip converting model back from accelerate after save_pretrained. "
        "Useful to avoid end-of-run CUDA OOM for very large models."
    ),
    default=True,
)
parser.add_argument(
    "--dispatch_extra_memory_gb",
    type=float,
    default=20.0,
    help="Reserved memory per GPU for dispatch_model. Lower this if dispatch fails.",
)
args = parser.parse_args()


@contextmanager
def maybe_skip_from_accelerate(skip_restore: bool):
    if not skip_restore:
        yield
        return

    import llmcompressor.transformers.compression.compressed_tensors_utils as ct_utils

    original_from_accelerate = ct_utils.from_accelerate
    ct_utils.from_accelerate = lambda model: ({}, None)
    try:
        yield
    finally:
        ct_utils.from_accelerate = original_from_accelerate


def patch_pipeline_dispatch(extra_memory_gb: float):
    extra_memory_bytes = int(extra_memory_gb * 1024**3)

    def _dispatch_with_cap(model):
        return _dispatch_model(model, extra_memory=extra_memory_bytes)

    basic_pipeline.dispatch_model = _dispatch_with_cap
    data_free_pipeline.dispatch_model = _dispatch_with_cap


patch_pipeline_dispatch(args.dispatch_extra_memory_gb)


model_id = args.model_id
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map=None,
    offload_folder=args.offload_folder,
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
# Confirm that model is dispatched correctly
devices = {offloaded for _onloaded, offloaded in get_device_map(model).values()}
print(f"Model was offloaded to the following devices: {devices}")


# Select calibration dataset.
DATASET_ID = args.dataset_id
DATASET_SPLIT = args.dataset_split

# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 8192

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# Apply chat template and tokenize inputs.
def preprocess_and_tokenize(example):
    # preprocess
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_img = f"data:image;base64,{encoded_image_text}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image_url": base64_img},
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

    return processor(
        messages=messages,
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

ignores = [
    # Ignore the output head
    r"re:vision_tower\.*",
    r"re:mm_projector\.*",
    r"re:.*\.mlp\.gate$",
    r"re:.*\.embed_tokens$",
    "lm_head",
]


# ---- Quantization config: mixed experts int4 + other linear int8 ----
# Weights: 4-bit, channelwise, symmetric, static
weights_args_4 = QuantizationArgs(
    num_bits=4,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.CHANNEL,
    symmetric=True,
    dynamic=False,
)

# Weights: 8-bit, channelwise, symmetric, static
weights_args_8 = QuantizationArgs(
    num_bits=8,
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

# Experts use int4 weights.
experts_w4_scheme = QuantizationScheme(
    targets=[
        r"re:.*mlp\.experts\..*\.(gate_proj|up_proj|down_proj)$",
    ],
    weights=weights_args_4,
    input_activations=activations_args,
)

# Other linear projections use int8 weights.
other_linear_w8_scheme = QuantizationScheme(
    targets=[
        r"re:.*mlp\.shared_experts\.(gate_proj|up_proj|down_proj)$",
        r"re:.*self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$",
        r"re:.*mlp\.(gate_proj|up_proj|down_proj)$",
    ],
    weights=weights_args_8,
    input_activations=activations_args,
)


# Configure the quantization algorithm to run.

config_groups = {
    "experts_w4": experts_w4_scheme,
    "other_linear_w8": other_linear_w8_scheme,
}

tail_name = "-W4A8Experts-W8A8Other"
recipes = []
if args.observer == "imatrix_mse":
    experts_w4_scheme.weights.observer = "imatrix_mse"
    other_linear_w8_scheme.weights.observer = "imatrix_mse"
    recipes.append(IMatrixGatherer(ignore=ignores))
    tail_name += "-IMatrix"

if args.modifier == "GPTQ":
    recipes.append(
        GPTQModifier(
            config_groups=config_groups,
            ignore=ignores,
            offload_hessians=True,
        )
    )
    tail_name += "-GPTQ"
elif args.modifier == "RTN":
    recipes.append(
        QuantizationModifier(
            config_groups=config_groups,
            ignore=ignores,
        )
    )
    tail_name += "-RTN"
else:
    raise ValueError(f"Invalid modifier selected: {args.modifier}")

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipes,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    batch_size=1,
    data_collator=data_collator,
    processor=processor,
    preprocessing_num_workers=2,
    dataloader_num_workers=2,
    sequential_prefetch=True,
)

# Save to disk compressed.
SAVE_NAME = model_id.rstrip("/").split("/")[-1] + tail_name + "-unpacked"
SAVE_DIR = os.path.join(args.save_dir, SAVE_NAME)

with maybe_skip_from_accelerate(args.skip_restore_from_accelerate):
    model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
