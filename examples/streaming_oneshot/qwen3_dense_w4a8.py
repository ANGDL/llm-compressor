"""Quantize a local Qwen3-0.6B checkpoint to W4A8 with streaming iMatrix + RTN.

The shared Sequential Pipeline tracer derives the ordered decoder subgraphs.
Each subgraph is loaded, calibrated, quantized, written once as a final shard,
and released before the next subgraph is processed. The source checkpoint is
never modified. Intermediate recovery data is not written unless
`checkpoint_progress` is enabled.
"""

import torch
from compressed_tensors.quantization import preset_name_to_scheme
from datasets import load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import streaming_oneshot

MODEL = "/Users/ang/models/Qwen3-0.6B"
OUTPUT_DIR = "/Users/ang/models/Qwen3-0.6B-W4A8-IMatrix-RTN"
WORK_DIR = "/Users/ang/models/streaming-work-qwen3-0.6b-w4a8"
# As with oneshot(), the dataset can instead be a pre-tokenized Hugging Face
# Dataset or a PyTorch DataLoader. The shared tracer and checkpoint-backed
# loader prepare the first boundary; users do not construct activations or list
# model-specific layer boundaries.
dataset = load_dataset(
    "/Users/ang/Downloads/llm-demo/datasets/ultrachat_200k",
    split="train_sft[:16]",
)

w4a8_scheme = preset_name_to_scheme("W4A8", ["Linear"])
w4a8_scheme.weights.observer = "imatrix_mse"
w4a8_scheme.weights.scale_dtype = torch.float32

recipe = [
    IMatrixGatherer(ignore=["lm_head"]),
    QuantizationModifier(
        config_groups={"group_0": w4a8_scheme},
        ignore=["lm_head"],
    ),
]

streaming_oneshot(
    model=MODEL,
    dataset=dataset,
    recipe=recipe,
    output_dir=OUTPUT_DIR,
    work_dir=WORK_DIR,
    num_calibration_samples=16,
    max_seq_length=2048,
    # Snapshot one completed subgraph on CPU and write it in the background.
    async_save=True,
    # Set True only when crash recovery is worth the intermediate disk writes.
    checkpoint_progress=False,
    # Replace a previous incomplete or completed quantization at OUTPUT_DIR.
    overwrite_output=True,
    pack_to_int8=True,
)
