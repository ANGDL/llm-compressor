"""Quantize a local Qwen3-0.6B checkpoint with streaming iMatrix + RTN.

The shared Sequential Pipeline tracer derives the ordered decoder subgraphs.
Each subgraph is loaded, calibrated, quantized, written once as a final shard,
and released before the next subgraph is processed. The source checkpoint is
never modified. Intermediate recovery data is not written unless
`checkpoint_progress` is enabled.
"""

from datasets import load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import streaming_oneshot

MODEL = "/Users/ang/models/Qwen3-0.6B"
OUTPUT_DIR = "/Users/ang/models/Qwen3-0.6B-W8A8-IMatrix-RTN"
WORK_DIR = "/Users/ang/models/streaming-work-qwen3-0.6b"

# As with oneshot(), the dataset can instead be a pre-tokenized Hugging Face
# Dataset or a PyTorch DataLoader. The shared tracer and checkpoint-backed
# loader prepare the first boundary; users do not construct activations or list
# model-specific layer boundaries.
dataset = load_dataset(
    "/Users/ang/Downloads/llm-demo/datasets/ultrachat_200k",
    split="train_sft[:16]",
)

recipe = [
    IMatrixGatherer(ignore=["lm_head"]),
    QuantizationModifier(
        scheme="W8A8",
        targets=["Linear"],
        weight_observer="imatrix_mse",
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
    # Omit device to use the same automatic accelerator selection as oneshot.
    # Set True only when crash recovery is worth the intermediate disk writes.
    checkpoint_progress=False,
    # Replace a previous incomplete or completed quantization at OUTPUT_DIR.
    overwrite_output=True,
)
