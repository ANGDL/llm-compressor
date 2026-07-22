"""Quantize a local Qwen3-0.6B checkpoint with out-of-core W8A8 iMatrix RTN."""

from datasets import load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import streaming_oneshot

MODEL = "/path/to/Qwen3-0.6B"
OUTPUT_DIR = "Qwen3-0.6B-W8A8-IMatrix-RTN"
WORK_DIR = "streaming-work-qwen3-0.6b"

# As with oneshot(), the dataset can instead be a pre-tokenized Hugging Face
# Dataset or a PyTorch DataLoader. The Qwen3 streaming adapter prepares the
# decoder-layer-zero boundary; users do not need to construct activations.
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:16]")

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
)
