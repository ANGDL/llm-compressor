import os
import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.gpt_oss import convert_model_for_quantization_gptoss
from llmcompressor.modifiers.quantization import QuantizationModifier


def main():
    MODEL_ID = "/ssd1/models/QwQ-32B"
    BASE_NAME = MODEL_ID.rstrip("/").split("/")[-1]
    OUTPUT_DIR = f"{BASE_NAME}-w4a8-blocklwise"
    OUTPUT_DIR = os.path.join("/ssd1/model", OUTPUT_DIR)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Weights: 8-bit, blockwise, symmetric, static
    weights_args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.BLOCK,
        symmetric=True,
        dynamic=False,
        block_structure=[16, 16],
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

    recipe = QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    )

    oneshot(
        model=model,
        recipe=recipe,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        trust_remote_code_model=True,
    )


if __name__ == "__main__":
    main()