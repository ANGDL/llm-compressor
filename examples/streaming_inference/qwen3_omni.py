"""Run streaming reference inference for a quantized Qwen3-Omni checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3OmniMoeForConditionalGeneration,
)

from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.streaming import load_streaming_model


class Qwen3OmniStreamingModel(Qwen3OmniMoeForConditionalGeneration):
    def forward(self, input_ids=None, **kwargs):
        return self.thinker(input_ids=input_ids, **kwargs)


def build_model(config):
    model = Qwen3OmniStreamingModel(config)
    linearize_moe(model)
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run streaming inference with a quantized Qwen3-Omni checkpoint"
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Path to the quantized Qwen3-Omni checkpoint",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt to send to the thinker model",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional local image path for multimodal inference",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--no-activation-qdq",
        action="store_true",
        help="Disable dynamic INT8 activation quantization during reference inference",
    )
    parser.add_argument(
        "--skip-thinking",
        action="store_true",
        help="Close the thinking block in the prompt and request a direct answer",
    )
    return parser.parse_args()


def build_inputs(processor, prompt, image_path, skip_thinking):
    content = []
    image = None
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    text = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if skip_thinking:
        if text.endswith("<think>\n"):
            text += "\n</think>\n\n"
        else:
            text += "<think>\n\n</think>\n\n"

    return processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=False,
    )


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(args.image)

    checkpoint = Path(args.model_id).expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")

    processor = AutoProcessor.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    inputs = build_inputs(
        processor,
        args.prompt,
        args.image,
        args.skip_thinking,
    )

    with load_streaming_model(
        checkpoint,
        model_factory=build_model,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        activation_qdq=not args.no_activation_qdq,
    ) as inference:
        generated = inference.generate(
            **inputs,
            return_audio=False,
            thinker_max_new_tokens=args.max_new_tokens,
            thinker_do_sample=False,
            thinker_use_cache=True,
            thinker_return_dict_in_generate=True,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = generated.sequences[:, input_length:]
        answer = processor.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=False,
        )
        print(answer)
        report = inference.report()
        print(
            "\nStreaming report: "
            f"device={report['device']}, "
            f"activation_mode={report['activation_mode']}, "
            f"tensor_reads={report['tensor_reads']}, "
            f"materializations={report['materializations']}"
        )


if __name__ == "__main__":
    main()
