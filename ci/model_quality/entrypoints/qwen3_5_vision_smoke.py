"""Smoke test a compressed Qwen3.5 vision checkpoint with a real image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    image = Image.new("RGB", (224, 224), color="blue")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What color is the image?"},
            ],
        }
    ]
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0"
    )
    model.eval()
    inputs = inputs.to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    response = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )[0].strip()
    if not response:
        raise RuntimeError("vision generation produced an empty response")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS",
                "runtime": "transformers",
                "model": str(args.model),
                "outputs": [{"prompt": "What color is the image?", "text": response}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(response)


if __name__ == "__main__":
    main()
