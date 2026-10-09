"""Smoke test a Qwen3.5 vision checkpoint through an xSGL OpenAI API."""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30008")
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    encoded = base64.b64encode(args.image.read_bytes()).decode("ascii")
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in exactly two concise sentences. "
                            "Mention the main panels and the selected path."
                        ),
                    },
                ],
            }
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 128,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        document = json.loads(response.read())
    choices = document.get("choices") or []
    if not choices:
        raise RuntimeError(f"xSGL returned no choices: {document}")
    message = choices[0].get("message") or {}
    text = message.get("content") or ""
    if not text.strip():
        raise RuntimeError(
            f"xSGL returned an empty final content response: {document}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS",
                "runtime": "xsgl",
                "base_url": args.base_url,
                "model": args.model,
                "outputs": [{"text": text}],
                "finish_reason": choices[0].get("finish_reason"),
                "usage": document.get("usage"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(text)


if __name__ == "__main__":
    main()
