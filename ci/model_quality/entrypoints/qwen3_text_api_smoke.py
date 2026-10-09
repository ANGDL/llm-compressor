"""Smoke-test a text model through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8025")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="What is 17 multiplied by 23?")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": 64,
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
        raise RuntimeError(f"API returned no choices: {document}")
    message = choices[0].get("message") or {}
    text = message.get("content") or ""
    if not text.strip():
        raise RuntimeError(f"API returned empty content: {document}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS",
                "runtime": "openai-compatible-api",
                "base_url": args.base_url,
                "model": args.model,
                "outputs": [{"prompt": args.prompt, "text": text}],
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
