"""Minimal causal language model smoke test against a running SGLang server.

This is the non-vLLM runtime adapter used by ``runtime_smoke.command``. It talks
to an already running SGLang (xSGL) OpenAI-compatible HTTP server, so the
server lifecycle stays with the operator who owns that runtime container.

The pinned SGLang build used for the first DeepSeek-V4 run does not return output
token ids from ``/generate`` or ``/v1/completions``. Token ids are therefore
re-derived from the generated text with the checkpoint tokenizer and the record
states ``token_ids_source`` so no reader mistakes them for server-reported ids.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--served-model-name", default="deepseek-v4-flash")
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--request-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--wait-ready-seconds", type=float, default=0.0)
    parser.add_argument("--ready-poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def _http_json(
    url: str, *, payload: dict | None, timeout: float
) -> tuple[int, dict | None]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, (json.loads(body) if body.strip() else None)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {error.code}: {body[:500]}") from error


def _wait_until_ready(base_url: str, timeout: float, poll_seconds: float) -> float:
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    last_error: Exception | None = None
    while True:
        try:
            status, _ = _http_json(
                f"{base_url}/health", payload=None, timeout=min(60.0, poll_seconds + 30.0)
            )
            if status == 200:
                return time.monotonic() - started
        except Exception as error:  # noqa: BLE001 - readiness probe retries everything
            last_error = error
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"{base_url}/health was not ready within {timeout}s: {last_error}"
            )
        time.sleep(poll_seconds)


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def main() -> int:
    args = parse_args()
    prompts = json.loads(args.prompts_json)
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) and prompt for prompt in prompts)
    ):
        raise ValueError("prompts-json must contain a non-empty list of strings")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")

    base_url = args.base_url.rstrip("/")
    ready_seconds = None
    if args.wait_ready_seconds > 0:
        ready_seconds = _wait_until_ready(
            base_url, args.wait_ready_seconds, args.ready_poll_seconds
        )

    tokenizer = None
    tokenizer_error = None
    try:
        tokenizer = _load_tokenizer(args.model)
    except Exception as error:  # noqa: BLE001 - token ids are advisory
        tokenizer_error = f"{type(error).__name__}: {error}"

    records = []
    for prompt in prompts:
        payload = {
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        }
        if args.served_model_name:
            payload["model"] = args.served_model_name
        status, body = _http_json(
            f"{base_url}/v1/completions",
            payload=payload,
            timeout=args.request_timeout_seconds,
        )
        if status != 200 or not isinstance(body, dict):
            raise RuntimeError(f"/v1/completions returned HTTP {status}: {body}")
        choices = body.get("choices")
        if not choices or not isinstance(choices, list):
            raise RuntimeError(f"/v1/completions returned no choices: {body}")
        text = choices[0].get("text")
        if not text:
            raise RuntimeError(f"/v1/completions returned no text: {body}")
        usage = body.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        token_ids = None
        token_ids_source = None
        if tokenizer is not None:
            token_ids = list(tokenizer.encode(text, add_special_tokens=False))
            token_ids_source = "local_checkpoint_tokenizer"
        records.append(
            {
                "prompt": prompt,
                "text": text,
                "token_ids": token_ids,
                "token_ids_source": token_ids_source,
                "completion_tokens": completion_tokens,
                "finish_reason": choices[0].get("finish_reason"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS",
                "runtime": "sglang",
                "base_url": base_url,
                "served_model_name": args.served_model_name,
                "model": args.model,
                "ready_seconds": ready_seconds,
                "tokenizer_error": tokenizer_error,
                "outputs": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
