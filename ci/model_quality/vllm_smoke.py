"""Minimal causal language model smoke test using the real vLLM runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from vllm import LLM, SamplingParams

    prompts = json.loads(args.prompts_json)
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) and prompt for prompt in prompts)
    ):
        raise ValueError("prompts-json must contain a non-empty list of strings")
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
    records = []
    for prompt, output in zip(prompts, outputs, strict=True):
        if not output.outputs:
            raise RuntimeError(f"vLLM returned no candidates for prompt {prompt!r}")
        candidate = output.outputs[0]
        if not candidate.token_ids:
            raise RuntimeError(f"vLLM returned no tokens for prompt {prompt!r}")
        records.append(
            {
                "prompt": prompt,
                "text": candidate.text,
                "token_ids": list(candidate.token_ids),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"status": "PASS", "outputs": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
