"""Measure CPU weight-decode scaling across PyTorch threads and processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

import torch
from transformers import Qwen3VLForConditionalGeneration

from llmcompressor.streaming import load_streaming_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument(
        "--threads", type=int, nargs="+", default=[1, 8, 16, 32, 104]
    )
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--shard", type=int)
    return parser.parse_args()


def decoder_weight_names(weights, layers: int) -> list[str]:
    names = []
    prefix = "model.language_model.layers."
    for name, weight in weights._weights.items():
        if weight.weight_bits is None or not name.startswith(prefix):
            continue
        index = int(name[len(prefix) :].split(".", 1)[0])
        if index < layers:
            names.append(name)
    return names


def decode_names(weights, names: list[str]) -> float:
    started = time.perf_counter()
    for name in names:
        weights._materialize(name)
    return time.perf_counter() - started


def load_weights(checkpoint: str):
    return load_streaming_model(
        checkpoint,
        device="cpu",
        dtype=torch.bfloat16,
        model_factory=Qwen3VLForConditionalGeneration,
        preload_module_classes=[
            "Qwen3VLVisionBlock",
            "Qwen3VLTextDecoderLayer",
        ],
    )


def child_main(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads[0])
    with load_weights(args.checkpoint) as stream:
        names = decoder_weight_names(stream.weights, args.layers)
        names = names[args.shard :: args.shards]
        print(f"RESULT {decode_names(stream.weights, names):.6f}", flush=True)


def run_processes(args: argparse.Namespace, thread_count: int) -> tuple[float, list]:
    started = time.perf_counter()
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--checkpoint",
                args.checkpoint,
                "--layers",
                str(args.layers),
                "--threads",
                str(thread_count),
                "--shards",
                str(args.shards),
                "--shard",
                str(index),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for index in range(args.shards)
    ]
    times = []
    for process in processes:
        stdout, _ = process.communicate(timeout=600)
        if process.returncode != 0:
            raise RuntimeError(f"Decode process failed with {process.returncode}")
        result = next(line for line in stdout.splitlines() if line.startswith("RESULT"))
        times.append(float(result.split()[1]))
    return time.perf_counter() - started, times


def main() -> None:
    args = parse_args()
    if args.shard is not None:
        child_main(args)
        return

    with load_weights(args.checkpoint) as stream:
        weights = stream.weights
        names = decoder_weight_names(weights, args.layers)
        if not names:
            raise RuntimeError("No quantized decoder weights were discovered")
        total_bytes = sum(weights._weights[name].nbytes for name in names)
        print(
            f"weights={len(names)} logical_bytes={total_bytes / 2**30:.2f}GiB "
            f"layers={args.layers}",
            flush=True,
        )
        decode_names(weights, [names[0]])

        for thread_count in args.threads:
            torch.set_num_threads(thread_count)
            serial = decode_names(weights, names)

            results = [0.0] * args.shards

            def run_thread(index: int) -> None:
                results[index] = decode_names(weights, names[index :: args.shards])

            started = time.perf_counter()
            workers = [
                threading.Thread(target=run_thread, args=(index,))
                for index in range(args.shards)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            thread_wall = time.perf_counter() - started

            process_wall, process_times = run_processes(args, thread_count)
            print(
                f"threads={thread_count:3d} serial={serial:.3f}s "
                f"two_threads={thread_wall:.3f}s "
                f"(shards={','.join(f'{value:.3f}' for value in results)}) "
                f"two_processes={process_wall:.3f}s "
                f"(shards={','.join(f'{value:.3f}' for value in process_times)})",
                flush=True,
            )


if __name__ == "__main__":
    main()
