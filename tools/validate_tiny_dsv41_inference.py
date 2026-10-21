"""Validate streaming inference against the existing full-topology tiny fixture.

This reads existing source/PTQ artifacts; it does not recreate or requantize them.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from llmcompressor.streaming import (
    DeepSeekV41WeightMaterializer,
    compare_cached_to_recompute,
    load_streaming_model,
)

if __package__:
    from .validate_tiny_dsv41 import load_cpu_reference
else:
    from validate_tiny_dsv41 import load_cpu_reference


@torch.inference_mode()
def uncached_prefix_consistency(model, ids):
    """Check whether future masked rows change an already computed prefix."""
    short = model(ids[:, :6], use_cache=False).logits
    longer = model(ids[:, :7], use_cache=False).logits[:, :6]
    difference = (short - longer).float()
    return {
        "short_length": 6,
        "long_length": 7,
        "max_abs": difference.abs().max().item(),
        "relative_l2": (
            difference.norm() / short.float().norm().clamp_min(1e-12)
        ).item(),
        "allclose": torch.allclose(short, longer, rtol=2e-5, atol=2e-5),
        "rtol": 2e-5,
        "atol": 2e-5,
    }


def validate(root: Path, modes):
    root = root.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        root / "native-finetuned", local_files_only=True
    )
    ids = tokenizer(
        "The capital of France is Paris. Paris is the largest city in France.",
        return_tensors="pt",
    )["input_ids"]
    results = {}
    for mode in modes:
        checkpoint = root / mode
        reference, _ = load_cpu_reference(checkpoint, tokenizer)
        reference.config.calibrate_dspark = False
        # FP32 parameters reduce rounding noise, but native attention still has
        # low-precision intermediate Q/DQ. Record uncached prefix sensitivity
        # independently instead of attributing every discrepancy to cache state.
        prefix_consistency = uncached_prefix_consistency(reference, ids)
        fp32_comparison = compare_cached_to_recompute(
            reference, ids[:, :5], ids[:, 5:8], rtol=2e-5, atol=2e-5
        )
        with (
            load_streaming_model(
                checkpoint,
                materializer=DeepSeekV41WeightMaterializer(),
                model_kwargs={"tokenizer": tokenizer, "engram_checkpoint": checkpoint},
            ) as inference,
            torch.no_grad(),
        ):
            assert inference.weights.tensor_reads == 0
            # Match the declared per-parameter compute/control dtype policy.
            for name, parameter in reference.named_parameters():
                dtype = inference.model.get_parameter(name).dtype
                parameter.data = parameter.data.to(dtype)
            inference.model.config.calibrate_dspark = False
            actual = inference(ids[:, :5], use_cache=True)
            expected = reference(ids[:, :5], use_cache=True)
            prefill_error = (actual.logits - expected.logits).abs().max().item()
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
            decode_errors = []
            for end in (6, 7, 8):
                token = ids[:, end - 1 : end]
                actual = inference(
                    token, use_cache=True, past_key_values=actual.past_key_values
                )
                expected = reference(
                    token, use_cache=True, past_key_values=expected.past_key_values
                )
                decode_errors.append(
                    (actual.logits - expected.logits).abs().max().item()
                )
                torch.testing.assert_close(
                    actual.logits, expected.logits, rtol=0, atol=0
                )
            assert not any(
                name.startswith("model.mtp.") for name in inference.weights.loads
            )
            recompute = compare_cached_to_recompute(inference, ids[:, :5], ids[:, 5:8])
            epsilon = torch.finfo(torch.bfloat16).eps
            bf16_within_precision_budget = all(
                row["finite"]
                and row["relative_l2"] <= epsilon
                and row["max_abs"] <= epsilon * max(1e-12, row["reference_max_abs"])
                and row["top1_agreement"] == 1
                and row["cache_length"] == row["processed_tokens"]
                for row in recompute["records"]
            )
            cached_generation = inference.generate(
                ids[:, :5], use_cache=True, do_sample=False, max_new_tokens=3
            )
            resident_generation = reference.generate(
                ids[:, :5], use_cache=True, do_sample=False, max_new_tokens=3
            )
            torch.testing.assert_close(
                cached_generation, resident_generation, rtol=0, atol=0
            )

            # Exercise all vision blocks through their ordinary module hooks.
            patches = torch.linspace(-1, 1, 36 * 3 * 2 * 2).reshape(36, 3, 2, 2)
            features = inference.call("encode_image", patches, 6, 6)
            resident_features = reference.encode_image(patches, 6, 6)
            torch.testing.assert_close(features, resident_features, rtol=0, atol=0)

            # Teacher-forced DSpark is a separate diagnostic, with numerical taps.
            taps = {"streaming": {}, "resident": {}}
            handles = []
            for label, model in (
                ("streaming", inference.model),
                ("resident", reference),
            ):
                model.config.calibrate_dspark = True
                for name in (
                    "model.mtp.0",
                    "model.mtp.1",
                    "model.mtp.2",
                    "model.mtp.2.markov_head",
                    "model.mtp.2.confidence_head",
                ):

                    def capture(_module, _args, output, label=label, name=name):
                        value = output[0] if isinstance(output, tuple) else output
                        taps[label][name] = value.detach().clone()

                    handles.append(
                        model.get_submodule(name).register_forward_hook(capture)
                    )
            try:
                logits = inference(ids, use_cache=False).logits
                resident_logits = reference(ids, use_cache=False).logits
                torch.testing.assert_close(logits, resident_logits, rtol=0, atol=0)
                for name, value in taps["streaming"].items():
                    torch.testing.assert_close(
                        value, taps["resident"][name], rtol=0, atol=0
                    )
            finally:
                for handle in handles:
                    handle.remove()
            assert len(taps["streaming"]) == 5
            assert all(parameter.is_meta for parameter in inference.model.parameters())
            results[mode] = {
                "passed": fp32_comparison["passed"] and recompute["passed"],
                "streaming_parity_passed": True,
                "prefill_max_abs": prefill_error,
                "decode_max_abs": decode_errors,
                "cached_vs_recompute": recompute,
                "fp32_cached_vs_recompute": fp32_comparison,
                "fp32_uncached_prefix_consistency": prefix_consistency,
                "bf16_within_precision_budget": bf16_within_precision_budget,
                "bf16_relative_l2_limit": epsilon,
                "bf16_normalized_max_abs_limit": epsilon,
                "generated_ids": cached_generation.tolist(),
                "generation_matches_resident": True,
                "vision_feature_max_abs": (features - resident_features)
                .abs()
                .max()
                .item(),
                "dspark_taps_match": list(taps["streaming"]),
                "text_layers": len(inference.model.model.layers),
                "vision_layers": len(inference.model.model.vision.blocks),
                "draft_layers": len(inference.model.model.mtp),
                "materializations": inference.weights.materializations,
                "all_parameters_meta": True,
                "execution_dtype": "bfloat16; explicit FP32 controls preserved",
            }
            print(f"{mode}: {json.dumps(results[mode])}", flush=True)
        del reference, inference
        gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["source-fp8-fp4", "wna8", "w8a8", "w4a8"],
        choices=["source-fp8-fp4", "wna8", "w8a8", "w4a8"],
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    results = validate(args.model, args.modes)
    destination = args.report or args.model / "streaming-inference.json"
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if not all(value["passed"] for value in results.values()):
        raise SystemExit("Cached vs recompute failed; inspect the numerical report")


if __name__ == "__main__":
    main()
