"""Separate checkpoint accuracy from cache consistency on existing tiny artifacts.

This CPU-only diagnostic loads resident tiny models. It neither requantizes nor
changes checkpoint files. The optional mHC intervention is restored before exit
and is not a production inference implementation.
"""

from __future__ import annotations

import argparse
import gc
import json
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from llmcompressor.modeling.deepseekv41.model import hc_split
from llmcompressor.streaming import (
    DeepSeekV41WeightMaterializer,
    compare_cached_to_recompute,
    load_streaming_model,
)

if __package__:
    from .create_tiny_dsv41 import TOY_TEXTS
    from .validate_tiny_dsv41 import load_cpu_reference
else:
    from create_tiny_dsv41 import TOY_TEXTS
    from validate_tiny_dsv41 import load_cpu_reference


def load_resident(checkpoint, tokenizer):
    model, schemes = load_cpu_reference(checkpoint, tokenizer)
    # Reuse the streaming loader's declared BF16/FP32 parameter policy without
    # reading its payloads; the actual values come from the independent loader.
    with load_streaming_model(
        checkpoint,
        materializer=DeepSeekV41WeightMaterializer(),
        model_kwargs={"tokenizer": tokenizer, "engram_checkpoint": checkpoint},
    ) as inference:
        assert inference.weights.tensor_reads == 0
        for name, parameter in model.named_parameters():
            parameter.data = parameter.data.to(
                inference.model.get_parameter(name).dtype
            )
    model.config.calibrate_dspark = False
    return model, schemes


def audit_pair(root, tokenizer):
    mixed, schemes = load_resident(root / "wna8", tokenizer)
    int8, int8_schemes = load_resident(root / "w8a8", tokenizer)
    result = {
        "identical_w8_modules": 0,
        "different_w8_modules": [],
        "w4_vs_w8_modules": 0,
        "different_nonquantized_parameters": [],
        "different_parameter_dtypes": [],
        "identical_activation_schemes": 0,
        "different_activation_schemes": [],
    }
    for name, scheme in schemes.items():
        other = int8_schemes[name]
        if (
            scheme.input_activations == other.input_activations
            and scheme.output_activations == other.output_activations
        ):
            result["identical_activation_schemes"] += 1
        else:
            result["different_activation_schemes"].append(name)
    for name, parameter in mixed.named_parameters():
        other = int8.get_parameter(name)
        if parameter.dtype != other.dtype:
            result["different_parameter_dtypes"].append(name)
        scheme = schemes.get(name.removesuffix(".weight"))
        if name.endswith(".weight") and scheme is not None:
            if scheme.weights.num_bits == 4:
                result["w4_vs_w8_modules"] += 1
            elif torch.equal(parameter, other):
                result["identical_w8_modules"] += 1
            else:
                result["different_w8_modules"].append(name)
        elif not torch.equal(parameter, other):
            result["different_nonquantized_parameters"].append(name)
    return result


def quality_metrics(model, batches, baseline):
    squared_error, elements, reference_energy = 0.0, 0, 0.0
    agreement, tokens, loss, targets = 0, 0, 0.0, 0
    records = []
    for ids, expected in zip(batches, baseline):
        actual = model(ids, use_cache=False).logits.float()
        difference = actual - expected
        error = difference.square().sum().item()
        energy = expected.square().sum().item()
        matched = (actual.argmax(-1) == expected.argmax(-1)).sum().item()
        records.append(
            {
                "tokens": ids.numel(),
                "mse": error / difference.numel(),
                "max_abs": difference.abs().max().item(),
                "relative_l2": (error / max(energy, 1e-24)) ** 0.5,
                "top1_agreement": matched / ids.numel(),
            }
        )
        squared_error += error
        reference_energy += energy
        elements += difference.numel()
        agreement += matched
        tokens += ids.numel()
        loss += F.cross_entropy(
            actual[:, :-1].reshape(-1, actual.shape[-1]),
            ids[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        targets += ids[:, 1:].numel()
    return {
        "comparison": "ptq_vs_source_same_full_sequence",
        "mse": squared_error / elements,
        "rmse": (squared_error / elements) ** 0.5,
        "max_abs": max(row["max_abs"] for row in records),
        "relative_l2": (squared_error / max(reference_energy, 1e-24)) ** 0.5,
        "top1_agreement": agreement / tokens,
        "tokens": tokens,
        "perplexity": torch.tensor(loss / targets).exp().item(),
        "records": records,
    }


@contextmanager
def replace_method(modules, name, factory):
    """Keep interventions instance-local and restore their original binding."""
    originals = []
    try:
        for module in modules:
            originals.append((module, name in module.__dict__, getattr(module, name)))
            setattr(module, name, factory(module))
        yield
    finally:
        for module, local, method in originals:
            if local:
                setattr(module, name, method)
            else:
                delattr(module, name)


def rowwise_mixes(self, x, kind):
    flat = x.flatten(2).float()
    rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.config.norm_eps)
    weight = getattr(self, f"hc_{kind}_fn").float()
    value = torch.stack(
        [F.linear(row, weight) for row in flat.reshape(-1, flat.shape[-1])]
    )
    value = value.reshape(*flat.shape[:-1], -1) * rms
    return hc_split(
        value,
        getattr(self, f"hc_{kind}_scale").float(),
        getattr(self, f"hc_{kind}_base").float(),
        self.config.hc_mult,
        self.config.hc_sinkhorn_iters,
        self.config.hc_eps,
    )


def trace_rounding(model, ids):
    prefill = model(ids[:, :5], use_cache=True)
    records, counts, first_inputs = {}, {}, {}
    phase = "decode"

    def capture_first_input(_module, args):
        first_inputs[phase] = args[0].detach().clone()

    def instrument(module):
        original = module.post
        layer = module.layer_id

        def post(x, residual, p, comb):
            call = counts.get((phase, layer), 0)
            counts[phase, layer] = call + 1
            value = p.unsqueeze(-1) * x.unsqueeze(-2) + (
                comb.unsqueeze(-1) * residual.unsqueeze(-2)
            ).sum(2)
            result = original(x, residual, p, comb)
            torch.testing.assert_close(value.to(x.dtype), result, rtol=0, atol=0)
            records.setdefault((layer, call), {})[phase] = {
                key: tensor[:, -1:].detach().clone()
                for key, tensor in (
                    ("x", x),
                    ("residual", residual),
                    ("float", value),
                    ("rounded", result),
                )
            }
            return result

        return post

    handle = model.model.layers[0].register_forward_pre_hook(capture_first_input)
    try:
        with replace_method(model.model.layers, "post", instrument):
            actual = model(
                ids[:, 5:6], use_cache=True, past_key_values=prefill.past_key_values
            ).logits[:, -1]
            phase = "recompute"
            expected = model(ids[:, :6], use_cache=False).logits[:, -1]
    finally:
        handle.remove()
    first = None
    for (layer, call), values in records.items():
        a, b = values["decode"], values["recompute"]
        if torch.equal(a["rounded"], b["rounded"]):
            continue
        position = tuple((a["rounded"] != b["rounded"]).nonzero()[0].tolist())
        first = {
            "layer_zero_based": layer,
            "merge": "attention" if call == 0 else "ffn",
            "local_coordinate": list(position),
            "x_identical": torch.equal(a["x"], b["x"]),
            "residual_identical": torch.equal(a["residual"], b["residual"]),
            "fp32_decode": a["float"][position].item(),
            "fp32_recompute": b["float"][position].item(),
            "bf16_decode": a["rounded"][position].item(),
            "bf16_recompute": b["rounded"][position].item(),
        }
        break
    flat = first_inputs["recompute"].flatten(2).float()
    weight = model.model.layers[0].hc_attn_fn.float()
    batched = F.linear(flat, weight)[:, -1:]
    single = F.linear(flat[:, -1:], weight)
    with replace_method(
        model.model.layers, "mixes", lambda _module: MethodType(rowwise_mixes, _module)
    ):
        intervention = compare_cached_to_recompute(model, ids[:, :5], ids[:, 5:8])
    return {
        "comparison": "same_checkpoint_cache_vs_recompute",
        "first_decode_logit_max_abs": (actual - expected).abs().max().item(),
        "first_bf16_post_divergence": first,
        "same_input_mhc_linear_batched_vs_singleton_max_abs": (batched - single)
        .abs()
        .max()
        .item(),
        "diagnostic_rowwise_mhc": intervention,
        "default_model_methods_restored": all(
            "mixes" not in layer.__dict__ and "post" not in layer.__dict__
            for layer in model.model.layers
        ),
    }


@torch.inference_mode()
def diagnose(root):
    creation = json.loads((root / "creation.json").read_text())
    if creation["parameters"] > 100_000_000:
        raise ValueError("Resident diagnostics are limited to tiny models under 100M")
    tokenizer = AutoTokenizer.from_pretrained(
        root / "native-finetuned", local_files_only=True
    )
    batches = [
        tokenizer(text, return_tensors="pt")["input_ids"][:, :64] for text in TOY_TEXTS
    ]
    ids = tokenizer(
        "The capital of France is Paris. Paris is the largest city in France.",
        return_tensors="pt",
    )["input_ids"]
    audit = audit_pair(root, tokenizer)
    source, _ = load_resident(root / "source-fp8-fp4", tokenizer)
    baseline = [source(tokens, use_cache=False).logits.float() for tokens in batches]
    del source
    gc.collect()
    results = {
        "reference": "source-fp8-fp4",
        "execution": "BF16 with explicitly declared FP32 controls; CPU",
        "software": {
            name: version(name)
            for name in ("torch", "transformers", "compressed-tensors")
        },
        "scope": "8 toy texts, at most 64 tokens each; functional tiny diagnostic",
        "checkpoint_audit": audit,
        "quality_vs_source": {},
        "cache_consistency": {},
    }
    for mode in ("wna8", "w8a8", "w4a8"):
        model, _ = load_resident(root / mode, tokenizer)
        quality = quality_metrics(model, batches, baseline)
        results["quality_vs_source"][mode] = quality
        results["cache_consistency"][mode] = {
            str(length): compare_cached_to_recompute(
                model, ids[:, :length], ids[:, length : length + 3]
            )
            for length in (3, 5, 7)
        }
        if mode == "w8a8":
            results["w8a8_rounding_diagnosis"] = trace_rounding(model, ids)
        print(
            f"{mode}: quality_mse={quality['mse']:.9g}, rmse={quality['rmse']:.9g}",
            flush=True,
        )
        del model
        gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.model.resolve()
    results = diagnose(root)
    destination = args.report or root / "precision-diagnosis.json"
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {destination}", flush=True)


if __name__ == "__main__":
    main()
