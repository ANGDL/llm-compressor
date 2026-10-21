"""Audit native mHC against local official source on existing tiny weights.

Only the official hc_mixes/hc_pre/hc_post functions are executed. Their GPU
Sinkhorn primitive is replaced by the existing CPU reference. Experiments are
instance-local and restore all original methods, dtypes and parameter values.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from llmcompressor.modeling.deepseekv41.model import hc_split
from llmcompressor.streaming import compare_cached_to_recompute

if __package__:
    from .diagnose_tiny_dsv41_precision import load_resident, replace_method
else:
    from diagnose_tiny_dsv41_precision import load_resident, replace_method


def official_hc_methods(directory):
    path = directory / "model.py"
    content = path.read_bytes()
    tree = ast.parse(content, filename=str(path))
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Block"
    )
    functions = [
        node
        for node in block.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("hc_mixes", "hc_pre", "hc_post")
    ]
    if len(functions) != 3:
        raise ValueError("Official Block must define hc_mixes, hc_pre and hc_post")
    namespace = {"torch": torch, "F": F, "hc_split_sinkhorn": hc_split}
    compile_unit = ast.Module(body=functions, type_ignores=[])
    exec(compile(compile_unit, str(path), "exec"), namespace)
    return namespace, {
        "model_path": str(path),
        "model_sha256": hashlib.sha256(content).hexdigest(),
        "method_lines": {node.name: node.lineno for node in functions},
        "kernel_path": str(directory / "kernel.py"),
        "kernel_sha256": hashlib.sha256(
            (directory / "kernel.py").read_bytes()
        ).hexdigest(),
        "kernel_execution": "existing CPU hc_split; official TileLang not executed",
    }


def official_mix_factory(function):
    def factory(module):
        context = SimpleNamespace(
            norm_eps=module.config.norm_eps,
            hc_mult=module.config.hc_mult,
            hc_sinkhorn_iters=module.config.hc_sinkhorn_iters,
            hc_eps=module.config.hc_eps,
        )

        def mixes(x, kind):
            return function(
                context,
                x,
                getattr(module, f"hc_{kind}_fn"),
                getattr(module, f"hc_{kind}_scale"),
                getattr(module, f"hc_{kind}_base"),
            )

        return mixes

    return factory


def fp64_projection(self, x, kind):
    flat = x.flatten(2).double()
    rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.config.norm_eps)
    # Deliberately diagnostic: official projection uses FP32, not FP64.
    value = (F.linear(flat, getattr(self, f"hc_{kind}_fn").double()) * rms).float()
    return hc_split(
        value,
        getattr(self, f"hc_{kind}_scale"),
        getattr(self, f"hc_{kind}_base"),
        self.config.hc_mult,
        self.config.hc_sinkhorn_iters,
        self.config.hc_eps,
    )


def fp32_parameter_exceptions(model):
    selected = []
    for name, parameter in model.named_parameters():
        owner_name = name.rpartition(".")[0]
        if owner_name.endswith((".compressor.wkv", ".compressor.wgate")):
            compressor = model.get_submodule(owner_name.rpartition(".")[0])
            if compressor.ratio > 1:
                selected.append((name, parameter))
        elif name.endswith(
            ("lm_head.weight", "markov_head.head.weight", "confidence_head.proj.weight")
        ):
            selected.append((name, parameter))
    return selected


@torch.inference_mode()
def audit(root, reference_directory):
    creation = json.loads((root / "creation.json").read_text())
    if creation["parameters"] > 100_000_000:
        raise ValueError("Resident diagnostics are limited to tiny models under 100M")
    functions, source = official_hc_methods(reference_directory)
    tokenizer = AutoTokenizer.from_pretrained(
        root / "native-finetuned", local_files_only=True
    )
    model, _ = load_resident(root / "w8a8", tokenizer)
    ids = tokenizer(
        "The capital of France is Paris. Paris is the largest city in France.",
        return_tensors="pt",
    )["input_ids"]
    results = {"official_source": source, "checkpoint": str(root / "w8a8")}

    def compare(label):
        value = compare_cached_to_recompute(model, ids[:, :5], ids[:, 5:8])
        results[label] = value
        print(label, [row["max_abs"] for row in value["records"]], flush=True)
        return value

    baseline = compare("baseline")
    with (
        replace_method(
            model.model.layers, "mixes", official_mix_factory(functions["hc_mixes"])
        ),
        replace_method(
            model.model.layers,
            "pre",
            lambda _module: lambda x, mix: functions["hc_pre"](None, x, mix),
        ),
        replace_method(
            model.model.layers,
            "post",
            lambda _module: (
                lambda x, residual, post, comb: functions["hc_post"](
                    None, x, residual, post, comb
                )
            ),
        ),
    ):
        official = compare("official_hc_methods")
    assert baseline == official
    exceptions = fp32_parameter_exceptions(model)
    original_values = [(name, parameter.data) for name, parameter in exceptions]
    results["parameter_exception_audit"] = [
        {
            "name": name,
            "current_dtype": str(parameter.dtype),
            "official_dtype": "float32",
        }
        for name, parameter in exceptions
    ]
    try:
        for _, parameter in exceptions:
            parameter.data = parameter.data.float()
        promoted = compare("official_fp32_parameter_exceptions")
        results["fp32_exceptions_changed_this_fixture"] = baseline != promoted
    finally:
        for name, value in original_values:
            model.get_parameter(name).data = value
    with replace_method(
        model.model.layers, "mixes", lambda module: MethodType(fp64_projection, module)
    ):
        compare("diagnostic_fp64_mhc_projection")
    block = model.model.layers[21]
    results["actual_mhc_parameter_dtypes"] = {
        name: str(parameter.dtype)
        for name, parameter in block.named_parameters(recurse=False)
        if name.startswith("hc_")
    }
    results["all_interventions_restored"] = all(
        not any(name in layer.__dict__ for name in ("mixes", "pre", "post"))
        for layer in model.model.layers
    ) and all(
        torch.equal(model.get_parameter(name), value) for name, value in original_values
    )
    assert results["all_interventions_restored"]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.model.resolve()
    results = audit(root, args.reference_dir.resolve())
    destination = args.report or root / "official-precision-audit.json"
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {destination}")


if __name__ == "__main__":
    main()
