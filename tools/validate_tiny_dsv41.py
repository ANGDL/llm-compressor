"""Validate the full-topology fixture, including the real streaming CLI.

python tools/validate_tiny_dsv41.py \
    .cache/tiny-models/DeepSeek-V4.1-Flash-Tiny

Raw and PTQ checkpoints are decoded here into a CPU reference model. This
checks serialization, dynamic INT8 activation Q/DQ and executable forward
paths; it does not validate a production inference engine or GPU kernels.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch
from compressed_tensors.quantization import QuantizationScheme, QuantizationStatus
from compressed_tensors.utils import match_named_modules
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer

from llmcompressor.modeling.deepseekv41 import (
    DeepseekV41NativeForConditionalGeneration,
    ModelConfig,
)
from llmcompressor.modeling.moe_context import moe_calibration_context
from llmcompressor.streaming import DeepSeekV41WeightMaterializer, build_meta_model
from llmcompressor.streaming.inference.quantization import reference_activation_qdq
from llmcompressor.streaming.materialization import materialize_weights

if __package__:
    from .create_tiny_dsv41 import perplexity, raw_name, toy_batches
else:
    from create_tiny_dsv41 import perplexity, raw_name, toy_batches


def index_tensors(directory):
    mapping = json.loads((directory / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    actual, size = {}, 0
    for filename in sorted(set(mapping.values())):
        with safe_open(directory / filename, framework="pt") as shard:
            for name in shard.keys():
                if name in actual or mapping.get(name) != filename:
                    raise AssertionError(f"Inconsistent checkpoint index: {name}")
                tensor = shard.get_tensor(name)
                if not torch.isfinite(tensor.float()).all():
                    raise AssertionError(f"Non-finite checkpoint tensor: {name}")
                actual[name] = filename
                size += tensor.numel() * tensor.element_size()
    if actual != mapping:
        raise AssertionError("Index contains missing tensors")
    metadata = json.loads((directory / "model.safetensors.index.json").read_text())[
        "metadata"
    ]
    assert metadata["total_size"] == size
    return mapping, size


def read_tensor(directory, mapping, name):
    with safe_open(directory / mapping[name], framework="pt") as shard:
        return shard.get_tensor(name)


def unpack_int4(weight):
    packed = weight.view(torch.uint8).int()
    low, high = packed & 15, packed >> 4
    signed = torch.stack(((low ^ 8) - 8, (high ^ 8) - 8), dim=-1)
    return signed.flatten(-2).to(torch.int8)


def _input_qdq(module, args):
    return (
        reference_activation_qdq(
            module, args[0], "input", module.quantization_scheme.input_activations
        ),
    )


def load_cpu_reference(directory, tokenizer):
    """Tiny-only loader for raw source or this workflow's packed INT4/INT8."""
    config = ModelConfig.from_pretrained(directory, local_files_only=True)
    model = build_meta_model(
        DeepseekV41NativeForConditionalGeneration,
        config,
        tokenizer=tokenizer,
        engram_checkpoint=directory,
        keep_nonpersistent_buffers=True,
    )
    qconfig = getattr(config, "quantization_config", {})
    compressed = qconfig.get("quantization_status") == "compressed"
    schemes = {}
    if compressed:
        for value in qconfig["config_groups"].values():
            scheme = QuantizationScheme.model_validate(value)
            for name, module in match_named_modules(
                model, scheme.targets, ignore=qconfig.get("ignore")
            ):
                if not isinstance(module, torch.nn.Linear):
                    raise AssertionError(f"Unexpected PTQ target type: {name}")
                schemes[name] = scheme
    else:
        materializer = DeepSeekV41WeightMaterializer()
        source = materializer.create_source(directory)
        weights = materialize_weights(
            source, model.state_dict(), materializer, target_dtype=torch.float32
        )
        model.load_state_dict(weights, strict=True, assign=True)
        return model.eval(), {}
    mapping, _ = index_tensors(directory)
    weights = {}
    for name, parameter in model.state_dict().items():
        raw = raw_name(name)
        tensor = read_tensor(directory, mapping, raw)
        module_name = name.removesuffix(".weight")
        if name.endswith(".weight") and module_name in schemes:
            scheme = schemes[module_name]
            assert tensor.dtype == torch.int8
            bits = scheme.weights.num_bits
            if bits == 4:
                assert tensor.shape[-1] * 2 == parameter.shape[-1]
                tensor = unpack_int4(tensor)
            else:
                assert bits == 8
            scale = read_tensor(
                directory, mapping, raw.removesuffix(".weight") + ".scale"
            )
            tensor = tensor.float() * scale.float()
            assert tuple(tensor.shape) == tuple(parameter.shape)
        weights[name] = tensor.float()
    model.load_state_dict(weights, strict=True, assign=True)
    for name, scheme in schemes.items():
        module = model.get_submodule(name)
        module.quantization_scheme = scheme
        module.quantization_status = QuantizationStatus.COMPRESSED
        module.register_forward_pre_hook(_input_qdq)
    return model.eval(), schemes


def mixed_image_inputs(model, tokenizer, *, alternate=False):
    ids = tokenizer(
        "The capital of France is Paris. Describe the picture.",
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"]
    # 6x6 ViT grid / 3x3 pixel unshuffle -> 2x2 image tokens.
    pixels = torch.linspace(-1, 1, 36 * 3 * 2 * 2).reshape(36, 3, 2, 2)
    if alternate:
        pixels = pixels.flip(0) + 0.5
    features = model.encode_image(pixels, 6, 6)
    assert features.shape == (4, model.config.text_config.dim)
    span = ids.new_full((1, 8), model.config.image_token_id)
    ids = torch.cat((ids[:, :2], span, ids[:, 2:]), dim=1)
    embeds = model.get_input_embeddings()(ids).clone()
    # start, image, image, newline, image, image, newline, end
    embeds[:, 2] = model.model.image_start
    embeds[:, [3, 4, 6, 7]] = features
    embeds[:, [5, 8]] = model.model.image_newline
    embeds[:, 9] = model.model.image_end
    image_mask = torch.zeros_like(ids, dtype=torch.bool)
    image_mask[:, 2:10] = True
    return dict(
        input_ids=ids,
        inputs_embeds=embeds,
        image_mask=image_mask,
        attention_mask=torch.ones_like(ids),
    )


@torch.no_grad()
def check_forward(model, tokenizer):
    counts, handles = Counter(), []
    expected = {}
    for name, module in model.named_modules():
        group = None
        if name.startswith("model.layers.") and name.count(".") == 2:
            group = "text"
        elif name.startswith("model.vision.blocks.") and name.count(".") == 3:
            group = "vision"
        elif name.startswith("model.mtp.") and name.count(".") == 2:
            group = "dspark"
        elif name.endswith(".engram"):
            group = "engram"
        elif name.endswith(".attn.compressor"):
            group = "compressor"
        elif name.endswith(".attn.indexer"):
            group = "indexer"
        elif name.endswith((".markov_head", ".confidence_head")):
            group = "draft_head"
        elif ".ffn.experts." in name and name.count(".") == 5:
            group = "expert"
        if group:
            expected[name] = group
            handles.append(
                module.register_forward_hook(
                    lambda _m, _a, _o, name=name: counts.update([name])
                )
            )
    text = tokenizer(
        "The capital of France is Paris. Paris is the largest city in France.",
        return_tensors="pt",
    )
    try:
        with moe_calibration_context(model, calibrate_all_experts=True):
            text_logits = model(**text).logits
            image_inputs = mixed_image_inputs(model, tokenizer)
            image_logits = model(**image_inputs).logits
            changed = model(
                **mixed_image_inputs(model, tokenizer, alternate=True)
            ).logits
        assert torch.isfinite(text_logits).all() and torch.isfinite(image_logits).all()
        assert not torch.equal(image_logits[:, 3:8], changed[:, 3:8])
        missing = sorted(set(expected) - set(counts))
        assert not missing, f"Unexecuted full-model modules: {missing}"
        coverage = dict(Counter(expected.values()))
        assert coverage["text"] == 40 and coverage["vision"] == 32
        assert coverage["dspark"] == 3 and coverage["engram"] == 2
        assert coverage["compressor"] == 4 and coverage["indexer"] == 8
        assert coverage["draft_head"] == 2 and coverage["expert"] == 332
        return (
            dict(
                coverage=coverage,
                text_logits_shape=list(text_logits.shape),
                image_logits_shape=list(image_logits.shape),
                image_response_max_abs=(image_logits - changed).abs().max().item(),
            ),
            text_logits,
            image_logits,
        )
    finally:
        for handle in handles:
            handle.remove()


def run_quantization(root, mode):
    output = root / mode
    if (output / "FINALIZED").is_file():
        return output
    example = Path(__file__).resolve().parents[1] / (
        "examples/streaming_oneshot/deepseek_v41_wNa8.py"
    )
    work = tempfile.mkdtemp(prefix=f"work-{mode}-cpu-", dir=root)
    command = [
        sys.executable,
        str(example),
        "--model-id",
        str(root / "source-fp8-fp4"),
        "--dataset-id",
        str(root / "calibration.jsonl"),
        "--output-dir",
        str(output),
        "--work-dir",
        work,
        "--quant-mode",
        mode,
        "--num-calibration-samples",
        "8",
        "--max-sequence-length",
        "64",
        "--batch-size",
        "2",
        "--device",
        "cpu",
        "--moe-calibrate-all-experts",
    ]
    if mode == "w8a8":
        command.append("--checkpoint-progress")
    if mode == "w4a8":
        command.append("--async-save")
    logfile = root / f"{mode}.log"
    print(f"Running real {mode.upper()} CLI; log: {logfile}", flush=True)
    environment = {
        **os.environ,
        "HF_HOME": str(root / "huggingface-cache"),
        "HF_DATASETS_CACHE": str(root / "huggingface-cache" / "datasets"),
        "HF_HUB_OFFLINE": "1",
    }
    with logfile.open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment
        )
    if result.returncode:
        raise RuntimeError(f"{mode} CLI failed with {result.returncode}; see {logfile}")
    assert (output / "FINALIZED").is_file()
    return output


def write_model_readme(root, results):
    creation = results["creation"]
    source = json.loads((Path(creation["source"]) / "config.json").read_text())
    tiny = json.loads((root / "native-finetuned" / "config.json").read_text())
    text, vision = tiny["text_config"], tiny["vision_config"]
    rows = [
        ("text", "num_hidden_layers"),
        ("vision", "num_hidden_layers"),
        ("text", "num_nextn_predict_layers"),
        ("text", "hidden_size"),
        ("text", "moe_intermediate_size"),
        ("text", "n_routed_experts"),
        ("text", "num_experts_per_tok"),
        ("text", "dspark_n_routed_experts"),
        ("text", "dspark_num_experts_per_tok"),
        ("text", "vocab_size"),
        ("text", "engram_num_embeddings"),
        ("vision", "hidden_size"),
    ]
    comparison = "| Config | Original | Tiny |\n|---|---:|---:|\n"
    for section, field in rows:
        comparison += (
            f"| {section}.{field} | {source[section + '_config'][field]} | "
            f"{tiny[section + '_config'][field]} |\n"
        )
    metrics = "| Checkpoint | Toy perplexity | Quantized linears | Tensor MiB |\n"
    metrics += "|---|---:|---:|---:|\n"
    metrics += f"| native-finetuned | {results['native_perplexity']:.4f} | 0 | — |\n"
    baseline = results["source"]
    metrics += (
        f"| source-fp8-fp4 | {baseline['perplexity']:.4f} | source FP8/FP4 | "
        f"{baseline['tensor_bytes'] / 2**20:.2f} |\n"
    )
    for mode, value in results["quantized"].items():
        metrics += (
            f"| {mode} | {value['perplexity']:.4f} | {value['quantized_linears']} | "
            f"{value['tensor_bytes'] / 2**20:.2f} |\n"
        )
    template = (
        Path(__file__).resolve().parents[1]
        / ".agents/skills/create-tiny-model/templates/README_TEMPLATE.md"
    ).read_text()
    readme = template.format_map(
        {
            "model_name": (
                f"DeepSeek-V4.1-Flash-Tiny-{creation['parameters'] / 1e6:.1f}M"
            ),
            "base_model_id": "deepseek-ai/DeepSeek-V4.1-Flash",
            "architecture": (
                "deepseek_v41_native / DeepseekV41NativeForConditionalGeneration"
            ),
            "total_params": f"{creation['parameters'] / 1e9:.6f}",
            "activated_params": (
                "Backbone top-6 of 8 experts; DSpark top-3 of 4. "
                "A precise per-token active parameter count is not estimated."
            ),
            "config_table": comparison,
            "checkpoint_description": (
                "`native-finetuned/`: reloadable, fine-tuned FP32 native checkpoint "
                f"(6 shards).\n\n`source-fp8-fp4/`: "
                f"{creation['source_checkpoint']['shards']} raw-format shards "
                "with original "
                "`embed/layers/head/vision/aligner/mtp` names, ordinary FP8 block32, "
                "packed E2M1 experts with group32 E8M0 scales and two dedicated Engram "
                "row32 FP8/E8M0 shards.\n\n`wna8/`, `w8a8/`, `w4a8/`: real streaming "
                "CLI exports, including DSpark and retained vision/Engram state."
            ),
            "creation process": (
                f"Preserved all {text['num_hidden_layers']} backbone, "
                f"{vision['num_hidden_layers']} ViT and "
                f"{text['num_nextn_predict_layers']} DSpark layers. Reduced widths, "
                "experts and Engram prime buckets for a CPU machine "
                "with limited memory. "
                "The original 129,280-token tokenizer is retained.\n\n"
                f"Fine-tuned text for {creation['training']['steps']} steps "
                "on the skill's "
                "8 toy texts. DSpark and vision are present and exercised but are not "
                "fine-tuned. No pretrained large-model weights are used.\n\n"
                "Rebuild with `python tools/create_tiny_dsv41.py --source "
                f"{creation['source']} --output /path/to/new-fixture`.\n\n"
                "Validate with `python tools/validate_tiny_dsv41.py "
                f"{root}`."
            ),
            "additional_notes": (
                "## Validation\n\n"
                + metrics
                + "\nReloaded generation: `"
                + results["generation"]
                + "`.\n\n"
                "Forward coverage includes all 40 text blocks, 32 vision blocks, "
                "3 DSpark blocks, 2 Engrams, 4 compressors, 8 indexers, 332 routed "
                "experts and both draft heads. Changing image patches changes text "
                "model logits. Exported checkpoints are reloaded into a CPU reference "
                "with dynamic INT8 activation Q/DQ; "
                "Engram bytes are compared exactly.\n\n"
                "Perplexities are measured on the same toy training corpus. This is "
                "a functional fixture, not a quality benchmark. GPU kernels, "
                "production engine loading, long-context behavior and speculative "
                "acceptance rates are not validated. Images use synthetic patches "
                "and prepared embeddings; "
                "automatic image prompt processing is outside this fixture.\n\n"
                "Local artifacts only. See `creation.json`, `validation.json` and "
                "`wna8.log` / `w8a8.log` / `w4a8.log` for evidence."
            ),
        }
    )
    start = readme.index("```python")
    end = readme.index("```", start + 3) + 3
    usage = f'''```python
import llmcompressor.modeling.deepseekv41  # register the native multimodal model
from transformers import AutoModelForImageTextToText, AutoTokenizer

path = "{root / "native-finetuned"}"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    path, tokenizer=tokenizer, local_files_only=True
)
model.config.calibrate_dspark = False
inputs = tokenizer("The capital of France is", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```'''
    readme = readme[:start] + usage + readme[end:]
    (root / "README.md").write_text(readme)
    (root / "native-finetuned" / "README.md").write_text(readme)


def validate(root, modes):
    root = root.resolve()
    creation = json.loads((root / "creation.json").read_text())
    native_path = root / "native-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(native_path, local_files_only=True)
    # Use the multimodal loader; source fixtures retain original raw naming and
    # therefore use the explicit CPU materializer below.
    native = AutoModelForImageTextToText.from_pretrained(
        native_path, local_files_only=True, tokenizer=tokenizer, dtype=torch.float32
    ).eval()
    native.config.calibrate_dspark = False
    native_ppl = perplexity(native, toy_batches(tokenizer))
    assert native_ppl <= creation["training"]["target_perplexity"]
    generated = native.generate(
        **tokenizer("The capital of France is", return_tensors="pt"),
        max_new_tokens=8,
        do_sample=False,
        use_cache=False,
    )
    generation = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Reloaded native toy perplexity: {native_ppl:.4f}; {generation}", flush=True)
    del native
    gc.collect()
    source_dir = root / "source-fp8-fp4"
    source_mapping, source_bytes = index_tensors(source_dir)
    source, _ = load_cpu_reference(source_dir, tokenizer)
    baseline, text, image = check_forward(source, tokenizer)
    source.config.calibrate_dspark = False
    source_ppl = perplexity(source, toy_batches(tokenizer))
    del source
    gc.collect()
    results = dict(
        creation=creation,
        native_perplexity=native_ppl,
        generation=generation,
        source=dict(**baseline, tensor_bytes=source_bytes, perplexity=source_ppl),
        quantized={},
        evaluation_scope="CPU reference; toy training corpus; no production engine",
    )
    for mode in modes:
        output = run_quantization(root, mode)
        mapping, size = index_tensors(output)
        model, schemes = load_cpu_reference(output, tokenizer)
        checks, actual_text, actual_image = check_forward(model, tokenizer)
        model.config.calibrate_dspark = False
        score = perplexity(model, toy_batches(tokenizer))
        assert math.isfinite(score)
        for name in source_mapping:
            if ".engram.embed." in name:
                expected = read_tensor(source_dir, source_mapping, name)
                actual = read_tensor(output, mapping, name)
                assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        # Vision is exported by the ordinary unquantized path, not lost when
        # the CLI calibrates text only.
        name = "vision.blocks.31.attn.wo.weight"
        expected = read_tensor(source_dir, source_mapping, name).bfloat16()
        actual = read_tensor(output, mapping, name)
        torch.testing.assert_close(actual.bfloat16(), expected, rtol=0, atol=0)
        checks.update(
            tensor_bytes=size,
            perplexity=score,
            quantized_linears=len(schemes),
            bits=dict(Counter(str(s.weights.num_bits) for s in schemes.values())),
            text_logit_mse=(text - actual_text).square().mean().item(),
            image_logit_mse=(image - actual_image).square().mean().item(),
            engram_bytes_preserved=True,
            vision_retained=True,
        )
        results["quantized"][mode] = checks
        print(
            f"{mode.upper()} PASS: {len(schemes)} linears, toy_ppl={score:.4f}, "
            f"text_mse={checks['text_logit_mse']:.6g}",
            flush=True,
        )
        del model
        gc.collect()
        (root / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    (root / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    write_model_readme(root, results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--quant-modes",
        nargs="*",
        choices=["wna8", "w8a8", "w4a8"],
        default=["wna8", "w8a8", "w4a8"],
    )
    args = parser.parse_args()
    results = validate(args.model, args.quant_modes)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
