"""Optional conformance against the model directory's independent implementation.

Run with DEEPSEEK_V41_REFERENCE_DIR=/path/to/DeepSeek-V4.1-Flash/inference.
Only GPU kernels are replaced with CPU equivalents. Released weights, vision,
Engram and GPU kernels are outside this tiny BF16 forward comparison.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from llmcompressor.modeling.deepseekv4.kernels import hc_split, sparse_attention
from llmcompressor.modeling.deepseekv41 import DeepseekV41NativeForCausalLM
from tests.llmcompressor.modeling.test_deepseek_v41 import tiny_config


@pytest.fixture
def reference_pair(monkeypatch):
    root = os.environ.get("DEEPSEEK_V41_REFERENCE_DIR")
    if root is None:
        pytest.skip("Set DEEPSEEK_V41_REFERENCE_DIR to enable reference comparison")
    runtime = Path(root)

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
        assert inplace
        return x

    def fp4_quant(x, block_size=32, inplace=False, scale_dtype=None):
        assert inplace
        return x

    kernel = types.ModuleType("kernel")
    kernel.act_quant = act_quant
    kernel.fp4_act_quant = fp4_quant
    kernel.hc_split_sinkhorn = hc_split
    kernel.sparse_attn = sparse_attention
    kernel.fp4_gemm = kernel.fp8_gemm = None
    monkeypatch.setitem(sys.modules, "kernel", kernel)
    # Isolate the reference's unqualified sibling imports from other models.
    for name in ("engram", "image_processor", "vision"):
        sibling_spec = importlib.util.spec_from_file_location(
            name, runtime / f"{name}.py"
        )
        sibling = importlib.util.module_from_spec(sibling_spec)
        monkeypatch.setitem(sys.modules, name, sibling)
        sibling_spec.loader.exec_module(sibling)
    spec = importlib.util.spec_from_file_location(
        "dsv41_reference_cpu", runtime / "model.py"
    )
    reference = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, reference)
    spec.loader.exec_module(reference)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        torch.manual_seed(42)
        args = reference.ModelArgs(
            max_batch_size=1,
            max_seq_len=64,
            dtype="bf16",
            expert_dtype=None,
            vocab_size=32,
            dim=32,
            moe_inter_dim=32,
            n_layers=5,
            n_mtp_layers=3,
            n_heads=2,
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            route_scale=1.5,
            swiglu_limit=10,
            q_lora_rank=32,
            head_dim=32,
            rope_head_dim=8,
            norm_eps=1e-20,
            o_groups=2,
            o_lora_rank=16,
            window_size=4,
            compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
            kv_source_layers=(1, 3),
            index_source_layers=(1, 3, 4),
            compress_rope_theta=160000,
            original_seq_len=65536,
            rope_theta=10000,
            rope_factor=16,
            beta_fast=32,
            beta_slow=1,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=2,
            candidate_source_layer=3,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            hc_mult=2,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            engram_layer_ids=(),
            vision_n_layers=0,
            dspark_block_size=3,
            dspark_noise_token_id=31,
            dspark_target_layer_ids=(2, 3, 4),
            dspark_markov_rank=8,
            dspark_n_routed_experts=2,
            dspark_n_activated_experts=1,
        )
        model = reference.Transformer(args)
        for name, parameter in model.named_parameters():
            if ("norm" in name and name.endswith("weight")) or name.endswith("_scale"):
                parameter.data.fill_(1)
            elif name.endswith(("_base", ".bias")) or "attn_sink" in name:
                parameter.data.zero_()
            else:
                parameter.data.normal_(0, 0.05)
        native = DeepseekV41NativeForCausalLM(tiny_config())
        mapping = {}
        for name, tensor in model.state_dict().items():
            if name.startswith("mtp.") and name.split(".", 2)[-1] in (
                "head.weight",
                "embed.weight",
            ):
                continue
            logical = (
                "model.lm_head." + name.removeprefix("head.")
                if name.startswith("head.")
                else "model." + name
            )
            mapping[logical] = tensor.clone()
        native.load_state_dict(mapping, strict=True, assign=True)
        yield reference, model, native
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("length", [2, 3, 6, 8, 9, 13])
def test_backbone_and_dspark_match_supplied_reference(
    reference_pair, monkeypatch, length
):
    reference, model, native = reference_pair
    ids = torch.arange(1, length + 1).remainder(32).unsqueeze(0)
    captures = {"reference": [], "native": []}
    handles = []
    for name, blocks in (("reference", model.layers), ("native", native.model.layers)):
        for block in blocks:
            handles.append(
                block.register_forward_hook(
                    lambda _m, _a, out, name=name: captures[name].append(out[0].clone())
                )
            )
    try:
        native.config.calibrate_dspark = False
        with torch.no_grad():
            _, expected, main_hidden = model(ids)
            actual = native(ids).logits[:, -1]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        for actual, expected in zip(captures["native"], captures["reference"]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()
    seed = length - model.mtp[0].block_size - 1
    if seed < 2:
        return
    with torch.no_grad():
        model.forward_spec(ids[:, seed], main_hidden[:, : seed - 1])
    draws = iter(ids[:, seed + offset + 1] for offset in range(model.mtp[0].block_size))
    monkeypatch.setattr(reference, "sample", lambda _logits, _temperature: next(draws))
    captures = {"reference": [], "native": []}
    handles = []
    for name, blocks in (("reference", model.mtp), ("native", native.model.mtp)):
        for block in blocks:
            handles.append(
                block.register_forward_hook(
                    lambda _m, _a, out, name=name: captures[name].append(out[0].clone())
                )
            )
    for module, name in (
        (native.model.lm_head, "logits"),
        (native.model.mtp[-1].markov_head, "markov"),
        (native.model.mtp[-1].confidence_head, "confidence"),
    ):
        handles.append(
            module.register_forward_hook(
                lambda _m, _a, out, name=name: captures.update({name: out})
            )
        )
    native.config.calibrate_dspark = True
    try:
        with torch.no_grad():
            _, expected, confidence = model.forward_spec(
                ids[:, seed], main_hidden[:, seed - 1 : seed], seed - 1
            )
            native(ids)
        actual = captures["logits"] + captures["markov"][0]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(captures["confidence"], confidence, rtol=0, atol=0)
        for actual, expected in zip(captures["native"], captures["reference"]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("prompt_length", [2, 3, 5])
def test_cached_decode_matches_supplied_reference(reference_pair, prompt_length):
    _, model, native = reference_pair
    native.config.calibrate_dspark = False
    ids = torch.arange(1, prompt_length + 5).remainder(32).unsqueeze(0)
    with torch.no_grad():
        actual = native(ids[:, :prompt_length], use_cache=True)
        _, expected, _ = model(ids[:, :prompt_length])
        torch.testing.assert_close(actual.logits[:, -1], expected, rtol=1e-5, atol=1e-5)
        for position in range(prompt_length, ids.shape[1]):
            token = ids[:, position : position + 1]
            actual = native(
                token, past_key_values=actual.past_key_values, use_cache=True
            )
            _, expected, _ = model(token, start_pos=position)
            torch.testing.assert_close(
                actual.logits[:, -1], expected, rtol=1e-5, atol=1e-5
            )
            assert actual.past_key_values.get_seq_length() == position + 1
