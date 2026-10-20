import importlib.util
import json
from copy import deepcopy
from functools import partial
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

from llmcompressor.modeling.deepseekv41 import (
    DeepseekV41Cache,
    DeepseekV41NativeForCausalLM,
    ModelConfig,
)
from llmcompressor.modeling.deepseekv41 import quantization as deepseekv41_quantization
from llmcompressor.modeling.deepseekv41.engram import (
    CheckpointEngramLookup,
    EngramHash,
)
from llmcompressor.modeling.deepseekv41.model import (
    DeepseekV41Block,
    DeepseekV41MoE,
    _candidate_mask,
    _draft_inputs,
)
from llmcompressor.modeling.moe_context import moe_calibration_context
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.pipelines.sequential.plan import trace_sequential_plan
from llmcompressor.streaming import (
    DeepSeekV41WeightMaterializer,
    DiskBoundaryActivationStore,
    HostMemoryBudget,
    compare_cached_to_recompute,
    load_streaming_model,
    streaming_oneshot,
)
from llmcompressor.streaming.inference.quantization import resolve_schemes
from llmcompressor.utils import ImatrixFallbackStats

EXAMPLE_PATH = (
    Path(__file__).parents[3] / "examples/streaming_oneshot/deepseek_v41_wNa8.py"
)
EXAMPLE_SPEC = importlib.util.spec_from_file_location(
    "deepseek_v41_example", EXAMPLE_PATH
)
EXAMPLE = importlib.util.module_from_spec(EXAMPLE_SPEC)
EXAMPLE_SPEC.loader.exec_module(EXAMPLE)


def tiny_config(*, engram=False, vision=False, dspark=True):
    stages = 3 if dspark else 0
    return ModelConfig(
        text_config=dict(
            vocab_size=32,
            hidden_size=32,
            moe_intermediate_size=32,
            num_hidden_layers=5,
            num_nextn_predict_layers=stages,
            num_attention_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=32,
            head_dim=32,
            qk_rope_head_dim=8,
            o_groups=2,
            o_lora_rank=16,
            sliding_window=4,
            max_position_embeddings=64,
            max_seq_len=64,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=2,
            hc_mult=2,
            compress_ratios=[0, 2, 2, 1, 1] + [0] * stages,
            kv_source_layer_ids=[1, 3],
            index_source_layer_ids=[1, 3, 4],
            candidate_source_layer_id=3,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            engram_layer_ids=[1] if engram else [],
            engram_num_embeddings=[5] if engram else [],
            engram_vocab_size=5,
            engram_n_heads=1,
            engram_head_dim=32,
            engram_max_ngram_size=2,
            engram_compressed_vocab_size=32,
            dspark_block_size=3 if dspark else 0,
            dspark_noise_token_id=31,
            dspark_target_layer_ids=[2, 3, 4] if dspark else [],
            dspark_markov_rank=8,
            dspark_n_routed_experts=2,
            dspark_num_experts_per_tok=1,
        ),
        vision_config=dict(
            num_hidden_layers=1 if vision else 0,
            hidden_size=32,
            num_attention_heads=2,
            intermediate_size=32,
            patch_size=2,
            downsample_ratio=2,
        ),
    )


def test_v41_linear_uses_native_gemm_for_matching_bf16_dtypes(monkeypatch):
    module = deepseekv41_quantization.Linear(8, 4, dtype=torch.bfloat16)
    inputs = torch.randn(3, 8, dtype=torch.bfloat16)
    calls = []
    original = deepseekv41_quantization.F.linear

    def record_linear(input, weight, bias=None):
        calls.append((input.dtype, weight.dtype, None if bias is None else bias.dtype))
        return original(input, weight, bias)

    monkeypatch.setattr(deepseekv41_quantization.F, "linear", record_linear)
    output = module(inputs)

    assert calls == [(torch.bfloat16, torch.bfloat16, None)]
    assert output.dtype == torch.bfloat16


def test_v41_linear_keeps_fp32_fallback_for_mismatched_dtypes(monkeypatch):
    module = deepseekv41_quantization.Linear(8, 4, dtype=torch.float32)
    inputs = torch.randn(3, 8, dtype=torch.bfloat16)
    calls = []
    original = deepseekv41_quantization.F.linear

    def record_linear(input, weight, bias=None):
        calls.append((input.dtype, weight.dtype, None if bias is None else bias.dtype))
        return original(input, weight, bias)

    monkeypatch.setattr(deepseekv41_quantization.F, "linear", record_linear)
    output = module(inputs)

    assert calls == [(torch.float32, torch.float32, None)]
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("calibrate_all_experts", [False, True])
def test_moe_sorted_routes_preserve_outputs_and_expert_calibration_inputs(
    dtype, calibrate_all_experts
):
    from llmcompressor.observers.base import Observer

    config = tiny_config(dspark=False).text_config
    config.n_routed_experts = 6
    torch.manual_seed(17)
    model = DeepseekV41MoE(config).to(dtype)
    model.calibrate_all_experts = calibrate_all_experts
    with torch.no_grad():
        model.gate.weight.normal_(0, 0.05)
        model.gate.bias.copy_(torch.tensor([100, 90, 0, 0, -100, -100]))
        model.gate.bias_vl.copy_(torch.tensor([0, 0, 100, 90, -100, -100]))
    inputs = torch.randn(2, 5, config.dim, dtype=dtype)
    image_mask = torch.tensor([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0]]).bool()
    linears = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    observers = {
        name: Observer.load_from_registry(
            "imatrix_mse",
            base_name="weight",
            args=EXAMPLE.QuantizationArgs(
                num_bits=4, strategy="channel", observer="imatrix_mse"
            ),
        )
        for name in linears
    }
    for name, observer in observers.items():
        observer.attach(linears[name])

    captured = {}
    handles = [
        module.register_forward_pre_hook(
            lambda _module, args, name=name: captured.update(
                {name: args[0].detach().clone()}
            )
        )
        for name, module in linears.items()
    ]
    try:
        with torch.no_grad():
            flat = inputs.flatten(0, 1)
            weights, indices = model.gate(flat, image_mask.flatten())
            expected = torch.zeros_like(flat, dtype=torch.float32)
            for expert_id, expert in enumerate(model.experts):
                rows, slots = torch.where(indices == expert_id)
                if calibrate_all_experts:
                    all_weights = flat.new_ones((flat.shape[0], 1), dtype=weights.dtype)
                    all_weights[rows, 0] = weights[rows, slots]
                    values = expert(flat, all_weights)[rows]
                elif rows.numel():
                    values = expert(flat[rows], weights[rows, slots, None])
                else:
                    continue
                expected.index_add_(0, rows, values.float())
            expected = (
                (expected + model.shared_experts(flat)).to(dtype).reshape_as(inputs)
            )
            expected_inputs = dict(captured)
            expected_statistics = {
                name: (observer._imatrix_sum.clone(), observer._imatrix_count.clone())
                for name, observer in observers.items()
            }
            for observer in observers.values():
                observer._imatrix_sum.zero_()
                observer._imatrix_count.zero_()
            captured.clear()
            actual = model(inputs, image_mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert captured.keys() == expected_inputs.keys()
        for name, tensor in captured.items():
            torch.testing.assert_close(tensor, expected_inputs[name], rtol=0, atol=0)
        for name, observer in observers.items():
            total, count = expected_statistics[name]
            torch.testing.assert_close(observer._imatrix_sum, total, rtol=0, atol=0)
            torch.testing.assert_close(observer._imatrix_count, count, rtol=0, atol=0)
        assert ("experts.5.w2" in captured) == calibrate_all_experts
    finally:
        for handle in handles:
            handle.remove()
        for name, observer in observers.items():
            observer.detach(linears[name])


def test_official_fp32_parameter_exceptions_survive_bf16_default():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = DeepseekV41NativeForCausalLM(tiny_config())
    finally:
        torch.set_default_dtype(previous)
    assert model.model.embed.weight.dtype == torch.bfloat16
    assert model.model.layers[0].attn.wq_a.weight.dtype == torch.bfloat16
    assert model.model.layers[1].attn.compressor.wkv.weight.dtype == torch.float32
    assert model.model.layers[1].attn.compressor.wgate.weight.dtype == torch.float32
    assert model.model.layers[3].attn.compressor.wkv.weight.dtype == torch.bfloat16
    assert model.model.lm_head.weight.dtype == torch.float32
    final_draft = model.model.mtp[-1]
    assert final_draft.markov_head.head.weight.dtype == torch.float32
    assert final_draft.confidence_head.proj.weight.dtype == torch.float32


def write_checkpoint(root, config):
    root.mkdir()
    model = DeepseekV41NativeForCausalLM(config, engram_token_map=list(range(32)))
    tensors = {}
    for name, tensor in model.state_dict().items():
        name = name.removeprefix("model.")
        if name.startswith("lm_head."):
            name = "head." + name.removeprefix("lm_head.")
        if ".engram.embed.weight" in name:
            tensors[name] = tensor.to(torch.float8_e4m3fn)
            tensors[name.replace(".weight", ".scale")] = torch.ones(
                tensor.shape[0], tensor.shape[1] // 32
            ).to(torch.float8_e8m0fnu)
        else:
            tensors[name] = tensor.contiguous()
    save_file(tensors, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in tensors.values()
                    )
                },
                "weight_map": {name: "model.safetensors" for name in tensors},
            }
        )
    )
    value = config.to_dict()
    value.update(model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"])
    (root / "config.json").write_text(json.dumps(value))
    return tensors


@pytest.mark.parametrize("use_auto_factory", [False, True])
def test_streaming_keeps_official_fp32_payloads_without_bf16_roundtrip(
    tmp_path, use_auto_factory
):
    checkpoint = tmp_path / "source"
    tensors = write_checkpoint(checkpoint, tiny_config())
    raw_names = [
        "layers.1.attn.compressor.wkv.weight",
        "layers.1.attn.compressor.wgate.weight",
        "head.weight",
        "mtp.2.markov_head.head.weight",
        "mtp.2.confidence_head.proj.weight",
    ]
    # This value cannot survive FP32 -> BF16 -> FP32.
    sentinel = 1 + 2**-16
    for name in raw_names:
        tensors[name].fill_(sentinel)
    save_file(tensors, checkpoint / "model.safetensors")
    with load_streaming_model(
        checkpoint,
        config=ModelConfig.from_pretrained(checkpoint, local_files_only=True),
        model_factory=None if use_auto_factory else DeepseekV41NativeForCausalLM,
        model_kwargs={"engram_token_map": list(range(32))},
        materializer=DeepSeekV41WeightMaterializer(),
    ) as inference:
        assert inference.weights.tensor_reads == 0
        for raw in raw_names:
            logical = "model." + (
                raw.replace("head.", "lm_head.", 1) if raw == "head.weight" else raw
            )
            actual = inference.weights[logical]
            assert actual.dtype == torch.float32
            torch.testing.assert_close(actual, tensors[raw], atol=0, rtol=0)
        assert all(p.is_meta for p in inference.model.parameters())


def test_native_and_raw_autoconfig_dispatch_roundtrip(tmp_path):
    config = tiny_config()
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert loaded.text_config.hidden_size == 32
    assert loaded.text_config.n_mtp_layers == 3
    assert isinstance(
        AutoModelForCausalLM.from_config(loaded), DeepseekV41NativeForCausalLM
    )
    value = config.to_dict()
    value["model_type"] = "deepseek_v41"
    (tmp_path / "config.json").write_text(json.dumps(value))
    raw = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert raw.model_type == "deepseek_v41"
    assert raw.text_config.num_hash_layers == 0


def test_native_checkpoint_reload_restores_hash_and_rope_buffers(tmp_path):
    config = tiny_config(engram=True, vision=True)
    model = AutoModelForImageTextToText.from_config(
        config, engram_token_map=list(range(32))
    ).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    expected = model(ids).logits.detach()
    model.save_pretrained(tmp_path)
    loaded = AutoModelForImageTextToText.from_pretrained(
        tmp_path, local_files_only=True, engram_token_map=list(range(32))
    ).eval()
    for name, buffer in model.named_buffers(remove_duplicate=False):
        torch.testing.assert_close(
            loaded.get_buffer(name), buffer, rtol=0, atol=0, msg=name
        )
    torch.testing.assert_close(loaded(ids).logits, expected, rtol=0, atol=0)


def test_nested_source_quantization_does_not_trigger_generic_fp8_loader():
    value = tiny_config().to_dict()
    value["quantization_config"] = {
        "quant_method": "fp8",
        "expert_dtype": "fp4",
        "weight_block_size": [32, 32],
    }
    config = ModelConfig.from_dict(value)
    assert not hasattr(config, "quantization_config")
    assert config.source_quantization_config["expert_dtype"] == "fp4"
    assert config.text_config.hidden_size == 32
    reloaded = ModelConfig.from_dict(config.to_dict())
    assert reloaded.source_quantization_config == config.source_quantization_config
    value = reloaded.to_dict()
    value["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }
    exported = ModelConfig.from_dict(value)
    assert exported.quantization_config["quant_method"] == "compressed-tensors"
    assert exported.source_quantization_config == config.source_quantization_config


@pytest.mark.parametrize("length", [1, 2, 3, 8, 9])
def test_csa2_forward_handles_short_and_incomplete_groups_without_state_leaks(length):
    torch.manual_seed(4)
    model = DeepseekV41NativeForCausalLM(tiny_config(dspark=False)).eval()
    ids = torch.arange(length).remainder(32).unsqueeze(0)
    with torch.no_grad():
        first = model(ids).logits
        model((ids + 1).remainder(32))
        repeated = model(ids).logits
    assert first.shape == (1, length, 32)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert model.model.layers[2].attn.compressor is None
    assert model.model.layers[4].attn.indexer is not None
    assert not hasattr(model.model.layers[4].attn.indexer, "wk")


@pytest.mark.parametrize("prompt_length", [1, 2, 3, 5])
def test_cached_decode_matches_recompute_across_compression_and_window(prompt_length):
    config = tiny_config(engram=True, dspark=True)
    config.calibrate_dspark = False
    config.text_config.engram_max_ngram_size = 4
    config.text_config.engram_num_embeddings = [5 + 7 + 11]
    torch.manual_seed(13)
    model = DeepseekV41NativeForCausalLM(
        config, engram_token_map=list(range(32))
    ).eval()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    lengths = []
    handle = model.model.layers[0].register_forward_pre_hook(
        lambda _module, args: lengths.append(args[0].shape[1])
    )
    with torch.no_grad():
        output = model(ids[:, :prompt_length], use_cache=True)
        torch.testing.assert_close(
            output.logits,
            model(ids[:, :prompt_length]).logits,
            rtol=2e-5,
            atol=2e-6,
        )
        cache = output.past_key_values
        for end in range(prompt_length + 1, ids.shape[1] + 1):
            output = model(ids[:, end - 1 : end], past_key_values=cache, use_cache=True)
            assert lengths[-1] == 1
            torch.testing.assert_close(
                output.logits,
                model(ids[:, :end]).logits[:, -1:],
                rtol=2e-5,
                atol=2e-6,
            )
            assert cache.get_seq_length() == end
        assert set(cache.sources) == {1, 3}
        assert cache.states[1]["compressed"].shape[1] == 5
        assert cache.states[3]["compressed"].shape[1] == 10
        assert cache.history_ids.shape[1] == 3
        for index, state in enumerate(cache.states):
            assert state["window"].shape[1] <= 3
            if index not in cache.sources:
                assert "compressed" not in state and "index_k" not in state
            for value in state.values():
                assert value.device.type == "cpu"
        cache.reset()
        assert cache.get_seq_length() == 0
        repeated = model(ids[:, :prompt_length], use_cache=True).logits
        reset = model(
            ids[:, :prompt_length], past_key_values=cache, use_cache=True
        ).logits
        torch.testing.assert_close(repeated, reset, rtol=0, atol=0)
    handle.remove()


def test_chunked_prefill_and_greedy_generation_use_native_cache():
    config = tiny_config(engram=True, dspark=True)
    config.calibrate_dspark = False
    torch.manual_seed(3)
    model = DeepseekV41NativeForCausalLM(
        config, engram_token_map=list(range(32))
    ).eval()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11]])
    with torch.no_grad():
        cache = DeepseekV41Cache(config.text_config)
        chunks = []
        for start, end in [(0, 3), (3, 4), (4, 7), (7, 9)]:
            chunks.append(
                model(ids[:, start:end], past_key_values=cache, use_cache=True).logits
            )
        torch.testing.assert_close(
            torch.cat(chunks, dim=1), model(ids).logits, rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            model.generate(ids, use_cache=True, max_new_tokens=4, do_sample=False),
            model.generate(ids, use_cache=False, max_new_tokens=4, do_sample=False),
        )


def test_cached_validation_and_failure_require_reset(monkeypatch):
    config = tiny_config(dspark=False)
    model = DeepseekV41NativeForCausalLM(config).eval()
    cache = DeepseekV41Cache(config.text_config)
    ids = torch.tensor([[3, 4, 5]])
    with pytest.raises(ValueError, match="unpadded"):
        model(ids, attention_mask=torch.tensor([[0, 1, 1]]), use_cache=True)
    assert not cache._active
    original = model.model.layers[2].forward

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected decode failure")

    monkeypatch.setattr(model.model.layers[2], "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model(ids, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == 0 and cache._streaming_failed
    monkeypatch.setattr(model.model.layers[2], "forward", original)
    with pytest.raises(RuntimeError, match="incomplete"):
        model(ids, past_key_values=cache, use_cache=True)
    cache.reset()
    assert (
        model(
            ids, past_key_values=cache, use_cache=True
        ).past_key_values.get_seq_length()
        == 3
    )


@pytest.mark.parametrize(
    "preload", [[], ["DeepseekV41Block"], ["Expert", "Compressor", "Indexer"]]
)
def test_streaming_cached_module_groups_keep_hooks_and_state(tmp_path, preload):
    checkpoint = tmp_path / "source"
    write_checkpoint(checkpoint, tiny_config(engram=True, dspark=False))
    with load_streaming_model(
        checkpoint,
        config=ModelConfig.from_pretrained(checkpoint, local_files_only=True),
        model_factory=DeepseekV41NativeForCausalLM,
        model_kwargs={
            "engram_token_map": list(range(32)),
            "engram_checkpoint": checkpoint,
        },
        materializer=DeepSeekV41WeightMaterializer(),
        preload_module_classes=preload,
    ) as inference:
        result = compare_cached_to_recompute(
            inference,
            torch.tensor([[3, 4, 5]]),
            torch.tensor([[6, 7, 8, 9]]),
        )
        assert result["passed"], result
        assert all(p.is_meta for p in inference.model.parameters())


def test_candidate_pool_pins_latest_block_and_limits_search():
    scores = torch.tensor([[[10.0, 9.0, 8.0, 7.0, 1.0, -torch.inf]]])
    keep = _candidate_mask(scores, torch.tensor([[[5]]]), 1, 2)
    assert keep.tolist() == [[[False, False, False, False, True, True]]]


def test_sparse_ties_do_not_change_past_outputs_when_future_is_appended():
    config = tiny_config(dspark=False)
    torch.manual_seed(13)
    model = DeepseekV41NativeForCausalLM(config).eval()
    # Zero indexer query projections force ties, including across candidate blocks.
    for module in model.modules():
        if type(module).__name__ == "Indexer":
            torch.nn.init.zeros_(module.wq_b.weight)
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    with torch.no_grad():
        short = model(ids[:, :7]).logits
        longer = model(ids).logits[:, :7]
    torch.testing.assert_close(short, longer, rtol=2e-5, atol=2e-6)


def test_single_pass_mhc_consumes_previous_pre_mix(monkeypatch):
    config = tiny_config(dspark=False).text_config
    block = DeepseekV41Block(config, 0, vision=False)
    block.attn_norm = block.ffn_norm = torch.nn.Identity()
    observed = {}

    class Attention(torch.nn.Module):
        def forward(self, x, mask, shared):
            observed["attention_input"] = x
            return torch.zeros_like(x), shared

    class FFN(torch.nn.Module):
        def forward(self, x, image_mask):
            observed["ffn_input"] = x
            return torch.zeros_like(x)

    block.attn, block.ffn = Attention(), FFN()
    x = torch.stack((torch.ones(32), torch.full((32,), 3.0))).view(1, 1, 2, 32)
    identity = torch.eye(2).view(1, 1, 2, 2)
    next_pre = torch.tensor([[[0.0, 1.0]]])
    monkeypatch.setattr(
        block,
        "mixes",
        lambda _x, _kind: (next_pre, torch.zeros_like(next_pre), identity),
    )
    block(
        x,
        torch.tensor([[[1.0, 0.0]]]),
        (None,) * 5,
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
        None,
    )
    torch.testing.assert_close(observed["attention_input"], torch.ones(1, 1, 32))
    torch.testing.assert_close(observed["ffn_input"], torch.full((1, 1, 32), 3.0))


def test_dspark_uses_target_inputs_and_observes_all_draft_experts():
    config = tiny_config()
    model = DeepseekV41NativeForCausalLM(config)
    observed, targets, counts = {}, {}, {}
    handles = []
    for index in config.text_config.dspark_target_layer_ids:
        handles.append(
            model.model.layers[index].register_forward_pre_hook(
                lambda _module, args, i=index: targets.update(
                    {i: args[0].mean(2).clone()}
                )
            )
        )
    handles.append(
        model.model.mtp[0].main_proj.register_forward_pre_hook(
            lambda _module, args: observed.update(context=args[0].clone())
        )
    )
    for stage, block in enumerate(model.model.mtp):
        assert block.ffn.n_routed_experts == 2
        assert block.ffn.gate.topk == 1
        for index, expert in enumerate(block.ffn.experts):
            key = (stage, index)
            counts[key] = 0

            def record(_module, _args, _output, key=key):
                counts[key] += 1

            handles.append(expert.w2.register_forward_hook(record))
    heads = []
    for head in (model.model.mtp[-1].markov_head, model.model.mtp[-1].confidence_head):
        handles.append(
            head.register_forward_hook(
                lambda _module, _args, _output: heads.append(True)
            )
        )
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 0, 0]])
    mask = torch.tensor([[1] * 8 + [0, 0]])
    with moe_calibration_context(model, calibrate_all_experts=False):
        output = model(ids, attention_mask=mask).logits
    expected = torch.cat(
        [targets[index][:, :4] for index in config.text_config.dspark_target_layer_ids],
        -1,
    )
    torch.testing.assert_close(observed["context"], expected)
    assert len(heads) == 2
    assert all(count == 1 for count in counts.values())
    assert output.shape == (1, 10, 32)
    assert torch.isfinite(output).all()
    for handle in handles:
        handle.remove()


def test_engram_hash_resets_at_padding_and_image_boundaries():
    config = tiny_config(engram=True, dspark=False).text_config
    hashes = EngramHash(config, token_map=list(range(32)))
    ids = torch.tensor([[3, 4, 5]])
    full = hashes(ids, torch.ones_like(ids, dtype=torch.bool))
    blocked = hashes(ids, torch.tensor([[True, False, True]]))
    standalone = hashes(ids[:, 2:], torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(blocked[:, 2:], standalone)
    assert not torch.equal(full[:, 2:], standalone)


def test_dspark_keeps_absolute_context_positions_with_mixed_padding():
    ids = torch.arange(1, 13).expand(2, -1)
    mask = torch.tensor([[1] * 12, [0] * 3 + [1] * 6 + [0] * 3]).bool()
    hidden = ids.float().unsqueeze(-1)
    context, valid, draft, teacher, positions = _draft_inputs(
        ids, mask, (hidden, hidden + 100), 4, 3, 31
    )
    assert positions.tolist() == [[4, 5, 6, 7], [1, 2, 3, 4]]
    assert valid.tolist() == [[True] * 4, [False, False, True, True]]
    assert teacher.tolist() == [[9, 10, 11], [6, 7, 8]]
    assert draft.tolist() == [[9, 31, 31], [6, 31, 31]]
    torch.testing.assert_close(
        context,
        torch.tensor(
            [
                [[5, 105], [6, 106], [7, 107], [8, 108]],
                [[2, 102], [3, 103], [4, 104], [5, 105]],
            ]
        ).float(),
    )


def test_engram_preservation_and_bounded_lookup(tmp_path):
    checkpoint = tmp_path / "source"
    tensors = write_checkpoint(checkpoint, tiny_config(engram=True, dspark=False))
    name = "layers.1.engram.embed.weight"
    lookup = CheckpointEngramLookup(checkpoint, 1, (5, 32))
    ids = torch.tensor([[[0, 4, 0], [2, 1, 4]]])
    actual = lookup(ids)
    expected = tensors[name].float()[ids].bfloat16()
    torch.testing.assert_close(actual, expected)
    materializer = DeepSeekV41WeightMaterializer()
    source = materializer.create_source(str(checkpoint))
    assert not any(".engram.embed." in name for name in source.tensor_names())
    output = tmp_path / "output"
    output.mkdir()
    mapping = materializer.save_preserved_tensors(checkpoint, output)
    assert materializer.save_preserved_tensors(checkpoint, output) == mapping
    # Source shares this shard with ordinary weights: preserve only the table.
    assert set(mapping) == {name, name.replace(".weight", ".scale")}
    for output_name, filename in mapping.items():
        with safe_open(output / filename, framework="pt") as shard:
            assert torch.equal(
                shard.get_tensor(output_name).view(torch.uint8),
                tensors[output_name].view(torch.uint8),
            )


def test_block32_and_row32_source_scales_decode_independently():
    materializer = DeepSeekV41WeightMaterializer(preserve_engram=False)
    weight = torch.arange(64 * 96).reshape(64, 96).remainder(7).float()
    scales = torch.tensor([[0.5, 1, 2], [4, 2, 0.5]])
    name = "model.layers.0.attn.wq_a.weight"
    actual = materializer.materialize_cpu(
        name,
        {
            name: weight.to(torch.float8_e4m3fn),
            name.replace(".weight", ".scale"): scales.to(torch.float8_e8m0fnu),
        },
        target_dtype=torch.float32,
    )
    expected = weight * scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    name = "model.layers.1.engram.embed.weight"
    rows = torch.ones(3, 64, dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[0.5, 1], [2, 4], [8, 16]])
    actual = materializer.materialize_cpu(
        name,
        {
            name: rows,
            name.replace(".weight", ".scale"): scales.to(torch.float8_e8m0fnu),
        },
        target_dtype=torch.float32,
    )
    torch.testing.assert_close(actual, scales.repeat_interleave(32, 1))


@pytest.mark.parametrize("mode", ["wna8", "w8a8", "w4a8"])
def test_raw_quantization_selectors_resolve_same_modules_as_native_recipe(mode):
    model = DeepseekV41NativeForCausalLM(tiny_config(vision=True))
    groups, ignored = EXAMPLE._schemes(mode)
    config = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "format": "int-quantized",
            "config_groups": {
                name: group.model_dump(mode="json") for name, group in groups.items()
            },
            "ignore": ignored,
        }
    }
    original = deepcopy(config)
    native_schemes, native_format = resolve_schemes(
        model, config["quantization_config"]
    )
    materializer = DeepSeekV41WeightMaterializer()
    saved = materializer.transform_output_config(config)
    raw_schemes, raw_format = resolve_schemes(
        model,
        saved["quantization_config"],
        output_module_name=materializer.output_module_name,
    )

    assert config == original
    assert raw_format == native_format
    assert raw_schemes.keys() == native_schemes.keys()
    for name, scheme in raw_schemes.items():
        assert scheme.weights == native_schemes[name].weights
        assert scheme.input_activations == native_schemes[name].input_activations
    assert "model.mtp.0.ffn.experts.0.w1" in raw_schemes
    assert "model.mtp.0.main_proj" in raw_schemes
    assert not any(
        name.startswith(("model.vision.", "model.aligner.")) for name in raw_schemes
    )


def test_vision_branch_and_image_routing_bias():
    config = tiny_config(vision=True, dspark=False)
    model = DeepseekV41NativeForCausalLM(config)
    encoded = model.encode_image(torch.randn(4, 3, 2, 2), 2, 2)
    assert encoded.shape == (1, 32)
    assert torch.isfinite(encoded).all()
    gate = model.model.layers[0].ffn.gate
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.copy_(torch.tensor([100.0, 90.0, 0.0, 0.0]))
        gate.bias_vl.copy_(torch.tensor([0.0, 0.0, 100.0, 90.0]))
    _, selected = gate(torch.ones(2, 32), torch.tensor([False, True]))
    assert selected.tolist() == [[0, 1], [2, 3]]


def test_raw_image_inputs_match_prebuilt_image_embeddings():
    config = tiny_config(vision=True, dspark=False)
    model = DeepseekV41NativeForCausalLM(config).eval()
    input_ids = torch.randint(0, config.text_config.vocab_size, (1, 9))
    patches = torch.randn(16, 3, 2, 2)
    image_token_types = torch.tensor([[-1, 0, 1, 1, 2, 1, 1, 2, 3]])

    with torch.no_grad():
        features = model.encode_image(patches, 4, 4)
        inputs_embeds = model.get_input_embeddings()(input_ids).clone()
        inputs_embeds[:, 1] = model.model.image_start
        inputs_embeds[:, [2, 3, 5, 6]] = features
        inputs_embeds[:, [4, 7]] = model.model.image_newline
        inputs_embeds[:, 8] = model.model.image_end
        expected = model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_mask=image_token_types.ge(0),
        ).logits
        actual = model(
            input_ids=input_ids,
            image_patches=patches,
            image_height=4,
            image_width=4,
            image_token_types=image_token_types,
        ).logits

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_vision_model_declares_prefix_parameter_dependencies():
    model = DeepseekV41NativeForCausalLM(tiny_config(vision=True))
    assert model.streaming_runtime_parameter_dependencies("prefix") == (
        "model.image_start",
        "model.image_end",
        "model.image_newline",
    )
    assert model.streaming_runtime_parameter_dependencies("target") == ()


def test_image_boundary_embeddings_are_initialized():
    model = DeepseekV41NativeForCausalLM(tiny_config(vision=True))
    for name in ("image_start", "image_end", "image_newline"):
        weight = getattr(model.model, name)
        assert torch.isfinite(weight).all()
        assert weight.count_nonzero() == weight.numel()
        assert weight.abs().max() < 1


def test_padded_text_training_has_finite_gradients():
    model = DeepseekV41NativeForCausalLM(
        tiny_config(engram=True, dspark=False), engram_token_map=list(range(32))
    )
    ids = torch.tensor([[1, 2, 3] + [0] * 13])
    mask = torch.tensor([[1] * 3 + [0] * 13]).bool()
    loss = model(ids, attention_mask=mask, labels=ids.masked_fill(~mask, -100)).loss
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_disk_boundaries_preserve_shared_csa2_tensor_identity(tmp_path):
    budget = HostMemoryBudget(
        safety_reserve_bytes=0, system_available=lambda: 1_000_000
    )
    store = DiskBoundaryActivationStore(tmp_path / "boundaries", host_budget=budget)
    tensor = torch.ones(16)
    store.put(0, 0, {"shared": (tensor, None), "consumer": tensor})
    loaded = store.get(0, 0)
    assert loaded["shared"][0] is loaded["consumer"]


def test_sequential_trace_matches_eager_backbone_and_draft_inputs():
    model = DeepseekV41NativeForCausalLM(
        tiny_config(engram=True), engram_token_map=list(range(32))
    ).eval()
    sample = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 0]]),
        "attention_mask": torch.tensor([[1] * 8 + [0]]),
    }
    plan = trace_sequential_plan(model, sample)
    draft_contexts, confidence = [], []
    handles = [
        model.model.mtp[0].main_proj.register_forward_pre_hook(
            lambda _m, args: draft_contexts.append(args[0].clone())
        ),
        model.model.mtp[-1].confidence_head.register_forward_hook(
            lambda _m, _args, out: confidence.append(out.clone())
        ),
    ]
    try:
        with torch.no_grad():
            expected = model(**sample).logits
            values = dict(sample)
            for subgraph in plan.subgraphs:
                output = subgraph.forward(
                    model, **{name: values[name] for name in subgraph.input_names}
                )
                values.update(output)
        torch.testing.assert_close(output["logits"], expected, rtol=0, atol=0)
        assert len(draft_contexts) == len(confidence) == 2
        torch.testing.assert_close(draft_contexts[0], draft_contexts[1], rtol=0, atol=0)
        torch.testing.assert_close(confidence[0], confidence[1], rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize(
    ("checkpoint_progress", "mode", "save_raw_checkpoint_format"),
    [
        (False, "wna8", True),
        (True, "w8a8", True),
        (False, "w4a8", True),
        (False, "wna8", False),
    ],
)
def test_streaming_quantizes_backbone_and_dspark_preserving_engram(
    tmp_path, checkpoint_progress, mode, save_raw_checkpoint_format
):
    config = tiny_config(engram=True, vision=True)
    checkpoint = tmp_path / "source"
    source_tensors = write_checkpoint(checkpoint, config)
    config = ModelConfig.from_pretrained(checkpoint, local_files_only=True)
    factory = partial(
        DeepseekV41NativeForCausalLM,
        engram_token_map=list(range(32)),
        engram_checkpoint=checkpoint,
    )
    schemes, ignored = EXAMPLE._schemes(mode)
    data = DataLoader(
        [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                "attention_mask": torch.ones(8, dtype=torch.long),
            }
        ],
        batch_size=1,
    )
    stats = ImatrixFallbackStats()
    stats.install_hooks()
    materializer = DeepSeekV41WeightMaterializer(
        save_raw_checkpoint_format=save_raw_checkpoint_format
    )
    with stats:
        output = streaming_oneshot(
            model=checkpoint,
            model_config=config,
            model_factory=factory,
            dataset=data,
            dataset_fingerprint="a" * 64,
            recipe=[
                IMatrixGatherer(ignore=ignored),
                QuantizationModifier(
                    config_groups=schemes,
                    ignore=ignored,
                ),
            ],
            output_dir=tmp_path / "output",
            work_dir=tmp_path / "work",
            num_calibration_samples=1,
            max_seq_length=8,
            target_dtype=torch.float32,
            device="cpu",
            moe_calibrate_all_experts=True,
            materializer=materializer,
            checkpoint_progress=checkpoint_progress,
        )
    assert (output / "FINALIZED").exists()
    index = json.loads((output / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    output_name = materializer.output_tensor_name
    source_index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    # Use the original index as the oracle, independently of the materializer
    # under test. Source quantization scales may be replaced or removed by PTQ.
    source_weights = {name for name in source_index if not name.endswith(".scale")}
    if save_raw_checkpoint_format:
        expected_weights = source_weights
    else:
        expected_weights = {
            "model." + ("lm_head.weight" if name == "head.weight" else name)
            for name in source_weights
        }
    assert {
        name for name in mapping if not name.endswith((".scale", ".weight_scale"))
    } == expected_weights
    actual_mapping = {}
    for shard in output.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as file:
            for name in file.keys():
                assert name not in actual_mapping, name
                actual_mapping[name] = shard.name
    assert actual_mapping == mapping
    for name in (
        "mtp.0.main_proj.weight",
        "mtp.1.ffn.experts.1.w2.weight",
        "mtp.2.attn.wq_a.weight",
    ):
        assert output_name(f"model.{name}") in mapping
    name = output_name("model.mtp.1.ffn.experts.1.w2.weight")
    expert = load_file(output / mapping[name])[name]
    assert expert.dtype == torch.int8
    assert expert.shape == (32, 32 if mode == "w8a8" else 16)
    assert not {
        name for name in stats.no_importance_counter if name.startswith("model.mtp.")
    }
    for name in ("layers.1.engram.embed.weight", "layers.1.engram.embed.scale"):
        saved_name = output_name(f"model.{name}")
        actual = load_file(output / mapping[saved_name])[saved_name]
        assert torch.equal(
            actual.view(torch.uint8), source_tensors[name].view(torch.uint8)
        )
    saved = json.loads((output / "config.json").read_text())
    assert saved["model_type"] == "deepseek_v41"
    assert saved["text_config"]["hidden_size"] == 32
    assert saved["engram_storage"]["group_size"] == 32
    routed_targets = saved["quantization_config"]["config_groups"][
        "routed_w4a8" if mode == "wna8" else "all"
    ]["targets"]
    assert routed_targets[0].startswith(
        "re:(layers|mtp)" if save_raw_checkpoint_format else r"re:model\.(layers|mtp)"
    )
    assert ("head" if save_raw_checkpoint_format else "model.lm_head") in saved[
        "quantization_config"
    ]["ignore"]
    # Match selectors against physical output names, without logical-name aliases.
    # Each INT8/packed-INT4 tensor must be covered once, with its emitted scale.
    physical_modules = torch.nn.Module()
    quantized_physical_names = set()
    for name, shard_name in mapping.items():
        if not name.endswith(".weight"):
            continue
        with safe_open(output / shard_name, framework="pt", device="cpu") as file:
            if file.get_slice(name).get_dtype() != "I8":
                continue
        module_name = name.removesuffix(".weight")
        quantized_physical_names.add(module_name)
        parent = physical_modules
        *parents, leaf = module_name.split(".")
        for part in parents:
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent._modules[part]
        parent.add_module(leaf, torch.nn.Linear(1, 1, bias=False))
        scale_name = f"{module_name}." + (
            "scale" if save_raw_checkpoint_format else "weight_scale"
        )
        assert scale_name in mapping
    physical_schemes, _ = resolve_schemes(
        physical_modules, saved["quantization_config"]
    )
    assert set(physical_schemes) == quantized_physical_names
    for name in ("vision.patch_embed.proj.weight", "aligner.w1.weight", "head.weight"):
        logical = "model." + ("lm_head.weight" if name == "head.weight" else name)
        saved_name = output_name(logical)
        actual = load_file(output / mapping[saved_name])[saved_name]
        torch.testing.assert_close(actual, source_tensors[name], rtol=0, atol=0)
    assert AutoConfig.from_pretrained(output).text_config.n_mtp_layers == 3
    with load_streaming_model(
        output,
        model_factory=DeepseekV41NativeForCausalLM,
        config=ModelConfig.from_pretrained(output, local_files_only=True),
        model_kwargs={
            "engram_token_map": list(range(32)),
            "engram_checkpoint": output,
        },
        materializer=materializer,
    ) as inference:
        assert inference.weights.tensor_reads == 0
        inference.model.config.calibrate_dspark = False
        result = compare_cached_to_recompute(
            inference,
            torch.tensor([[3, 4, 5]]),
            torch.tensor([[6, 7, 8]]),
        )
        assert result["passed"], result
        assert not any(
            name.startswith("model.mtp.") for name in inference.weights.loads
        )
        assert all(p.is_meta for p in inference.model.parameters())
