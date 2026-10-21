"""Native cache/generation and quantized checkpoint round-trip integration tests."""

from __future__ import annotations

import json
from copy import deepcopy
from types import MethodType

import pytest
import torch
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    CompressedTensorsConfig,
    LlamaConfig,
    Qwen3Config,
)

from llmcompressor.streaming import (
    SafetensorsWeightSource,
    compare_cached_to_recompute,
    load_streaming_model,
)
from llmcompressor.utils.int4_packing import pack_int4_to_int8


def make_model(family="llama", *, tied=False):
    config_cls = LlamaConfig if family == "llama" else Qwen3Config
    config = config_cls(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=tied,
        eos_token_id=None,
        pad_token_id=0,
    )
    torch.manual_seed(13)
    return AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()


def quantized_checkpoint(model, root, *, bits, format_name, mixed=False):
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    groups = {}
    for index, (name, module) in enumerate(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        width = 8 if mixed and index % 2 else bits
        scheme = QuantizationScheme(
            targets=[name],
            format=format_name,
            weights=QuantizationArgs(
                num_bits=width,
                type="int",
                symmetric=True,
                strategy="channel",
            ),
            input_activations=QuantizationArgs(
                num_bits=8,
                type="int",
                symmetric=False,
                dynamic=True,
                strategy="token",
            ),
        )
        groups[name] = scheme
        weight = state.pop(f"{name}.weight")
        scale = weight.float().abs().amax(-1, keepdim=True) / (2 ** (width - 1) - 1)
        scale = scale.clamp_min(1e-6).to(torch.bfloat16)
        compressor = BaseCompressor.get_value_from_registry(format_name)
        values = compressor.compress({"weight": weight, "weight_scale": scale}, scheme)
        state.update({f"{name}.{key}": value for key, value in values.items()})
    root.mkdir()
    config = model.config
    config.quantization_config = QuantizationConfig(
        config_groups=groups,
        format=format_name,
        quantization_status=QuantizationStatus.COMPRESSED,
    ).model_dump(mode="json")
    config.save_pretrained(root)
    save_file(
        {name: value.contiguous() for name, value in state.items()},
        root / "model.safetensors",
    )


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("tied", [False, True])
def test_native_prefill_decode_generate_and_one_block_residency(tmp_path, family, tied):
    reference = make_model(family, tied=tied)
    reference.save_pretrained(tmp_path)
    ids = torch.tensor([[1, 3, 5, 7, 9]])
    decoder_class = type(reference.model.layers[0]).__name__
    with load_streaming_model(
        tmp_path, preload_module_classes=[decoder_class]
    ) as stream:
        assert stream.weights.tensor_reads == 0
        assert len(stream.weights) > 0
        assert "model.embed_tokens.weight" in stream.weights
        assert stream.weights.tensor_reads == 0
        observed = []
        for index, layer in enumerate(stream.model.model.layers):
            original = layer._old_forward

            def check(*args, index=index, original=original, **kwargs):
                resident = [
                    not all(p.is_meta for p in block.parameters())
                    for block in stream.model.model.layers
                ]
                assert resident == [j == index for j in range(2)]
                observed.append(args[0].shape[1])
                return original(*args, **kwargs)

            layer._old_forward = check
        with torch.no_grad():
            actual = stream(ids, use_cache=True)
            expected = reference(ids, use_cache=True)
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
            prefix = ids
            for token in (11, 12, 13):
                next_ids = torch.tensor([[token]])
                actual = stream(
                    next_ids, past_key_values=actual.past_key_values, use_cache=True
                )
                expected = reference(
                    next_ids, past_key_values=expected.past_key_values, use_cache=True
                )
                prefix = torch.cat((prefix, next_ids), dim=1)
                torch.testing.assert_close(
                    actual.logits, expected.logits, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual.logits[:, -1],
                    reference(prefix).logits[:, -1],
                    rtol=0.02,
                    atol=0.002,
                )
                assert actual.past_key_values.get_seq_length() == prefix.shape[1]
                assert all(p.is_meta for p in stream.model.parameters())
            assert observed[:8] == [5, 5, 1, 1, 1, 1, 1, 1]
            torch.testing.assert_close(
                stream.generate(ids, max_new_tokens=4, do_sample=False),
                reference.generate(ids, max_new_tokens=4, do_sample=False),
            )
        assert stream.report()["materializations"] > 0
    assert all(p.is_meta for p in stream.model.parameters())
    reads = stream.weights.tensor_reads
    stream.close()
    assert stream.weights.tensor_reads == reads
    with pytest.raises(RuntimeError, match="closed"):
        stream(ids)


def test_cache_comparison_accepts_replacement_cache_objects():
    class ReplacingCacheModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = make_model()
            self.device = self.model.device

        def forward(self, *args, **kwargs):
            output = self.model(*args, **kwargs)
            if output.past_key_values is not None:
                output.past_key_values = deepcopy(output.past_key_values)
            return output

    comparison = compare_cached_to_recompute(
        ReplacingCacheModel(),
        torch.tensor([[1, 3, 5]]),
        torch.tensor([[7, 9, 11]]),
    )
    assert comparison["passed"], comparison
    assert [row["cache_length"] for row in comparison["records"]] == [3, 4, 5, 6]


def test_use_cache_requires_a_returned_cache_and_releases_weights(tmp_path):
    reference = make_model()
    reference.save_pretrained(tmp_path)
    with load_streaming_model(tmp_path) as inference:
        original = inference.model._old_forward

        def omit_cache(*args, **kwargs):
            output = original(*args, **kwargs)
            return type(output)(logits=output.logits)

        inference.model._old_forward = omit_cache
        with pytest.raises(RuntimeError, match="did not return past_key_values"):
            inference(torch.tensor([[1, 3, 5]]), use_cache=True)
        assert all(p.is_meta for p in inference.model.parameters())


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize(
    "bits,format_name,nibbles",
    [
        (8, "int-quantized", False),
        (4, "int-quantized", False),
        (4, "int-quantized", True),
        (4, "pack-quantized", False),
    ],
)
def test_quantized_bf16_load_with_a8_prefill_decode_generate(
    tmp_path, monkeypatch, family, bits, format_name, nibbles
):
    root = tmp_path / "quantized"
    quantized_checkpoint(make_model(family), root, bits=bits, format_name=format_name)
    reference = AutoModelForCausalLM.from_pretrained(
        root,
        local_files_only=True,
        dtype=torch.bfloat16,
        quantization_config=CompressedTensorsConfig(
            **json.loads((root / "config.json").read_text())["quantization_config"],
            dequantize=True,
        ),
    ).eval()
    if nibbles:
        tensors = load_file(root / "model.safetensors")
        for name, value in list(tensors.items()):
            if name.endswith(".weight") and value.dtype == torch.int8:
                tensors[name] = pack_int4_to_int8(value)
        save_file(tensors, root / "model.safetensors")

    calls = []
    from llmcompressor.streaming.inference import quantization

    original = quantization.forward_quantize

    def count(module, value, base_name, args):
        calls.append(base_name)
        return original(module, value, base_name, args)

    monkeypatch.setattr(quantization, "forward_quantize", count)
    ids = torch.tensor([[1, 3, 5, 7, 9]])
    with load_streaming_model(root) as stream, torch.no_grad():
        assert stream.weights.tensor_reads == 0
        assert not hasattr(stream.model, "ct_decompress_hook")
        report = stream.report()
        quantized = [w for w in report["weights"] if w["weight_bits"] is not None]
        assert len(quantized) == 15
        assert {w["dtype"] for w in quantized} == {"torch.bfloat16"}
        assert {w["backend"] for w in quantized} == {"dequantized_float"}
        if nibbles:
            assert {w["packing"] for w in quantized} == {"signed_nibbles"}
        actual = stream(ids, use_cache=True)
        expected = reference(ids, use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        for token in (11, 12, 13):
            next_ids = torch.tensor([[token]])
            actual = stream(
                next_ids, past_key_values=actual.past_key_values, use_cache=True
            )
            expected = reference(
                next_ids, past_key_values=expected.past_key_values, use_cache=True
            )
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        assert actual.past_key_values.get_seq_length() == 8
        torch.testing.assert_close(
            stream.generate(ids, max_new_tokens=3, do_sample=False),
            reference.generate(ids, max_new_tokens=3, do_sample=False),
        )
        assert calls and set(calls) == {"input"}
        assert all(p.is_meta for p in stream.model.parameters())


def test_mixed_precision_and_explicit_activation_bypass(tmp_path, monkeypatch):
    quantized_checkpoint(
        make_model(),
        tmp_path / "mixed",
        bits=4,
        format_name="int-quantized",
        mixed=True,
    )
    from llmcompressor.streaming.inference import quantization

    def unexpected(*_args):
        pytest.fail("Explicit weight-only diagnostics must bypass activation Q/DQ")

    monkeypatch.setattr(quantization, "forward_quantize", unexpected)
    with load_streaming_model(tmp_path / "mixed", activation_qdq=False) as stream:
        assert {w["weight_bits"] for w in stream.report()["weights"]} == {None, 4, 8}
        assert stream.report()["activation_mode"] == "dequantized_weights"
        assert torch.isfinite(stream(torch.tensor([[1, 3, 5]])).logits).all()


def test_forward_failure_unloads_partial_block_and_invalidates_cache(tmp_path):
    reference = make_model()
    reference.save_pretrained(tmp_path)
    with load_streaming_model(
        tmp_path, preload_module_classes=["LlamaDecoderLayer"]
    ) as stream:
        ids = torch.tensor([[1, 3, 5]])
        cache = stream(ids, use_cache=True).past_key_values
        layer = stream.model.model.layers[1]
        original = layer._old_forward

        def fail(*_args, **_kwargs):
            raise RuntimeError("injected failure")

        layer._old_forward = fail
        with pytest.raises(RuntimeError, match="injected"):
            stream(torch.tensor([[7]]), past_key_values=cache, use_cache=True)
        assert all(p.is_meta for p in stream.model.parameters())
        layer._old_forward = original
        with pytest.raises(RuntimeError, match="partially updated"):
            stream(torch.tensor([[7]]), past_key_values=cache, use_cache=True)
        assert stream(ids, use_cache=True).past_key_values.get_seq_length() == 3


def test_budgets_and_malformed_headers_fail_before_read(tmp_path, monkeypatch):
    model = make_model()
    model.save_pretrained(tmp_path)

    def forbidden(*_args):
        pytest.fail("Preflight must not read checkpoint payloads")

    monkeypatch.setattr(SafetensorsWeightSource, "load_tensors_cpu", forbidden)
    with pytest.raises(MemoryError, match="working set"):
        load_streaming_model(tmp_path, max_device_weight_bytes=1)
    with pytest.raises(MemoryError, match="CPU preparation"):
        load_streaming_model(tmp_path, max_preparation_bytes=1)


def test_missing_quantization_scale_is_not_silently_cast(tmp_path):
    root = tmp_path / "quantized"
    quantized_checkpoint(make_model(), root, bits=4, format_name="int-quantized")
    state = load_file(root / "model.safetensors")
    state.pop("lm_head.weight_scale")
    save_file(state, root / "model.safetensors")
    with pytest.raises(KeyError):
        load_streaming_model(root)


def test_unknown_format_is_rejected(tmp_path):
    root = tmp_path / "quantized"
    quantized_checkpoint(make_model(), root, bits=4, format_name="int-quantized")
    path = root / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"]["format"] = "float-quantized"
    for group in config["quantization_config"]["config_groups"].values():
        group["format"] = "float-quantized"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unsupported compressed format"):
        load_streaming_model(root)


def test_token_qdq_is_independent_of_other_moe_rows():
    from llmcompressor.streaming.inference.quantization import reference_activation_qdq

    module = torch.nn.Linear(4, 4)
    module.quantization_status = QuantizationStatus.DECOMPRESSED
    args = QuantizationArgs(
        num_bits=8, type="int", strategy="token", dynamic=True, symmetric=False
    )
    rows = torch.tensor(
        [[0.2, 0.3, -0.7, 0.8], [20.0, 300.0, -70.0, 8.0], [1e-3, 0, -0.01, 0.002]]
    )
    together = reference_activation_qdq(module, rows, "input", args)
    individually = torch.stack(
        [reference_activation_qdq(module, row, "input", args) for row in rows]
    )
    torch.testing.assert_close(together, individually, rtol=0, atol=0)
    from compressed_tensors.quantization.lifecycle.forward import forward_quantize

    expected = forward_quantize(module, rows.unsqueeze(0), "input", args).squeeze(0)
    torch.testing.assert_close(together, expected, rtol=0, atol=0)
    assert not torch.equal(together, forward_quantize(module, rows, "input", args))


def test_instance_bound_forward_and_keyword_input_are_preserved(tmp_path):
    from llmcompressor.streaming.inference.quantization import reference_activation_qdq

    class CustomModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.proj = torch.nn.Linear(4, 4, bias=False)

            def custom_forward(module, x):
                return (
                    torch.nn.functional.linear(x.float(), module.weight.float()).to(
                        x.dtype
                    )
                    * 2
                )

            self.proj.forward = MethodType(custom_forward, self.proj)

        def forward(self, x):
            return self.proj(x=x)

    reference = CustomModel(LlamaConfig()).to(torch.bfloat16).eval()
    root = tmp_path / "custom"
    quantized_checkpoint(reference, root, bits=4, format_name="int-quantized")
    raw = load_file(root / "model.safetensors")
    scheme = QuantizationScheme.model_validate(
        reference.config.quantization_config["config_groups"]["proj"]
    )
    decoded = BaseCompressor.get_value_from_registry("int-quantized").decompress(
        {"weight": raw["proj.weight"], "weight_scale": raw["proj.weight_scale"]},
        scheme,
    )["weight"]
    reference.proj.weight.data.copy_(decoded)
    reference.proj.quantization_status = QuantizationStatus.DECOMPRESSED
    rows = torch.tensor(
        [[0.2, 0.3, -0.7, 0.8], [20, 300, -70, 8]], dtype=torch.bfloat16
    )
    expected_input = reference_activation_qdq(
        reference.proj, rows, "input", scheme.input_activations
    )
    with load_streaming_model(root, model_factory=CustomModel) as inference:
        actual = inference(rows)
        torch.testing.assert_close(actual, reference(expected_input), rtol=0, atol=0)
        torch.testing.assert_close(actual[:1], inference(rows[:1]), rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.accelerator.is_available()
    or torch.accelerator.current_accelerator().type != "cuda",
    reason="CUDA is unavailable",
)
def test_cuda_weight_and_native_kv_offload(tmp_path):
    reference = make_model()
    reference.save_pretrained(tmp_path)
    reference = reference.to("cuda")
    ids = torch.tensor([[1, 3, 5, 7, 9]], device="cuda")
    with (
        load_streaming_model(
            tmp_path, device="cuda:0", preload_module_classes=["LlamaDecoderLayer"]
        ) as inference,
        torch.no_grad(),
    ):
        actual = inference.generate(
            ids,
            use_cache=True,
            cache_implementation="offloaded",
            do_sample=False,
            max_new_tokens=3,
        )
        expected = reference.generate(
            ids, use_cache=True, do_sample=False, max_new_tokens=3
        )
        torch.testing.assert_close(actual, expected)
        assert all(p.is_meta for p in inference.model.parameters())
        for module in inference.model.modules():
            for name in module._non_persistent_buffers_set:
                value = module._buffers.get(name)
                assert value is None or value.device.type == "cpu"
