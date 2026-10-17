import json

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText

from llmcompressor.modeling.deepseekv41 import (
    DeepseekV41NativeForConditionalGeneration,
    ModelConfig,
)
from llmcompressor.modeling.deepseekv41.quantization import fp4_qdq
from llmcompressor.streaming import DeepSeekV41WeightMaterializer
from llmcompressor.utils.int4_packing import pack_int4_to_int8
from tests.llmcompressor.modeling.test_deepseek_v41 import tiny_config
from tools.create_tiny_dsv41 import (
    build_tiny_config,
    encode_fp4,
    encode_fp8,
    raw_name,
    save_source_fixture,
)
from tools.validate_tiny_dsv41 import unpack_int4


def test_full_topology_tiny_keeps_every_layer_and_source(tmp_path):
    original = ModelConfig()
    (tmp_path / "config.json").write_text(json.dumps(original.to_dict()))
    config = build_tiny_config(tmp_path)
    assert config.text_config.compress_ratios == original.text_config.compress_ratios
    assert config.text_config.num_experts_per_tok == 6
    assert config.text_config.dspark_num_experts_per_tok == 3
    assert config.text_config.hc_mult == 4
    assert config.text_config.engram_max_ngram_size == 4
    assert config.vision_config.downsample_ratio == 3
    token_map = [
        i % config.text_config.engram_compressed_vocab_size
        for i in range(config.text_config.vocab_size)
    ]
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(
            config, engram_token_map=token_map
        )
    assert isinstance(model, DeepseekV41NativeForConditionalGeneration)
    assert len(model.model.layers) == 40
    assert len(model.model.vision.blocks) == 32
    assert len(model.model.mtp) == 3
    assert model.num_parameters() == 32_202_678
    assert [
        i
        for i, layer in enumerate(model.model.layers)
        if layer.attn.compressor is not None
    ] == [2, 8, 14, 20]
    assert [
        i
        for i, layer in enumerate(model.model.layers)
        if layer.attn.indexer is not None
    ] == [2, 8, 14, 20, 24, 28, 32, 36]
    assert [
        i for i, layer in enumerate(model.model.layers) if layer.engram is not None
    ] == [1, 14]
    for index, layer in enumerate(model.model.mtp):
        assert layer.ffn.n_routed_experts == 4
        assert layer.ffn.gate.topk == 3
        assert hasattr(layer, "main_proj") == (index == 0)
        assert hasattr(layer, "markov_head") == (index == 2)


def test_fp8_fixture_roundtrips_non_square_partial_blocks():
    weight = torch.randn(45, 70) * 0.02
    encoded, scale = encode_fp8(weight)
    assert encoded.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.shape == (2, 3)
    name = "model.layers.0.attn.wq_a.weight"
    decoded = DeepSeekV41WeightMaterializer().materialize_cpu(
        name,
        {name: encoded, name.replace(".weight", ".scale"): scale},
        target_dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, weight, rtol=0.07, atol=1e-5)


def test_engram_fixture_uses_row32_instead_of_block32():
    weight = torch.randn(7, 64) * torch.arange(1, 8)[:, None]
    encoded, scale = encode_fp8(weight, rowwise=True)
    assert scale.shape == (7, 2)
    name = "model.layers.1.engram.embed.weight"
    decoded = DeepSeekV41WeightMaterializer(preserve_engram=False).materialize_cpu(
        name,
        {name: encoded, name.replace(".weight", ".scale"): scale},
        target_dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, weight, rtol=0.07, atol=1e-4)


def test_source_fp4_fixture_uses_e2m1_codes_and_even_ties():
    weight = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 4])
    packed, scale = encode_fp4(weight)
    assert packed.shape == (1, 16)
    assert packed.dtype == torch.int8
    name = "model.layers.0.ffn.experts.0.w1.weight"
    decoded = DeepSeekV41WeightMaterializer().materialize_cpu(
        name,
        {name: packed, name.replace(".weight", ".scale"): scale},
        target_dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, fp4_qdq(weight), rtol=0, atol=0)


def test_ptq_int4_unpack_uses_signed_nibbles_instead_of_fp4_table():
    weight = torch.arange(-8, 8).to(torch.int8).view(2, 8)
    torch.testing.assert_close(unpack_int4(pack_int4_to_int8(weight)), weight)


def test_source_fixture_keeps_bf16_execution_and_fp32_control_state(tmp_path):
    model = DeepseekV41NativeForConditionalGeneration(
        tiny_config(vision=True, engram=True), engram_token_map=list(range(32))
    )
    source = tmp_path / "original"
    source.mkdir()
    names = {raw_name(name): "unused.safetensors" for name in model.state_dict()}
    names["layers.0.attn.wq_a.scale"] = "unused.safetensors"
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": names})
    )
    output = tmp_path / "fixture"
    stats = save_source_fixture(model, source, output, max_shard_bytes=1024**2)
    mapping = json.loads((output / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    for name, dtype in (
        ("embed.weight", torch.bfloat16),
        ("head.weight", torch.bfloat16),
        ("vision.patch_embed.proj.weight", torch.bfloat16),
        ("vision.patch_embed.proj.bias", torch.bfloat16),
        ("image_start", torch.bfloat16),
        ("layers.0.hc_attn_fn", torch.float32),
        ("layers.0.ffn.gate.weight", torch.float32),
        ("layers.0.attn.wq_a.weight", torch.float8_e4m3fn),
        ("layers.0.attn.wq_a.scale", torch.float8_e8m0fnu),
        ("layers.0.ffn.experts.0.w1.weight", torch.int8),
        ("layers.1.engram.embed.weight", torch.float8_e4m3fn),
    ):
        with safe_open(output / mapping[name], framework="pt") as shard:
            assert shard.get_tensor(name).dtype == dtype, name
    assert stats["kinds"]["engram_fp8_row32"] == 1
