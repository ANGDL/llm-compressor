"""Runtime-compatible Engram export, independent of the large model weights."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from llmcompressor.streaming import DeepSeekV41WeightMaterializer
from llmcompressor.streaming.materialization import deepseek_v41 as adapter
from tools.create_tiny_dsv41 import encode_fp8


def checkpoint(root, tensors):
    root.mkdir()
    save_file(tensors, root / "source.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "source.safetensors" for name in tensors}})
    )
    (root / "config.json").write_text("{}")
    return root


def read_output(output, mapping):
    result = {}
    for filename in set(mapping.values()):
        tensors = load_file(output / filename)
        assert not result.keys() & tensors.keys()
        result.update(tensors)
    assert result.keys() == mapping.keys()
    return result


@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("raw_names", [False, True])
def test_export_bounds_rows_and_preserves_original_checkpoint(
    tmp_path, monkeypatch, source_dtype, raw_names
):
    torch.manual_seed(17)
    # Two tables share a source shard: each needs its own unique export file.
    tensors = {
        f"layers.{layer}.engram.embed.weight": (
            torch.randn(7, 64) * torch.arange(1, 8)[:, None]
        ).to(source_dtype)
        for layer in (1, 14)
    }
    tensors["layers.1.engram.q_weight"] = torch.ones(4, 32, dtype=source_dtype)
    source = checkpoint(tmp_path / "source", tensors)
    original_bytes = (source / "source.safetensors").read_bytes()
    output = tmp_path / "output"
    output.mkdir()
    materializer = DeepSeekV41WeightMaterializer(
        engram_chunk_rows=3, save_raw_checkpoint_format=raw_names
    )
    weight_source = materializer.create_source(source)
    assert tuple(weight_source.tensor_names()) == ("model.layers.1.engram.q_weight",)
    chunks = []
    original = adapter._encode_engram_rows

    def encode_chunk(rows):
        # A small view can still pin the whole file's storage/mmap. Require
        # independent chunk storage, not just a bounded logical tensor shape.
        assert rows.untyped_storage().nbytes() == rows.numel() * rows.element_size()
        chunks.append(tuple(rows.shape))
        return original(rows)

    monkeypatch.setattr(adapter, "_encode_engram_rows", encode_chunk)
    mapping = materializer.save_preserved_tensors(source, output)
    assert chunks == [(3, 64), (3, 64), (1, 64)] * 2
    assert len(set(mapping.values())) == 2
    actual = read_output(output, mapping)
    for name, tensor in tensors.items():
        if not name.endswith(".weight"):
            continue
        output_name = name if raw_names else f"model.{name}"
        scale_name = output_name.removesuffix(".weight") + ".scale"
        weight, scale = encode_fp8(tensor, rowwise=True)
        for key, expected in ((output_name, weight), (scale_name, scale)):
            assert actual[key].dtype == expected.dtype
            assert torch.equal(
                actual[key].view(torch.uint8), expected.view(torch.uint8)
            )
        decoded = (
            actual[output_name].float().unflatten(-1, (-1, 32))
            * actual[scale_name].float().unsqueeze(-1)
        ).flatten(-2)
        torch.testing.assert_close(decoded, tensor.float(), rtol=0.07, atol=1e-6)
    assert materializer.save_preserved_tensors(source, output) == mapping
    assert len(chunks) == 6  # Completed files must not be encoded a second time.
    assert (source / "source.safetensors").read_bytes() == original_bytes
    config = materializer.transform_output_config({"engram_storage": {"stale": True}})
    assert config["engram_storage"] == adapter._ENGRAM_FP8_STORAGE
    for filename in set(mapping.values()):
        with safe_open(output / filename, framework="pt", device="cpu") as shard:
            assert {shard.get_slice(name).get_dtype() for name in shard.keys()} == {
                "F8_E4M3",
                "F8_E8M0",
            }


@pytest.mark.parametrize("mixed", [False, True])
def test_official_fp8_payloads_are_never_reencoded(tmp_path, monkeypatch, mixed):
    name = "layers.1.engram.embed.weight"
    scale_name = "layers.1.engram.embed.scale"
    weight, scale = encode_fp8(torch.randn(7, 64), rowwise=True)
    tensors = {name: weight, scale_name: scale}
    if mixed:
        tensors["layers.14.engram.embed.weight"] = torch.randn(5, 64).bfloat16()
    source = checkpoint(tmp_path / "source", tensors)
    output = tmp_path / "output"
    output.mkdir()
    materializer = DeepSeekV41WeightMaterializer(engram_chunk_rows=2)
    materializer.create_source(source)
    original = adapter._encode_engram_rows
    chunks = []

    def record(rows):
        assert rows.dtype == torch.bfloat16
        chunks.append(rows.shape[0])
        return original(rows)

    monkeypatch.setattr(adapter, "_encode_engram_rows", record)
    mapping = materializer.save_preserved_tensors(source, output)
    actual = read_output(output, mapping)
    assert chunks == ([2, 2, 1] if mixed else [])
    for key in (name, scale_name):
        assert actual[key].dtype == tensors[key].dtype
        assert torch.equal(
            actual[key].view(torch.uint8), tensors[key].view(torch.uint8)
        )


def test_encoder_pow2_boundaries_zero_and_bf16_extremes():
    maximum = torch.tensor(
        [
            0,
            448,
            torch.nextafter(torch.tensor(448.0), torch.tensor(float("inf"))),
            2**120,
            torch.finfo(torch.bfloat16).tiny,
        ]
    )
    weight = maximum[:, None].expand(-1, 64).clone()
    weight[:, 32:] *= -0.25
    encoded, scale = adapter._encode_engram_rows(weight)
    decoded_scale = scale.float()
    assert torch.isfinite(encoded.float()).all()
    assert torch.isfinite(decoded_scale).all()
    assert (decoded_scale > 0).all()
    assert decoded_scale[1, 0] == 1
    assert decoded_scale[2, 0] == 2  # Not rounded down across the pow2 boundary.
    assert decoded_scale[1, 1] == 0.25
    assert (encoded[0].float() == 0).all()
    assert torch.equal(torch.log2(decoded_scale), torch.log2(decoded_scale).round())
    decoded = encoded.float().unflatten(-1, (-1, 32)) * decoded_scale.unsqueeze(-1)
    assert torch.isfinite(decoded).all()
    assert torch.isfinite(decoded.bfloat16()).all()


@pytest.mark.parametrize("sign", [-1, 1])
def test_finite_input_with_unrepresentable_fp8_decode_is_rejected(sign):
    value = torch.finfo(torch.bfloat16).max * sign
    with pytest.raises(ValueError, match="overflow BF16"):
        adapter._encode_engram_rows(torch.full((1, 32), value, dtype=torch.bfloat16))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_encode_is_not_published_and_can_retry(tmp_path, bad):
    name = "layers.1.engram.embed.weight"
    weight = torch.ones(5, 32, dtype=torch.bfloat16)
    weight[3, 0] = bad  # Fail after at least one completed chunk.
    source = checkpoint(tmp_path / "source", {name: weight})
    output = tmp_path / "output"
    output.mkdir()
    materializer = DeepSeekV41WeightMaterializer(engram_chunk_rows=2)
    materializer.create_source(source)
    with pytest.raises(ValueError, match="non-finite"):
        materializer.save_preserved_tensors(source, output)
    assert list(output.iterdir()) == []
    weight[3, 0] = 1
    save_file({name: weight}, source / "source.safetensors")
    mapping = materializer.save_preserved_tensors(source, output)
    assert len(read_output(output, mapping)) == 2
    # A matching header alone cannot make a truncated export resumable.
    shard = output / next(iter(mapping.values()))
    with shard.open("r+b") as file:
        file.truncate(shard.stat().st_size - 1)
    with pytest.raises(ValueError, match="Encoded Engram shard disagrees"):
        materializer.save_preserved_tensors(source, output)


def test_source_table_opt_out_stays_lossless(tmp_path):
    name = "layers.1.engram.embed.weight"
    weight = torch.randn(7, 64).bfloat16()
    source = checkpoint(tmp_path / "source", {name: weight})
    output = tmp_path / "output"
    output.mkdir()
    materializer = DeepSeekV41WeightMaterializer(engram_table_dtype="source")
    materializer.create_source(source)
    actual = read_output(output, materializer.save_preserved_tensors(source, output))
    assert actual.keys() == {name}
    assert torch.equal(actual[name], weight)
    assert materializer.transform_output_config({})["engram_storage"] == {
        "weight_dtype": "bf16"
    }


def test_export_format_is_in_fingerprint():
    default = DeepSeekV41WeightMaterializer().manifest_info(target_dtype=torch.bfloat16)
    dense = DeepSeekV41WeightMaterializer(engram_table_dtype="source").manifest_info(
        target_dtype=torch.bfloat16
    )
    assert default != dense


@pytest.mark.parametrize("shape", [(5, 7), (5, 0), (5, 32, 1)])
def test_bad_table_shape_rejected_before_calibration(tmp_path, shape):
    source = checkpoint(
        tmp_path / "source",
        {"layers.1.engram.embed.weight": torch.ones(shape, dtype=torch.bfloat16)},
    )
    with pytest.raises(
        ValueError, match="width divisible by 32|Invalid Engram table shape"
    ):
        DeepSeekV41WeightMaterializer().create_source(source)


@pytest.mark.parametrize(
    "kwargs", [{"engram_table_dtype": "int8"}, {"engram_chunk_rows": 0}]
)
def test_invalid_export_options(kwargs):
    with pytest.raises(ValueError):
        DeepSeekV41WeightMaterializer(**kwargs)
