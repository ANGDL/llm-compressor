import json
from math import prod

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ci.model_quality import executor


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    model = {
        "id": "model-a",
        "validation": {
            "streaming": True,
            "quantization_auxiliary_suffixes": [".scale"],
            "quantization_scale_suffixes": [".scale"],
        },
    }
    output = tmp_path / "run-a/model-a/model"
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quantization_status": "compressed",
                    "format": "int-quantized",
                    "config_groups": {"group_0": {"targets": ["Linear"]}},
                }
            }
        )
    )
    (output / "FINALIZED").touch()
    return model, output


def _save_shards(output, shards):
    mapping = {}
    for filename, tensors in shards.items():
        save_file(tensors, output / filename)
        mapping.update({name: filename for name in tensors})
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )


def test_validate_preserved_fp8_engram_shards(artifact, monkeypatch):
    model, output = artifact
    monkeypatch.setattr(executor, "_TENSOR_INSPECTION_CHUNK_ELEMENTS", 16)
    shards = {
        "model-00000.safetensors": {
            "layers.1.engram.wkv.weight": torch.ones(2, 32, dtype=torch.int8),
            "layers.1.engram.wkv.scale": torch.ones(2, 1),
        }
    }
    for number, layer in enumerate((1, 14)):
        shards[f"model-preserved-engram-{number:05d}.safetensors"] = {
            f"layers.{layer}.engram.embed.weight": torch.ones(5, 32).to(
                torch.float8_e4m3fn
            ),
            f"layers.{layer}.engram.embed.scale": torch.ones(5, 1).to(
                torch.float8_e8m0fnu
            ),
        }
    _save_shards(output, shards)

    result = executor.run_validate(model, "run-a")

    assert result["status"] == "PASS"
    assert result["shard_count"] == 3
    assert result["tensor_count"] == 6
    assert result["non_finite_tensor_count"] == 0
    assert result["quantization_auxiliary_tensor_count"] == 3
    assert result["checksummed_file_count"] == 6
    health = {item["name"]: item for item in result["quantization_tensor_health"]}
    assert len(health) == 3
    for layer in (1, 14):
        item = health[f"layers.{layer}.engram.embed.scale"]
        assert item["dtype"] == "torch.float8_e8m0fnu"
        assert item["shape"] == [5, 1]
        assert item["min"] == item["max"] == item["mean"] == 1.0
        assert item["zero_fraction"] == 0.0


@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu, torch.float32],
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_validate_rejects_nonfinite_values_in_last_chunk(
    artifact, monkeypatch, dtype, bad_value
):
    model, output = artifact
    monkeypatch.setattr(executor, "_TENSOR_INSPECTION_CHUNK_ELEMENTS", 16)
    values = torch.ones(5, 32)
    values[-1, -1] = bad_value
    name = (
        "layers.1.engram.embed.scale"
        if dtype == torch.float8_e8m0fnu
        else "layers.1.engram.embed.weight"
    )
    _save_shards(
        output,
        {"model-preserved-engram-00000.safetensors": {name: values.to(dtype)}},
    )

    with pytest.raises(executor.StageError) as error:
        executor.run_validate(model, "run-a")

    assert error.value.reason_code == "VALIDATION_FAILED"
    assert f"non-finite tensors: {name}" in str(error.value)
    assert "unable to inspect" not in str(error.value)


@pytest.mark.parametrize("shape", [(), (0, 32), (41,), (7, 4), (2, 41), (2, 3, 41)])
def test_tensor_inspection_bounds_reads_and_preserves_statistics(
    tmp_path, monkeypatch, shape
):
    limit = 16
    monkeypatch.setattr(executor, "_TENSOR_INSPECTION_CHUNK_ELEMENTS", limit)
    tensor = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
    path = tmp_path / "model.safetensors"
    save_file({"layer.weight_scale": tensor}, path)
    reads = []

    class CheckedSlice:
        def __init__(self, view):
            self.view = view

        def get_shape(self):
            return self.view.get_shape()

        def get_dtype(self):
            return self.view.get_dtype()

        def __getitem__(self, index):
            value = self.view[index]
            assert value.numel() <= limit
            reads.append(value.numel())
            return value

    class CheckedHandle:
        def __init__(self, handle):
            self.handle = handle

        def get_slice(self, name):
            return CheckedSlice(self.handle.get_slice(name))

        def get_tensor(self, name):
            assert tensor.numel() <= limit, "must not load a whole large tensor"
            reads.append(tensor.numel())
            return self.handle.get_tensor(name)

    with safe_open(path, framework="pt", device="cpu") as handle:
        finite, health = executor._inspect_tensor(
            CheckedHandle(handle), "layer.weight_scale", collect_health=True
        )

    assert finite
    assert sum(reads) == tensor.numel()
    assert health["dtype"] == "torch.float32"
    assert health["shape"] == list(shape)
    if tensor.numel():
        assert health["min"] == tensor.min().item()
        assert health["max"] == tensor.max().item()
        assert health["mean"] == pytest.approx(tensor.double().mean().item())
        assert health["zero_fraction"] == pytest.approx(
            (tensor == 0).double().mean().item()
        )
    else:
        assert all(
            health[key] is None for key in ("min", "max", "mean", "zero_fraction")
        )


def test_tensor_inspection_keeps_finite_float64_range(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"layer.weight": torch.tensor([1e300], dtype=torch.float64)}, path)

    with safe_open(path, framework="pt", device="cpu") as handle:
        finite, health = executor._inspect_tensor(
            handle, "layer.weight", collect_health=False
        )

    assert finite
    assert health is None
