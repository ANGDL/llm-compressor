import gc
import json
import weakref

import pytest
import torch
from safetensors.torch import save_file

from llmcompressor.streaming.checkpoint import SafetensorsWeightSource, WeightMap


@pytest.fixture
def sharded_checkpoint(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    shard_1 = checkpoint / "model-00001-of-00002.safetensors"
    shard_2 = checkpoint / "model-00002-of-00002.safetensors"
    save_file(
        {
            "layer0.bias": torch.arange(2, dtype=torch.float32),
            "layer0.weight": torch.arange(6, dtype=torch.float16).reshape(2, 3),
        },
        shard_1,
    )
    save_file(
        {
            "layer1.scale": torch.full((2, 1), 0.5),
            "layer1.weight": torch.arange(6, dtype=torch.int8).reshape(2, 3),
        },
        shard_2,
    )
    index = {
        "metadata": {"total_size": 44},
        "weight_map": {
            "layer0.bias": shard_1.name,
            "layer0.weight": shard_1.name,
            "layer1.scale": shard_2.name,
            "layer1.weight": shard_2.name,
        },
    }
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    return checkpoint


def test_weight_map_reads_metadata_without_loading_tensors(sharded_checkpoint):
    weight_map = WeightMap.from_checkpoint(sharded_checkpoint)

    metadata = weight_map.metadata("layer0.weight")
    assert metadata.shape == (2, 3)
    assert metadata.dtype == torch.float16
    assert metadata.shard.name == "model-00001-of-00002.safetensors"
    assert set(weight_map) == {
        "layer0.bias",
        "layer0.weight",
        "layer1.scale",
        "layer1.weight",
    }
    shard_indices = {
        name: weight_map.metadata(name).storage_index
        for name in ("layer0.bias", "layer0.weight")
    }
    assert len(set(shard_indices.values())) == 2


def test_loads_only_requested_tensors_across_shards(sharded_checkpoint):
    source = SafetensorsWeightSource(sharded_checkpoint)

    assert not hasattr(source, "load_tensors")
    tensors = source.load_tensors_cpu(["layer0.weight", "layer1.scale"])

    assert set(tensors) == {"layer0.weight", "layer1.scale"}
    assert torch.equal(
        tensors["layer0.weight"],
        torch.arange(6, dtype=torch.float16).reshape(2, 3),
    )
    assert torch.equal(tensors["layer1.scale"], torch.full((2, 1), 0.5))


def test_opens_each_requested_shard_once(sharded_checkpoint, monkeypatch):
    from llmcompressor.streaming.checkpoint import weight_source

    source = SafetensorsWeightSource(sharded_checkpoint)
    real_safe_open = weight_source.safe_open
    opened = []

    def recording_safe_open(path, *args, **kwargs):
        opened.append(path)
        return real_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(weight_source, "safe_open", recording_safe_open)
    source.load_tensors_cpu(["layer0.weight", "layer0.bias", "layer1.scale"])

    assert len(opened) == 2
    assert len(set(opened)) == 2


def test_iter_tensor_groups_reuses_shards_and_releases_previous_group(
    sharded_checkpoint, monkeypatch
):
    from llmcompressor.streaming.checkpoint import weight_source

    source = SafetensorsWeightSource(sharded_checkpoint)
    real_safe_open = weight_source.safe_open
    opened = []

    def recording_safe_open(path, *args, **kwargs):
        opened.append(path)
        return real_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(weight_source, "safe_open", recording_safe_open)
    groups = source.iter_tensor_groups_cpu(
        [
            ("layer0.weight", "layer0.bias"),
            ("layer0.bias",),
            ("layer1.weight", "layer1.scale"),
        ]
    )

    first = next(groups)
    reference = weakref.ref(first["layer0.weight"])
    del first
    second = next(groups)

    assert reference() is None
    assert set(second) == {"layer0.bias"}
    assert len(opened) == 1

    third = next(groups)
    assert set(third) == {"layer1.weight", "layer1.scale"}
    assert len(opened) == 2


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"), reason="PyTorch has no E8M0 dtype"
)
def test_native_e8m0_loading_preserves_storage_bytes(tmp_path):
    path = tmp_path / "model.safetensors"
    first = torch.tensor([127, 128], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    second = torch.tensor([129, 130], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    save_file({"first": first, "second": second}, path)
    source = SafetensorsWeightSource(path)
    groups = list(source.iter_tensor_groups_cpu([("first",), ("second",)]))

    assert torch.equal(
        groups[0]["first"].view(torch.uint8),
        torch.tensor([127, 128], dtype=torch.uint8),
    )
    assert torch.equal(
        groups[1]["second"].view(torch.uint8),
        torch.tensor([129, 130], dtype=torch.uint8),
    )


def test_supports_single_safetensors_file(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2, dtype=torch.bfloat16)}, checkpoint)

    source = SafetensorsWeightSource(checkpoint)

    assert source.metadata("weight").dtype == torch.bfloat16
    assert torch.equal(
        source.load_tensors_cpu(["weight"])["weight"],
        torch.ones(2, dtype=torch.bfloat16),
    )


def test_reports_missing_tensor_and_index_dependency(sharded_checkpoint):
    source = SafetensorsWeightSource(sharded_checkpoint)
    with pytest.raises(KeyError, match="missing.weight"):
        source.load_tensors_cpu(["missing.weight"])

    index_path = sharded_checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["missing.weight"] = (
        "model-00001-of-00002.safetensors"
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(KeyError, match="missing.weight"):
        SafetensorsWeightSource(sharded_checkpoint)


def test_source_does_not_retain_loaded_tensor(sharded_checkpoint):
    source = SafetensorsWeightSource(sharded_checkpoint)
    tensors = source.load_tensors_cpu(["layer0.weight"])
    reference = weakref.ref(tensors["layer0.weight"])

    del tensors
    gc.collect()

    assert reference() is None


def test_rejects_index_shard_outside_checkpoint(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    save_file({"weight": torch.ones(1)}, tmp_path / "outside.safetensors")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes checkpoint directory"):
        SafetensorsWeightSource(checkpoint)
