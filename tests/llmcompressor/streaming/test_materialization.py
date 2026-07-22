import json
from typing import Mapping

import pytest
import torch
from safetensors.torch import save_file

from llmcompressor.streaming.checkpoint import (
    SafetensorsWeightSource,
    TensorMetadata,
)
from llmcompressor.streaming.materialization import (
    CastWeightMaterializer,
    DeepSeekV4WeightMaterializer,
    WeightMaterializer,
    materialize_weights,
)


class ScaledIntMaterializer(WeightMaterializer):
    def dependencies(
        self, tensor_name: str, metadata: TensorMetadata
    ) -> list[str]:
        return [tensor_name.replace(".weight", ".scale")]

    def materialize(
        self,
        tensor_name: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        target_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        scale_name = tensor_name.replace(".weight", ".scale")
        return (tensors[tensor_name] * tensors[scale_name]).to(
            device=device, dtype=target_dtype
        )


class InvalidMaterializer(WeightMaterializer):
    def __init__(self, result):
        self.result = result

    def materialize(
        self, tensor_name, tensors, *, target_dtype, device
    ) -> torch.Tensor:
        return self.result


@pytest.fixture
def sharded_checkpoint(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    shard_1 = checkpoint / "model-00001-of-00002.safetensors"
    shard_2 = checkpoint / "model-00002-of-00002.safetensors"
    save_file(
        {"layer0.weight": torch.arange(6, dtype=torch.float16).reshape(2, 3)},
        shard_1,
    )
    save_file(
        {
            "layer1.scale": torch.full((2, 1), 0.5),
            "layer1.weight": torch.arange(6, dtype=torch.int8).reshape(2, 3),
        },
        shard_2,
    )
    weight_map = {
        "layer0.weight": shard_1.name,
        "layer1.scale": shard_2.name,
        "layer1.weight": shard_2.name,
    }
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    return checkpoint


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_default_materializer_casts_supported_floats(tmp_path, source_dtype):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2, 3, dtype=source_dtype)}, path)
    source = SafetensorsWeightSource(path)

    result = materialize_weights(
        source,
        ["weight"],
        CastWeightMaterializer(),
        target_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert result["weight"].dtype == torch.bfloat16
    assert result["weight"].device.type == "cpu"


def test_custom_materializer_loads_declared_dependency(sharded_checkpoint):
    source = SafetensorsWeightSource(sharded_checkpoint)

    result = materialize_weights(
        source,
        ["layer1.weight"],
        ScaledIntMaterializer(),
        target_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    expected = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3) * 0.5
    assert torch.equal(result["layer1.weight"], expected)


def test_missing_materializer_dependency_reports_name(sharded_checkpoint):
    source = SafetensorsWeightSource(sharded_checkpoint)

    with pytest.raises(KeyError, match="layer0.scale"):
        materialize_weights(
            source,
            ["layer0.weight"],
            ScaledIntMaterializer(),
            target_dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )


def test_default_materializer_rejects_integer_source(sharded_checkpoint):
    source = SafetensorsWeightSource(sharded_checkpoint)

    with pytest.raises(TypeError, match="source dtype"):
        materialize_weights(
            source,
            ["layer1.weight"],
            CastWeightMaterializer(),
            target_dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (torch.ones(2, 3, dtype=torch.int8), "non-floating dtype"),
        (torch.ones(3, 2), "returned shape"),
        (torch.ones(2, 3, dtype=torch.float16), "returned dtype"),
    ],
)
def test_rejects_invalid_materializer_output(
    sharded_checkpoint, result, message
):
    source = SafetensorsWeightSource(sharded_checkpoint)

    with pytest.raises((TypeError, ValueError), match=message):
        materialize_weights(
            source,
            ["layer0.weight"],
            InvalidMaterializer(result),
            target_dtype=torch.float32,
            device=torch.device("cpu"),
        )


def test_materializer_manifest_identity_is_stable_and_dtype_sensitive():
    materializer = CastWeightMaterializer()

    first = materializer.manifest_info(target_dtype=torch.bfloat16)
    second = materializer.manifest_info(target_dtype=torch.bfloat16)
    fp32 = materializer.manifest_info(target_dtype=torch.float32)

    assert first == second
    assert first.identifier.endswith("CastWeightMaterializer")
    assert first.config_sha256 != fp32.config_sha256


def test_deepseek_v4_materializer_unpacks_fp4_blocks(tmp_path):
    materializer = DeepSeekV4WeightMaterializer(fp4_block_size=32)
    name = "model.layers.0.ffn.experts.0.w1.weight"
    metadata = TensorMetadata(
        name=name,
        shape=(2, 2),
        dtype=torch.int8,
        shard=tmp_path / "model.safetensors",
    )
    packed = torch.tensor([[0x01, 0x29], [-0x6D, -0x0C]], dtype=torch.int8)
    scale = torch.full((2, 1), 127, dtype=torch.uint8)

    result = materializer.materialize(
        name,
        {name: packed, "model.layers.0.ffn.experts.0.w1.scale": scale},
        target_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert materializer.logical_shape(name, metadata) == (2, 4)
    assert torch.equal(
        result,
        torch.tensor(
            [[0.5, 0.0, -0.5, 1.0], [1.5, -0.5, 2.0, -6.0]],
            dtype=torch.bfloat16,
        ),
    )
