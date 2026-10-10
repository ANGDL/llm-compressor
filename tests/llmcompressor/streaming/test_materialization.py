import json
import weakref
from typing import Mapping

import pytest
import torch
from safetensors.torch import save_file
from transformers.conversion_mapping import Concatenate
from transformers.core_model_loading import WeightConverter, WeightRenaming

from llmcompressor.streaming.checkpoint import (
    SafetensorsWeightSource,
    TensorMetadata,
)
from llmcompressor.streaming.loading import TargetWeightLoader
from llmcompressor.streaming.materialization import (
    CastWeightMaterializer,
    DeepSeekV4WeightMaterializer,
    KimiK3WeightMaterializer,
    KimiK3WeightSource,
    StreamingDTypePolicy,
    TransformersWeightMaterializer,
    WeightMaterializer,
    materialize_weights,
)


def test_streaming_dtype_policy_preserves_control_state_by_default():
    policy = StreamingDTypePolicy.for_quantized_modules(
        torch.bfloat16, ("layer.proj",)
    )

    assert policy.resolve("layer.proj.weight", torch.float32) == torch.bfloat16
    assert policy.resolve("layer.proj.bias", torch.float32) == torch.bfloat16
    assert policy.resolve("layer.attn_sink", torch.float32) == torch.float32
    assert policy.resolve("layer.norm.weight", torch.bfloat16) == torch.bfloat16
    assert (
        policy.resolve("layer.experts.weight", torch.int8)
        == torch.bfloat16
    )


def test_streaming_dtype_policy_allows_model_adapter_override():
    policy = StreamingDTypePolicy.for_quantized_modules(
        torch.bfloat16,
        ("layer.proj",),
        overrides=((r".*\.attention_sink_bias", torch.float32),),
    )

    assert (
        policy.resolve("layer.attention_sink_bias", torch.bfloat16)
        == torch.float32
    )


class ScaledIntMaterializer(WeightMaterializer):
    def dependencies(
        self, tensor_name: str, metadata: TensorMetadata
    ) -> list[str]:
        return [tensor_name.replace(".weight", ".scale")]

    def materialize_cpu(
        self,
        tensor_name: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        scale_name = tensor_name.replace(".weight", ".scale")
        return (tensors[tensor_name] * tensors[scale_name]).to(dtype=target_dtype)


class InvalidMaterializer(WeightMaterializer):
    def __init__(self, result):
        self.result = result

    def materialize_cpu(self, tensor_name, tensors, *, target_dtype) -> torch.Tensor:
        return self.result


class RawLifetimeMaterializer(ScaledIntMaterializer):
    def __init__(self):
        self.previous_raw = []

    def materialize_cpu(self, tensor_name, tensors, *, target_dtype):
        assert all(reference() is None for reference in self.previous_raw)
        self.previous_raw = [weakref.ref(value) for value in tensors.values()]
        return super().materialize_cpu(
            tensor_name,
            tensors,
            target_dtype=target_dtype,
        )


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
    materializer = CastWeightMaterializer()

    result = materialize_weights(
        source,
        ["weight"],
        materializer,
        target_dtype=torch.bfloat16,
    )

    assert not hasattr(materializer, "materialize")
    assert result["weight"].dtype == torch.bfloat16
    assert result["weight"].device.type == "cpu"


def test_custom_materializer_loads_declared_dependency(sharded_checkpoint):
    source = SafetensorsWeightSource(sharded_checkpoint)

    result = materialize_weights(
        source,
        ["layer1.weight"],
        ScaledIntMaterializer(),
        target_dtype=torch.bfloat16,
    )

    expected = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3) * 0.5
    assert torch.equal(result["layer1.weight"], expected)


def test_materializer_releases_each_raw_group_before_loading_next(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "layer0.scale": torch.full((2, 1), 0.5),
            "layer0.weight": torch.arange(6, dtype=torch.int8).reshape(2, 3),
            "layer1.scale": torch.full((2, 1), 0.25),
            "layer1.weight": torch.arange(6, dtype=torch.int8).reshape(2, 3),
        },
        path,
    )
    materializer = RawLifetimeMaterializer()

    result = materialize_weights(
        SafetensorsWeightSource(path),
        ["layer0.weight", "layer1.weight"],
        materializer,
        target_dtype=torch.bfloat16,
    )

    assert set(result) == {"layer0.weight", "layer1.weight"}


def test_materializer_supports_legacy_source_without_group_iterator(
    sharded_checkpoint,
):
    source = SafetensorsWeightSource(sharded_checkpoint)

    class LegacySource:
        metadata = source.metadata
        load_tensors_cpu = source.load_tensors_cpu

    result = materialize_weights(
        LegacySource(),
        ["layer1.weight"],
        ScaledIntMaterializer(),
        target_dtype=torch.bfloat16,
    )

    assert result["layer1.weight"].dtype == torch.bfloat16


def test_large_materialization_reports_debug_progress(tmp_path, monkeypatch):
    from llmcompressor.streaming.materialization import base

    path = tmp_path / "model.safetensors"
    save_file(
        {f"layer{index}.weight": torch.ones(1) for index in range(65)},
        path,
    )
    messages = []
    monkeypatch.setattr(base.streaming_logger, "debug", messages.append)

    result = materialize_weights(
        SafetensorsWeightSource(path),
        [f"layer{index}.weight" for index in range(65)],
        CastWeightMaterializer(),
        target_dtype=torch.bfloat16,
    )

    assert len(result) == 65
    assert any("tensors=1-64/65" in message for message in messages)
    assert any("progress=64/65" in message for message in messages)
    assert any("progress=65/65" in message for message in messages)


def test_materializer_reads_tensors_in_physical_storage_order(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "layer.1.weight": torch.ones(1),
            "layer.10.weight": torch.ones(1),
            "layer.2.weight": torch.ones(1),
        },
        path,
    )
    source = SafetensorsWeightSource(path)
    requested = list(reversed(source.tensor_names()))
    expected = sorted(
        requested,
        key=lambda name: source.metadata(name).storage_index,
    )
    observed = []
    original_iter = source.iter_tensor_groups_cpu

    def recording_iter(groups):
        groups = tuple(tuple(group) for group in groups)
        observed.extend(group[0] for group in groups)
        return original_iter(groups)

    monkeypatch.setattr(source, "iter_tensor_groups_cpu", recording_iter)

    materialize_weights(
        source,
        requested,
        CastWeightMaterializer(),
        target_dtype=torch.bfloat16,
    )

    assert observed == expected


def test_missing_materializer_dependency_reports_name(sharded_checkpoint):
    source = SafetensorsWeightSource(sharded_checkpoint)

    with pytest.raises(KeyError, match="layer0.scale"):
        materialize_weights(
            source,
            ["layer0.weight"],
            ScaledIntMaterializer(),
            target_dtype=torch.bfloat16,
        )


def test_default_materializer_rejects_integer_source(sharded_checkpoint):
    source = SafetensorsWeightSource(sharded_checkpoint)

    with pytest.raises(TypeError, match="source dtype"):
        materialize_weights(
            source,
            ["layer1.weight"],
            CastWeightMaterializer(),
            target_dtype=torch.bfloat16,
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
        )


def test_materializer_manifest_identity_is_stable_and_dtype_sensitive():
    materializer = CastWeightMaterializer()
    policy = StreamingDTypePolicy.for_quantized_modules(
        torch.bfloat16, ("layer.proj",)
    )

    first = materializer.manifest_info(target_dtype=torch.bfloat16)
    second = materializer.manifest_info(target_dtype=torch.bfloat16)
    fp32 = materializer.manifest_info(target_dtype=torch.float32)
    policy_first = materializer.manifest_info(
        target_dtype=torch.bfloat16, dtype_policy=policy
    )
    policy_second = materializer.manifest_info(
        target_dtype=torch.bfloat16, dtype_policy=policy
    )

    assert first == second
    assert first.identifier.endswith("CastWeightMaterializer")
    assert first.config_sha256 != fp32.config_sha256
    assert policy_first == policy_second
    assert policy_first.config_sha256 != first.config_sha256


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

    result = materializer.materialize_cpu(
        name,
        {name: packed, "model.layers.0.ffn.experts.0.w1.scale": scale},
        target_dtype=torch.bfloat16,
    )

    assert materializer.logical_shape(name, metadata) == (2, 4)
    assert torch.equal(
        result,
        torch.tensor(
            [[0.5, 0.0, -0.5, 1.0], [1.5, -0.5, 2.0, -6.0]],
            dtype=torch.bfloat16,
        ),
    )


def test_deepseek_v4_materializer_decodes_ue8m0_endpoints():
    decoded = DeepSeekV4WeightMaterializer._decode_e8m0(
        torch.tensor([0, 127, 255], dtype=torch.uint8)
    )

    assert decoded[0].item() == 0.0
    assert decoded[1].item() == 1.0
    assert torch.isinf(decoded[2])


@pytest.mark.parametrize("mtp_index", [0, 1, 2])
def test_deepseek_v4_materializer_raw_output_names_match_sglang_contract(
    mtp_index,
):
    materializer = DeepSeekV4WeightMaterializer(
        save_raw_checkpoint_format=True
    )

    assert materializer.output_tensor_name(
        f"model.mtp.{mtp_index}.ffn.experts.7.w1.weight"
    ) == f"mtp.{mtp_index}.ffn.experts.7.w1.weight"
    assert materializer.output_tensor_name(
        f"model.mtp.{mtp_index}.ffn.shared_experts.w2.weight_scale"
    ) == f"mtp.{mtp_index}.ffn.shared_experts.w2.scale"
    assert materializer.output_tensor_name(
        f"model.mtp.{mtp_index}.lm_head.weight"
    ) == f"mtp.{mtp_index}.head.weight"
    assert materializer.output_module_name(
        f"model.mtp.{mtp_index}.ffn.experts.7.w1"
    ) == f"mtp.{mtp_index}.ffn.experts.7.w1"
    assert materializer.output_weight_scale_name(
        f"mtp.{mtp_index}.ffn.experts.7.w1"
    ) == f"mtp.{mtp_index}.ffn.experts.7.w1.scale"


@pytest.mark.parametrize("save_raw_checkpoint_format", [False, True])
def test_deepseek_v4_materializer_preserves_source_model_identity(
    save_raw_checkpoint_format,
):
    updates = DeepSeekV4WeightMaterializer(
        save_raw_checkpoint_format=save_raw_checkpoint_format
    ).output_config_updates()

    assert updates == {"torch_dtype": "bfloat16"}


def test_deepseek_v4_materializer_disables_absent_mtp_in_output_config():
    materializer = DeepSeekV4WeightMaterializer(source_has_mtp=False)

    output = materializer.transform_output_config(
        {
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "compress_ratios": [0, 4, 0],
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [0, 1],
        }
    )

    assert output["num_nextn_predict_layers"] == 0
    assert output["compress_ratios"] == [0, 4]
    assert output["dspark_block_size"] == 0
    assert output["dspark_target_layer_ids"] == []


def test_transformers_materializer_applies_registered_weight_conversions(
    tmp_path,
):
    class ConvertedModel(torch.nn.Module):
        base_model_prefix = ""

        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Module()
            self.layer.forget_gate = torch.nn.Module()
            self.layer.forget_gate.dt_bias = torch.nn.Parameter(torch.empty(2))
            self.layer.conv = torch.nn.Conv1d(
                6,
                6,
                kernel_size=4,
                groups=6,
                bias=False,
            )
            self.layer.experts = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2, bias=False)]
            )

    path = tmp_path / "model.safetensors"
    query = torch.arange(8, dtype=torch.float16).reshape(2, 1, 4)
    key = query + 10
    value = query + 20
    dt_bias = torch.arange(2, dtype=torch.float32)
    expert = torch.eye(2, dtype=torch.float16)
    save_file(
        {
            "layer.q_conv.weight": query,
            "layer.k_conv.weight": key,
            "layer.v_conv.weight": value,
            "layer.dt_bias": dt_bias,
            "layer.experts.0.weight": expert,
        },
        path,
    )
    conversions = [
        WeightRenaming(
            r"layer\.dt_bias",
            "layer.forget_gate.dt_bias",
        ),
        WeightConverter(
            source_patterns=[
                "layer.q_conv.weight",
                "layer.k_conv.weight",
                "layer.v_conv.weight",
            ],
            target_patterns="layer.conv.weight",
            operations=[Concatenate(dim=0)],
        ),
        WeightConverter(
            source_patterns="layer.experts.*.weight",
            target_patterns="layer.packed_experts",
            operations=[Concatenate(dim=0)],
        ),
    ]
    with torch.device("meta"):
        model = ConvertedModel()
    materializer = TransformersWeightMaterializer(
        CastWeightMaterializer(), model, conversions
    )
    source = materializer.create_source(str(path))

    assert set(source.tensor_names()) == {
        "layer.forget_gate.dt_bias",
        "layer.conv.weight",
        "layer.experts.0.weight",
    }
    loader = TargetWeightLoader(model, source, materializer)
    assert loader.plan("layer.forget_gate").parameter_sources
    assert loader.plan("layer.conv").parameter_sources
    assert loader.plan("layer.experts.0").parameter_sources
    conv_metadata = source.metadata("layer.conv.weight")
    assert (
        materializer.estimate_workspace_bytes(
            "layer.conv.weight",
            conv_metadata,
            torch.bfloat16,
        )
        >= 3 * query.numel() * torch.bfloat16.itemsize
    )

    result = materialize_weights(
        source,
        source.tensor_names(),
        materializer,
        target_dtype=torch.bfloat16,
    )

    assert torch.equal(
        result["layer.conv.weight"],
        torch.cat((query, key, value), dim=0).to(torch.bfloat16),
    )
    assert torch.equal(
        result["layer.forget_gate.dt_bias"],
        dt_bias.to(torch.bfloat16),
    )
    assert torch.equal(
        result["layer.experts.0.weight"],
        expert.to(torch.bfloat16),
    )


def test_kimi_k3_source_maps_packed_expert_to_logical_weight(tmp_path):
    path = tmp_path / "model.safetensors"
    raw_name = (
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"
    )
    scale_name = raw_name.removesuffix("_packed") + "_scale"
    save_file(
        {
            raw_name: torch.tensor([[0x01, 0x29], [0x93, 0xF4]], dtype=torch.uint8),
            scale_name: torch.full((2, 1), 127, dtype=torch.uint8),
        },
        path,
    )
    source = KimiK3WeightSource(path)
    logical_name = raw_name.removesuffix("_packed")

    assert set(source.tensor_names()) == {logical_name}
    assert source.metadata(logical_name).shape == (2, 2)
    assert source.metadata(logical_name).dtype == torch.uint8

    result = materialize_weights(
        source,
        [logical_name],
        KimiK3WeightMaterializer(fp4_block_size=4),
        target_dtype=torch.bfloat16,
    )[logical_name]

    assert torch.equal(
        result,
        torch.tensor(
            [[0.5, 0.0, -0.5, 1.0], [1.5, -0.5, 2.0, -6.0]],
            dtype=torch.bfloat16,
        ),
    )


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"), reason="PyTorch has no E8M0 dtype"
)
def test_kimi_k3_source_normalizes_native_e8m0_scale_to_bytes(tmp_path):
    path = tmp_path / "model.safetensors"
    raw_name = (
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"
    )
    scale_name = raw_name.removesuffix("_packed") + "_scale"
    scale = torch.tensor([127, 128], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    save_file(
        {
            raw_name: torch.tensor([[0x01, 0x29]], dtype=torch.uint8),
            scale_name: scale.reshape(1, 2),
        },
        path,
    )
    source = KimiK3WeightSource(path)
    logical_name = raw_name.removesuffix("_packed")

    values = source.load_tensors_cpu([logical_name, scale_name])

    assert values[scale_name].dtype == torch.uint8
    assert values[scale_name].tolist() == [[127, 128]]


def test_kimi_k3_materializer_rejects_mismatched_scale_shape(tmp_path):
    materializer = KimiK3WeightMaterializer(fp4_block_size=4)
    name = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"

    with pytest.raises(ValueError, match="scale shape"):
        materializer.materialize_cpu(
            name,
            {
                name: torch.zeros((2, 2), dtype=torch.uint8),
                name.removesuffix(".weight") + ".weight_scale": torch.zeros(
                    (2, 2), dtype=torch.uint8
                ),
            },
            target_dtype=torch.bfloat16,
        )


def test_kimi_k3_materializer_decodes_ue8m0_endpoints():
    decoded = KimiK3WeightMaterializer._decode_e8m0(
        torch.tensor([0, 127, 255], dtype=torch.uint8)
    )

    assert decoded[0].item() == 0.0
    assert decoded[1].item() == 1.0
    assert torch.isinf(decoded[2])


def test_kimi_k3_materializer_kda_a_log_configuration_and_shape(tmp_path):
    materializer = KimiK3WeightMaterializer(kda_num_heads=96)
    name = "language_model.model.layers.0.self_attn.A_log"
    metadata = TensorMetadata(
        name=name,
        shape=(128,),
        dtype=torch.float32,
        shard=tmp_path / "model.safetensors",
    )
    source = torch.arange(128, dtype=torch.float32)

    result = materializer.materialize_cpu(
        name,
        {name: source},
        target_dtype=torch.bfloat16,
    )

    assert materializer.configuration()["kda_num_heads"] == 96
    assert (
        materializer.configuration()["routed_expert_loading"]
        == "resident-decoder-layer"
    )
    assert materializer.logical_shape(name, metadata) == (96,)
    assert torch.equal(result, source[:96].to(torch.bfloat16))


@pytest.mark.parametrize("shape", [(95,), (1, 1, 128, 1)])
def test_kimi_k3_materializer_rejects_invalid_kda_a_log_shape(tmp_path, shape):
    materializer = KimiK3WeightMaterializer(kda_num_heads=96)
    name = "language_model.model.layers.0.self_attn.A_log"
    metadata = TensorMetadata(
        name=name,
        shape=shape,
        dtype=torch.float32,
        shard=tmp_path / "model.safetensors",
    )

    with pytest.raises(ValueError, match="KDA A_log checkpoint tensor"):
        materializer.logical_shape(name, metadata)


def test_kimi_k3_output_config_removes_source_mxfp4_metadata():
    source_config = {
        "model_type": "kimi_k3",
        "quantization_config": {"format": "int-quantized"},
        "text_config": {
            "model_type": "kimi_linear",
            "quantization_config": {"format": "mxfp4-pack-quantized"},
        },
    }

    output_config = KimiK3WeightMaterializer().transform_output_config(source_config)

    assert output_config["quantization_config"] == {"format": "int-quantized"}
    assert "quantization_config" not in output_config["text_config"]
    assert source_config["text_config"]["quantization_config"] == {
        "format": "mxfp4-pack-quantized"
    }
