from __future__ import annotations

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
)

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.streaming.output import (
    build_quantization_config,
    prepare_quantized_tensor_for_save,
)
from tests.llmcompressor.streaming.test_quantize import scheme


def test_build_quantization_config_coalesces_identical_exact_schemes():
    first = scheme()
    second = scheme()

    config = build_quantization_config(
        {"layers.0": first, "layers.1": second}
    )

    assert len(config["config_groups"]) == 1
    assert config["config_groups"]["group_0"]["targets"] == [
        "layers.0",
        "layers.1",
    ]
    assert config["quantization_status"] == QuantizationStatus.COMPRESSED.value


def test_build_quantization_config_preserves_recipe_groups_and_ignore():
    modifier = QuantizationModifier(
        scheme="W8A8",
        targets=["Linear"],
        weight_observer="imatrix_mse",
        ignore=["lm_head"],
    )

    config = build_quantization_config(modifier.resolved_config)

    assert len(config["config_groups"]) == 1
    assert config["config_groups"]["group_0"]["targets"] == ["Linear"]
    assert config["ignore"] == ["lm_head"]
    assert config["format"] != CompressionFormat.mixed_precision.value
    assert config["quantization_status"] == QuantizationStatus.COMPRESSED.value


def _w4a8_scheme() -> QuantizationScheme:
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            strategy="channel",
            symmetric=True,
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type="int",
            strategy="token",
            dynamic=True,
        ),
    )


def test_prepare_quantized_tensor_packs_only_int4_int_quantized_weight():
    tensor = torch.tensor(
        [[-8, -1, 0, 7], [1, 2, 3, 4]], dtype=torch.int8
    )
    scheme = _w4a8_scheme()

    packed = prepare_quantized_tensor_for_save(
        "layers.0.weight",
        tensor,
        scheme,
        format_name=CompressionFormat.int_quantized.value,
        pack_to_int8=True,
    )

    assert packed.tolist() == [[-8, 112], [33, 67]]
    assert packed.device.type == "cpu"
    assert packed.data_ptr() != tensor.data_ptr()
    for name, format_name, enabled in (
        ("layers.0.bias", CompressionFormat.int_quantized.value, True),
        ("layers.0.weight", CompressionFormat.pack_quantized.value, True),
        ("layers.0.weight", CompressionFormat.int_quantized.value, False),
    ):
        assert (
            prepare_quantized_tensor_for_save(
                name,
                tensor,
                scheme,
                format_name=format_name,
                pack_to_int8=enabled,
            )
            is tensor
        )

    scheme.weights.num_bits = 8
    assert (
        prepare_quantized_tensor_for_save(
            "layers.0.weight",
            tensor,
            scheme,
            format_name=CompressionFormat.int_quantized.value,
            pack_to_int8=True,
        )
        is tensor
    )
