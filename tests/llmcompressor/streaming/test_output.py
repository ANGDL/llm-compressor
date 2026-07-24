from __future__ import annotations

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationStatus

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.streaming.output import build_quantization_config
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
