import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "llmcompressor"
    / "utils"
    / "replace_tensor.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("replace_tensor", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

replace_tensors = MODULE.replace_tensors


def _tensor_size_in_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _write_json(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


@pytest.mark.unit
def test_replace_tensors_generates_suffixed_outputs_for_full_precision_source(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()

    source_tensor = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
    source_file = "model-00001-of-00001.safetensors"
    save_file(
        {"model.layers.3.self_attn.q_proj.weight": source_tensor},
        str(source_dir / source_file),
    )
    _write_json(
        source_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": _tensor_size_in_bytes(source_tensor)},
            "weight_map": {"model.layers.3.self_attn.q_proj.weight": source_file},
        },
    )
    _write_json(source_dir / "config.json", {"architectures": ["TestModel"]})

    target_original = torch.ones((2, 2), dtype=torch.int8)
    target_other = torch.full((2, 2), 2, dtype=torch.int8)
    target_file = "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.layers.3.self_attn.q_proj.weight": target_original,
            "model.layers.0.mlp.down_proj.weight": target_other,
        },
        str(target_dir / target_file),
    )
    _write_json(
        target_dir / "model.safetensors.index.json",
        {
            "metadata": {
                "total_size": _tensor_size_in_bytes(target_original)
                + _tensor_size_in_bytes(target_other)
            },
            "weight_map": {
                "model.layers.3.self_attn.q_proj.weight": target_file,
                "model.layers.0.mlp.down_proj.weight": target_file,
            },
        },
    )
    original_target_config = {
        "architectures": ["QuantizedModel"],
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "type": "int"},
                }
            },
            "format": "int-quantized",
            "ignore": ["lm_head"],
            "quantization_status": "compressed",
        },
    }
    _write_json(target_dir / "config.json", original_target_config)

    result = replace_tensors(
        source_model_path=str(source_dir),
        target_model_path=str(target_dir),
        tensor_name_pattern=r"model\.layers\.3\.self_attn\.q_proj\.weight$",
        suffix="bf16copy",
    )

    assert result["output_shard"] == "matched_tensors.bf16copy.safetensors"
    assert result["output_index"] == "model.safetensors.index.json.bf16copy"
    assert result["output_config"] == "config.json.bf16copy"

    with safe_open(
        str(target_dir / "matched_tensors.bf16copy.safetensors"), framework="pt"
    ) as handle:
        assert set(handle.keys()) == {"model.layers.3.self_attn.q_proj.weight"}
        extracted = handle.get_tensor("model.layers.3.self_attn.q_proj.weight")
        assert torch.equal(extracted, source_tensor)

    with (target_dir / "model.safetensors.index.json.bf16copy").open("r") as file:
        updated_index = json.load(file)

    assert (
        updated_index["weight_map"]["model.layers.3.self_attn.q_proj.weight"]
        == "matched_tensors.bf16copy.safetensors"
    )
    assert (
        updated_index["weight_map"]["model.layers.0.mlp.down_proj.weight"] == target_file
    )
    expected_total_size = (
        _tensor_size_in_bytes(target_original)
        + _tensor_size_in_bytes(target_other)
        - _tensor_size_in_bytes(target_original)
        + _tensor_size_in_bytes(source_tensor)
    )
    assert updated_index["metadata"]["total_size"] == expected_total_size

    with (target_dir / "config.json.bf16copy").open("r") as file:
        updated_config = json.load(file)
    assert "model.layers.3.self_attn.q_proj" in updated_config["quantization_config"]["ignore"]
    assert updated_config["quantization_config"]["config_groups"] == original_target_config[
        "quantization_config"
    ]["config_groups"]

    with (target_dir / "model.safetensors.index.json").open("r") as file:
        original_index = json.load(file)
    with (target_dir / "config.json").open("r") as file:
        original_config = json.load(file)

    assert original_index["weight_map"]["model.layers.3.self_attn.q_proj.weight"] == target_file
    assert original_config == original_target_config


@pytest.mark.unit
def test_replace_tensors_copies_quantized_groups_for_quantized_source(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()

    qweight = torch.ones((2, 2), dtype=torch.int8)
    qscale = torch.full((2, 2), 0.25, dtype=torch.float16)
    source_file = "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.layers.2.self_attn.q_proj.weight": qweight,
            "model.layers.2.self_attn.q_proj.weight_scale": qscale,
        },
        str(source_dir / source_file),
    )
    _write_json(
        source_dir / "model.safetensors.index.json",
        {
            "metadata": {
                "total_size": _tensor_size_in_bytes(qweight) + _tensor_size_in_bytes(qscale)
            },
            "weight_map": {
                "model.layers.2.self_attn.q_proj.weight": source_file,
                "model.layers.2.self_attn.q_proj.weight_scale": source_file,
            },
        },
    )
    _write_json(
        source_dir / "config.json",
        {
            "quantization_config": {
                "config_groups": {
                    "source_group": {
                        "targets": [r"re:.*self_attn\.q_proj$"],
                        "weights": {
                            "group_size": 128,
                            "num_bits": 4,
                            "strategy": "channel",
                            "type": "int",
                        },
                    }
                },
                "format": "pack-quantized",
                "ignore": [],
                "quantization_status": "compressed",
            }
        },
    )

    target_tensor = torch.zeros((2, 2), dtype=torch.int8)
    target_file = "model-00001-of-00001.safetensors"
    save_file(
        {"model.layers.0.mlp.down_proj.weight": target_tensor},
        str(target_dir / target_file),
    )
    _write_json(
        target_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": _tensor_size_in_bytes(target_tensor)},
            "weight_map": {"model.layers.0.mlp.down_proj.weight": target_file},
        },
    )
    _write_json(
        target_dir / "config.json",
        {
            "quantization_config": {
                "config_groups": {
                    "target_group": {
                        "targets": ["Linear"],
                        "weights": {"num_bits": 8, "type": "int"},
                    }
                },
                "format": "int-quantized",
                "ignore": ["lm_head"],
                "quantization_status": "compressed",
            }
        },
    )

    replace_tensors(
        source_model_path=str(source_dir),
        target_model_path=str(target_dir),
        tensor_name_pattern=r"model\.layers\.2\.self_attn\.q_proj\.(weight|weight_scale)$",
        suffix="quantcopy",
    )

    with (target_dir / "config.json.quantcopy").open("r") as file:
        updated_config = json.load(file)

    config_groups = updated_config["quantization_config"]["config_groups"]
    assert "target_group" in config_groups
    assert "source_group" in config_groups
    assert config_groups["source_group"]["targets"] == ["model.layers.2.self_attn.q_proj"]
    assert (
        config_groups["source_group"]["weights"]["num_bits"] == 4
    )
    assert updated_config["quantization_config"]["ignore"] == ["lm_head"]

    with (target_dir / "model.safetensors.index.json.quantcopy").open("r") as file:
        updated_index = json.load(file)

    assert (
        updated_index["weight_map"]["model.layers.2.self_attn.q_proj.weight"]
        == "matched_tensors.quantcopy.safetensors"
    )
    assert (
        updated_index["weight_map"]["model.layers.2.self_attn.q_proj.weight_scale"]
        == "matched_tensors.quantcopy.safetensors"
    )


@pytest.mark.unit
def test_replace_tensors_raises_when_no_tensor_matches(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()

    tensor = torch.ones((2, 2), dtype=torch.float32)
    source_file = "model.safetensors"
    save_file({"model.layers.0.self_attn.q_proj.weight": tensor}, str(source_dir / source_file))
    _write_json(
        source_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": _tensor_size_in_bytes(tensor)},
            "weight_map": {"model.layers.0.self_attn.q_proj.weight": source_file},
        },
    )

    save_file({"model.layers.1.mlp.down_proj.weight": tensor}, str(target_dir / source_file))
    _write_json(
        target_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": _tensor_size_in_bytes(tensor)},
            "weight_map": {"model.layers.1.mlp.down_proj.weight": source_file},
        },
    )
    _write_json(target_dir / "config.json", {"quantization_config": {"ignore": []}})

    with pytest.raises(ValueError, match="No tensors matched pattern"):
        replace_tensors(
            source_model_path=str(source_dir),
            target_model_path=str(target_dir),
            tensor_name_pattern=r"model\.layers\.9\..*",
            suffix="nomatch",
        )