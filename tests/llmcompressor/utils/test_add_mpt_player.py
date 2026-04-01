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
    / "add_mpt_player.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("add_mpt_player", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

add_mpt_player = MODULE.add_mpt_player
update_config_ignores = MODULE.update_config_ignores


def _tensor_size_in_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@pytest.mark.unit
def test_add_mpt_player_updates_index_and_logs_progress(tmp_path, capsys):
    bf16_dir = tmp_path / "bf16"
    quant_dir = tmp_path / "quant"
    bf16_dir.mkdir()
    quant_dir.mkdir()

    t_q = torch.ones((2, 2), dtype=torch.float32)
    t_k = torch.full((2, 2), 2.0, dtype=torch.float32)
    t_other = torch.zeros((2, 2), dtype=torch.float32)

    bf16_file = "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.layers.3.self_attn.q_proj.weight": t_q,
            "model.layers.3.self_attn.k_proj.weight": t_k,
            "model.layers.2.self_attn.q_proj.weight": t_other,
        },
        str(bf16_dir / bf16_file),
    )

    bf16_index = {
        "metadata": {
            "total_size": _tensor_size_in_bytes(t_q)
            + _tensor_size_in_bytes(t_k)
            + _tensor_size_in_bytes(t_other)
        },
        "weight_map": {
            "model.layers.3.self_attn.q_proj.weight": bf16_file,
            "model.layers.3.self_attn.k_proj.weight": bf16_file,
            "model.layers.2.self_attn.q_proj.weight": bf16_file,
        },
    }
    with open(bf16_dir / "model.safetensors.index.json", "w") as f:
        json.dump(bf16_index, f)

    quant_tensor = torch.randn((2, 2), dtype=torch.float32)
    quant_file = "model.safetensors"
    save_file({"model.layers.0.mlp.down_proj.weight": quant_tensor}, str(quant_dir / quant_file))

    quant_total_size = _tensor_size_in_bytes(quant_tensor)
    quant_index = {
        "metadata": {"total_size": quant_total_size},
        "weight_map": {"model.layers.0.mlp.down_proj.weight": quant_file},
    }
    with open(quant_dir / "model.safetensors.index.json", "w") as f:
        json.dump(quant_index, f)

    add_mpt_player(str(bf16_dir), str(quant_dir), 3)

    captured = capsys.readouterr().out
    assert "Found 2 tensors for model.layers.3" in captured
    assert "[1/2] Processing tensor: model.layers.3.self_attn.q_proj.weight" in captured
    assert "[2/2] Processing tensor: model.layers.3.self_attn.k_proj.weight" in captured

    with open(quant_dir / "model.safetensors.index.json", "r") as f:
        updated_index = json.load(f)

    updated_map = updated_index["weight_map"]
    assert updated_map["model.layers.0.mlp.down_proj.weight"] == quant_file
    assert updated_map["model.layers.3.self_attn.q_proj.weight"] == "mtp.safetensors"
    assert updated_map["model.layers.3.self_attn.k_proj.weight"] == "mtp.safetensors"

    expected_added_size = _tensor_size_in_bytes(t_q) + _tensor_size_in_bytes(t_k)
    assert updated_index["metadata"]["total_size"] == quant_total_size + expected_added_size
    assert (quant_dir / "model.safetensors.index.json.bak").exists()

    with safe_open(str(quant_dir / "mtp.safetensors"), framework="pt") as f:
        keys = set(f.keys())
    assert keys == {
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
    }


@pytest.mark.unit
def test_add_mpt_player_raises_when_no_matching_tensor(tmp_path):
    bf16_dir = tmp_path / "bf16"
    quant_dir = tmp_path / "quant"
    bf16_dir.mkdir()
    quant_dir.mkdir()

    bf16_file = "model-00001-of-00001.safetensors"
    tensor = torch.ones((2, 2), dtype=torch.float32)
    save_file({"model.layers.1.self_attn.q_proj.weight": tensor}, str(bf16_dir / bf16_file))

    with open(bf16_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            {
                "metadata": {"total_size": _tensor_size_in_bytes(tensor)},
                "weight_map": {"model.layers.1.self_attn.q_proj.weight": bf16_file},
            },
            f,
        )

    quant_file = "model.safetensors"
    save_file({"model.layers.0.mlp.down_proj.weight": tensor}, str(quant_dir / quant_file))
    with open(quant_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            {
                "metadata": {"total_size": _tensor_size_in_bytes(tensor)},
                "weight_map": {"model.layers.0.mlp.down_proj.weight": quant_file},
            },
            f,
        )

    with pytest.raises(ValueError, match="No tensors found for layer model.layers.3"):
        add_mpt_player(str(bf16_dir), str(quant_dir), 3)


@pytest.mark.unit
def test_add_mpt_player_supports_layer_name_matching(tmp_path, capsys):
    bf16_dir = tmp_path / "bf16"
    quant_dir = tmp_path / "quant"
    bf16_dir.mkdir()
    quant_dir.mkdir()

    t_mtp = torch.ones((2, 2), dtype=torch.float32)
    t_other = torch.zeros((2, 2), dtype=torch.float32)

    bf16_file = "model-00001-of-00001.safetensors"
    save_file(
        {
            "mtp.layers.0.self_attn.q_proj.weight": t_mtp,
            "mtp.layers.1.self_attn.q_proj.weight": t_other,
        },
        str(bf16_dir / bf16_file),
    )

    with open(bf16_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            {
                "metadata": {
                    "total_size": _tensor_size_in_bytes(t_mtp)
                    + _tensor_size_in_bytes(t_other)
                },
                "weight_map": {
                    "mtp.layers.0.self_attn.q_proj.weight": bf16_file,
                    "mtp.layers.1.self_attn.q_proj.weight": bf16_file,
                },
            },
            f,
        )

    quant_tensor = torch.randn((2, 2), dtype=torch.float32)
    quant_file = "model.safetensors"
    save_file({"model.layers.0.mlp.down_proj.weight": quant_tensor}, str(quant_dir / quant_file))

    with open(quant_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            {
                "metadata": {"total_size": _tensor_size_in_bytes(quant_tensor)},
                "weight_map": {"model.layers.0.mlp.down_proj.weight": quant_file},
            },
            f,
        )

    add_mpt_player(
        str(bf16_dir),
        str(quant_dir),
        mtp_layer_name="mtp.layers.0",
    )

    captured = capsys.readouterr().out
    assert "Found 1 tensors for mtp.layers.0" in captured

    with open(quant_dir / "model.safetensors.index.json", "r") as f:
        updated_index = json.load(f)

    assert (
        updated_index["weight_map"]["mtp.layers.0.self_attn.q_proj.weight"]
        == "mtp.safetensors"
    )
    assert "mtp.layers.1.self_attn.q_proj.weight" not in updated_index["weight_map"]


@pytest.mark.unit
def test_update_config_ignores_adds_and_is_idempotent(tmp_path):
    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()

    config_path = quant_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "quantization_config": {
                    "ignore": ["lm_head"],
                }
            },
            f,
        )

    update_config_ignores(str(quant_dir), 3)
    update_config_ignores(str(quant_dir), 3)

    with open(config_path, "r") as f:
        config_data = json.load(f)

    ignores = config_data["quantization_config"]["ignore"]
    assert ignores.count("model.layers.3") == 1
    assert ignores.count(r"re:.*layers\.3\..*") == 1


@pytest.mark.unit
def test_update_config_ignores_supports_layer_name(tmp_path):
    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()

    config_path = quant_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({"quantization_config": {"ignore": []}}, f)

    update_config_ignores(str(quant_dir), mtp_layer_name="mtp.layers.0")

    with open(config_path, "r") as f:
        config_data = json.load(f)

    ignores = config_data["quantization_config"]["ignore"]
    assert "mtp.layers.0" in ignores
    assert r"re:.*mtp\.layers\.0\..*" in ignores