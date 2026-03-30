import importlib.util
import json
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "llmcompressor"
    / "utils"
    / "pack_int4_to_int8.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("pack_int4_to_int8", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

QuantConfigParser = MODULE.QuantConfigParser
_pack_int4_to_int8 = MODULE._pack_int4_to_int8


@pytest.mark.unit
def test_pack_int4_to_int8_packs_rowwise():
    tensor = torch.tensor(
        [[0x1, 0x2, 0xF, 0x0], [0x3, 0x4, 0x5, 0x6]],
        dtype=torch.int8,
    )

    packed = _pack_int4_to_int8(tensor)

    expected = torch.tensor(
        [[0x21, 0x0F], [0x43, 0x65]],
        dtype=torch.int8,
    )
    assert torch.equal(packed, expected)


@pytest.mark.unit
def test_pack_int4_to_int8_raises_on_odd_columns():
    tensor = torch.tensor([[0x1, 0x2, 0x3]], dtype=torch.int8)

    with pytest.raises(ValueError, match="even number of columns"):
        _pack_int4_to_int8(tensor)


@pytest.mark.unit
def test_quant_config_parser_matches_linear_target_list(tmp_path):
    config = {
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "int", "strategy": "channel"},
                    "targets": ["Linear"],
                }
            }
        }
    }

    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    parser = QuantConfigParser(str(tmp_path))
    assert parser.is_int4_layer("model.layers.0.self_attn.q_proj.weight")


@pytest.mark.unit
def test_quant_config_parser_matches_regex_against_module_name(tmp_path):
    config = {
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "int", "strategy": "channel"},
                    "targets": [r"re:.*self_attn\.q_proj$"],
                }
            }
        }
    }

    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    parser = QuantConfigParser(str(tmp_path))
    assert parser.is_int4_layer("model.layers.0.self_attn.q_proj.weight")
    assert not parser.is_int4_layer("model.layers.0.self_attn.k_proj.weight")
