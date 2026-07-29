import importlib.util
import json
from pathlib import Path

import pytest
import torch
from compressed_tensors.compressors.pack_quantized import pack_to_int32

from llmcompressor.utils import int4_packing
from llmcompressor.utils.int4_packing import (
    pack_int4_to_int8,
    pack_int4_to_int8_cpu_snapshot,
)

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
Int8Packer = MODULE.Int8Packer
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
@pytest.mark.parametrize("columns", range(2, 18, 2))
def test_pack_int4_to_int8_matches_oneshot_int32_packing(columns):
    values = torch.arange(columns * 2, dtype=torch.int8).reshape(2, columns)
    values = (values % 16) - 8

    packed = _pack_int4_to_int8(values)

    # compressed-tensors uses offset-binary nibbles in int32 containers. The
    # INT8 checkpoint contract uses two's-complement nibbles, which differ by
    # the sign bit in each nibble.
    packed_int32 = pack_to_int32(values, num_bits=4)
    expected = torch.stack(
        [
            (packed_int32 >> shift) & 0xFF
            for shift in (0, 8, 16, 24)
        ],
        dim=-1,
    ).reshape(2, -1).to(torch.uint8)
    expected = (expected ^ 0x88)[:, : columns // 2].contiguous().view(torch.int8)
    assert torch.equal(packed, expected)


@pytest.mark.unit
def test_pack_int4_to_int8_requires_int8_storage():
    with pytest.raises(ValueError, match="torch.int8"):
        _pack_int4_to_int8(torch.ones((2, 2), dtype=torch.int16))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("shape", "max_workspace_bytes", "expected_blocks"),
    (
        ((5, 8), 8, 5),
        ((2, 20), 8, 6),
    ),
)
def test_cpu_snapshot_packs_in_bounded_blocks(
    shape, max_workspace_bytes, expected_blocks, monkeypatch
):
    values = torch.arange(shape[0] * shape[1], dtype=torch.int8).reshape(shape)
    values = (values % 16) - 8
    expected = pack_int4_to_int8(values)
    pack_block = int4_packing._pack_int4_block
    block_shapes = []

    def recording_pack_block(tensor):
        block_shapes.append(tuple(tensor.shape))
        return pack_block(tensor)

    monkeypatch.setattr(
        int4_packing, "_pack_int4_block", recording_pack_block
    )
    actual = pack_int4_to_int8_cpu_snapshot(
        values,
        max_workspace_bytes=max_workspace_bytes,
    )

    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)
    assert len(block_shapes) == expected_blocks
    assert all(
        (rows * columns) <= max_workspace_bytes
        for rows, columns in block_shapes
    )


@pytest.mark.unit
def test_cpu_snapshot_does_not_alias_source_storage():
    values = torch.tensor([[-8, -1, 0, 7]], dtype=torch.int8)

    snapshot = pack_int4_to_int8_cpu_snapshot(values, max_workspace_bytes=2)
    values.zero_()

    assert snapshot.tolist() == [[-8, 112]]


@pytest.mark.unit
def test_cpu_snapshot_rejects_too_small_workspace():
    with pytest.raises(ValueError, match="at least 2"):
        pack_int4_to_int8_cpu_snapshot(
            torch.ones((1, 2), dtype=torch.int8),
            max_workspace_bytes=1,
        )


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


@pytest.mark.unit
def test_int8_packer_copies_model_subdirectories(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    save_path = tmp_path / "packed"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "config_groups": {
                        "group_0": {
                            "weights": {"num_bits": 4, "type": "int"},
                            "targets": ["Linear"],
                        }
                    }
                }
            }
        )
    )
    MODULE.save_file(
        {"model.layers.0.weight": torch.tensor([[1, 2]], dtype=torch.int8)},
        model_path / "model-00001.safetensors",
    )
    (model_path / "inference").mkdir()
    (model_path / "inference" / "generation_config.json").write_text("{}")

    class SynchronousPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def imap_unordered(self, function, iterable):
            return map(function, iterable)

    monkeypatch.setattr(MODULE, "Pool", SynchronousPool)

    Int8Packer(str(model_path), str(save_path)).save()

    assert (save_path / "inference" / "generation_config.json").read_text() == "{}"
