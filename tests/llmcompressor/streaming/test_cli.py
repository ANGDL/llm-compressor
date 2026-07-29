import pytest
import torch

from llmcompressor.streaming.cli import _dtype, build_parser


@pytest.mark.parametrize("command", ["collect", "quantize", "finalize", "run"])
def test_cli_exposes_all_streaming_stages(command):
    parser = build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args([command, "--help"])
    assert error.value.code == 0


def test_cli_validates_computation_dtype():
    assert _dtype("bfloat16") is torch.bfloat16
    with pytest.raises(ValueError, match="floating-point"):
        _dtype("int8")


def test_cli_can_disable_packing_to_int8():
    parser = build_parser()

    args = parser.parse_args(
        [
            "quantize",
            "--checkpoint",
            "checkpoint",
            "--artifacts",
            "artifacts",
            "--staging",
            "staging",
            "--schemes",
            "schemes.json",
            "--no-pack-to-int8",
        ]
    )

    assert args.no_pack_to_int8 is True
