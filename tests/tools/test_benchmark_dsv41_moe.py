from types import SimpleNamespace

import pytest
import torch

from tests.llmcompressor.modeling.test_deepseek_v41 import (
    tiny_config,
    write_checkpoint,
)
from tools.benchmark_dsv41_moe import _checkpoint_moe, benchmark


@pytest.mark.parametrize("vision", [False, True])
def test_checkpoint_moe_preserves_source_controls_and_decodes_linear_weights(
    tmp_path, vision
):
    source = tmp_path / "source"
    tensors = write_checkpoint(source, tiny_config(dspark=False, vision=vision))
    model = _checkpoint_moe(
        SimpleNamespace(checkpoint=source, layer_id=0),
        torch.device("cpu"),
        torch.bfloat16,
    )
    assert model.gate.bias.dtype == torch.float32
    assert (model.gate.bias_vl is not None) == vision
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear:
            assert module.weight.dtype == torch.bfloat16
            torch.testing.assert_close(
                module.weight,
                tensors[f"layers.0.ffn.{name}.weight"].bfloat16(),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("calibrate_all_experts", [False, True])
def test_benchmark_checks_output_and_observer_alignment(calibrate_all_experts):
    report = benchmark(
        SimpleNamespace(
            checkpoint=None,
            device="cpu",
            dtype="bfloat16",
            seed=42,
            hidden_size=32,
            intermediate_size=32,
            experts=6,
            topk=2,
            tokens=[3, 13],
            warmup=1,
            repeats=2,
            variants=["native_where", "current"],
            with_imatrix=True,
            calibrate_all_experts=calibrate_all_experts,
            profile=False,
            output=None,
        )
    )
    for case in report["cases"]:
        current = case["variants"]["current"]
        assert current["finite"]
        assert current["vs_native_where"] == {
            "exact": True,
            "max_abs": 0.0,
            "imatrix_exact": True,
        }
