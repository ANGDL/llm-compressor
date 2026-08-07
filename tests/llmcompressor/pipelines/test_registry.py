from types import SimpleNamespace

import pytest
import torch

from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.modifiers.pruning import SparseGPTModifier, WandaPruningModifier
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.transform import (
    AWQModifier,
    QuIPModifier,
    SmoothQuantModifier,
    SpinQuantModifier,
)
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.pipelines import (
    CalibrationPipeline,
    DataFreePipeline,
    SequentialPipeline,
)


@pytest.mark.parametrize(
    "modifiers,exp_pipeline",
    [
        ([QuantizationModifier(scheme="FP8")], SequentialPipeline),
        ([QuantizationModifier(scheme="W4A16")], DataFreePipeline),
        (
            [QuantizationModifier(scheme="W4A16", weight_observer="imatrix_mse")],
            SequentialPipeline,
        ),
        (
            [QuantizationModifier(scheme="W4A16", observer={"weights": "imatrix_mse"})],
            SequentialPipeline,
        ),
        ([GPTQModifier(scheme="FP8")], SequentialPipeline),
        ([GPTQModifier(scheme="W4A16")], SequentialPipeline),
        ([SmoothQuantModifier(), GPTQModifier(scheme="W4A16")], SequentialPipeline),
        ([AWQModifier(), QuantizationModifier(scheme="W4A16")], SequentialPipeline),
        ([AWQModifier(), QuantizationModifier(scheme="FP8")], SequentialPipeline),
        ([SparseGPTModifier(sparsity=1.0)], SequentialPipeline),
        ([IMatrixGatherer()], SequentialPipeline),
        ([WandaPruningModifier(sparsity=1.0)], SequentialPipeline),
        ([QuIPModifier()], DataFreePipeline),
        ([SpinQuantModifier()], DataFreePipeline),
        ([QuIPModifier(), QuantizationModifier(scheme="FP8")], SequentialPipeline),
        ([QuIPModifier(), QuantizationModifier(scheme="W4A16")], DataFreePipeline),
        ([AutoSmoothModifier()], SequentialPipeline),
        ([AutoRoundModifier()], SequentialPipeline),
    ],
)
def test_infer_pipeline(modifiers, exp_pipeline):
    pipeline = CalibrationPipeline.from_modifiers(modifiers)
    assert isinstance(pipeline, exp_pipeline)


def test_legacy_imatrix_gatherer_is_a_noop():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    modifier = IMatrixGatherer(
        targets=["Linear"],
        ignore=[],
        weight_observer="imatrix_mse",
        attach_by_initialize=False,
    )

    assert modifier.on_initialize(SimpleNamespace(model=model))
    assert modifier.requires_calibration_data
    assert modifier.targets == ["Linear"]
    assert modifier.ignore == []
    assert modifier.weight_observer == "imatrix_mse"
    assert modifier.attach_by_initialize is False
    assert not model[0]._forward_pre_hooks
    assert not hasattr(model[0], "weight_observer")
