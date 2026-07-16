import pytest
import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.quantization import QuantizationArgs

from llmcompressor.entrypoints.model_free.lifecycle import (
    calibrate_weight,
    initialize_quantized_linear,
)
from llmcompressor.observers import StrictSymmetricMinMaxObserver


def test_strict_symmetric_minmax_uses_127():
    args = QuantizationArgs(
        num_bits=8,
        type="int",
        strategy="channel",
        symmetric=True,
        dynamic=False,
        observer="strict_symmetric_minmax",
    )
    observer = StrictSymmetricMinMaxObserver("weight", args)
    observer(torch.tensor([[1.0, -0.5], [0.25, -0.75]], dtype=torch.float32))

    scale = observer.get_qparams()["scale"]

    assert scale.flatten().tolist() == pytest.approx([1.0 / 127, 0.75 / 127])
    assert scale.dtype == torch.float32


def test_strict_symmetric_minmax_clamps_scale_after_dividing():
    args = QuantizationArgs(
        num_bits=8,
        type="int",
        strategy="channel",
        symmetric=True,
        dynamic=False,
        observer="strict_symmetric_minmax",
    )
    observer = StrictSymmetricMinMaxObserver("weight", args)
    observer(torch.zeros((2, 3), dtype=torch.float32))

    scale = observer.get_qparams()["scale"]

    assert scale.flatten().tolist() == pytest.approx([1e-8, 1e-8])


def test_model_free_strict_symmetric_scale_is_float32():
    from llmcompressor.entrypoints.model_free.validate import validate_scheme

    _, scheme = validate_scheme(
        "W8A8", strict_symmetric=True, scale_dtype=torch.float32
    )
    module = initialize_quantized_linear(
        torch.tensor([[1.0, -0.5], [0.25, -0.75]], dtype=torch.bfloat16),
        scheme,
        "cpu",
    )
    calibrate_weight(module)

    assert module.weight_scale.dtype == torch.float32
    assert module.weight_scale.detach().flatten().tolist() == pytest.approx(
        [1.0 / 127, 0.75 / 127]
    )

    compress_module(module)
    expected = torch.round(
        torch.tensor([[1.0, -0.5], [0.25, -0.75]], dtype=torch.float32)
        / module.weight_scale.detach()
    ).clamp(-128, 127).to(torch.int8)
    assert torch.equal(module.weight.detach(), expected)


def test_scale_dtype_is_independent_from_strict_symmetric():
    from llmcompressor.entrypoints.model_free.validate import validate_scheme

    _, scheme = validate_scheme("W8A8", strict_symmetric=True)

    assert scheme.weights.observer == "strict_symmetric_minmax"
    assert scheme.weights.scale_dtype is None