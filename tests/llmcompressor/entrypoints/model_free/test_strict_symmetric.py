import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs

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