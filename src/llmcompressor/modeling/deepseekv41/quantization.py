"""Reference linear layers used by the V4.1 quantization pipeline."""

from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


def _linear_forward(self, x):
    if (
        x.dtype == self.weight.dtype
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and (self.bias is None or self.bias.dtype == x.dtype)
    ):
        return F.linear(x, self.weight, self.bias)
    bias = None if self.bias is None else self.bias.float()
    return F.linear(x.float(), self.weight.float(), bias).to(x.dtype)


def Linear(in_features, out_features, *, bias=False, dtype=None):
    """Create a compressible nn.Linear with the reference accumulation rules.

    compressed-tensors selects formats by exact module type. Keep nn.Linear,
    and bind the floating reference forward before PTQ wraps it. A PTQ scheme
    then owns activation quantization instead of the source FP8 simulation.
    """
    module = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    module.forward = MethodType(_linear_forward, module)
    return module
