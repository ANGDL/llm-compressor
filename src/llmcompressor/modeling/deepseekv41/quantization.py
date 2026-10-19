"""Reference quantize/dequantize operations used by V4.1 attention."""

from functools import lru_cache
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=16)
def _supports_fp8_cast(device_type: str, device_index: int | None) -> bool:
    """Return whether a device accepts the native E4M3 cast operation."""

    if device_type in {"cpu", "meta"}:
        return True
    try:
        device = torch.device(device_type, device_index)
        torch.zeros(1, device=device).to(torch.float8_e4m3fn)
    except (NotImplementedError, RuntimeError):
        return False
    return True


def _fp8_e4m3_round(x):
    """Round finite values to E4M3FN without requiring an FP8 storage cast."""

    absolute = x.abs().clamp_max(448)
    exponent = torch.floor(torch.log2(absolute.clamp_min(2**-9))).clamp(-6, 8)
    step = torch.exp2(exponent - 3)
    rounded = (torch.round(absolute / step) * step).clamp_max(448)
    return rounded.copysign(x)


def fp8_qdq(x, group_size=32):
    groups = x.float().unflatten(-1, (-1, group_size))
    amax = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448)))
    normalized = (groups / scale).clamp(-448, 448)
    if _supports_fp8_cast(x.device.type, x.device.index):
        result = normalized.to(torch.float8_e4m3fn).float()
    else:
        result = _fp8_e4m3_round(normalized)
    return (result * scale).flatten(-2).to(x.dtype)


def fp4_qdq(x, group_size=32, *, e4m3_scale=False):
    groups = x.float().unflatten(-1, (-1, group_size))
    minimum = 6 * (2**-9 if e4m3_scale else 2**-126)
    amax = groups.abs().amax(-1, keepdim=True).clamp_min(minimum)
    if e4m3_scale:
        scale = (amax / 6).to(torch.float8_e4m3fn).float()
    else:
        scale = torch.exp2(torch.ceil(torch.log2(amax / 6)))
    value = (groups / scale).clamp(-6, 6)
    table = value.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    midpoints = (table[:-1] + table[1:]) / 2
    indices = torch.bucketize(value.abs().contiguous(), midpoints)
    lower = indices.clamp(max=6)
    ties = (value.abs() == midpoints[lower]) & (indices.remainder(2) == 1)
    indices = indices + ties.long()
    rounded = table[indices] * value.sign()
    return (rounded * scale).flatten(-2).to(x.dtype)


def _linear_forward(self, x):
    if self.source_fp8 and getattr(self, "quantization_scheme", None) is None:
        x = fp8_qdq(x)
    bias = None if self.bias is None else self.bias.float()
    return F.linear(x.float(), self.weight.float(), bias).to(x.dtype)


def Linear(in_features, out_features, *, source_fp8=False, bias=False, dtype=None):
    """Create a compressible nn.Linear with the reference accumulation rules.

    compressed-tensors selects formats by exact module type. Keep nn.Linear,
    and bind the floating reference forward before PTQ wraps it. A PTQ scheme
    then owns activation quantization instead of the source FP8 simulation.
    """
    module = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    module.source_fp8 = source_fp8
    module.forward = MethodType(_linear_forward, module)
    return module
