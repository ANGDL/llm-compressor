"""Default materializer for ordinary floating-point checkpoints."""

from __future__ import annotations

from typing import Mapping

import torch

from .base import WeightMaterializer


class CastWeightMaterializer(WeightMaterializer):
    """Cast FP32, FP16, or BF16 checkpoint weights for computation."""

    _SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}

    def materialize_cpu(
        self,
        tensor_name: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = tensors[tensor_name]
        if tensor.dtype not in self._SUPPORTED_DTYPES:
            raise TypeError(
                f"CastWeightMaterializer does not support source dtype "
                f"{tensor.dtype} for {tensor_name!r}"
            )
        return tensor.to(dtype=target_dtype, copy=True)
