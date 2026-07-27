import torch
from compressed_tensors.quantization import FP8_E4M3_DATA

__all__ = ["get_supported_scale_dtypes", "validate_scale_dtype"]


def get_supported_scale_dtypes(tensor_dtype: torch.dtype) -> tuple[torch.dtype, ...]:
    supported = (tensor_dtype, torch.float32, FP8_E4M3_DATA.dtype)
    return tuple(
        dtype
        for index, dtype in enumerate(supported)
        if dtype not in supported[:index] and dtype != torch.float64
    )


def validate_scale_dtype(
    scale_dtype: torch.dtype | None,
    tensor_dtype: torch.dtype | None = None,
) -> None:
    if scale_dtype is None:
        return

    if tensor_dtype is None:
        if scale_dtype == torch.float64:
            raise ValueError("scale_dtype=torch.float64 is not supported")
        return

    supported = get_supported_scale_dtypes(tensor_dtype)
    if scale_dtype not in supported:
        supported_names = ", ".join(str(dtype) for dtype in supported)
        raise ValueError(
            f"scale_dtype={scale_dtype} is not supported for tensor dtype "
            f"{tensor_dtype}; supported dtypes are: {supported_names}"
        )
