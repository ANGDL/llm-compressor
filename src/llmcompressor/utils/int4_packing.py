"""Memory-efficient packing helpers for INT4 checkpoint weights."""

from __future__ import annotations

import torch

DEFAULT_INT4_PACKING_WORKSPACE_BYTES = 64 * 1024 * 1024

__all__ = [
    "DEFAULT_INT4_PACKING_WORKSPACE_BYTES",
    "pack_int4_to_int8",
    "pack_int4_to_int8_cpu_snapshot",
]


def _validate_int4_tensor(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.dtype is not torch.int8:
        raise ValueError(
            f"Expected torch.int8 tensor for INT4 packing, got {tensor.dtype}"
        )
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor, but got {tensor.ndim}D")

    rows, columns = tensor.shape
    if columns % 2 != 0:
        raise ValueError(
            "Expected even number of columns for INT4 packing, "
            f"but got shape {tensor.shape}"
        )
    return rows, columns


def _pack_int4_block(tensor: torch.Tensor) -> torch.Tensor:
    """Pack a validated block using two half-sized temporary tensors."""

    low = tensor[:, 0::2].clone()
    low.bitwise_and_(0x0F)
    high = tensor[:, 1::2].clone()
    high.bitwise_and_(0x0F)
    high.bitwise_left_shift_(4)
    low.bitwise_or_(high)
    return low


def pack_int4_to_int8(tensor: torch.Tensor) -> torch.Tensor:
    """Pack adjacent signed INT4 values into one two's-complement byte."""

    _validate_int4_tensor(tensor)
    return _pack_int4_block(tensor)


def pack_int4_to_int8_cpu_snapshot(
    tensor: torch.Tensor,
    *,
    max_workspace_bytes: int = DEFAULT_INT4_PACKING_WORKSPACE_BYTES,
) -> torch.Tensor:
    """Pack into CPU storage while bounding temporary device allocations.

    The returned tensor never aliases model-owned storage. Accelerator inputs are
    packed in rectangular blocks and copied to their final CPU positions before
    the next block is allocated. ``max_workspace_bytes`` bounds the two INT8
    temporaries used for each block; it does not include the CPU result.
    """

    rows, columns = _validate_int4_tensor(tensor)
    if max_workspace_bytes < 2:
        raise ValueError("max_workspace_bytes must be at least 2")

    packed_columns = columns // 2
    snapshot = torch.empty((rows, packed_columns), dtype=torch.int8, device="cpu")
    if rows == 0 or packed_columns == 0:
        return snapshot

    # Each packed element needs one low-nibble and one high-nibble temporary.
    max_packed_elements = max_workspace_bytes // 2
    if packed_columns <= max_packed_elements:
        rows_per_block = max(1, max_packed_elements // packed_columns)
        for row_start in range(0, rows, rows_per_block):
            row_end = min(row_start + rows_per_block, rows)
            packed = _pack_int4_block(tensor[row_start:row_end])
            snapshot[row_start:row_end].copy_(packed)
    else:
        # Extremely wide rows are split by columns to retain the workspace bound.
        for row in range(rows):
            for packed_start in range(0, packed_columns, max_packed_elements):
                packed_end = min(
                    packed_start + max_packed_elements, packed_columns
                )
                input_start = packed_start * 2
                input_end = packed_end * 2
                packed = _pack_int4_block(
                    tensor[row : row + 1, input_start:input_end]
                )
                snapshot[row : row + 1, packed_start:packed_end].copy_(packed)

    return snapshot
