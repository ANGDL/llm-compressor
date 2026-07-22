"""On-demand decoding for original DeepSeek-V4 FP8/FP4 checkpoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from llmcompressor.streaming.checkpoint import (
    CheckpointWeightSource,
    SafetensorsWeightSource,
    TensorMetadata,
)

from .base import WeightMaterializer


def _raw_to_logical(name: str) -> str:
    if name.startswith("model."):
        return name
    if name.startswith("head."):
        return f"model.lm_head.{name.removeprefix('head.')}"
    return f"model.{name}"


class DeepSeekV4WeightSource(CheckpointWeightSource):
    """Expose raw DeepSeek-V4 keys as native Transformers model keys."""

    def __init__(self, checkpoint: str | Path):
        self._source = SafetensorsWeightSource(checkpoint)
        self._logical_to_raw = {
            _raw_to_logical(name): name for name in self._source.tensor_names()
        }
        if len(self._logical_to_raw) != len(self._source.tensor_names()):
            raise ValueError("DeepSeek-V4 key normalization produced duplicates")
        self._primary_names = tuple(
            name
            for name in self._logical_to_raw
            if not name.endswith(".scale")
        )

    def tensor_names(self):
        # Source quantization scales are dependencies, not output model tensors.
        return self._primary_names

    def metadata(self, name: str) -> TensorMetadata:
        try:
            raw_name = self._logical_to_raw[name]
        except KeyError as error:
            raise KeyError(f"Unknown DeepSeek-V4 tensor {name!r}") from error
        return replace(self._source.metadata(raw_name), name=name)

    def load_tensors(
        self, names, *, device: torch.device
    ) -> dict[str, torch.Tensor]:
        requested = list(dict.fromkeys(names))
        raw_names = []
        for name in requested:
            try:
                raw_names.append(self._logical_to_raw[name])
            except KeyError as error:
                raise KeyError(f"Unknown DeepSeek-V4 tensor {name!r}") from error
        raw_values = self._source.load_tensors(raw_names, device=device)
        return {
            name: raw_values[self._logical_to_raw[name]] for name in requested
        }


class DeepSeekV4WeightMaterializer(WeightMaterializer):
    """Decode DeepSeek-V4 FP8 blocks and packed FP4 experts on demand."""

    _FP4_TABLE = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    )

    def __init__(
        self,
        *,
        fp8_block_size: tuple[int, int] = (128, 128),
        fp4_block_size: int = 32,
    ):
        if min(fp8_block_size) <= 0 or fp4_block_size <= 0:
            raise ValueError("DeepSeek-V4 block sizes must be positive")
        self.fp8_block_size = tuple(fp8_block_size)
        self.fp4_block_size = fp4_block_size

    def configuration(self) -> Mapping[str, Any]:
        return {
            "fp8_block_size": self.fp8_block_size,
            "fp4_block_size": self.fp4_block_size,
            "key_layout": "deepseek-v4-raw",
        }

    def create_source(self, checkpoint: str) -> CheckpointWeightSource:
        return DeepSeekV4WeightSource(checkpoint)

    def output_config_updates(self) -> Mapping[str, Any]:
        return {
            "architectures": ["DeepseekV4NativeForCausalLM"],
            "model_type": "deepseek_v4_native",
            "torch_dtype": "bfloat16",
        }

    def dependencies(
        self, tensor_name: str, metadata: TensorMetadata
    ) -> list[str]:
        if tensor_name.endswith(".weight") and metadata.dtype in {
            torch.float8_e4m3fn,
            torch.int8,
            torch.uint8,
        }:
            return [f"{tensor_name.removesuffix('.weight')}.scale"]
        return []

    def logical_shape(
        self, tensor_name: str, metadata: TensorMetadata
    ) -> tuple[int, ...]:
        if (
            tensor_name.endswith(".weight")
            and metadata.dtype in {torch.int8, torch.uint8}
        ):
            return (*metadata.shape[:-1], metadata.shape[-1] * 2)
        return metadata.shape

    @staticmethod
    def _decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
        return torch.exp2(scale.to(torch.float32) - 127.0)

    def _dequantize_fp8(
        self, weight: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        out_dim, in_dim = weight.shape
        block_rows, block_columns = self.fp8_block_size
        row_blocks = (out_dim + block_rows - 1) // block_rows
        column_blocks = (in_dim + block_columns - 1) // block_columns
        padded = F.pad(
            weight.float(),
            (
                0,
                column_blocks * block_columns - in_dim,
                0,
                row_blocks * block_rows - out_dim,
            ),
        )
        blocks = padded.reshape(
            row_blocks, block_rows, column_blocks, block_columns
        ).transpose(1, 2)
        decoded_scale = self._decode_e8m0(scale)
        if tuple(decoded_scale.shape) != (row_blocks, column_blocks):
            raise ValueError(
                "DeepSeek-V4 FP8 scale shape does not match weight blocks: "
                f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
            )
        result = blocks * decoded_scale[..., None, None]
        return result.transpose(1, 2).reshape(
            row_blocks * block_rows, column_blocks * block_columns
        )[:out_dim, :in_dim]

    def _dequantize_fp4(
        self, weight: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        out_dim, packed_in_dim = weight.shape
        in_dim = packed_in_dim * 2
        packed = weight.to(torch.uint8)
        table = torch.tensor(
            self._FP4_TABLE, dtype=torch.float32, device=weight.device
        )
        low = table[(packed & 0x0F).long()]
        high = table[((packed >> 4) & 0x0F).long()]
        unpacked = torch.stack((low, high), dim=-1).flatten(1)
        decoded_scale = self._decode_e8m0(scale)
        expected = (out_dim, (in_dim + self.fp4_block_size - 1) // self.fp4_block_size)
        if tuple(decoded_scale.shape) != expected:
            raise ValueError(
                "DeepSeek-V4 FP4 scale shape does not match weight blocks: "
                f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
            )
        expanded_scale = decoded_scale.repeat_interleave(
            self.fp4_block_size, dim=-1
        )[:, :in_dim]
        return unpacked * expanded_scale

    def materialize(
        self,
        tensor_name: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        target_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        weight = tensors[tensor_name].to(device)
        scale_name = f"{tensor_name.removesuffix('.weight')}.scale"
        if scale_name in tensors:
            scale = tensors[scale_name].to(device)
            if weight.dtype == torch.float8_e4m3fn:
                result = self._dequantize_fp8(weight, scale)
            elif weight.dtype in {torch.int8, torch.uint8}:
                result = self._dequantize_fp4(weight, scale)
            else:
                raise TypeError(
                    f"Unsupported scaled dtype {weight.dtype} for {tensor_name!r}"
                )
        elif weight.dtype.is_floating_point:
            result = weight
        else:
            raise TypeError(
                f"Unsupported unscaled dtype {weight.dtype} for {tensor_name!r}"
            )
        return result.to(device=device, dtype=target_dtype)
