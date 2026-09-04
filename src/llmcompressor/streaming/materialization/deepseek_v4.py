"""On-demand decoding for original DeepSeek-V4 FP8/FP4 checkpoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping

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


def _logical_to_raw(name: str) -> str:
    return (
        name.removeprefix("model.")
        .replace(".weight_scale", ".scale")
        .replace("lm_head.", "head.")
    )


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

    def load_tensors_cpu(self, names) -> dict[str, torch.Tensor]:
        requested = list(dict.fromkeys(names))
        raw_names = []
        for name in requested:
            try:
                raw_names.append(self._logical_to_raw[name])
            except KeyError as error:
                raise KeyError(f"Unknown DeepSeek-V4 tensor {name!r}") from error
        raw_values = self._source.load_tensors_cpu(raw_names)
        return {
            name: raw_values[self._logical_to_raw[name]] for name in requested
        }

    def iter_tensor_groups_cpu(self, groups) -> Iterator[dict[str, torch.Tensor]]:
        logical_groups = [tuple(dict.fromkeys(group)) for group in groups]
        raw_groups = []
        for group in logical_groups:
            try:
                raw_groups.append(tuple(self._logical_to_raw[name] for name in group))
            except KeyError as error:
                raise KeyError(
                    f"Unknown DeepSeek-V4 tensor {error.args[0]!r}"
                ) from error

        for logical, raw, raw_values in zip(
            logical_groups,
            raw_groups,
            self._source.iter_tensor_groups_cpu(raw_groups),
        ):
            yield {
                logical_name: raw_values[raw_name]
                for logical_name, raw_name in zip(logical, raw)
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
        save_raw_checkpoint_format: bool = False,
        source_has_mtp: bool | None = None,
    ):
        if min(fp8_block_size) <= 0 or fp4_block_size <= 0:
            raise ValueError("DeepSeek-V4 block sizes must be positive")
        self.fp8_block_size = tuple(fp8_block_size)
        self.fp4_block_size = fp4_block_size
        self.save_raw_checkpoint_format = save_raw_checkpoint_format
        self.source_has_mtp = source_has_mtp

    def configuration(self) -> Mapping[str, Any]:
        return {
            "fp8_block_size": self.fp8_block_size,
            "fp4_block_size": self.fp4_block_size,
            "key_layout": "deepseek-v4-raw",
            "save_raw_checkpoint_format": self.save_raw_checkpoint_format,
            "source_has_mtp": self.source_has_mtp,
        }

    def create_source(self, checkpoint: str) -> CheckpointWeightSource:
        return DeepSeekV4WeightSource(checkpoint)

    def output_config_updates(self) -> Mapping[str, Any]:
        return {"torch_dtype": "bfloat16"}

    def transform_output_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.source_has_mtp is not False:
            return config
        updated = dict(config)
        updated["num_nextn_predict_layers"] = 0
        updated["dspark_block_size"] = 0
        updated["dspark_target_layer_ids"] = []
        compress_ratios = updated.get("compress_ratios")
        num_hidden_layers = updated.get("num_hidden_layers")
        if isinstance(compress_ratios, list) and isinstance(num_hidden_layers, int):
            updated["compress_ratios"] = compress_ratios[:num_hidden_layers]
        return updated

    def output_tensor_name(self, tensor_name: str) -> str:
        if not self.save_raw_checkpoint_format:
            return tensor_name
        return _logical_to_raw(tensor_name)

    def output_weight_scale_name(self, output_module_name: str) -> str:
        if self.save_raw_checkpoint_format:
            return f"{output_module_name}.scale"
        return super().output_weight_scale_name(output_module_name)

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
        if scale.dtype == getattr(torch, "float8_e8m0fnu", None):
            scale = scale.view(torch.uint8)
        return (scale.to(torch.int32) << 23).view(torch.float32)

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

    def materialize_cpu(
        self,
        tensor_name: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        weight = tensors[tensor_name]
        scale_name = f"{tensor_name.removesuffix('.weight')}.scale"
        if scale_name in tensors:
            scale = tensors[scale_name]
            if scale.dtype == getattr(torch, "float8_e8m0fnu", None):
                scale = scale.view(torch.uint8)
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
        return result.to(dtype=target_dtype)

    def estimate_workspace_bytes(
        self, tensor_name, metadata, target_dtype
    ) -> int:
        if self.dependencies(tensor_name, metadata):
            elements = 1
            for dimension in self.logical_shape(tensor_name, metadata):
                elements *= dimension
            return 4 * elements * torch.float32.itemsize
        return 0
