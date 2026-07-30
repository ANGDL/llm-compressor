"""On-demand decoding for the checkpoint-native Kimi K3 MXFP4 experts.

Kimi K3 stores routed experts as ``weight_packed`` (two E2M1 values per
``uint8`` byte) and ``weight_scale`` (one UE8M0 exponent per group of 32
input values).  The Transformers reference model still exposes each expert as
an ordinary ``nn.Linear`` with a logical ``weight`` parameter.  This adapter
keeps the checkpoint bytes untouched and materializes that logical weight only
for the currently active streaming subgraph.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from llmcompressor.streaming.checkpoint import (
    CheckpointWeightSource,
    SafetensorsWeightSource,
    TensorMetadata,
)

from .base import WeightMaterializer

__all__ = ["KimiK3WeightMaterializer", "KimiK3WeightSource"]


_PACKED_SUFFIX = ".weight_packed"
_SCALE_SUFFIX = ".weight_scale"
_EXPERT_MARKER = ".block_sparse_moe.experts."
_KDA_A_LOG_SUFFIX = ".self_attn.A_log"


def _logical_name(raw_name: str) -> str:
    if raw_name.endswith(_PACKED_SUFFIX):
        return f"{raw_name.removesuffix(_PACKED_SUFFIX)}.weight"
    return raw_name


class KimiK3WeightSource(CheckpointWeightSource):
    """Expose K3 raw names using the model's logical parameter names."""

    def __init__(self, checkpoint: str | Path):
        self._source = SafetensorsWeightSource(checkpoint)
        raw_names = tuple(self._source.tensor_names())
        self._logical_to_raw = {_logical_name(name): name for name in raw_names}
        if len(self._logical_to_raw) != len(raw_names):
            raise ValueError(
                "Kimi K3 checkpoint key normalization produced duplicate "
                "logical tensor names"
            )
        self._primary_names = tuple(
            logical
            for logical, raw in self._logical_to_raw.items()
            if not raw.endswith(_SCALE_SUFFIX)
        )

    def tensor_names(self):
        # ``weight_scale`` is an input dependency of a packed expert, not a
        # standalone model parameter to be copied during finalization.
        return self._primary_names

    def metadata(self, name: str) -> TensorMetadata:
        try:
            raw_name = self._logical_to_raw[name]
        except KeyError as error:
            raise KeyError(f"Unknown Kimi K3 tensor {name!r}") from error
        return replace(self._source.metadata(raw_name), name=name)

    def load_tensors(self, names, *, device: torch.device) -> dict[str, torch.Tensor]:
        requested = list(dict.fromkeys(names))
        raw_names = []
        for name in requested:
            try:
                raw_names.append(self._logical_to_raw[name])
            except KeyError as error:
                raise KeyError(f"Unknown Kimi K3 tensor {name!r}") from error
        raw_values = self._source.load_tensors(raw_names, device=device)
        return {name: raw_values[self._logical_to_raw[name]] for name in requested}

    def iter_tensor_groups(
        self, groups, *, device: torch.device
    ) -> Iterator[dict[str, torch.Tensor]]:
        logical_groups = [tuple(dict.fromkeys(group)) for group in groups]
        raw_groups = []
        for group in logical_groups:
            try:
                raw_groups.append(tuple(self._logical_to_raw[name] for name in group))
            except KeyError as error:
                raise KeyError(f"Unknown Kimi K3 tensor {error.args[0]!r}") from error

        for logical, raw, raw_values in zip(
            logical_groups,
            raw_groups,
            self._source.iter_tensor_groups(raw_groups, device=device),
        ):
            yield {
                logical_name: raw_values[raw_name]
                for logical_name, raw_name in zip(logical, raw)
            }


class KimiK3WeightMaterializer(WeightMaterializer):
    """Materialize K3 checkpoint layouts into model-compatible weights."""

    _FP4_TABLE = torch.tensor(
        (
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
        ),
        dtype=torch.float32,
    )

    def __init__(
        self,
        *,
        fp4_block_size: int = 32,
        kda_num_heads: int | None = None,
    ):
        if fp4_block_size <= 0:
            raise ValueError("Kimi K3 fp4_block_size must be positive")
        if kda_num_heads is not None and (
            isinstance(kda_num_heads, bool)
            or not isinstance(kda_num_heads, int)
            or kda_num_heads <= 0
        ):
            raise ValueError("Kimi K3 kda_num_heads must be a positive integer")
        self.fp4_block_size = fp4_block_size
        self.kda_num_heads = kda_num_heads

    def configuration(self) -> Mapping[str, Any]:
        return {
            "fp4_block_size": self.fp4_block_size,
            "kda_num_heads": self.kda_num_heads,
            "key_layout": "kimi-k3-weight-packed-weight-scale",
            "fp4_layout": "e2m1-low-nibble-first",
            "scale_layout": "ue8m0",
            "routed_expert_loading": "deferred-per-linear",
        }

    def create_source(self, checkpoint: str) -> CheckpointWeightSource:
        return KimiK3WeightSource(checkpoint)

    def output_config_updates(self) -> Mapping[str, Any]:
        # The output is still loaded by the K3 checkpoint's remote model code.
        # In particular, do not replace its model_type with DeepSeek-V4's.
        return {}

    def transform_output_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        # The source stores its MXFP4 metadata inside text_config and mirrors it
        # to the root while loading KimiK3Config.  The streamed checkpoint uses
        # the new root WNA8 config; retaining the nested source entry would leave
        # two contradictory storage contracts in config.json.
        updated = dict(config)
        text_config = updated.get("text_config")
        if isinstance(text_config, Mapping):
            text_config = dict(text_config)
            text_config.pop("quantization_config", None)
            updated["text_config"] = text_config
        return updated

    @staticmethod
    def _is_packed_expert(tensor_name: str, metadata: TensorMetadata) -> bool:
        return (
            tensor_name.endswith(".weight")
            and _EXPERT_MARKER in tensor_name
            and metadata.dtype == torch.uint8
        )

    def dependencies(self, tensor_name: str, metadata: TensorMetadata) -> list[str]:
        if self._is_packed_expert(tensor_name, metadata):
            return [f"{tensor_name.removesuffix('.weight')}{_SCALE_SUFFIX}"]
        return []

    def deferred_module(
        self, tensor_name: str, metadata: TensorMetadata
    ) -> str | None:
        if not self._is_packed_expert(tensor_name, metadata):
            return None
        return tensor_name.removesuffix(".weight")

    def logical_shape(
        self, tensor_name: str, metadata: TensorMetadata
    ) -> tuple[int, ...]:
        if self._is_packed_expert(tensor_name, metadata):
            return (*metadata.shape[:-1], metadata.shape[-1] * 2)
        if self._is_kda_a_log(tensor_name):
            self._validate_kda_a_log_shape(metadata.shape)
            return (self.kda_num_heads,)
        return metadata.shape

    def _is_kda_a_log(self, tensor_name: str) -> bool:
        return self.kda_num_heads is not None and tensor_name.endswith(
            _KDA_A_LOG_SUFFIX
        )

    def _validate_kda_a_log_shape(self, shape: tuple[int, ...]) -> None:
        if len(shape) != 1:
            raise ValueError(
                f"Kimi K3 KDA A_log checkpoint tensor must be 1D; got shape {shape}"
            )
        if shape[0] < self.kda_num_heads:
            raise ValueError(
                "Kimi K3 KDA A_log checkpoint tensor is shorter than num_heads: "
                f"{shape[0]} < {self.kda_num_heads}"
            )

    @staticmethod
    def _decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
        # Match the K3 vLLM loader exactly: UE8M0 is the exponent byte of an
        # FP32 value.  This also preserves the endpoint encodings (0 -> zero,
        # 255 -> infinity), where a plain exp2(byte - 127) is not equivalent.
        return (scale.to(torch.int32) << 23).view(torch.float32)

    def _dequantize_fp4(
        self, weight: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        out_dim, packed_in_dim = weight.shape
        in_dim = packed_in_dim * 2
        packed = weight.to(torch.uint8)
        table = self._FP4_TABLE.to(device=weight.device)
        low = table[(packed & 0x0F).long()]
        high = table[((packed >> 4) & 0x0F).long()]
        unpacked = torch.stack((low, high), dim=-1).flatten(1)

        decoded_scale = self._decode_e8m0(scale)
        expected = (
            out_dim,
            (in_dim + self.fp4_block_size - 1) // self.fp4_block_size,
        )
        if tuple(decoded_scale.shape) != expected:
            raise ValueError(
                "Kimi K3 FP4 scale shape does not match weight blocks: "
                f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
            )
        expanded_scale = decoded_scale.repeat_interleave(self.fp4_block_size, dim=-1)[
            :, :in_dim
        ]
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
        if self._is_kda_a_log(tensor_name):
            self._validate_kda_a_log_shape(tuple(weight.shape))
            result = weight[: self.kda_num_heads]
        elif (
            weight.dtype == torch.uint8
            and tensor_name.endswith(".weight")
            and _EXPERT_MARKER in tensor_name
        ):
            scale_name = f"{tensor_name.removesuffix('.weight')}{_SCALE_SUFFIX}"
            try:
                scale = tensors[scale_name].to(device)
            except KeyError as error:
                raise KeyError(f"Missing Kimi K3 MXFP4 scale {scale_name!r}") from error
            result = self._dequantize_fp4(weight, scale)
        elif weight.dtype.is_floating_point:
            result = weight
        else:
            raise TypeError(
                f"Unsupported Kimi K3 source dtype {weight.dtype} for {tensor_name!r}"
            )
        return result.to(device=device, dtype=target_dtype)
