"""V4.1 block32 decoding and bounded-memory Engram checkpoint export."""

import json
import os
import struct
from copy import deepcopy
from pathlib import Path

import torch
from safetensors import safe_open

from .._logging import streaming_logger
from .deepseek_v4 import (
    DeepSeekV4WeightMaterializer,
    DeepSeekV4WeightSource,
    _logical_to_raw,
    _raw_to_logical,
)

_ENGRAM_FP8_STORAGE = {
    "weight_dtype": "fp8_e4m3",
    "scale_dtype": "ue8m0",
    "group_size": 32,
    "axis": "row",
}


def _encode_engram_rows(weight):
    """Encode row-local groups of 32; never use 32x32 linear-weight blocks."""
    groups = weight.float().unflatten(-1, (-1, 32))
    if not torch.isfinite(groups).all():
        raise ValueError("Cannot encode non-finite Engram weights")
    # Keep scales positive and normal in FP32, including all-zero groups.
    maximum = groups.abs().amax(-1).clamp_min(448 * 2**-126)
    raw_scale = maximum / 448
    # frexp avoids log2 rounding down just above a power-of-two boundary.
    mantissa, exponent = torch.frexp(raw_scale)
    exponent = exponent - (mantissa == 0.5).int()
    scale = torch.ldexp(torch.ones_like(raw_scale), exponent)
    encoded = (groups / scale.unsqueeze(-1)).clamp(-448, 448)
    encoded = encoded.to(torch.float8_e4m3fn).flatten(-2).contiguous()
    if (maximum > 2.0**126).any():
        # An FP8 value and its scale can each be finite while their product
        # overflows FP32/BF16 (e.g. BF16 max rounds to 256 * 2**120).
        decoded_max = encoded.float().unflatten(-1, (-1, 32)).abs().amax(-1) * scale
        if not torch.isfinite(decoded_max.bfloat16()).all():
            raise ValueError("Engram FP8 encoding would overflow BF16 on lookup")
    return encoded, scale.to(torch.float8_e8m0fnu).contiguous()


def _raw_quantization_target(target):
    """Translate V4.1 recipe roots without rewriting arbitrary regex contents."""
    if not target.startswith("re:"):
        return _logical_to_raw(f"{target}.weight").removesuffix(".weight")
    pattern = target.removeprefix("re:")
    anchor = "^" if pattern.startswith("^") else ""
    pattern = pattern.removeprefix("^").removeprefix(r"model\.")
    if pattern == "lm_head" or pattern.startswith(("lm_head$", r"lm_head\.")):
        pattern = "head" + pattern.removeprefix("lm_head")
    return f"re:{anchor}{pattern}"


def _is_engram_table(name):
    return ".engram.embed." in name


class DeepSeekV41WeightSource(DeepSeekV4WeightSource):
    def __init__(self, checkpoint, *, preserve_engram=True):
        super().__init__(checkpoint)
        self._primary_names = tuple(
            name
            for name in self._primary_names
            if not (preserve_engram and _is_engram_table(name))
        )

    def engram_storage(self, *, table_dtype="source"):
        """Validate source metadata and describe the table export format."""
        storage = None
        for name in self._logical_to_raw:
            if not name.endswith(".engram.embed.weight"):
                continue
            weight = self.metadata(name)
            if len(weight.shape) != 2 or min(weight.shape) <= 0:
                raise ValueError(f"Invalid Engram table shape: {name}")
            scale_name = f"{name.removesuffix('.weight')}.scale"
            if weight.dtype in {torch.float8_e4m3fn, torch.uint8}:
                scale = self.metadata(scale_name)
                if (
                    scale.dtype not in {torch.float8_e8m0fnu, torch.uint8}
                    or weight.shape[-1] % 32
                    or scale.shape != (weight.shape[0], weight.shape[1] // 32)
                ):
                    raise ValueError(f"Invalid FP8 Engram row scales: {scale_name}")
                current = dict(_ENGRAM_FP8_STORAGE)
            else:
                dtype = {
                    torch.bfloat16: "bf16",
                    torch.float16: "fp16",
                    torch.float32: "fp32",
                }.get(weight.dtype)
                if dtype is None:
                    raise TypeError(f"Unsupported Engram dtype: {weight.dtype}")
                if scale_name in self._logical_to_raw:
                    raise ValueError("Unquantized Engram weights must not have scales")
                current = {"weight_dtype": dtype}
            if table_dtype == "fp8":
                if weight.shape[1] % 32:
                    raise ValueError(
                        "FP8 Engram tables require a width divisible by 32"
                    )
                current = dict(_ENGRAM_FP8_STORAGE)
            if storage is not None and storage != current:
                raise ValueError(
                    "Preserved Engram tables must use the same storage format"
                )
            storage = current
        return storage


class DeepSeekV41WeightMaterializer(DeepSeekV4WeightMaterializer):
    """Keep tables externally backed and export runtime-compatible FP8 rows.

    Calibration reads the original table. Only checkpoint export converts dense
    rows; existing FP8 weights/scales are copied byte-for-byte. ``source`` is an
    opt-out for consumers that support dense tables, not the XSGL INT8 runtime.
    """

    def __init__(
        self,
        *,
        save_raw_checkpoint_format=True,
        preserve_engram=True,
        source_has_mtp=None,
        engram_table_dtype="fp8",
        engram_chunk_rows=65536,
    ):
        if engram_table_dtype not in {"fp8", "source"}:
            raise ValueError("engram_table_dtype must be 'fp8' or 'source'")
        if engram_chunk_rows <= 0:
            raise ValueError("engram_chunk_rows must be positive")
        super().__init__(
            fp8_block_size=(32, 32),
            fp4_block_size=32,
            save_raw_checkpoint_format=save_raw_checkpoint_format,
            source_has_mtp=source_has_mtp,
        )
        self.preserve_engram = preserve_engram
        self.engram_table_dtype = engram_table_dtype
        self.engram_chunk_rows = engram_chunk_rows
        self._source_quantization_config = {}
        self._engram_storage = None

    def configuration(self):
        result = dict(super().configuration())
        result.update(
            key_layout="deepseek-v41-raw",
            preserve_engram=self.preserve_engram,
            engram_table_dtype=self.engram_table_dtype,
            engram_encoding="row32-e4m3-ue8m0-v1",
            engram_chunk_rows=self.engram_chunk_rows,
        )
        return result

    def create_source(self, checkpoint):
        config = json.loads((Path(checkpoint) / "config.json").read_text())
        self._source_quantization_config = config.get(
            "source_quantization_config", config.get("quantization_config", {})
        )
        source = DeepSeekV41WeightSource(
            checkpoint, preserve_engram=self.preserve_engram
        )
        # Validate before calibration starts, not after a full-model run.
        self._engram_storage = (
            source.engram_storage(table_dtype=self.engram_table_dtype)
            if self.preserve_engram
            else None
        )
        return source

    @staticmethod
    def _decode_e8m0(scale):
        if scale.dtype == torch.uint8:
            scale = scale.view(torch.float8_e8m0fnu)
        return scale.float()

    def materialize_cpu(self, tensor_name, tensors, *, target_dtype):
        if _is_engram_table(tensor_name):
            weight = tensors[tensor_name]
            scale_name = f"{tensor_name.removesuffix('.weight')}.scale"
            if weight.dtype == torch.float8_e4m3fn:
                scales = self._decode_e8m0(tensors[scale_name])
                expected = (weight.shape[0], weight.shape[1] // 32)
                if tuple(scales.shape) != expected:
                    raise ValueError(
                        "Engram scales must cover each row in groups of 32"
                    )
                result = weight.float().unflatten(-1, (-1, 32))
                return (result * scales.unsqueeze(-1)).flatten(-2).to(target_dtype)
        return super().materialize_cpu(tensor_name, tensors, target_dtype=target_dtype)

    def output_config_updates(self):
        return {
            "dtype": "bfloat16",
            "source_quantization_config": self._source_quantization_config,
        }

    def transform_output_config(self, config):
        updated = deepcopy(dict(config))
        if self.save_raw_checkpoint_format:
            quantization = updated.get("quantization_config") or {}
            for group in (quantization.get("config_groups") or {}).values():
                group["targets"] = [
                    _raw_quantization_target(target)
                    for target in group.get("targets") or []
                ]
            if quantization.get("ignore") is not None:
                quantization["ignore"] = [
                    _raw_quantization_target(target)
                    for target in quantization["ignore"]
                ]
        text = dict(updated.get("text_config", {}))
        if self.source_has_mtp is False:
            text["num_nextn_predict_layers"] = 0
            text["dspark_block_size"] = 0
            text["dspark_target_layer_ids"] = []
            text["compress_ratios"] = text["compress_ratios"][
                : text["num_hidden_layers"]
            ]
        updated["text_config"] = text
        updated.pop("engram_storage", None)
        if self.preserve_engram and self._engram_storage is not None:
            updated["engram_storage"] = dict(self._engram_storage)
        return updated

    def save_preserved_tensors(self, checkpoint, output_dir):
        if not self.preserve_engram:
            return {}
        checkpoint = Path(checkpoint).resolve()
        index_path = checkpoint / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())["weight_map"]
        groups = {}
        for name, filename in index.items():
            if _is_engram_table(name):
                source = (checkpoint / filename).resolve()
                if not source.is_relative_to(checkpoint):
                    raise ValueError("Engram source shard escapes checkpoint")
                output_name = self.output_tensor_name(_raw_to_logical(name))
                groups.setdefault(source, {})[name] = output_name
        result = {}
        for number, (source, names) in enumerate(sorted(groups.items())):
            copied = {}
            for table_number, (name, output_name) in enumerate(sorted(names.items())):
                if self.engram_table_dtype == "fp8" and name.endswith(".weight"):
                    with safe_open(source, framework="pt", device="cpu") as shard:
                        dtype = shard.get_slice(name).get_dtype()
                    if dtype in {"BF16", "F16", "F32"}:
                        filename = (
                            f"model-engram-fp8-{number:05d}-"
                            f"{table_number:03d}.safetensors"
                        )
                        scale_name = f"{output_name.removesuffix('.weight')}.scale"
                        self._encode_engram_table(
                            source,
                            name,
                            Path(output_dir) / filename,
                            output_name,
                            scale_name,
                        )
                        result.update({output_name: filename, scale_name: filename})
                        continue
                copied[name] = output_name
            if not copied:
                continue
            filename = f"model-preserved-engram-{number:05d}.safetensors"
            destination = Path(output_dir) / filename
            self._copy_tensor_payloads(source, destination, copied)
            result.update({name: filename for name in copied.values()})
        return result

    def _encode_engram_table(self, source, name, destination, weight_name, scale_name):
        """Stream a huge table to one atomic safetensors file, using bounded RAM.

        safetensors supplies bounded input slices. Its save_file API requires
        complete output tensors, so emit the two payloads by offset instead of
        allocating the full FP8 table and scale (over 100 GB per real layer).
        """
        with safe_open(source, framework="pt", device="cpu") as shard:
            rows, width = shard.get_slice(name).get_shape()
        if width % 32:
            raise ValueError("FP8 Engram tables require a width divisible by 32")
        weight_bytes = rows * width
        scale_bytes = rows * (width // 32)
        header = {
            weight_name: {
                "dtype": "F8_E4M3",
                "shape": [rows, width],
                "data_offsets": [0, weight_bytes],
            },
            scale_name: {
                "dtype": "F8_E8M0",
                "shape": [rows, width // 32],
                "data_offsets": [weight_bytes, weight_bytes + scale_bytes],
            },
        }
        payload = json.dumps(header, separators=(",", ":")).encode()
        payload += b" " * (-len(payload) % 8)
        data_start = 8 + len(payload)
        if destination.exists():
            with destination.open("rb") as existing:
                length = struct.unpack("<Q", existing.read(8))[0]
                actual = json.loads(existing.read(length))
            if actual != header or destination.stat().st_size != (
                8 + length + weight_bytes + scale_bytes
            ):
                raise ValueError(f"Encoded Engram shard disagrees: {destination}")
            return
        temporary = destination.with_suffix(".pending")
        streaming_logger.info(
            f"engram | encode FP8 row32 | tensor={name} | rows={rows}"
        )
        try:
            with temporary.open("wb") as output:
                output.write(struct.pack("<Q", len(payload)))
                output.write(payload)
                for start in range(0, rows, self.engram_chunk_rows):
                    end = min(rows, start + self.engram_chunk_rows)
                    # Clone a bounded slice, then close the entire file mapping.
                    # Keeping one mmap for the whole scan can accumulate resident
                    # source pages up to the size of the hundreds-of-GB table.
                    with safe_open(source, framework="pt", device="cpu") as shard:
                        chunk = shard.get_slice(name)[start:end].clone()
                    weight, scale = _encode_engram_rows(chunk)
                    del chunk
                    output.seek(data_start + start * width)
                    output.write(memoryview(weight.view(torch.uint8).numpy()))
                    output.seek(data_start + weight_bytes + start * (width // 32))
                    output.write(memoryview(scale.view(torch.uint8).numpy()))
                    del weight, scale
                    if start // self.engram_chunk_rows % 64 == 0 or end == rows:
                        streaming_logger.info(
                            f"engram | encode FP8 row32 | tensor={name} | "
                            f"rows={end}/{rows}"
                        )
                output.flush()
                os.fsync(output.fileno())
            with safe_open(temporary, framework="pt", device="cpu") as encoded:
                if set(encoded.keys()) != set(header):
                    raise ValueError("Encoded Engram tensor names disagree")
                for key, spec in header.items():
                    item = encoded.get_slice(key)
                    if (
                        item.get_shape() != spec["shape"]
                        or item.get_dtype() != spec["dtype"]
                    ):
                        raise ValueError(f"Invalid encoded Engram header: {key}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _copy_tensor_payloads(source, destination, names):
        with source.open("rb") as original:
            header_size = struct.unpack("<Q", original.read(8))[0]
            header = json.loads(original.read(header_size))
            data_start = 8 + header_size
            names = dict(
                sorted(
                    names.items(), key=lambda item: header[item[0]]["data_offsets"][0]
                )
            )
            selected, cursor = {}, 0
            for name, output in names.items():
                item = dict(header[name])
                start, end = item["data_offsets"]
                item["data_offsets"] = [cursor, cursor + end - start]
                selected[output] = item
                cursor += end - start
            if destination.exists():
                with destination.open("rb") as existing:
                    length = struct.unpack("<Q", existing.read(8))[0]
                    actual = json.loads(existing.read(length))
                actual.pop("__metadata__", None)
                if actual != selected:
                    raise ValueError(f"Preserved Engram shard disagrees: {destination}")
                return
            actual_names = set(header) - {"__metadata__"}
            if actual_names == set(names) and all(k == v for k, v in names.items()):
                try:
                    os.link(source, destination)
                    return
                except OSError:
                    pass
            payload = json.dumps(selected, separators=(",", ":")).encode()
            payload += b" " * (-len(payload) % 8)
            temporary = destination.with_suffix(".pending")
            try:
                with temporary.open("wb") as output:
                    output.write(struct.pack("<Q", len(payload)))
                    output.write(payload)
                    for name in names:
                        start, end = header[name]["data_offsets"]
                        original.seek(data_start + start)
                        remaining = end - start
                        while remaining:
                            chunk = original.read(min(4 * 1024**2, remaining))
                            if not chunk:
                                raise ValueError("Truncated source Engram shard")
                            output.write(chunk)
                            remaining -= len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
