"""Finalize completed streaming staging shards into a model directory."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from safetensors import safe_open

from .artifacts import ArtifactStore, fingerprint_checkpoint
from .materialization import CastWeightMaterializer, WeightMaterializer

__all__ = ["finalize_streaming_checkpoint"]

_INDEX_NAME = "model.safetensors.index.json"
_FINALIZED = "FINALIZED"


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dtype_size(dtype_name: str) -> int:
    names = {
        "BOOL": torch.bool,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "U8": torch.uint8,
    }
    try:
        return torch.empty((), dtype=names[dtype_name]).element_size()
    except KeyError as error:
        raise ValueError(
            f"Unsupported output safetensors dtype {dtype_name!r}"
        ) from error


def _read_shard_headers(shard: Path) -> tuple[dict[str, tuple[int, ...]], int]:
    tensors = {}
    total_size = 0
    with safe_open(shard, framework="pt", device="cpu") as file:
        for name in file.keys():
            if name in tensors:
                raise ValueError(f"Duplicate tensor {name!r} in {shard.name}")
            tensor_slice = file.get_slice(name)
            shape = tuple(tensor_slice.get_shape())
            tensors[name] = shape
            numel = 1
            for dimension in shape:
                numel *= dimension
            total_size += numel * _dtype_size(tensor_slice.get_dtype())
    return tensors, total_size


def _update_config(
    config: dict[str, Any],
    quantization_config: Mapping[str, Any] | None,
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(config)
    updated.update(updates)
    qconfig = dict(quantization_config or updated.get("quantization_config", {}))
    qconfig.setdefault("quant_method", "compressed-tensors")
    qconfig.setdefault("format", "pack-quantized")
    qconfig["quantization_status"] = "compressed"
    updated["quantization_config"] = qconfig
    return updated


def _validate_transformers_config(path: Path) -> None:
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(path, local_files_only=True)
    except Exception as error:
        raise ValueError(
            f"Final config cannot be loaded by AutoConfig: {error}"
        ) from error


def finalize_streaming_checkpoint(
    *,
    checkpoint: str | Path,
    artifact_dir: str | Path,
    staging_dir: str | Path,
    output_dir: str | Path,
    quantization_config: Mapping[str, Any] | None = None,
    materializer: WeightMaterializer | None = None,
    copy_auxiliary_files: bool = True,
    validate_config: bool = True,
) -> Path:
    """Publish complete staging shards as a standard indexed checkpoint."""
    source_dir = Path(checkpoint).expanduser().resolve()
    staging = Path(staging_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise ValueError(f"checkpoint must be a directory: {source_dir}")
    if output == source_dir:
        raise ValueError("output_dir must differ from the source checkpoint")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    store = ArtifactStore(artifact_dir)
    manifest = store.load_manifest()
    if (
        fingerprint_checkpoint(source_dir).content_fingerprint
        != manifest.source.content_fingerprint
    ):
        raise ValueError(
            "Calibration artifacts belong to a different source checkpoint"
        )
    materializer = materializer or CastWeightMaterializer()
    source = materializer.create_source(str(source_dir))
    source_names = set(source.tensor_names())
    expected_shards = {
        source.metadata(name).shard.name for name in source_names
    }
    shards_dir = staging / "shards"
    states_dir = staging / "state"
    if not shards_dir.is_dir() or not states_dir.is_dir():
        raise RuntimeError("Staging directory is incomplete")
    actual_shards = {path.name for path in shards_dir.glob("*.safetensors")}
    actual_states = {
        path.name.removesuffix(".json") for path in states_dir.glob("*.json")
    }
    if actual_shards != expected_shards or actual_states != expected_shards:
        raise RuntimeError(
            "Staging directory is incomplete: shard set does not match the "
            "source checkpoint; "
            f"expected={sorted(expected_shards)}, "
            f"shards={sorted(actual_shards)}, states={sorted(actual_states)}"
        )

    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(parents=True)
        output_shards: dict[str, dict[str, tuple[int, ...]]] = {}
        total_size = 0
        for shard_name in sorted(expected_shards):
            logger.info(f"streaming finalize: validating shard {shard_name}")
            shard = shards_dir / shard_name
            state_path = states_dir / f"{shard_name}.json"
            if not shard.is_file() or not state_path.is_file():
                raise RuntimeError(f"Staging shard {shard_name!r} is incomplete")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("completed") is not True:
                raise RuntimeError(f"Staging shard {shard_name!r} is not complete")
            headers, shard_size = _read_shard_headers(shard)
            if state.get("tensor_names") != sorted(headers):
                raise ValueError(f"State tensor list disagrees with {shard_name!r}")
            if state.get("total_size") != shard_size:
                raise ValueError(f"State total_size disagrees with {shard_name!r}")
            output_shards[shard_name] = headers
            total_size += shard_size
            shutil.copy2(shard, temporary / shard_name)

        weight_map: dict[str, str] = {}
        for shard_name, headers in output_shards.items():
            for tensor_name in headers:
                if tensor_name in weight_map:
                    raise ValueError(
                        f"Tensor {tensor_name!r} appears in multiple shards"
                    )
                weight_map[tensor_name] = shard_name
        if set(weight_map) != {
            name for headers in output_shards.values() for name in headers
        }:
            raise ValueError("Final weight map does not cover all output tensors")
        quantized_modules = {
            module
            for shard_name in expected_shards
            for module in json.loads(
                (states_dir / f"{shard_name}.json").read_text(encoding="utf-8")
            ).get("quantized_modules", [])
        }
        module_formats = {
            module: format_name
            for shard_name in expected_shards
            for module, format_name in json.loads(
                (states_dir / f"{shard_name}.json").read_text(encoding="utf-8")
            ).get("module_formats", {}).items()
        }
        omitted_tied_weights = {
            alias: canonical
            for shard_name in expected_shards
            for alias, canonical in json.loads(
                (states_dir / f"{shard_name}.json").read_text(encoding="utf-8")
            ).get("omitted_tied_weights", {}).items()
        }
        for alias, canonical in omitted_tied_weights.items():
            if alias in weight_map:
                raise ValueError(f"Omitted tied weight {alias!r} is still present")
            if alias not in source_names:
                raise ValueError(f"Unknown omitted tied weight {alias!r}")
            if canonical not in weight_map:
                raise ValueError(
                    f"Canonical tensor {canonical!r} for tied weight {alias!r} "
                    "is missing"
                )
        for module_name in quantized_modules:
            names = {
                name for name in weight_map if name.startswith(f"{module_name}.")
            }
            format_name = module_formats.get(module_name)
            expected_weights = (
                {f"{module_name}.weight"}
                if format_name in {"int-quantized", "float-quantized"}
                else {
                    f"{module_name}.weight_packed",
                    f"{module_name}.weight_compressed",
                }
            )
            if names.isdisjoint(expected_weights):
                raise ValueError(
                    f"Quantized module {module_name!r} has no weight for format "
                    f"{format_name!r}"
                )
            if f"{module_name}.weight_scale" not in names:
                raise ValueError(
                    f"Quantized module {module_name!r} has no weight scale"
                )
        for source_name in source_names:
            module_name, separator, tensor_name = source_name.rpartition(".")
            if (
                separator
                and tensor_name == "weight"
                and module_name in quantized_modules
            ):
                continue
            if source_name not in weight_map:
                if source_name in omitted_tied_weights:
                    continue
                raise ValueError(
                    f"Non-quantized source tensor {source_name!r} is missing"
                )

        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        _atomic_json(temporary / _INDEX_NAME, index)

        config_path = source_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing source config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        _atomic_json(
            temporary / "config.json",
            _update_config(
                config,
                quantization_config,
                materializer.output_config_updates(),
            ),
        )

        if copy_auxiliary_files:
            for path in source_dir.iterdir():
                if (
                    path.name in {"config.json", _INDEX_NAME}
                    or path.suffix == ".safetensors"
                ):
                    continue
                destination = temporary / path.name
                if path.is_dir():
                    shutil.copytree(path, destination)
                else:
                    shutil.copy2(path, destination)

        if validate_config:
            _validate_transformers_config(temporary)
        (temporary / _FINALIZED).write_bytes(b"")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output
