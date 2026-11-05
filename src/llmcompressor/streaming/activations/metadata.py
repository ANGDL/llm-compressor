"""Content and structure metadata for persisted activation boundaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from typing import Any

import torch


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().to(device="cpu").contiguous()
    return memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B").tobytes()


def _describe(
    value: Any,
    *,
    path: str,
    tensor_ids: dict[int, int],
    aliases: dict[str, int],
) -> tuple[Any, int]:
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity in tensor_ids:
            tensor_id = tensor_ids[identity]
            aliases[path] = tensor_id
            return {"type": "tensor_ref", "id": tensor_id}, 0
        tensor_id = len(tensor_ids)
        tensor_ids[identity] = tensor_id
        aliases[path] = tensor_id
        raw = _tensor_bytes(value)
        return (
            {
                "type": "tensor",
                "id": tensor_id,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            len(raw),
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": type(value).__name__, "value": value}, 0
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "sha256": hashlib.sha256(value).hexdigest(),
            "size": len(value),
        }, 0
    if is_dataclass(value) and not isinstance(value, type):
        fields_value = {}
        size = 0
        for field in fields(value):
            description, field_size = _describe(
                getattr(value, field.name),
                path=f"{path}.{field.name}",
                tensor_ids=tensor_ids,
                aliases=aliases,
            )
            fields_value[field.name] = description
            size += field_size
        return (
            {
                "type": "dataclass",
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": fields_value,
            },
            size,
        )
    if isinstance(value, list):
        values = []
        size = 0
        for index, item in enumerate(value):
            description, item_size = _describe(
                item,
                path=f"{path}[{index}]",
                tensor_ids=tensor_ids,
                aliases=aliases,
            )
            values.append(description)
            size += item_size
        return {"type": "list", "items": values}, size
    if isinstance(value, tuple):
        values = []
        size = 0
        for index, item in enumerate(value):
            description, item_size = _describe(
                item,
                path=f"{path}[{index}]",
                tensor_ids=tensor_ids,
                aliases=aliases,
            )
            values.append(description)
            size += item_size
        return {"type": "tuple", "items": values}, size
    if isinstance(value, dict):
        items = []
        size = 0
        for key, item in value.items():
            if not isinstance(key, (type(None), bool, int, float, str, bytes)):
                raise TypeError(
                    f"Unsupported boundary mapping key {type(key).__qualname__} "
                    f"at {path}"
                )
            description, item_size = _describe(
                item,
                path=f"{path}.{key}",
                tensor_ids=tensor_ids,
                aliases=aliases,
            )
            items.append(
                {
                    "key": key.hex() if isinstance(key, bytes) else key,
                    "key_type": type(key).__name__,
                    "value": description,
                }
            )
            size += item_size
        return {"type": "dict", "items": items}, size
    raise TypeError(
        f"Unsupported boundary value type {type(value).__qualname__} at {path}"
    )


def boundary_summary(value: Any) -> dict[str, Any]:
    """Describe boundary shape, aliases, tensor bytes, and content digest."""

    aliases: dict[str, int] = {}
    structure, size = _describe(
        value,
        path="value",
        tensor_ids={},
        aliases=aliases,
    )
    payload = {
        "schema_version": 1,
        "size": size,
        "structure": structure,
        "aliases": aliases,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload
