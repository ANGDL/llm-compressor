"""Remove partial quantized tensors left by a failed quantization task."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

_TENSOR_SUFFIXES = (
    ".safetensors",
    ".safetensors.index.json",
    ".pt",
    ".pth",
    ".bin",
)
_TENSOR_MARKERS = ("FINALIZED",)


def _is_remnant(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    return path.name in _TENSOR_MARKERS or path.name.endswith(_TENSOR_SUFFIXES)


def _iter_output_remnants(output_dir: Path) -> Iterable[Path]:
    if not output_dir.is_dir():
        return
    root = output_dir.resolve()
    for path in output_dir.rglob("*"):
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            continue
        if _is_remnant(path):
            yield path


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def quantized_remnants(model_dir: str | Path) -> dict[str, Any]:
    """Describe partial output tensors and the transient streaming workspace."""

    root = Path(model_dir)
    output_dir = root / "model"
    work_dir = root / "work"
    paths = list(_iter_output_remnants(output_dir))
    files = len(paths)
    total_bytes = sum(_size(path) for path in paths)
    if work_dir.is_dir():
        for path in work_dir.rglob("*"):
            if path.is_file() and not path.is_symlink():
                files += 1
                total_bytes += _size(path)
    relative = [
        str(path.relative_to(root))
        for path in sorted(paths, key=lambda item: str(item))
    ]
    if work_dir.is_dir():
        relative.append("work/")
    return {
        "files": files,
        "bytes": total_bytes,
        "paths": relative[:100],
    }


def remove_quantized_remnants(model_dir: str | Path) -> dict[str, Any]:
    """Delete partial output tensors and the streaming work directory.

    Only files inside the model run directory are touched. Configuration,
    tokenizer files, logs, reports and attempt state are preserved so the
    failure can still be inspected and retried.
    """

    root = Path(model_dir)
    before = quantized_remnants(root)
    for path in list(_iter_output_remnants(root / "model")):
        try:
            path.unlink()
        except OSError:
            continue
    work_dir = root / "work"
    if work_dir.is_symlink():
        work_dir.unlink()
    elif work_dir.is_dir():
        shutil.rmtree(work_dir)
    after = quantized_remnants(root)
    return {
        "files": before["files"],
        "bytes": before["bytes"],
        "paths": before["paths"],
        "remaining_files": after["files"],
        "remaining_bytes": after["bytes"],
    }
