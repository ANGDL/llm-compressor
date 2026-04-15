from __future__ import annotations

import re
import subprocess
from importlib import metadata
from pathlib import Path

__all__ = ["__version__", "__version_tuple__", "version", "version_tuple"]


def _normalize_tag(tag: str) -> str:
    # Accept both "v0.10.0" and "0.10.0".
    return tag[1:] if tag.startswith("v") else tag


def _version_from_metadata() -> str | None:
    try:
        return metadata.version("llmcompressor")
    except metadata.PackageNotFoundError:
        return None


def _version_from_git() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        raw = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "describe",
                "--tags",
                "--long",
                "--dirty",
                "--abbrev=8",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

    match = re.match(
        r"^(?P<tag>.+)-(?P<distance>\d+)-g(?P<sha>[0-9a-f]+)(?P<dirty>-dirty)?$",
        raw,
    )
    if not match:
        return None

    tag = _normalize_tag(match.group("tag"))
    distance = int(match.group("distance"))
    sha = match.group("sha")
    dirty = match.group("dirty") is not None

    if distance == 0 and not dirty:
        return tag

    local = f"g{sha}"
    if dirty:
        local = f"{local}.dirty"
    return f"{tag}.dev{distance}+{local}"


def _resolve_version() -> str:
    # Prefer installed package metadata; if unavailable, derive from git.
    return _version_from_metadata() or _version_from_git() or "0.0.0"


def _to_version_tuple(v: str) -> tuple[object, ...]:
    tokens = re.split(r"[.+-]", v)
    out: list[object] = []
    for token in tokens:
        if not token:
            continue
        out.append(int(token) if token.isdigit() else token)
    return tuple(out)


__version__ = version = _resolve_version()
__version_tuple__ = version_tuple = _to_version_tuple(__version__)
