from __future__ import annotations

import re
import subprocess
from importlib import metadata
from pathlib import Path

__all__ = [
    "__commit__",
    "__commit_short__",
    "__version__",
    "__version_tuple__",
    "version",
    "version_tuple",
]

try:
    from ._build_info import __commit__ as _BUILD_COMMIT
    from ._build_info import __commit_short__ as _BUILD_COMMIT_SHORT
    from ._build_info import __dirty__ as _BUILD_DIRTY
except ImportError:
    _BUILD_COMMIT = None
    _BUILD_COMMIT_SHORT = None
    _BUILD_DIRTY = False

_COMMIT_RE = re.compile(r"\+g([0-9a-f]{7,40})(\.dirty)?")


def _normalize_tag(tag: str) -> str:
    # Fork releases use k-v* to avoid colliding with upstream tags.
    if tag.startswith("k-v"):
        return tag[3:]
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


def _commit_from_git() -> tuple[str | None, str | None, bool]:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except Exception:
        return None, None, False
    if not commit:
        return None, None, False
    return commit, commit[:12], dirty


def _resolve_commit() -> tuple[str | None, str | None, bool]:
    if _BUILD_COMMIT:
        return _BUILD_COMMIT, _BUILD_COMMIT_SHORT, bool(_BUILD_DIRTY)

    metadata_version = _version_from_metadata() or ""
    match = _COMMIT_RE.search(metadata_version)
    if match:
        commit = match.group(1)
        return commit, commit[:12], match.group(2) is not None

    return _commit_from_git()


def _with_commit(version: str, commit_short: str | None, dirty: bool) -> str:
    if not commit_short or _COMMIT_RE.search(version):
        return version
    suffix = f"+g{commit_short}"
    return f"{version}{suffix}.dirty" if dirty else f"{version}{suffix}"


def _resolve_version() -> str:
    version = _version_from_metadata() or _version_from_git() or "0.0.0"
    _commit, commit_short, dirty = _resolve_commit()
    return _with_commit(version, commit_short, dirty)


def _to_version_tuple(v: str) -> tuple[object, ...]:
    tokens = re.split(r"[.+-]", v)
    out: list[object] = []
    for token in tokens:
        if not token:
            continue
        out.append(int(token) if token.isdigit() else token)
    return tuple(out)


__commit__, __commit_short__, __commit_dirty__ = _resolve_commit()
__version__ = version = _resolve_version()
__version_tuple__ = version_tuple = _to_version_tuple(__version__)
