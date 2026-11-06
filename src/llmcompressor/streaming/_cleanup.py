"""Cleanup of temporary streaming PTQ artifacts after publication."""

from __future__ import annotations

import shutil
from pathlib import Path

from ._logging import streaming_logger

_OWNED_WORK_DIR_ENTRIES = ("artifacts", "boundaries", "inputs", "publish", "staging")


def cleanup_successful_work_dir(work: Path) -> None:
    """Remove owned artifacts and an empty work root, preserving caller files."""
    for name in _OWNED_WORK_DIR_ENTRIES:
        path = work / name
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError as error:
            streaming_logger.warning(
                f"finalize | work cleanup incomplete | path={path} | error={error}"
            )
    try:
        work.rmdir()
    except OSError:
        pass
