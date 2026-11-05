"""Durable single-participant coordination for streaming transactions."""

from __future__ import annotations

import fcntl
import functools
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_json
from ..parallel import ParallelContext

__all__ = [
    "CommittedTransactionCorruptionError",
    "SingleParticipantCoordinator",
    "StreamingWorkDirectoryLock",
    "TransactionCoordinationError",
    "fault_point",
    "with_work_directory_lock",
]


class TransactionCoordinationError(RuntimeError):
    """The local transaction cannot be accepted by the coordinator."""


class CommittedTransactionCorruptionError(TransactionCoordinationError):
    """A durable transaction exists but fails integrity validation."""


class StreamingWorkDirectoryLock:
    """An OS-level non-blocking lock for one streaming work directory."""

    filename = ".streaming.lock"

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir).expanduser()
        self.path = self.work_dir / self.filename
        self._file = None

    def acquire(self) -> StreamingWorkDirectoryLock:
        if self._file is not None:
            raise RuntimeError("Streaming work directory lock is already held")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            file.close()
            raise RuntimeError(
                f"Streaming work directory is already locked: {self.work_dir}"
            ) from error
        file.seek(0)
        file.truncate()
        file.write(f"pid={os.getpid()}\n")
        file.flush()
        os.fsync(file.fileno())
        self._file = file
        return self

    def release(self) -> None:
        file = self._file
        self._file = None
        if file is None:
            return
        try:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        finally:
            file.close()
        try:
            self.work_dir.rmdir()
        except OSError:
            pass

    def __enter__(self) -> StreamingWorkDirectoryLock:
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def fault_point(name: str) -> None:
    """Pause at an internal test fault point when explicitly requested.

    The hook is inert unless ``LLMCOMPRESSOR_STREAMING_FAULT_AT`` matches
    ``name``. A parent process can wait for the optional ready marker and then
    terminate the worker, which exercises real process-failure windows without
    adding a public algorithm option.
    """

    if os.environ.get("LLMCOMPRESSOR_STREAMING_FAULT_AT") != name:
        return
    ready = os.environ.get("LLMCOMPRESSOR_STREAMING_FAULT_READY")
    release = os.environ.get("LLMCOMPRESSOR_STREAMING_FAULT_RELEASE")
    if ready:
        ready_path = Path(ready)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.write_text(name, encoding="utf-8")
    if not release:
        return
    release_path = Path(release)
    while not release_path.exists():
        time.sleep(0.01)


def with_work_directory_lock(function):
    """Serialize one process at a time for a keyword-only streaming entrypoint."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        work_dir = kwargs.get("work_dir")
        if work_dir is None:
            raise TypeError("A work_dir is required for the streaming lock")
        with StreamingWorkDirectoryLock(work_dir):
            return function(*args, **kwargs)

    return wrapped


class SingleParticipantCoordinator:
    """Persist local prepared state and one global commit for EP=1.

    The file layout intentionally retains participant and global commit names,
    so the same transaction reader can validate a future multi-rank run.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        run_fingerprint: str,
        context: ParallelContext | None = None,
    ):
        self.root = Path(root)
        self.run_fingerprint = run_fingerprint
        self.context = context or ParallelContext.single_device()
        if self.context.world_size != 1:
            raise ValueError(
                "SingleParticipantCoordinator only supports world_size=1"
            )
        self.prepared_dir = self.root / "prepared"
        self.commits_dir = self.root / "commits"
        self.prepared_dir.mkdir(parents=True, exist_ok=True)
        self.commits_dir.mkdir(parents=True, exist_ok=True)

    @property
    def participant_id(self) -> str:
        return self.context.participant_id

    def _prepared_path(self, transaction_id: str) -> Path:
        return (
            self.prepared_dir
            / transaction_id
            / f"{self.participant_id}.json"
        )

    def _commit_path(self, transaction_id: str) -> Path:
        return self.commits_dir / f"{transaction_id}.json"

    def prepare(
        self,
        transaction_id: str,
        metadata: Mapping[str, Any],
    ) -> Path:
        payload = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "run_fingerprint": self.run_fingerprint,
            "participant": self.participant_id,
            "participants": list(self.context.participants),
            "metadata_sha256": fingerprint_json(metadata),
            "metadata": dict(metadata),
        }
        path = self._prepared_path(transaction_id)
        _atomic_json(path, payload)
        return path

    def commit(self, transaction_id: str) -> Path:
        prepared_paths = sorted(
            self.prepared_dir.joinpath(transaction_id).glob("*.json")
        )
        expected = set(self.context.participants)
        actual = {path.stem for path in prepared_paths}
        if actual != expected:
            raise TransactionCoordinationError(
                f"Transaction {transaction_id!r} has participants "
                f"{sorted(actual)}, expected {sorted(expected)}"
            )
        prepared = [_read_json(path) for path in prepared_paths]
        for value in prepared:
            if value.get("run_fingerprint") != self.run_fingerprint:
                raise TransactionCoordinationError(
                    f"Prepared transaction {transaction_id!r} belongs to another run"
                )
        payload = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "run_fingerprint": self.run_fingerprint,
            "participants": list(self.context.participants),
            "prepared_sha256": {
                value["participant"]: fingerprint_json(value)
                for value in prepared
            },
        }
        path = self._commit_path(transaction_id)
        _atomic_json(path, payload)
        return path

    def ensure_committed(
        self,
        transaction_id: str,
        metadata: Mapping[str, Any],
    ) -> Path:
        path = self._commit_path(transaction_id)
        if not path.is_file():
            self.prepare(transaction_id, metadata)
            fault_point("transaction.after_prepared")
            self.commit(transaction_id)
            fault_point("transaction.after_global_commit")
        self.validate_commit(transaction_id)
        return path

    def validate_commit(self, transaction_id: str) -> None:
        path = self._commit_path(transaction_id)
        try:
            value = _read_json(path)
        except (OSError, json.JSONDecodeError, TypeError) as error:
            raise CommittedTransactionCorruptionError(
                f"Global commit for transaction {transaction_id!r} is unreadable"
            ) from error
        if value.get("run_fingerprint") != self.run_fingerprint:
            raise CommittedTransactionCorruptionError(
                f"Global commit for transaction {transaction_id!r} belongs to "
                "another run"
            )
        if set(value.get("participants", ())) != set(self.context.participants):
            raise CommittedTransactionCorruptionError(
                f"Global commit for transaction {transaction_id!r} has invalid "
                "participants"
            )
        prepared_dir = self.prepared_dir / transaction_id
        for participant in self.context.participants:
            prepared_path = prepared_dir / f"{participant}.json"
            if not prepared_path.is_file():
                raise CommittedTransactionCorruptionError(
                    f"Prepared transaction {transaction_id!r} is missing "
                    f"participant {participant!r}"
                )
            prepared = _read_json(prepared_path)
            expected_digest = value.get("prepared_sha256", {}).get(participant)
            if expected_digest != fingerprint_json(prepared):
                raise CommittedTransactionCorruptionError(
                    f"Prepared transaction {transaction_id!r} has a digest "
                    f"mismatch for participant {participant!r}"
                )

    def is_committed(self, transaction_id: str) -> bool:
        path = self._commit_path(transaction_id)
        if not path.is_file():
            return False
        self.validate_commit(transaction_id)
        return True


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
