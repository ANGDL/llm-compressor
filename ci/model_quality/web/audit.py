"""Append-only audit trail and idempotency records for write operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from ..state import atomic_write_json, utc_now
from .store import read_json

_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")

# Secrets must never reach the audit trail. The audit records a request
# fingerprint plus a redacted argument summary instead of raw credentials.
_REDACT_KEYS = {
    "token",
    "password",
    "secret",
    "authorization",
    "api_key",
    "access_key",
    "secret_key",
    "credential",
}


class ConflictError(RuntimeError):
    """Raised when a request conflicts with recorded state."""


class ValidationError(ValueError):
    """Raised when a request body is structurally invalid."""


@contextmanager
def exclusive_lock(path: Path, *, timeout_seconds: float = 10.0) -> Iterator[None]:
    """Hold a best-effort exclusive file lock for a short critical section."""

    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ConflictError(f"timed out waiting for lock {path.name}")
            time.sleep(0.02)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def timestamp_token() -> str:
    """Return a filename-safe UTC timestamp token."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def request_fingerprint(value: Any) -> str:
    """Return a stable sha256 over the canonical request payload."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def redact(value: Any) -> Any:
    """Recursively drop secret-looking fields from a request payload."""

    if isinstance(value, dict):
        return {
            key: ("***redacted***" if key.lower() in _REDACT_KEYS else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str) and len(value) > 512:
        return value[:512] + "...<truncated>"
    return value


class AuditLog:
    """Write-once audit records under ``<runs_root>/_audit``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    @property
    def directory(self) -> Path:
        return self.root / "_audit"

    @property
    def events_directory(self) -> Path:
        return self.directory / "events"

    @property
    def journal_path(self) -> Path:
        return self.directory / "audit.jsonl"

    def _next_sequence(self) -> int:
        sequences = []
        for path in self.events_directory.glob("*.json"):
            if path.stem.isdigit():
                sequences.append(int(path.stem))
        return (max(sequences) + 1) if sequences else 1

    def record(
        self,
        *,
        actor: str,
        action: str,
        target: str,
        result: str,
        request: Any = None,
        details: Any = None,
    ) -> dict[str, Any]:
        """Append one audit record and return it."""

        sequence = self._next_sequence()
        payload = {
            "schema_version": 1,
            "sequence": sequence,
            "recorded_at": utc_now(),
            "actor": actor,
            "action": action,
            "target": target,
            "result": result,
            "request_fingerprint": (
                request_fingerprint(request) if request is not None else None
            ),
            "details": redact(details) if details is not None else None,
        }
        atomic_write_json(self.events_directory / f"{sequence:012d}.json", payload)
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(payload) + "\n")
        return payload

    def read(
        self,
        *,
        action: str | None = None,
        actor: str | None = None,
        target: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return audit records newest first."""

        if offset < 0 or limit < 1 or limit > 1000:
            raise ValidationError("invalid audit pagination")
        records = []
        for path in sorted(self.events_directory.glob("*.json"), reverse=True):
            value = read_json(path)
            if not isinstance(value, dict):
                continue
            if action and value.get("action") != action:
                continue
            if actor and value.get("actor") != actor:
                continue
            if target and value.get("target") != target:
                continue
            records.append(value)
        return {
            "items": records[offset : offset + limit],
            "total": len(records),
            "offset": offset,
            "limit": limit,
        }


class IdempotencyStore:
    """Replay-safe record of write requests keyed by ``idempotency_key``."""

    def __init__(self, root: str | Path, *, ttl_seconds: int = 86400) -> None:
        self.root = Path(root).expanduser()
        self.ttl_seconds = int(ttl_seconds)

    @property
    def directory(self) -> Path:
        return self.root / "_audit" / "idempotency"

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or not _IDEMPOTENCY_KEY.fullmatch(key):
            raise ValidationError(
                "idempotency_key must be 8-128 characters of letters, numbers, "
                "dot, underscore, colon, or dash"
            )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.directory / f"{digest}.json"

    @staticmethod
    def _expired(record: dict[str, Any], ttl_seconds: int) -> bool:
        created = record.get("created_at")
        if not isinstance(created, str):
            return False
        try:
            parsed = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - parsed > timedelta(seconds=ttl_seconds)

    def begin(self, key: str, payload: Any) -> dict[str, Any]:
        """Claim ``key`` for ``payload``.

        Returns ``{"replayed": True, "response": ...}`` when the same key and
        payload were already processed, and ``{"replayed": False}`` for a new
        claim. Reusing a key with a different payload is a conflict.
        """

        path = self._path(key)
        fingerprint = request_fingerprint(payload)
        existing = read_json(path)
        if isinstance(existing, dict):
            if existing.get("request_fingerprint") != fingerprint:
                raise ConflictError(
                    "idempotency_key was already used for a different request"
                )
            if (
                not self._expired(existing, self.ttl_seconds)
                and existing.get("status") != "FAILED"
            ):
                if existing.get("status") == "COMPLETED":
                    return {"replayed": True, "response": existing.get("response")}
                raise ConflictError(
                    "an identical request with this idempotency_key is in progress"
                )
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "idempotency_key": key,
                "request_fingerprint": fingerprint,
                "status": "IN_PROGRESS",
                "created_at": utc_now(),
                "response": None,
            },
        )
        return {"replayed": False, "response": None}

    def lookup(self, key: str) -> Any:
        """Return the stored response of a completed claim, or ``None``.

        Schedulers use this to recognise an occurrence whose response is
        already recorded even when the caller cannot reproduce the original
        request payload (for example after a crash between the write and the
        schedule update).
        """

        existing = read_json(self._path(key))
        if isinstance(existing, dict) and existing.get("status") == "COMPLETED":
            return existing.get("response")
        return None

    def complete(self, key: str, response: Any) -> None:
        path = self._path(key)
        existing = read_json(path)
        if not isinstance(existing, dict):
            raise ConflictError("idempotency record is missing")
        atomic_write_json(
            path,
            {
                **existing,
                "status": "COMPLETED",
                "completed_at": utc_now(),
                "response": response,
            },
        )

    def fail(self, key: str, error: str) -> None:
        """Release a claim so the operation may be retried after a failure."""

        path = self._path(key)
        existing = read_json(path)
        if not isinstance(existing, dict):
            return
        atomic_write_json(
            path,
            {
                **existing,
                "status": "FAILED",
                "failed_at": utc_now(),
                "error": redact(error),
                "response": None,
            },
        )
