"""Thread-safe host-memory reservations for streaming preparation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import RLock


_GIB = 1024**3


class HostMemoryError(MemoryError):
    """Raised before a streaming task exceeds the configured host budget."""


def _system_available_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024

    try:
        import psutil
    except ImportError as error:  # pragma: no cover - accelerate installs psutil
        raise RuntimeError("Cannot determine available host memory") from error
    return int(psutil.virtual_memory().available)


def _read_limit(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed < (1 << 60) else None


def _cgroup_available_bytes() -> int | None:
    candidates = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    remaining = []
    for limit_path, usage_path in candidates:
        limit = _read_limit(limit_path)
        usage = _read_limit(usage_path)
        if limit is not None and usage is not None:
            remaining.append(max(0, limit - usage))
    return min(remaining) if remaining else None


class HostMemoryBudget:
    """Reserve promised host capacity before checkpoint reads begin."""

    def __init__(
        self,
        *,
        safety_reserve_bytes: int | None = None,
        system_available: Callable[[], int] = _system_available_bytes,
        cgroup_available: Callable[[], int | None] = _cgroup_available_bytes,
    ):
        if safety_reserve_bytes is not None and safety_reserve_bytes < 0:
            raise ValueError("safety_reserve_bytes must be non-negative")
        self.safety_reserve_bytes = safety_reserve_bytes
        self._system_available = system_available
        self._cgroup_available = cgroup_available
        self._reservations: set[HostMemoryReservation] = set()
        self._lock = RLock()

    def effective_available_bytes(self) -> int:
        """Return unpromised capacity without double-counting live allocations."""

        with self._lock:
            available = self._system_available()
            cgroup = self._cgroup_available()
            if available < 0 or (cgroup is not None and cgroup < 0):
                raise ValueError("Available memory providers must be non-negative")
            if cgroup is not None:
                available = min(available, cgroup)
            safety_reserve = (
                min(4 * _GIB, available // 10)
                if self.safety_reserve_bytes is None
                else self.safety_reserve_bytes
            )
            promised = sum(item._reserved_bytes for item in self._reservations)
            return max(0, available - safety_reserve - promised)

    def reserve(self, owner: str, required_bytes: int) -> HostMemoryReservation:
        if not owner:
            raise ValueError("A host-memory reservation requires an owner")
        if required_bytes < 0:
            raise ValueError("required_bytes must be non-negative")
        with self._lock:
            available = self.effective_available_bytes()
            if required_bytes > available:
                raise HostMemoryError(
                    f"Insufficient host memory for {owner}: requested "
                    f"{required_bytes / _GIB:.2f} GiB, available "
                    f"{available / _GIB:.2f} GiB after safety reserve"
                )
            reservation = HostMemoryReservation(self, owner, required_bytes)
            self._reservations.add(reservation)
            return reservation


class HostMemoryReservation:
    """Own one component's promised and physically allocated host bytes."""

    def __init__(self, budget: HostMemoryBudget, owner: str, reserved_bytes: int):
        self._budget = budget
        self.owner = owner
        self._reserved_bytes = reserved_bytes
        self._committed_bytes = 0
        self._closed = False

    @property
    def reserved_bytes(self) -> int:
        with self._budget._lock:
            return self._reserved_bytes

    @property
    def committed_bytes(self) -> int:
        with self._budget._lock:
            return self._committed_bytes

    @property
    def closed(self) -> bool:
        with self._budget._lock:
            return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Host-memory reservation {self.owner!r} is closed")

    @staticmethod
    def _validate_bytes(value: int) -> None:
        if value < 0:
            raise ValueError("Reservation byte counts must be non-negative")

    def commit(self, value: int) -> None:
        """Move bytes from promised capacity to live allocation accounting."""

        self._validate_bytes(value)
        with self._budget._lock:
            self._require_open()
            if value > self._reserved_bytes:
                raise ValueError("Cannot commit more bytes than remain reserved")
            self._reserved_bytes -= value
            self._committed_bytes += value

    def uncommit(self, value: int) -> None:
        """Release live workspace while retaining its capacity claim."""

        self._validate_bytes(value)
        with self._budget._lock:
            self._require_open()
            if value > self._committed_bytes:
                raise ValueError("Cannot uncommit more bytes than are committed")
            self._committed_bytes -= value
            self._reserved_bytes += value

    def release_reserved(self, value: int) -> None:
        self._validate_bytes(value)
        with self._budget._lock:
            self._require_open()
            if value > self._reserved_bytes:
                raise ValueError("Cannot release more bytes than remain reserved")
            self._reserved_bytes -= value

    def release_committed(self, value: int) -> None:
        self._validate_bytes(value)
        with self._budget._lock:
            self._require_open()
            if value > self._committed_bytes:
                raise ValueError("Cannot release more bytes than are committed")
            self._committed_bytes -= value

    def close(self) -> None:
        with self._budget._lock:
            if self._closed:
                return
            self._reserved_bytes = 0
            self._committed_bytes = 0
            self._closed = True
            self._budget._reservations.discard(self)

    def __enter__(self) -> HostMemoryReservation:
        with self._budget._lock:
            self._require_open()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()
