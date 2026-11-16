"""Thread-local metadata for calibration calls coalesced by an executor."""

from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar

_BATCH_LENGTHS: ContextVar[tuple[int, ...] | None] = ContextVar(
    "calibration_batch_lengths", default=None
)


def calibration_sample_count(input_ndim: int) -> int | None:
    lengths = _BATCH_LENGTHS.get()
    if lengths is None:
        return None
    return len(lengths) if input_ndim == 2 else sum(lengths)


@contextmanager
def calibration_batch_context(batch_lengths: Sequence[int]):
    token = _BATCH_LENGTHS.set(tuple(batch_lengths))
    try:
        yield
    finally:
        _BATCH_LENGTHS.reset(token)
