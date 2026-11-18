"""Thread-local metadata for calibration calls coalesced by an executor."""

from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar

_BATCH_LENGTHS: ContextVar[tuple[int, ...] | None] = ContextVar(
    "calibration_batch_lengths", default=None
)
_SAMPLE_COUNT: ContextVar[int | None] = ContextVar(
    "calibration_sample_count", default=None
)


def calibration_sample_count(input_ndim: int) -> int | None:
    count = _SAMPLE_COUNT.get()
    if count is not None:
        return count
    lengths = _BATCH_LENGTHS.get()
    if lengths is None:
        return None
    return len(lengths) if input_ndim == 2 else sum(lengths)


@contextmanager
def calibration_batch_context(
    batch_lengths: Sequence[int], *, sample_count: int | None = None
):
    if sample_count is not None and sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    token = _BATCH_LENGTHS.set(tuple(batch_lengths))
    count_token = _SAMPLE_COUNT.set(sample_count)
    try:
        yield
    finally:
        _BATCH_LENGTHS.reset(token)
        _SAMPLE_COUNT.reset(count_token)
