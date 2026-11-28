"""Order background weight H2D after the current target's first forward."""

from __future__ import annotations

from threading import Event, Lock
from typing import Any

import torch


class CopyGate:
    """Signal when the consumer compute stream has reached a safe copy point.

    The producer thread materializes weights on CPU independently. Before
    enqueueing H2D it waits for ``compute_done``, recorded on the consumer
    stream after the first calibration forward. This prevents the weight copy
    stream from overlapping the first-forward/boundary-transfer window.
    """

    def __init__(self, consumer_stream: Any | None = None):
        self.consumer_stream = consumer_stream
        self._ready = Event()
        self._compute_event: Any | None = None
        self._lock = Lock()

    @property
    def is_open(self) -> bool:
        return self._ready.is_set()

    def record_compute_done(self, stream: Any | None = None) -> None:
        """Record ``compute_done`` and release waiting producers."""

        with self._lock:
            if self._ready.is_set():
                return
            stream = stream if stream is not None else self.consumer_stream
            if stream is not None:
                event = torch.cuda.Event()
                event.record(stream)
                self._compute_event = event
            self._ready.set()

    def open_without_event(self) -> None:
        """Release producers when no compute event will be recorded."""

        with self._lock:
            self._ready.set()

    def wait_for_compute(self, timeout: float | None = None) -> Any | None:
        """Block until the compute gate is open and return its event."""

        if not self._ready.wait(timeout):
            raise TimeoutError("Timed out waiting for the copy compute gate")
        return self._compute_event
