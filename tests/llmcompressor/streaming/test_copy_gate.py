from threading import Thread
from time import sleep

import pytest
import torch

from llmcompressor.streaming.loading.copy_gate import CopyGate


def test_copy_gate_blocks_until_compute_done():
    gate = CopyGate()
    observed = []

    def wait():
        observed.append(gate.wait_for_compute(timeout=2))

    thread = Thread(target=wait)
    thread.start()
    sleep(0.05)
    assert thread.is_alive()

    gate.record_compute_done()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert observed == [None]


def test_copy_gate_open_without_event_is_idempotent():
    gate = CopyGate()
    gate.open_without_event()
    gate.record_compute_done()
    assert gate.is_open
    assert gate.wait_for_compute() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_copy_gate_records_cuda_event():
    gate = CopyGate(torch.cuda.current_stream())
    gate.record_compute_done()
    event = gate.wait_for_compute()
    assert event is not None
    event.synchronize()
