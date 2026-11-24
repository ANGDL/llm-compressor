import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from llmcompressor.modifiers.gptq import gptq_quantize
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.streaming import numerics
from llmcompressor.streaming.numerics import (
    factorize_hessian,
    native_gptq_factorization,
)


def test_factorization_bridge_restores_original_after_nested_contexts():
    original = gptq_quantize.factorize_hessian
    with native_gptq_factorization():
        assert gptq_quantize.factorize_hessian is factorize_hessian
        with native_gptq_factorization():
            assert gptq_quantize.factorize_hessian is factorize_hessian
        assert gptq_quantize.factorize_hessian is factorize_hessian
    assert gptq_quantize.factorize_hessian is original


@pytest.mark.parametrize("batch,expected_chunks", [(81, (81,)), (96, (81, 15))])
def test_factorization_bounds_hessian_stride_without_copying(
    monkeypatch, batch, expected_chunks
):
    width = 5120
    rows = 2048
    calls = []

    def factorize(weights, hessians, damping, fallback):
        calls.append((weights, hessians, fallback))
        assert damping == 0.01
        return hessians

    monkeypatch.setattr(numerics, "_factorize_hessian", factorize)
    with FakeTensorMode():
        weights = torch.empty((batch, rows, width), device="cuda:0")
        hessians = torch.empty((batch, width, width), device="cuda:0")
        fallback = torch.zeros(batch, dtype=torch.bool, device="cuda:0")
        assert factorize_hessian(weights, hessians, 0.01, fallback) is hessians

        offset = 0
        assert len(calls) == len(expected_chunks)
        for tensors, count in zip(calls, expected_chunks):
            for chunk, original in zip(tensors, (weights, hessians, fallback)):
                assert chunk.shape[0] == count
                assert chunk.untyped_storage() is original.untyped_storage()
                assert chunk.storage_offset() == offset * original.stride(0)
            offset += count


def test_local_hook_disabling_does_not_disable_another_owner():
    modifier = HooksMixin()
    module = torch.nn.Identity()
    barrier = threading.Barrier(2)
    observed = []
    modifier.register_hook(
        module, lambda *_args: observed.append(threading.get_ident()), "forward"
    )

    def disabled():
        with HooksMixin.disable_hooks_local():
            barrier.wait(timeout=5)
            module(torch.ones(1))
            barrier.wait(timeout=5)

    def enabled():
        barrier.wait(timeout=5)
        module(torch.ones(1))
        barrier.wait(timeout=5)
        return threading.get_ident()

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            disabled_future = workers.submit(disabled)
            enabled_future = workers.submit(enabled)
            disabled_future.result()
            identity = enabled_future.result()
        assert observed == [identity]
    finally:
        modifier.remove_hooks()
