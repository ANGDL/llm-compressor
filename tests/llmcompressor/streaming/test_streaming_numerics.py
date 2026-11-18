import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from llmcompressor.modifiers.gptq import gptq_quantize
from llmcompressor.modifiers.utils.hooks import HooksMixin
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
