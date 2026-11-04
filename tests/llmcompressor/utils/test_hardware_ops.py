from inspect import unwrap
from types import SimpleNamespace

import pytest
import torch

from llmcompressor.utils import hardware_ops


@pytest.fixture(autouse=True)
def reset_providers(monkeypatch):
    monkeypatch.setattr(hardware_ops, "_PROVIDERS", None)
    monkeypatch.delenv("LLMCOMPRESSOR_OPS_BACKEND", raising=False)
    monkeypatch.setattr(hardware_ops, "entry_points", lambda **kwargs: ())


def descriptor(**changes):
    fields = dict(
        name="example",
        contract_version=1,
        backend="kunlunxin",
        implementation="cluster",
        execution_kind="precompiled_native",
        priority=0,
        supports=lambda *args, **kwargs: True,
        invoke=lambda *args, **kwargs: "hardware",
    )
    return SimpleNamespace(**(fields | changes))


def install(monkeypatch, *operators):
    provider = SimpleNamespace(protocol_version=1, operators=operators)
    entry = SimpleNamespace(name="test", load=lambda: lambda: provider)
    monkeypatch.setattr(hardware_ops, "entry_points", lambda **kwargs: (entry,))


def entrypoint():
    def eager(value):
        return "torch"

    def internal(value):
        return "internal"

    internal.__wrapped__ = eager
    return hardware_ops.optional_hardware_backend("example")(internal)


def test_without_provider_keeps_existing_dispatch():
    assert entrypoint()(1) == "internal"


@pytest.mark.parametrize("mode", (None, "native", "torch"))
@pytest.mark.parametrize("symmetric", (True, False))
@torch.no_grad()
def test_without_provider_quantize_weight_matches_eager(monkeypatch, mode, symmetric):
    from compressed_tensors.quantization import ActivationOrdering, QuantizationArgs

    from llmcompressor.modifiers.gptq import gptq_quantize

    if mode is not None:
        monkeypatch.setenv("LLMCOMPRESSOR_OPS_BACKEND", mode)
    generator = torch.Generator().manual_seed(20261104)
    weights = torch.randn(2, 3, 10, generator=generator)
    samples = torch.randn(2, 10, 12, generator=generator)
    hessians = torch.bmm(samples, samples.transpose(1, 2)) + torch.eye(10)
    scale = torch.rand(2, 3, 5, generator=generator) * 0.2 + 0.1
    zero_point = (
        torch.zeros_like(scale)
        if symmetric
        else torch.randint(-2, 3, scale.shape, generator=generator).float()
    )
    quant_args = QuantizationArgs(
        num_bits=4,
        symmetric=symmetric,
        strategy="group",
        group_size=2,
        actorder=ActivationOrdering.WEIGHT,
    )
    inputs = (weights, hessians, scale, zero_point, None, quant_args)
    original_inputs = tuple(tensor.clone() for tensor in inputs[:4])

    with monkeypatch.context() as context:
        context.setattr(
            gptq_quantize,
            "gptq_block_update",
            unwrap(gptq_quantize.gptq_block_update),
        )
        expected = gptq_quantize.quantize_weight(*inputs, blocksize=4)

    actual = gptq_quantize.quantize_weight(*inputs, blocksize=4)

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert torch.isfinite(actual[0]).all()
    assert (actual[1] > 0).all()
    assert not actual[2].any()
    for tensor, original in zip(inputs[:4], original_inputs):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def test_native_provider_does_not_change_eager_protection(monkeypatch):
    from compressed_tensors.utils import impl_backend

    monkeypatch.setattr(impl_backend, "ENFORCE_EAGER", "1")
    monkeypatch.setenv("CT_ENFORCE_EAGER", "1")
    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_TRITON", "1")
    install(monkeypatch, descriptor())
    assert entrypoint()(1) == "hardware"
    assert impl_backend.ENFORCE_EAGER == "1"
    assert hardware_ops.os.environ["CT_ENFORCE_EAGER"] == "1"
    assert hardware_ops.os.environ["LLMCOMPRESSOR_DISABLE_TRITON"] == "1"


@pytest.mark.parametrize(
    "operator",
    (
        descriptor(name="other"),
        descriptor(contract_version=2),
        descriptor(supports=lambda *args: False),
    ),
)
def test_missing_or_unsupported_operator_uses_existing_fallback(monkeypatch, operator):
    install(monkeypatch, operator)
    assert entrypoint()(1) == "internal"


@pytest.mark.parametrize("mode,expected", (("torch", "torch"), ("native", "internal")))
def test_explicit_fallback_does_not_load_plugins(monkeypatch, mode, expected):
    def unexpected(**kwargs):
        raise AssertionError("plugin discovery is unexpected")

    monkeypatch.setattr(hardware_ops, "entry_points", unexpected)
    monkeypatch.setenv("LLMCOMPRESSOR_OPS_BACKEND", mode)
    assert entrypoint()(1) == expected


def test_forced_backend_reports_unsupported_inputs(monkeypatch):
    install(monkeypatch, descriptor(supports=lambda *args: False))
    monkeypatch.setenv("LLMCOMPRESSOR_OPS_BACKEND", "kunlunxin")
    with pytest.raises(RuntimeError, match="No supported hardware implementation"):
        entrypoint()(1)


@pytest.mark.parametrize("stage", ("supports", "invoke"))
def test_backend_failure_never_retries_mutated_inputs(monkeypatch, stage):
    values = []

    def broken(value):
        value.append("changed")
        raise RuntimeError("backend failed")

    install(monkeypatch, descriptor(**{stage: broken}))
    with pytest.raises(RuntimeError, match="backend failed"):
        entrypoint()(values)
    assert values == ["changed"]


def test_jit_descriptor_is_rejected(monkeypatch):
    install(monkeypatch, descriptor(execution_kind="triton_jit"))
    with pytest.warns(RuntimeWarning, match="invalid precompiled"):
        assert entrypoint()(1) == "internal"


def test_provider_discovery_is_cached(monkeypatch):
    install(monkeypatch, descriptor())
    assert entrypoint()(1) == "hardware"
    monkeypatch.setattr(hardware_ops, "entry_points", lambda **kwargs: ())
    assert entrypoint()(1) == "hardware"


def test_duplicate_priority_is_an_error(monkeypatch):
    install(monkeypatch, descriptor(), descriptor(implementation="duplicate"))
    with pytest.raises(ValueError, match="Conflicting"):
        entrypoint()(1)
