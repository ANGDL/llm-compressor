from types import SimpleNamespace

import pytest
import torch

from llmcompressor.modeling.deepseekv4 import kernels


def test_fp32_bmm_falls_back_to_explicit_fp32_inputs(monkeypatch):
    original_bmm = torch.bmm

    def fake_bmm(left, right, **kwargs):
        if "out_dtype" in kwargs:
            raise NotImplementedError("aten::bmm.dtype")
        return original_bmm(left, right, **kwargs)

    monkeypatch.setattr(kernels.torch, "bmm", fake_bmm)
    left = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    right = torch.randn(2, 4, 5, dtype=torch.bfloat16)

    actual = kernels._bmm_fp32(left, right)
    expected = original_bmm(left.float(), right.float())

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reference_sparse_attention_restores_fp32_sink(monkeypatch):
    observed = {}

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        observed["sink_dtype"] = attn_sink.dtype
        return q

    monkeypatch.setattr(
        kernels,
        "_load_reference_kernel_module",
        lambda: SimpleNamespace(sparse_attn=sparse_attn),
    )
    monkeypatch.setattr(kernels, "_can_use_reference_kernels", lambda _q: True)

    q = torch.ones((1, 1, 1, 4), dtype=torch.bfloat16)
    kv = torch.ones((1, 1, 4), dtype=torch.bfloat16)
    attn_sink = torch.ones((1,), dtype=torch.bfloat16)
    topk_idxs = torch.zeros((1, 1, 1), dtype=torch.int64)

    output = kernels.sparse_attention(q, kv, attn_sink, topk_idxs, 1.0)

    assert output is q
    assert observed["sink_dtype"] == torch.float32


def test_all_masked_reference_queries_have_zero_output_and_finite_gradients(
    monkeypatch,
):
    monkeypatch.setattr(kernels, "_load_reference_kernel_module", lambda: None)
    q = torch.randn(1, 2, 2, 4, requires_grad=True)
    kv = torch.randn(1, 3, 4, requires_grad=True)
    sink = torch.randn(2, requires_grad=True)
    indices = torch.tensor([[[-1, -1], [0, 1]]])
    output = kernels.sparse_attention(q, kv, sink, indices, 0.5)
    torch.testing.assert_close(output[:, 0], torch.zeros_like(output[:, 0]))
    output.sum().backward()
    for tensor in (q, kv, sink):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_sparse_online_rescaling_keeps_earlier_block_weights_in_fp32(monkeypatch):
    monkeypatch.setattr(kernels, "_load_reference_kernel_module", lambda: None)
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.bfloat16)
    kv = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]], dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 65), dtype=torch.int32)
    indices[..., -1] = 1
    actual = kernels.sparse_attention(q, kv, torch.zeros(1), indices, 1.0)
    # Block 1 has 64 equal weights of 1. Block 2 changes the maximum from 0
    # to 2, rescaling its FP32 accumulator by exp(-2), without a second BF16
    # cast. There is one more exp(-2) term in the denominator for the sink.
    correction = torch.exp(torch.tensor(-2.0))
    denominator = 1 + 65 * correction
    expected = torch.stack((2 / denominator, 64 * correction / denominator))
    torch.testing.assert_close(
        actual, expected.view(1, 1, 1, 2).bfloat16(), atol=0, rtol=0
    )
    global_value = 64 * correction.bfloat16().float() / denominator
    assert actual[..., 1].item() != global_value.bfloat16().item()


@pytest.mark.parametrize("topk", [65, 129])
def test_sparse_leading_masked_blocks_keep_outputs_and_gradients_finite(
    monkeypatch, topk
):
    monkeypatch.setattr(kernels, "_load_reference_kernel_module", lambda: None)
    torch.manual_seed(36)
    q = torch.randn(2, 3, 2, 4, requires_grad=True)
    kv = torch.randn(2, 5, 4, requires_grad=True)
    sink = torch.randn(2, requires_grad=True)
    indices = torch.full((2, 3, topk), -1, dtype=torch.int32)
    indices[:, 1, -1] = 4
    indices[:, 2, 64:] = 0
    actual = kernels.sparse_attention(q, kv, sink, indices, 0.5)
    torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]))
    assert torch.isfinite(actual).all()
    actual.sum().backward()
    for tensor in (q, kv, sink):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    torch.testing.assert_close(q.grad[:, 0], torch.zeros_like(q.grad[:, 0]))
