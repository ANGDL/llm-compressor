from types import SimpleNamespace

import torch

from llmcompressor.modeling.deepseekv4 import kernels


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
