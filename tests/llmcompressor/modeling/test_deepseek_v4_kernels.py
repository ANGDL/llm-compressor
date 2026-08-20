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
