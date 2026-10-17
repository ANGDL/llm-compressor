"""Fixed-token numerical checks for native incremental model adaptation."""

from __future__ import annotations

import torch
from torch import nn

from .api import StreamingInference


@torch.inference_mode()
def compare_cached_to_recompute(
    inference: StreamingInference | nn.Module,
    input_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    *,
    rtol: float = 0.02,
    atol: float = 0.002,
) -> dict:
    """Compare cached decode with full-prefix recomputation on identical tokens.

    Only the last-position logits are retained. This checks the model's cache
    implementation; checkpoint/kernel quality requires an independent reference.
    """
    if (
        input_ids.ndim != 2
        or continuation_ids.ndim != 2
        or input_ids.shape[0] != continuation_ids.shape[0]
        or input_ids.shape[1] == 0
    ):
        raise ValueError("Expected nonempty [batch, prompt] and [batch, continuation]")
    prefix = input_ids.to(inference.device)
    continuation_ids = continuation_ids.to(inference.device)
    output = inference(prefix, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    if cache is None:
        raise RuntimeError("Model did not return a cache with use_cache=True")
    records = []

    def compare(actual, expected, stage):
        actual, expected = (
            actual.detach().float().cpu(),
            expected.detach().float().cpu(),
        )
        difference = actual - expected
        records.append(
            {
                "stage": stage,
                "processed_tokens": prefix.shape[1],
                "cache_length": int(cache.get_seq_length()),
                "finite": bool(
                    torch.isfinite(actual).all() and torch.isfinite(expected).all()
                ),
                "max_abs": difference.abs().max().item(),
                "rms_abs": difference.square().mean().sqrt().item(),
                "relative_l2": (
                    difference.norm() / expected.norm().clamp_min(1e-12)
                ).item(),
                "reference_max_abs": expected.abs().max().item(),
                "top1_agreement": (actual.argmax(-1) == expected.argmax(-1))
                .float()
                .mean()
                .item(),
                "allclose": torch.allclose(actual, expected, rtol=rtol, atol=atol),
            }
        )

    expected = inference(prefix, use_cache=False, logits_to_keep=1)
    compare(output.logits[:, -1], expected.logits[:, -1], "prefill")
    for offset in range(continuation_ids.shape[1]):
        token = continuation_ids[:, offset : offset + 1]
        output = inference(
            token, past_key_values=cache, use_cache=True, logits_to_keep=1
        )
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("Model did not return a cache with use_cache=True")
        prefix = torch.cat((prefix, token), dim=1)
        expected = inference(prefix, use_cache=False, logits_to_keep=1)
        compare(output.logits[:, -1], expected.logits[:, -1], f"decode_{offset + 1}")
    return {
        "passed": all(
            row["finite"]
            and row["allclose"]
            and row["cache_length"] == row["processed_tokens"]
            for row in records
        ),
        "rtol": rtol,
        "atol": atol,
        "activation_mode": (
            inference.report()["activation_mode"]
            if isinstance(inference, StreamingInference)
            else "native_model"
        ),
        "records": records,
    }
