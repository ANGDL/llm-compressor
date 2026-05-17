"""Distributed integration tests for AutoSmooth all-reduce paths."""

import pytest
import torch
import torch.distributed as dist

import llmcompressor.modifiers.autosmooth.base as autosmooth_base
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.transform.awq.mappings import ResolvedMapping
from tests.testing_utils import torchrun


class _ToyParent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quantized_balance = torch.nn.Linear(4, 4, bias=False)


def _init_gloo_process_group():
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")


@pytest.mark.integration
@torchrun(world_size=2)
def test_autosmooth_distributed_allreduce_pipeline(monkeypatch):
    _init_gloo_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Keep collective tensors on CPU so gloo works on machines without CUDA/NCCL.
    monkeypatch.setattr(
        autosmooth_base,
        "accelerator_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )

    try:
        parent = _ToyParent()
        mapping = ResolvedMapping(
            smooth_name="toy.smooth",
            smooth_layer=torch.nn.LayerNorm(4),
            balance_layers=[parent.quantized_balance],
            balance_names=["toy.quantized_balance"],
            parent=parent,
            parent_name="toy",
        )

        # Rank-local activation means should be reduced into a global mean.
        modifier = AutoSmoothModifier(activation_scale_type="mean")
        local_mean = (
            torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
            if rank == 0
            else torch.tensor([3.0, 4.0, 5.0, 6.0], dtype=torch.float32)
        )
        modifier._smooth_activation_scales[mapping.smooth_name] = (local_mean, 1)

        scales = modifier._resolve_activation_scales(mapping, torch.device("cpu"))
        torch.testing.assert_close(
            scales,
            torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32),
        )

        gathered_mean_scales = [torch.zeros_like(scales) for _ in range(world_size)]
        dist.all_gather(gathered_mean_scales, scales)
        for gathered_scale in gathered_mean_scales[1:]:
            torch.testing.assert_close(gathered_mean_scales[0], gathered_scale)

        # Rank-local activation max values should be reduced element-wise.
        modifier.activation_scale_type = "max"
        local_max = (
            torch.tensor([1.0, 7.0, 3.0, 2.0], dtype=torch.float32)
            if rank == 0
            else torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float32)
        )
        modifier._smooth_activation_scales[mapping.smooth_name] = (local_max, 1)

        max_scales = modifier._resolve_activation_scales(mapping, torch.device("cpu"))
        torch.testing.assert_close(
            max_scales,
            torch.tensor([2.0, 7.0, 6.0, 8.0], dtype=torch.float32),
        )

        # Rank-local activation min/max should be reduced to global span.
        modifier.activation_scale_type = "minmax"
        local_min_vals = (
            torch.tensor([-1.0, -2.0, -3.0, -1.0], dtype=torch.float32)
            if rank == 0
            else torch.tensor([-2.0, -1.0, -4.0, -3.0], dtype=torch.float32)
        )
        local_max_vals = (
            torch.tensor([2.0, 4.0, 3.0, 5.0], dtype=torch.float32)
            if rank == 0
            else torch.tensor([1.0, 6.0, 5.0, 4.0], dtype=torch.float32)
        )
        modifier._smooth_activation_scales[mapping.smooth_name] = (
            (local_min_vals, local_max_vals),
            1,
        )

        minmax_scales = modifier._resolve_activation_scales(mapping, torch.device("cpu"))
        torch.testing.assert_close(
            minmax_scales,
            torch.tensor([4.0, 8.0, 9.0, 8.0], dtype=torch.float32),
        )

        # Rank-local losses should be reduced before normalization.
        fp16_outputs = [torch.tensor([[2.0, -2.0], [1.0, -1.0]], dtype=torch.float32)]
        int_w_outputs = [
            torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float32)
            if rank == 0
            else torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
        ]
        loss = modifier._compute_loss(fp16_outputs, int_w_outputs)
        assert loss == pytest.approx(1.25)

        gathered = [torch.zeros(1) for _ in range(world_size)]
        dist.all_gather(gathered, torch.tensor([loss]))
        for gathered_loss in gathered[1:]:
            torch.testing.assert_close(gathered[0], gathered_loss)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()