import torch
from compressed_tensors.quantization.quant_args import round_to_quantized_type_dtype
from torch import distributed as dist

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.helpers import lerp

__all__ = [
    "MemorylessMinMaxObserver",
    "StrictSymmetricMinMaxObserver",
    "StaticMinMaxObserver",
    "MinMaxObserver",
]


@Observer.register("memoryless_minmax")
class MemorylessMinMaxObserver(Observer):
    """
    Compute quantization parameters by taking the min/max of the observed value.
    """

    _act_sync_dict = {}

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        self.min_vals, self.max_vals = _get_min_max(observed)


@Observer.register("strict_symmetric_minmax")
class StrictSymmetricMinMaxObserver(MemorylessMinMaxObserver):
    """Min/max observer using the symmetric integer range [-127, 127]."""

    @torch.no_grad()
    def get_qparams(self):
        qparams = super().get_qparams()
        if self.args.type != "int" or not self.args.symmetric:
            raise ValueError(
                "strict_symmetric_minmax requires symmetric integer quantization"
            )

        max_val_pos = torch.max(
            torch.abs(self.min_vals), torch.abs(self.max_vals)
        )
        qmax = 2.0 ** (self.args.num_bits - 1) - 1
        scales = max_val_pos / qmax

        if self.args.scale_dtype is not None:
            scales = round_to_quantized_type_dtype(
                scales, dtype=self.args.scale_dtype
            )

        # Keep the same nonzero safeguard as the standard qparam path.
        scales = torch.where(scales == 0, qparams["scale"], scales)
        qparams["scale"] = scales
        return qparams


@Observer.register("static_minmax")
class StaticMinMaxObserver(MemorylessMinMaxObserver):
    """
    Compute quantization parameters by taking the min/max of all observed values.
    """

    _act_sync_dict = {
        "min_vals": dist.ReduceOp.MIN,
        "max_vals": dist.ReduceOp.MAX,
    }

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        min_vals, max_vals = _get_min_max(observed)

        if hasattr(self, "min_vals"):
            self.min_vals = torch.min(min_vals, self.min_vals)
            self.max_vals = torch.max(max_vals, self.max_vals)
        else:
            self.min_vals = min_vals
            self.max_vals = max_vals


@Observer.register("minmax")
class MinMaxObserver(Observer):
    """
    Compute quantization parameters by taking the moving average of min/max values.
    """

    _act_sync_dict = {
        "min_vals": dist.ReduceOp.AVG,
        "max_vals": dist.ReduceOp.AVG,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_constant = self.args.observer_kwargs.get("averaging_constant", 0.01)

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        min_vals, max_vals = _get_min_max(observed)

        if hasattr(self, "min_vals") and self.avg_constant != 1.0:
            min_vals = lerp(self.min_vals, min_vals, self.avg_constant)
            max_vals = lerp(self.max_vals, max_vals, self.avg_constant)

        self.min_vals = min_vals
        self.max_vals = max_vals


def _get_min_max(observed: torch.Tensor) -> MinMaxTuple:
    min_vals = torch.amin(observed, dim=(0, -1))
    max_vals = torch.amax(observed, dim=(0, -1))

    return min_vals, max_vals
