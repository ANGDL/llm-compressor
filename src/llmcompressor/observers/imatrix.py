import math

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.quantization.utils.mxfp_utils import (
    should_generate_mx_scales,
)
from compressed_tensors.utils import patch_attr
from loguru import logger
from torch import distributed as dist
from torch.utils.hooks import RemovableHandle

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.grid_search import (
    _default_triton_error_buffer,
    _grid_search_observer,
    _resolve_grid_type,
)
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = [
    "IMatrixMSEObserver",
    "accumulate_imatrix_statistics",
    "make_empty_imatrix_statistics",
]

_GROUP_STRATEGIES = (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP)

IMATRIX_PRECISION = torch.float32


def make_empty_imatrix_statistics(
    in_features: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(in_features, int) or in_features <= 0:
        raise ValueError(f"in_features must be a positive integer, got {in_features!r}")
    return (
        torch.zeros(in_features, dtype=IMATRIX_PRECISION, device=device),
        torch.zeros((), dtype=torch.int64, device=device),
    )


def accumulate_imatrix_statistics(
    inputs: torch.Tensor,
    imatrix_sum: torch.Tensor,
    imatrix_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.ndim == 0:
        raise ValueError("iMatrix inputs must have at least one dimension")
    if inputs.shape[-1] != imatrix_sum.numel():
        raise ValueError(
            f"iMatrix input width {inputs.shape[-1]} does not match accumulator "
            f"width {imatrix_sum.numel()}"
        )
    values = inputs.detach().to(device=imatrix_sum.device, dtype=IMATRIX_PRECISION)
    imatrix_sum.add_(values.square().sum(dim=tuple(range(values.ndim - 1))))
    imatrix_count.add_(math.prod(values.shape[:-1]))
    return imatrix_sum, imatrix_count


@Observer.register("imatrix_mse")
class IMatrixMSEObserver(Observer):
    """
    Search candidate min/max ranges using per-input-channel activation
    importance (E[x²]) to weight QDQ error.

    Supports CHANNEL, GROUP, and TENSOR_GROUP for weight-only Linear modules.
    Falls back to uniform MSE when importance data is unavailable.

    Importance is accumulated on the observer as raw ``_imatrix_sum`` /
    ``_imatrix_count`` and synced across DDP ranks via ``_act_sync_dict``
    before observation.

    The base objective is ``importance * abs(QDQ(w) - w) ** norm``, summed over
    each quantization group. If ``magnitude_normalization`` is enabled, it also
    multiplies each element by ``sqrt(w**2 + group_absmax**2)``. Importance is
    accumulated from observer inputs as E[x²].

    Observer kwargs are read from ``QuantizationArgs.observer_kwargs``:
        maxshrink (float): Maximum fraction of range to shrink. Defaults to
            ``0.95``.
        patience (int): Number of consecutive candidates outside the current
            best-score tolerance before early stopping. Triton uses
            ``triton_error_buffer`` for this tolerance; eager search counts
            strict non-improvements. ``0`` runs the full search. Defaults to
            ``5``.
        grid (float): Linear grid resolution or exponential candidate budget.
            Defaults to ``20``.
        norm (float): Power applied to absolute QDQ error before importance
            weighting. Defaults to ``3.0``.
        expand (float): Multiplier applied to observed min/max before the
            shrink search; must be at least ``1``. Defaults to ``1.0``.
        strict (bool): Raise an error if importance is unavailable or has an
            unsupported shape. If false, warn and fall back to uniform MSE.
            Defaults to ``False``.
        grid_type (str or None): ``"linear"`` or ``"exponential"``. ``None``
            selects exponential spacing when MX scales are generated (E8M0),
            and linear spacing otherwise. Exponential candidates are powers
            of two starting at 1. Defaults to ``None``.
        magnitude_normalization (bool): Multiply each error by
            ``sqrt(w**2 + group_absmax**2)``. Defaults to ``False``.
        triton_error_buffer (float): Relative tolerance used by the Triton
            implementation for per-group patience; it affects early stopping,
            not the error score. Defaults to ``1.0`` for 4-bit floating point
            quantization and ``0.30`` otherwise. The eager implementation does
            not use this option.
    """

    _act_sync_dict = {
        "_imatrix_sum": dist.ReduceOp.SUM,
        "_imatrix_count": dist.ReduceOp.SUM,
    }

    _stats_attrs = ["min_vals", "max_vals", "_imatrix_sum", "_imatrix_count"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = self.args.observer_kwargs
        self.maxshrink = kw.get("maxshrink", 0.95)
        self.patience = kw.get("patience", 5)
        self.grid = kw.get("grid", 20)
        self.norm = kw.get("norm", 3.0)
        self.strict = kw.get("strict", False)
        self.expand = kw.get("expand", 1.0)
        self.chunk_size = kw.get("chunk_size", 0)
        self.magnitude_normalization = kw.get("magnitude_normalization", False)
        self.grid_type = _resolve_grid_type(kw.get("grid_type"), self.args)
        self.triton_error_buffer = kw.get(
            "triton_error_buffer", _default_triton_error_buffer(self.args)
        )

        self._imatrix_sum: torch.Tensor | None = None
        self._imatrix_count: torch.Tensor = torch.tensor(0, dtype=torch.int64)
        self._imatrix_hook: RemovableHandle | None = None

        if self.grid <= 0:
            raise ValueError(f"grid must be > 0, got {self.grid}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        if self.chunk_size < 0:
            raise ValueError(f"chunk_size must be >= 0, got {self.chunk_size}")
        if not (0 <= self.maxshrink <= 1):
            raise ValueError(f"maxshrink must be in [0, 1], got {self.maxshrink}")
        if not isinstance(self.magnitude_normalization, bool):
            raise ValueError(
                "magnitude_normalization must be a bool, got "
                f"{self.magnitude_normalization!r}"
            )
        if (
            not isinstance(self.norm, (int, float))
            or not math.isfinite(self.norm)
            or self.norm <= 0
        ):
            raise ValueError(f"norm must be a finite positive number, got {self.norm}")

    # ------------------------------------------------------------------
    # Hook lifecycle: collect E[x²] per input channel
    # ------------------------------------------------------------------

    def attach(self, module: torch.nn.Module) -> None:
        """Attach a forward-pre hook to accumulate E[x²] per input channel."""
        if self._imatrix_hook is not None:
            self._imatrix_hook.remove()
            self._imatrix_hook = None

        if not hasattr(module, "in_features"):
            return

        in_features = module.in_features
        param = next(module.parameters(), None)
        device = (
            param.device
            if param is not None and param.device.type != "meta"
            else "cpu"
        )
        if hasattr(module, "_imatrix_sum") and hasattr(module, "_imatrix_count"):
            self._imatrix_sum = module._imatrix_sum
            self._imatrix_count = module._imatrix_count
            del module._imatrix_sum
            del module._imatrix_count
            return
        self._imatrix_sum, self._imatrix_count = make_empty_imatrix_statistics(
            in_features, device=device
        )

        def _hook(mod, args):
            if (
                HooksMixin._HOOKS_DISABLED
                and getattr(self, "_imatrix_hook", None)
                not in HooksMixin._HOOKS_KEEP_ENABLED
            ):
                return
            x = args[0] if isinstance(args, tuple) else args
            if isinstance(x, tuple):
                x = x[0]
            if x is None or not isinstance(x, torch.Tensor):
                return

            x_f = x.detach().to(IMATRIX_PRECISION)
            device = x_f.device
            n_tokens = math.prod(x_f.shape[:-1])
            token_sum = x_f.pow(2).sum(dim=list(range(x_f.dim() - 1)))

            self._imatrix_sum = self._imatrix_sum.to(device)
            self._imatrix_count = self._imatrix_count.to(device)

            self._imatrix_sum.add_(token_sum)
            self._imatrix_count += n_tokens

        self._imatrix_hook = module.register_forward_pre_hook(_hook)

    def detach(self, module: torch.nn.Module) -> None:
        """Remove the activation collection hook."""
        if self._imatrix_hook is not None:
            self._imatrix_hook.remove()
            self._imatrix_hook = None

    # ------------------------------------------------------------------

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        importance_weights = self._prepare_importance(observed)
        self.min_vals, self.max_vals = _grid_search_observer(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            self.triton_error_buffer,
            expand=self.expand,
            importance_weights=importance_weights,
            chunk_size=self.chunk_size,
            use_imatrix_error=True,
            magnitude_normalization=self.magnitude_normalization,
            grid_type=self.grid_type,
        )

    # ------------------------------------------------------------------

    def _prepare_importance(self, observed: torch.Tensor) -> torch.Tensor | None:
        """Validate → normalize → broadcast to match observed shape."""
        imp = self._get_validated_importance(observed)
        if imp is None:
            return None

        imp = imp.to(device=observed.device, dtype=torch.float32)
        imp = imp / (imp.mean() + torch.finfo(torch.float32).tiny)

        out_features = observed.shape[1]
        imp_2d = imp.unsqueeze(0).expand(out_features, -1)
        return flatten_for_calibration(imp_2d, self.base_name, self.args)

    def _get_validated_importance(self, observed: torch.Tensor) -> torch.Tensor | None:
        """Compute importance from sum/count, validate, and return 1D tensor or None."""
        if self.base_name != "weight":
            if self.strict:
                raise NotImplementedError(
                    "imatrix_mse: only supported for weight observers"
                )
            logger.warning(
                "imatrix_mse: only supported for weight observers."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None

        if self.args.strategy == QuantizationStrategy.TENSOR:
            if self.strict:
                raise NotImplementedError("imatrix_mse: TENSOR strategy not supported")
            logger.warning(
                "imatrix_mse: TENSOR strategy not supported."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None

        if self._imatrix_sum is None or self._imatrix_count.item() == 0:
            if self.strict:
                raise ValueError("imatrix_mse: no importance data available")
            logger.warning(
                "imatrix_mse: no importance data available."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None

        imp = self._imatrix_sum / self._imatrix_count.float()

        if not torch.isfinite(imp).all():
            if self.strict:
                raise ValueError("imatrix_mse: contains non-finite values")
            logger.warning(
                "imatrix_mse: contains non-finite values. Falling back to uniform MSE.",
                log_once=True,
            )
            return None
        if (imp < 0).any():
            if self.strict:
                raise ValueError("imatrix_mse: contains negative values")
            logger.warning(
                "imatrix_mse: contains negative values. Falling back to uniform MSE.",
                log_once=True,
            )
            return None
        if torch.all(imp == 0):
            if self.strict:
                raise ValueError("imatrix_mse: all zeros")
            logger.warning(
                "imatrix_mse: all zeros. Falling back to uniform MSE.", log_once=True
            )
            return None

        if self.args.strategy == QuantizationStrategy.CHANNEL:
            expected = observed.shape[-1]
        elif self.args.strategy in _GROUP_STRATEGIES:
            expected = observed.shape[2] * observed.shape[3]
        else:
            expected = None
        if expected is None:
            if self.strict:
                raise NotImplementedError(
                    f"imatrix_mse: unsupported strategy {self.args.strategy}"
                )
            logger.warning(
                f"imatrix_mse: unsupported strategy {self.args.strategy}."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None
        if imp.numel() != expected:
            if self.strict:
                raise ValueError(
                    "imatrix_mse: size mismatch:"
                    f" expected {expected}, got {imp.numel()}"
                )
            logger.warning(
                "imatrix_mse: size mismatch:"
                f" expected {expected}, got {imp.numel()}."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None
        return imp


# ---------------------------------------------------------------------------
# TODO: refactor to replace memoryless_mse's grid search, this function
# subsumes it when importance_weights=None.
# ---------------------------------------------------------------------------


def _grid_search(
    observed: torch.Tensor,
    args: QuantizationArgs,
    maxshrink: float,
    patience: int,
    grid: int,
    norm: float,
    expand: float = 1.0,
    importance_weights: torch.Tensor | None = None,
    chunk_size: int = 0,
) -> MinMaxTuple:
    """Grid search for min/max minimizing (importance-weighted) quant error.

    Note: global_scale is NOT used during optimization since it cancels out when
    using FP32 scales. After optimization, global_scale is computed from the final
    min/max values in get_qparams().
    """
    if (
        args.strategy == QuantizationStrategy.TENSOR_GROUP
        and args.scale_dtype is not None
    ):
        args = args.model_copy(update={"scale_dtype": None})

    min_val = torch.amin(observed, dim=(0, -1)) * expand
    max_val = torch.amax(observed, dim=(0, -1)) * expand
    if args.scale_dtype == torch.float32:
        min_val = min_val.float()
        max_val = max_val.float()
    best_error = torch.full(
        min_val.shape,
        torch.finfo(torch.float32).max,
        device=min_val.device,
        dtype=torch.float32,
    )
    best_min = min_val.clone()
    best_max = max_val.clone()

    no_improve = 0
    observed_f = observed.float()

    shrink_steps = max(1, int(maxshrink * grid))
    for i in range(shrink_steps + 1):
        p = 1 - i / grid
        shrink_min = p * min_val
        shrink_max = p * max_val

        scales, zps = calculate_qparams(
            min_vals=shrink_min,
            max_vals=shrink_max,
            quantization_args=args,
            global_scale=None,
        )

        with patch_attr(args, "strategy", QuantizationStrategy.TOKEN):
            try:
                err = _compute_err(
                    observed=observed,
                    observed_f=observed_f,
                    observed_flat=observed.reshape(
                        observed.shape[0], scales.numel(), observed.shape[-1]
                    ),
                    observed_f_flat=observed_f.reshape(
                        observed.shape[0], scales.numel(), observed.shape[-1]
                    ),
                    scales=scales,
                    zps=zps,
                    args=args,
                    norm=norm,
                    importance_weights=importance_weights,
                    importance_flat=(
                        importance_weights.reshape(
                            observed.shape[0], scales.numel(), observed.shape[-1]
                        )
                        if importance_weights is not None
                        else None
                    ),
                    effective_chunk_size=chunk_size,
                )
            except RuntimeError as error:
                if observed.device.type == "cpu" or not _is_oom_error(error):
                    raise
                logger.warning(
                    "imatrix_mse: out of memory during grid search; retrying on CPU.",
                    log_once=True,
                )
                return _grid_search(
                    observed.cpu(),
                    args,
                    maxshrink,
                    patience,
                    grid,
                    norm,
                    expand=expand,
                    importance_weights=(
                        importance_weights.cpu()
                        if importance_weights is not None
                        else None
                    ),
                    chunk_size=chunk_size,
                )

        improved = err < best_error
        if torch.any(improved):
            best_error[improved] = err[improved]
            best_min[improved] = shrink_min[improved]
            best_max[improved] = shrink_max[improved]
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                break

    return best_min, best_max


def _compute_err(
    observed: torch.Tensor,
    observed_f: torch.Tensor,
    observed_flat: torch.Tensor,
    observed_f_flat: torch.Tensor,
    scales: torch.Tensor,
    zps: torch.Tensor,
    args: QuantizationArgs,
    norm: float,
    importance_weights: torch.Tensor | None,
    importance_flat: torch.Tensor | None,
    effective_chunk_size: int,
) -> torch.Tensor:
    if effective_chunk_size <= 0 or scales.numel() <= effective_chunk_size:
        q = fake_quantize(
            observed, scales.unsqueeze(-1), zps.unsqueeze(-1), args
        ).float()
        q.sub_(observed_f).abs_().pow_(norm)
        if importance_weights is not None:
            q.mul_(importance_weights)
        return q.sum(dim=(0, -1), dtype=torch.float32)

    observations = observed_flat
    observations_f = observed_f_flat
    importance = importance_flat
    err = torch.empty_like(scales, dtype=torch.float32)
    flat_scales = scales.reshape(-1)
    flat_zps = zps.reshape(-1)
    for start in range(0, flat_scales.numel(), effective_chunk_size):
        end = min(start + effective_chunk_size, flat_scales.numel())
        q_chunk = fake_quantize(
            observations[:, start:end],
            flat_scales[start:end].unsqueeze(-1),
            flat_zps[start:end].unsqueeze(-1),
            args,
        ).float()
        q_chunk.sub_(observations_f[:, start:end]).abs_().pow_(norm)
        if importance is not None:
            q_chunk.mul_(importance[:, start:end])
        err.reshape(-1)[start:end] = q_chunk.sum(dim=(0, -1), dtype=torch.float32)
    return err


def _is_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "oom" in message


@Observer.register("nvfp4_expanded_imatrix")
class NVFP4ExpandedIMatrixObserver(IMatrixMSEObserver):
    """
    IMatrix observer with defaults tuned for NVFP4 range expansion.

    Same search as :class:`IMatrixMSEObserver` but covers 1.8x down to
    ~0.8x of the per-group range in 112 steps, matching
    :class:`NVFP4ExpandedMSEObserver`.

    Usage::

        QuantizationArgs(
            ...
            observer="nvfp4_expanded_imatrix",
        )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = self.args.observer_kwargs
        if should_generate_mx_scales(self.args):
            self.expand = kw.get("expand", 1.0)
            self.maxshrink = kw.get("maxshrink", 1.0 - 1.0 / 64.0)
        else:
            self.expand = kw.get("expand", 1.8)
            self.maxshrink = kw.get("maxshrink", 1 - 0.8 / 1.8)
        self.grid = kw.get("grid", 200)
        self.norm = kw.get("norm", 2.4)
        self.patience = kw.get("patience", 1000)
