import math

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.utils import patch_attr
from loguru import logger

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["IMatrixMSEObserver"]

_GROUP_STRATEGIES = (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP)

IMATRIX_PRECISION = torch.float32


@Observer.register("imatrix_mse")
class IMatrixMSEObserver(Observer):
    """
    MSE observer weighted by per-input-channel importance.

    Supports CHANNEL, GROUP, and TENSOR_GROUP for weight-only Linear modules.
    Falls back to uniform MSE for global_scale search.
    Extra observer_kwargs: maxshrink, patience, grid, norm, strict.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = self.args.observer_kwargs
        self.maxshrink = kw.get("maxshrink", 0.95)
        self.patience = kw.get("patience", 5)
        self.grid = kw.get("grid", 20)
        self.norm = kw.get("norm", 3.0)
        self.strict = kw.get("strict", False)
        self.chunk_size = kw.get("chunk_size", 0)

        if self.grid <= 0:
            raise ValueError(f"grid must be > 0, got {self.grid}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        if self.chunk_size < 0:
            raise ValueError(f"chunk_size must be >= 0, got {self.chunk_size}")
        if not (0 <= self.maxshrink <= 1):
            raise ValueError(f"maxshrink must be in [0, 1], got {self.maxshrink}")
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
        """Attach a forward-pre hook to accumulate E[x²] per input channel.

        If ``_imatrix_importance`` already exists on the module (second pass,
        e.g. QuantizationModifier after IMatrixGatherer), skip hook attachment.
        """
        if hasattr(module, "_imatrix_importance"):
            return

        if not hasattr(module, "in_features"):
            return

        in_features = module.in_features
        module._imatrix_sum = torch.zeros(in_features, dtype=IMATRIX_PRECISION)
        module._imatrix_count = 0

        def _hook(mod, args):
            x = args[0] if isinstance(args, tuple) else args
            if isinstance(x, tuple):
                x = x[0]
            if x is None or not isinstance(x, torch.Tensor):
                return

            x_f = x.detach().to(IMATRIX_PRECISION)
            n_tokens = math.prod(x_f.shape[:-1])
            token_sum = x_f.pow(2).sum(dim=list(range(x_f.dim() - 1)))

            if mod._imatrix_sum.device != token_sum.device:
                mod._imatrix_sum = mod._imatrix_sum.to(token_sum.device)

            mod._imatrix_sum.add_(token_sum)
            mod._imatrix_count += n_tokens

        module._imatrix_hook = module.register_forward_pre_hook(_hook)

    def detach(self, module: torch.nn.Module) -> None:
        """Remove hooks and compute / clean up importance data.

        Case 1 – accumulators present (``_imatrix_sum``): compute importance,
        remove the hook and accumulators, **leave** ``_imatrix_importance`` on
        the module so the next quantization pass can use it.

        Case 2 – only ``_imatrix_importance`` present (no accumulators): this
        is the final cleanup pass — delete it so it doesn't end up in the
        checkpoint.
        """
        if hasattr(module, "_imatrix_sum"):
            if module._imatrix_count > 0:
                importance = module._imatrix_sum / module._imatrix_count
                module._imatrix_importance = importance
            if hasattr(module, "_imatrix_hook"):
                module._imatrix_hook.remove()
                del module._imatrix_hook
            del module._imatrix_sum
            del module._imatrix_count
            return

        if hasattr(module, "_imatrix_importance"):
            del module._imatrix_importance

    # ------------------------------------------------------------------

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _grid_search(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            global_scale=self._get_module_param("global_scale"),
            importance_weights=self._prepare_importance(observed),
            chunk_size=self.chunk_size,
        )

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        # TODO: support importance weights here by deferring the reshape
        # to the grid search call. Currently the base class reshapes
        # observed to (1, 1, -1) which loses the channel layout needed
        # for importance broadcasting.
        return _grid_search(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            optimize_global_scale=True,
            chunk_size=self.chunk_size,
        )

    # ------------------------------------------------------------------

    def _prepare_importance(self, observed: torch.Tensor) -> torch.Tensor | None:
        """Validate → reorder (g_idx) → normalize → broadcast."""
        imp = self._get_validated_importance(observed)
        if imp is None:
            return None

        imp = imp.to(device=observed.device, dtype=torch.float32)
        imp = imp / (imp.mean() + torch.finfo(torch.float32).tiny)

        # Expand to weight shape and use flatten_for_calibration
        # to handle all strategies and g_idx
        module = self.module() if self.module is not None else None
        if module is None or not hasattr(module, "weight"):
            return None
        out_features = module.weight.shape[0]
        imp_2d = imp.unsqueeze(0).expand(out_features, -1)
        g_idx = getattr(module, f"{self.base_name}_g_idx", None)
        return flatten_for_calibration(imp_2d, self.base_name, self.args, g_idx)

    def _get_validated_importance(self, observed: torch.Tensor) -> torch.Tensor | None:
        """Return 1D importance tensor or None (with warning/raise)."""
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

        module = self.module() if self.module is not None else None

        if module is not None and not isinstance(module, torch.nn.Linear):
            if self.strict:
                raise TypeError(
                    "imatrix_mse: only supported for Linear,"
                    f" got {type(module).__name__}"
                )
            logger.warning(
                "imatrix_mse: only supported for Linear,"
                f" got {type(module).__name__}."
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

        imp = getattr(module, "_imatrix_importance", None) if module else None
        if imp is None:
            if self.strict:
                raise ValueError("imatrix_mse: no _imatrix_importance on module")
            logger.warning(
                "imatrix_mse: no _imatrix_importance on module."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None
        if imp.dim() != 1:
            if self.strict:
                raise ValueError(
                    f"imatrix_mse: expected 1D, got shape {tuple(imp.shape)}"
                )
            logger.warning(
                f"imatrix_mse: expected 1D, got shape {tuple(imp.shape)}."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None
        if not torch.is_floating_point(imp):
            imp = imp.float()
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
    global_scale: torch.Tensor | None = None,
    optimize_global_scale: bool = False,
    importance_weights: torch.Tensor | None = None,
    chunk_size: int = 0,
) -> MinMaxTuple:
    """Grid search for min/max minimizing (importance-weighted) quant error."""
    try:
        return _grid_search_impl(
            observed=observed,
            args=args,
            maxshrink=maxshrink,
            patience=patience,
            grid=grid,
            norm=norm,
            global_scale=global_scale,
            optimize_global_scale=optimize_global_scale,
            importance_weights=importance_weights,
            chunk_size=chunk_size,
        )
    except torch.OutOfMemoryError:
        if observed.device.type != "cuda":
            raise

        with torch.cuda.device(observed.device):
            torch.cuda.empty_cache()

        retry_chunk_size = _get_oom_retry_chunk_size(
            observed=observed,
            requested_chunk_size=chunk_size,
        )
        if retry_chunk_size is not None:
            logger.warning(
                "imatrix_mse grid search hit CUDA OOM. Retrying on GPU with "
                f"adaptive chunk_size={retry_chunk_size}.",
                log_once=True,
            )
            try:
                return _grid_search_impl(
                    observed=observed,
                    args=args,
                    maxshrink=maxshrink,
                    patience=patience,
                    grid=grid,
                    norm=norm,
                    global_scale=global_scale,
                    optimize_global_scale=optimize_global_scale,
                    importance_weights=importance_weights,
                    chunk_size=retry_chunk_size,
                )
            except torch.OutOfMemoryError:
                with torch.cuda.device(observed.device):
                    torch.cuda.empty_cache()

        logger.warning(
            "imatrix_mse grid search hit CUDA OOM. Falling back to CPU for this "
            "module; this is slower but avoids calibration failure.",
            log_once=True,
        )

        observed_cpu = observed.detach().to("cpu")
        global_scale_cpu = (
            global_scale.detach().to("cpu") if global_scale is not None else None
        )
        importance_cpu = (
            importance_weights.detach().to("cpu")
            if importance_weights is not None
            else None
        )

        best_min, best_max = _grid_search_impl(
            observed=observed_cpu,
            args=args,
            maxshrink=maxshrink,
            patience=patience,
            grid=grid,
            norm=norm,
            global_scale=global_scale_cpu,
            optimize_global_scale=optimize_global_scale,
            importance_weights=importance_cpu,
            chunk_size=chunk_size,
        )
        return best_min.to(observed.device), best_max.to(observed.device)


def _grid_search_impl(
    observed: torch.Tensor,
    args: QuantizationArgs,
    maxshrink: float,
    patience: int,
    grid: int,
    norm: float,
    global_scale: torch.Tensor | None = None,
    optimize_global_scale: bool = False,
    importance_weights: torch.Tensor | None = None,
    chunk_size: int = 0,
) -> MinMaxTuple:
    """Core grid search implementation used by both CUDA and CPU fallback paths."""
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    best_error = torch.full(
        min_val.shape,
        torch.finfo(torch.float32).max,
        device=min_val.device,
        dtype=torch.float32,
    )
    best_min = min_val.clone()
    best_max = max_val.clone()

    no_improve = 0
    observed_f = observed if observed.dtype == torch.float32 else observed.float()
    if importance_weights is not None:
        importance_weights = importance_weights.to(observed_f.dtype)

    qparam_count = min_val.numel()
    num_observations = observed.shape[0]
    group_size = observed.shape[-1]
    effective_chunk_size = _get_effective_chunk_size(
        requested_chunk_size=chunk_size,
        qparam_count=qparam_count,
        num_observations=num_observations,
        group_size=group_size,
    )

    observed_flat = observed.reshape(num_observations, qparam_count, group_size)
    observed_f_flat = observed_f.reshape(num_observations, qparam_count, group_size)
    importance_flat = (
        importance_weights.reshape(num_observations, qparam_count, group_size)
        if importance_weights is not None
        else None
    )

    shrink_steps = max(1, int(maxshrink * grid))
    for i in range(shrink_steps + 1):
        p = 1 - i / grid
        shrink_min = p * min_val
        shrink_max = p * max_val

        if optimize_global_scale:
            global_scale = generate_gparam(shrink_min, shrink_max)

        scales, zps = calculate_qparams(
            min_vals=shrink_min,
            max_vals=shrink_max,
            quantization_args=args,
            global_scale=global_scale,
        )

        with patch_attr(args, "strategy", QuantizationStrategy.TOKEN):
            err = _compute_err(
                observed=observed,
                observed_f=observed_f,
                observed_flat=observed_flat,
                observed_f_flat=observed_f_flat,
                scales=scales,
                zps=zps,
                args=args,
                norm=norm,
                global_scale=global_scale,
                importance_weights=importance_weights,
                importance_flat=importance_flat,
                effective_chunk_size=effective_chunk_size,
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
    global_scale: torch.Tensor | None,
    importance_weights: torch.Tensor | None,
    importance_flat: torch.Tensor | None,
    effective_chunk_size: int,
) -> torch.Tensor:
    if effective_chunk_size >= scales.numel():
        q = fake_quantize(
            observed,
            scales.unsqueeze(-1),
            zps.unsqueeze(-1),
            args,
            global_scale=global_scale,
        ).to(observed_f.dtype)

        q.sub_(observed_f).abs_().pow_(norm)
        if importance_weights is not None:
            q.mul_(importance_weights)
        return q.sum(dim=(0, -1), dtype=torch.float32)

    scales_flat = scales.reshape(-1)
    zps_flat = zps.reshape(-1)
    err_flat = torch.empty(
        scales_flat.numel(), dtype=torch.float32, device=scales_flat.device
    )

    for start in range(0, scales_flat.numel(), effective_chunk_size):
        end = min(start + effective_chunk_size, scales_flat.numel())

        q_chunk = fake_quantize(
            observed_flat[:, start:end, :],
            scales_flat[start:end].unsqueeze(-1),
            zps_flat[start:end].unsqueeze(-1),
            args,
            global_scale=global_scale,
        ).to(observed_f.dtype)

        q_chunk.sub_(observed_f_flat[:, start:end, :]).abs_().pow_(norm)
        if importance_flat is not None:
            q_chunk.mul_(importance_flat[:, start:end, :])

        err_flat[start:end] = q_chunk.sum(dim=(0, -1), dtype=torch.float32)

    return err_flat.reshape_as(scales)


def _get_effective_chunk_size(
    requested_chunk_size: int,
    qparam_count: int,
    num_observations: int,
    group_size: int,
) -> int:
    if requested_chunk_size > 0:
        return min(requested_chunk_size, qparam_count)

    # Target <= 64MB temporary q tensor for error computation.
    target_elements = 16 * 1024 * 1024
    denom = max(1, num_observations * group_size)
    auto_chunk = max(1, target_elements // denom)
    return min(auto_chunk, qparam_count)


def _get_oom_retry_chunk_size(
    observed: torch.Tensor,
    requested_chunk_size: int,
) -> int | None:
    """Estimate a smaller GPU chunk size based on currently free CUDA memory."""
    if observed.device.type != "cuda":
        return None

    qparam_count = observed.shape[1] * observed.shape[2]
    if qparam_count <= 1:
        return None

    num_observations = observed.shape[0]
    group_size = observed.shape[-1]

    # q_chunk is promoted to float32 during error computation.
    bytes_per_elem = torch.finfo(torch.float32).bits // 8
    per_qparam_bytes = max(1, num_observations * group_size * bytes_per_elem)

    free_bytes, _ = torch.cuda.mem_get_info(observed.device)
    target_bytes = max(1, int(free_bytes * 0.35))
    memory_based_chunk = max(1, target_bytes // per_qparam_bytes)

    if requested_chunk_size > 0:
        memory_based_chunk = min(memory_based_chunk, requested_chunk_size)

    return min(memory_based_chunk, qparam_count)
