import math
from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.utils import patch_attr
from loguru import logger
from torch import distributed as dist

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["IMatrixMSEObserver"]

_GROUP_STRATEGIES = (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP)

IMATRIX_PRECISION = torch.float32


@Observer.register("imatrix_mse")
class IMatrixMSEObserver(Observer):
    """
    MSE observer weighted by per-input-channel importance (E[x²]).

    Supports CHANNEL, GROUP, and TENSOR_GROUP for weight-only Linear modules.
    Falls back to uniform MSE when importance data is unavailable.

    Importance is accumulated as raw ``_imatrix_sum`` / ``_imatrix_count``
    and synced across DDP ranks via ``_act_sync_dict`` before observation.
    """

    _act_sync_dict = {
        "_imatrix_sum": dist.ReduceOp.SUM,
        "_imatrix_count": dist.ReduceOp.SUM,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = self.args.observer_kwargs
        self.maxshrink = kw.get("maxshrink", 0.95)
        self.patience = kw.get("patience", 5)
        self.grid = kw.get("grid", 20)
        self.norm = kw.get("norm", 3.0)
        self.strict = kw.get("strict", False)
        self.chunk_size = kw.get("chunk_size", 0)

        self._imatrix_sum: Optional[torch.Tensor] = None
        self._imatrix_count: torch.Tensor = torch.tensor(0, dtype=torch.int64)

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

        If raw accumulators (``_imatrix_sum`` / ``_imatrix_count``) already
        exist on the module (second pass after IMatrixGatherer), copy them
        to the observer and skip hook registration.
        """
        if hasattr(module, "_imatrix_sum"):
            self._imatrix_sum = module._imatrix_sum
            self._imatrix_count = module._imatrix_count
            del module._imatrix_sum
            del module._imatrix_count
            return

        if not hasattr(module, "in_features"):
            return

        in_features = module.in_features
        module._imatrix_sum = torch.zeros(in_features, dtype=IMATRIX_PRECISION)
        module._imatrix_count = torch.tensor(0, dtype=torch.int64)

        def _hook(mod, args):
            if isinstance(args, tuple):
                if len(args) == 0:
                    # Some modules can be invoked with kwargs-only inputs.
                    # In this case we cannot read the activation tensor here.
                    return
                x = args[0]
            else:
                x = args
            if isinstance(x, tuple):
                x = x[0]
            if x is None or not isinstance(x, torch.Tensor):
                return

            x_f = x.detach().to(IMATRIX_PRECISION)
            device = x_f.device
            n_tokens = math.prod(x_f.shape[:-1])
            token_sum = x_f.pow(2).sum(dim=list(range(x_f.dim() - 1)))

            mod._imatrix_sum = mod._imatrix_sum.to(device)
            mod._imatrix_count = mod._imatrix_count.to(device)

            mod._imatrix_sum.add_(token_sum)
            mod._imatrix_count += n_tokens

        module._imatrix_hook = module.register_forward_pre_hook(_hook)

    def detach(self, module: torch.nn.Module) -> None:
        """Remove hooks and leave raw sum/count on module for second-pass pickup.

        Case 1 – accumulators present on module: leave them for next
        observer's ``attach()`` to pick up.

        Case 2 – no accumulators (second-pass cleanup): nothing to do.
        """
        if hasattr(module, "_imatrix_hook"):
            module._imatrix_hook.remove()
            del module._imatrix_hook

    # ------------------------------------------------------------------

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        importance_weights = self._prepare_importance(observed)
        self.min_vals, self.max_vals = _grid_search(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            importance_weights=importance_weights,
            chunk_size=self.chunk_size,
        )

    # ------------------------------------------------------------------

    def _prepare_importance(self, observed: torch.Tensor) -> Optional[torch.Tensor]:
        """Validate → normalize → broadcast to match observed shape."""
        imp = self._get_validated_importance(observed)
        if imp is None:
            return None

        imp = imp.to(device=observed.device, dtype=torch.float32)
        imp = imp / (imp.mean() + torch.finfo(torch.float32).tiny)

        out_features = observed.shape[1]
        imp_2d = imp.unsqueeze(0).expand(out_features, -1)
        return flatten_for_calibration(imp_2d, self.base_name, self.args)

    def _get_validated_importance(
        self, observed: torch.Tensor
    ) -> Optional[torch.Tensor]:
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
    importance_weights: Optional[torch.Tensor] = None,
    chunk_size: int = 0,
    _allow_oom_fallback: bool = True,
) -> MinMaxTuple:
    """Grid search for min/max minimizing (importance-weighted) quant error.

    Note: global_scale is NOT used during optimization since it cancels out when
    using FP32 scales. After optimization, global_scale is computed from the final
    min/max values in get_qparams().
    """
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
    observed_f = observed.float()
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
    fallback_to_cpu = False

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
                    observed_flat=observed_flat,
                    observed_f_flat=observed_f_flat,
                    scales=scales,
                    zps=zps,
                    args=args,
                    norm=norm,
                    importance_weights=importance_weights,
                    importance_flat=importance_flat,
                    effective_chunk_size=effective_chunk_size,
                )
            except RuntimeError as error:
                if (
                    not _allow_oom_fallback
                    or observed.device.type == "cpu"
                    or not _is_oom_error(error)
                ):
                    raise

                if observed.device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass

                retry_chunk_size = _get_oom_retry_chunk_size(
                    observed=observed,
                    requested_chunk_size=effective_chunk_size,
                )
                if (
                    retry_chunk_size is None
                    or retry_chunk_size >= effective_chunk_size
                ):
                    fallback_to_cpu = True
                    break

                logger.warning(
                    "imatrix_mse: out of memory during grid search on "
                    f"{observed.device.type}. Retrying with chunk_size={retry_chunk_size}.",
                    log_once=True,
                )
                effective_chunk_size = retry_chunk_size

                try:
                    err = _compute_err(
                        observed=observed,
                        observed_f=observed_f,
                        observed_flat=observed_flat,
                        observed_f_flat=observed_f_flat,
                        scales=scales,
                        zps=zps,
                        args=args,
                        norm=norm,
                        importance_weights=importance_weights,
                        importance_flat=importance_flat,
                        effective_chunk_size=effective_chunk_size,
                    )
                except RuntimeError as retry_error:
                    if not _is_oom_error(retry_error):
                        raise
                    if observed.device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except RuntimeError:
                            pass
                    fallback_to_cpu = True
                    break

        if fallback_to_cpu:
            break

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

    if fallback_to_cpu:
        logger.warning(
            "imatrix_mse: out of memory during grid search on "
            f"{observed.device.type}. Retrying on CPU.",
            log_once=True,
        )
        best_min, best_max = _grid_search(
            observed.cpu(),
            args,
            maxshrink,
            patience,
            grid,
            norm,
            importance_weights.cpu() if importance_weights is not None else None,
            chunk_size,
            _allow_oom_fallback=False,
        )
        return best_min.to(observed.device), best_max.to(observed.device)

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
    importance_weights: Optional[torch.Tensor],
    importance_flat: Optional[torch.Tensor],
    effective_chunk_size: int,
) -> torch.Tensor:
    if effective_chunk_size >= scales.numel():
        q = fake_quantize(observed, scales.unsqueeze(-1), zps.unsqueeze(-1), args)
        q = q.float()

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
        )
        q_chunk = q_chunk.float()

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
) -> Optional[int]:
    """Estimate a smaller chunk size for OOM retry on accelerator devices."""
    qparam_count = max(1, math.prod(observed.shape[1:-1]))
    if qparam_count <= 1:
        return None

    num_observations = observed.shape[0]
    group_size = observed.shape[-1]
    current_chunk_size = _get_effective_chunk_size(
        requested_chunk_size=requested_chunk_size,
        qparam_count=qparam_count,
        num_observations=num_observations,
        group_size=group_size,
    )
    if current_chunk_size <= 1:
        return None

    if observed.device.type != "cuda":
        # No portable free-memory API for all accelerators (e.g. MPS),
        # so shrink the current chunk conservatively.
        return max(1, current_chunk_size // 2)

    bytes_per_elem = torch.finfo(torch.float32).bits // 8
    per_qparam_bytes = max(1, num_observations * group_size * bytes_per_elem)

    try:
        free_bytes, _ = torch.cuda.mem_get_info(observed.device)
    except RuntimeError:
        return max(1, current_chunk_size // 2)

    target_bytes = max(1, int(free_bytes * 0.35))
    memory_based_chunk = max(1, target_bytes // per_qparam_bytes)
    memory_based_chunk = min(memory_based_chunk, current_chunk_size - 1)

    if requested_chunk_size > 0:
        memory_based_chunk = min(memory_based_chunk, requested_chunk_size)

    return min(memory_based_chunk, qparam_count)


def _is_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "oom" in message
