import math
import os
from collections.abc import Iterator, Sequence

import torch
import transformers
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import calculate_range
from compressed_tensors.utils import (
    get_execution_device,
    getattr_chain,
    update_offload_parameter,
)

from llmcompressor._torch_accelerator_compat import accelerator_is_available

GPTQ_PRECISION = torch.float32
MIN_BATCHED_CHOLESKY_SIZE = 16
_AUTO_BATCH_MEMORY_FRACTION = 0.75
_USE_INPLACE_HESSIAN = os.getenv("LLM_COMPRESSOR_USE_INPLACE_HESSIAN", "0").lower() in {
    "1",
    "true",
}


class FusedQuantType:
    """Numeric quantizer identifiers consumed by the fused GPTQ Triton kernel."""

    INT = 0
    FP4_E2M1 = 1
    FP8_E4M3 = 2


def make_empty_hessian(
    module: torch.nn.Module, device: torch.device | None = None
) -> torch.Tensor:
    """Allocate the FP32 square Hessian accumulator for a module's input width."""
    weight = module.weight
    num_columns = weight.shape[1]
    device = device if device is not None else weight.device
    return torch.zeros((num_columns, num_columns), device=device, dtype=GPTQ_PRECISION)


def make_empty_gptq_statistics(
    module: torch.nn.Module,
    device: torch.device | str | None = None,
    count_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the Hessian and sample counter used by streaming collection."""
    resolved_device = device if device is not None else module.weight.device
    resolved_count_device = (
        count_device if count_device is not None else resolved_device
    )
    return (
        make_empty_hessian(module, device=torch.device(resolved_device)),
        torch.zeros((), dtype=torch.int64, device=resolved_count_device),
    )


def accumulate_hessian(
    inp: torch.Tensor,
    module: torch.nn.Module,
    hessian: torch.Tensor,
    num_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate one module-input batch into its GPTQ Hessian statistics.

    The Hessian and sample counter are updated in place and returned for the
    hook caller to retain. Linear, Conv1D, and Conv2d inputs are reshaped into
    the common ``[input_features, observations]`` representation first.
    """
    inp = inp.to(device=hessian.device)
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
    elif len(inp.shape) > 3:
        inp = inp.reshape(inp.shape[0], -1, inp.shape[-1])

    num_added = inp.shape[0]
    match module:
        case torch.nn.Linear() | transformers.Conv1D():
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        case torch.nn.Conv2d():
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            inp = unfold(inp).permute([1, 0, 2]).flatten(1)

    num_samples += num_added
    inp = inp.to(dtype=GPTQ_PRECISION)
    if _USE_INPLACE_HESSIAN:
        inp.mul_(math.sqrt(2))
        hessian.addmm_(inp, inp.t())
    else:
        inp = math.sqrt(2) * inp
        hessian += inp.matmul(inp.t())
    return hessian, num_samples


def prepare_batch(
    batch: list[torch.nn.Module],
    batch_qparams: list[dict[str, torch.Tensor]],
    hessians_by_module: dict[torch.nn.Module, torch.Tensor],
    num_samples_by_module: dict[torch.nn.Module, torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Build disposable stacked GPTQ inputs and consume per-module statistics.

    Removes each module's accumulated Hessian and sample count from the passed
    dictionaries, normalizes its Hessian, and stacks it with weights and
    observer qparams for a single call to ``quantize_weight``.
    """
    first_hessian = hessians_by_module[batch[0]]
    hessians = first_hessian.new_empty((len(batch), *first_hessian.shape))
    del first_hessian
    for index, module in enumerate(batch):
        hessian = hessians_by_module.pop(module)
        num_samples = num_samples_by_module.pop(module).to(device=hessian.device)
        torch.div(hessian, num_samples, out=hessians[index])
        del hessian

    weights = torch.empty(
        (len(batch), *batch[0].weight.shape),
        device=batch[0].weight.device,
        dtype=GPTQ_PRECISION,
    )
    torch.stack([module.weight for module in batch], out=weights)
    scales = torch.stack([qparam["scale"] for qparam in batch_qparams])
    zero_points = torch.stack([qparam["zero_point"] for qparam in batch_qparams])
    global_scales = (
        torch.stack([qparam["global_scale"].reshape(-1)[0] for qparam in batch_qparams])
        if batch_qparams[0]["global_scale"] is not None
        else None
    )
    return weights, hessians, scales, zero_points, global_scales


def update_batch_qparams(
    modules: list[torch.nn.Module],
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    global_scales: torch.Tensor | None,
    quant_args: QuantizationArgs,
) -> None:
    """Write a batch's quantized weights and qparams back to its modules.

    ``update_offload_parameter`` preserves offload-cache semantics while each
    stacked result is cast to the module's storage dtype.
    """
    for index, module in enumerate(modules):
        q_param_dict = {
            "weight": quantized[index].to(dtype=module.weight.dtype),
            "weight_scale": scales[index].to(
                dtype=quant_args.scale_dtype or module.weight.dtype
            ),
            "weight_zero_point": zero_points[index].to(dtype=quant_args.zp_dtype),
        }
        if global_scales is not None:
            q_param_dict["weight_global_scale"] = (
                global_scales[index]
                .reshape_as(module.weight_global_scale)
                .to(dtype=torch.float32)
            )
        for attr, val in q_param_dict.items():
            update_offload_parameter(module, attr, val)


def assign_batches(
    module_list: list[torch.nn.Module],
    batched_quantization: str | int | None,
    block_size: int,
    *,
    batch_size_cap: int | None = None,
) -> list[list[torch.nn.Module]]:
    """Partition modules into compatible GPTQ execution batches."""
    return list(
        iter_batches(
            module_list, batched_quantization, block_size, batch_size_cap=batch_size_cap
        )
    )


def iter_batches(
    module_list: list[torch.nn.Module],
    batched_quantization: str | int | None,
    block_size: int,
    *,
    batch_size_cap: int | None = None,
) -> Iterator[list[torch.nn.Module]]:
    """Yield compatible GPTQ batches, sizing each auto batch from live memory.

    The caller must finish and release one batch before requesting the next.
    ``batch_size_cap`` is an optional upper bound from the caller.
    """
    if batch_size_cap is not None and (
        isinstance(batch_size_cap, bool)
        or not isinstance(batch_size_cap, int)
        or batch_size_cap <= 0
    ):
        raise ValueError("batch_size_cap must be a positive integer or None")

    if batched_quantization is None:
        for module in module_list:
            yield [module]
        return

    if batched_quantization != "auto" and not isinstance(batched_quantization, int):
        raise ValueError(
            f"Unknown batched_quantization value: {batched_quantization!r}"
        )
    pending: dict[tuple, list[torch.nn.Module]] = {}
    for module in module_list:
        key = batch_key(module)
        if key is None:
            yield [module]
        else:
            pending.setdefault(key, []).append(module)

    for members in pending.values():
        start = 0
        while start < len(members):
            upper = min(len(members) - start, batch_size_cap or len(members))
            if isinstance(batched_quantization, int):
                size = min(upper, batched_quantization)
            else:
                device = get_execution_device(members[start])
                available_bytes = available_batch_bytes(device) if upper > 1 else None
                size = (
                    _fit_batch_size(members[start], upper, block_size, available_bytes)
                    if available_bytes is not None
                    else min(upper, batch_size_cap or 1)
                )
            yield members[start : start + size]
            start += size


def batch_key(module: torch.nn.Module) -> tuple | None:
    """Return the shape, device, dtype, and qparam key needed to share a batch.

    Return ``None`` for modules without a compatible two-dimensional weight or
    quantization configuration, forcing them into a singleton batch.
    """
    weight = getattr(module, "weight", None)
    if weight is None or weight.dim() != 2:
        return None
    quant_args = getattr_chain(module, "quantization_scheme.weights", None)
    if quant_args is None:
        return None
    try:
        args_repr = quant_args.model_dump_json()
    except Exception:
        return None
    return (
        tuple(weight.shape),
        str(weight.dtype),
        str(get_execution_device(module)),
        args_repr,
    )


def available_batch_bytes(device: torch.device | str) -> int | None:
    """Return reusable accelerator memory after the GPTQ batch headroom."""

    device = torch.device(device)
    if device.type not in {"cuda", "xpu"} or not accelerator_is_available():
        return None
    device_module = torch.get_device_module(device)
    mem_get_info = getattr(device_module, "mem_get_info", None)
    if not callable(mem_get_info):
        return None
    free_bytes, _ = mem_get_info(device)
    memory_reserved = getattr(device_module, "memory_reserved", None)
    memory_allocated = getattr(device_module, "memory_allocated", None)
    if callable(memory_reserved) and callable(memory_allocated):
        cached_bytes = int(memory_reserved(device)) - int(memory_allocated(device))
        free_bytes += max(0, cached_bytes)
    return int(free_bytes * _AUTO_BATCH_MEMORY_FRACTION)


def quantization_batch_peak_bytes(
    module: torch.nn.Module, batch_size: int, block_size: int
) -> int:
    """Estimate additional peak bytes while quantizing a resident batch.

    Existing module weights and Hessian accumulators are already reflected in
    live memory. The stack is normalized into its destination and reused by
    the solve. Temporary buffers are counted by the phase where they are live.
    The device budget keeps headroom for allocator and backend-specific scratch.
    """
    out_features = module.weight.shape[0]
    in_features = module.weight.numel() // out_features
    width = min(block_size, in_features)
    itemsize = GPTQ_PRECISION.itemsize
    weight_bytes = out_features * in_features * itemsize
    hessian_bytes = in_features**2 * itemsize
    stack_bytes = batch_size * (weight_bytes + hessian_bytes)

    actorder = getattr_chain(module, "quantization_scheme.weights.actorder", None)
    permutation_bytes = batch_size * in_features * torch.int64.itemsize
    ordering_scratch = (
        max(hessian_bytes, weight_bytes) + 2 * permutation_bytes if actorder else 0
    )
    factorization_batch = (
        batch_size if batch_size >= MIN_BATCHED_CHOLESKY_SIZE else 1
    )
    factorization_scratch = (2 * factorization_batch + 1) * hessian_bytes
    solve_scratch = batch_size * itemsize * (
        out_features * (in_features + 8 * width + 2) + width * in_features
    )
    if actorder:
        solve_scratch += 2 * permutation_bytes
        factorization_scratch += 2 * permutation_bytes
    return stack_bytes + max(ordering_scratch, factorization_scratch, solve_scratch)


def _fit_batch_size(
    module: torch.nn.Module, upper: int, block_size: int, available_bytes: int
) -> int:
    low, high = 1, upper
    while low < high:
        middle = (low + high + 1) // 2
        if quantization_batch_peak_bytes(module, middle, block_size) <= available_bytes:
            low = middle
        else:
            high = middle - 1
    return low


def predict_batch_size(
    modules: Sequence[torch.nn.Module],
    batched_quantization: str | int | None,
    block_size: int,
    available_bytes: int | None,
) -> int:
    """Predict one compatible batch upper bound from module and device bytes.

    Modules are grouped by :func:`batch_key`. The returned cap is the minimum
    safe cap across those groups so the streaming owner update can pass one
    integer to every GPTQ batch. Activation and token workspaces are already
    reflected in ``available_bytes`` because this prediction runs after the
    owner window is resident.
    """
    if batched_quantization is None:
        return 1
    if isinstance(batched_quantization, int):
        return batched_quantization
    if batched_quantization != "auto":
        raise ValueError(
            f"Unknown batched_quantization value: {batched_quantization!r}"
        )
    if not modules or available_bytes is None or available_bytes <= 0:
        return 1

    groups: dict[object, list[torch.nn.Module]] = {}
    for module in modules:
        key = batch_key(module)
        if key is not None:
            groups.setdefault(key, []).append(module)
    if not groups:
        return 1

    upper = max(len(members) for members in groups.values())
    return min(
        _fit_batch_size(members[0], upper, block_size, available_bytes)
        for members in groups.values()
    )


def quantization_peak_bytes(module: torch.nn.Module, block_size: int) -> int:
    """Estimate a single-module GPTQ update from its shape and scheme."""
    return quantization_batch_peak_bytes(module, 1, block_size)


def permute_weight_columns(weights: torch.Tensor, permutation: torch.Tensor) -> None:
    """Reorder working weights using a single matrix of scratch."""
    scratch = torch.empty_like(weights[0])
    for weight, indices in zip(weights, permutation):
        torch.index_select(weight, -1, indices, out=scratch)
        weight.copy_(scratch)


def apply_activation_ordering(
    weights: torch.Tensor,
    hessians: torch.Tensor,
    actorder: ActivationOrdering | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Reorder weight columns and both Hessian axes by descending activation.

    The supplied working tensors are overwritten in place. The returned
    permutation is used to restore the quantized weights' original order.
    """
    if not actorder:
        return weights, hessians, None
    if actorder not in (ActivationOrdering.WEIGHT, ActivationOrdering.STATIC):
        raise ValueError(
            f"Invalid activation ordering {actorder}. Only 'weight' and 'static'"
            " are supported for GPTQ."
        )

    perm = torch.argsort(
        torch.diagonal(hessians, dim1=-2, dim2=-1), dim=-1, descending=True
    )
    weight_perm = perm.to(device=weights.device)
    scratch = torch.empty_like(hessians[0])
    for hessian, indices in zip(hessians, perm):
        torch.index_select(hessian, -2, indices, out=scratch)
        torch.index_select(scratch, -1, indices, out=hessian)
    del scratch
    permute_weight_columns(weights, weight_perm)
    return weights, hessians, weight_perm


def factorize_hessian(
    weights: torch.Tensor,
    hessians: torch.Tensor,
    percdamp: float,
    used_rtn_fallback: torch.Tensor,
) -> torch.Tensor:
    """Dampen and factorize Hessians in place into GPTQ update factors.

    Dead columns are zeroed in ``weights``. Non-positive-definite Hessians are
    replaced with identity factors and marked for RTN fallback. Small batches
    use per-item linear algebra because it is faster than batched CUDA calls.
    """
    batch_size, _, num_columns = weights.shape
    diag = torch.diagonal(hessians, dim1=-2, dim2=-1)
    dead = diag == 0
    if dead.any():
        diag.masked_fill_(dead, 1.0)
        weights.masked_fill_(dead.unsqueeze(1), 0)

    # For small batches its faster to do it per-item than to use the batched CUDA ops.
    if batch_size < MIN_BATCHED_CHOLESKY_SIZE:
        info = torch.empty((), dtype=torch.int32, device=hessians.device)
        identity = None
        for index in range(batch_size):
            damp = percdamp * torch.mean(torch.diag(hessians[index]))
            torch.diagonal(hessians[index]).add_(damp)
            torch.linalg.cholesky_ex(
                hessians[index], check_errors=False, out=(hessians[index], info)
            )
            if info.item() == 0:
                torch.cholesky_inverse(hessians[index], out=hessians[index])
                torch.linalg.cholesky(hessians[index], upper=True, out=hessians[index])
            else:
                if identity is None:
                    identity = torch.eye(
                        num_columns, dtype=hessians.dtype, device=hessians.device
                    )
                hessians[index].copy_(identity)
                used_rtn_fallback[index] = True
        return hessians

    damp = percdamp * diag.mean(dim=-1)
    diag.add_(damp.unsqueeze(-1))
    info = torch.empty(batch_size, dtype=torch.int32, device=hessians.device)
    torch.linalg.cholesky_ex(hessians, check_errors=False, out=(hessians, info))
    bad = info.nonzero(as_tuple=False).flatten()
    if bad.numel():
        hessians.index_copy_(
            0,
            bad,
            torch.eye(num_columns, dtype=hessians.dtype, device=hessians.device).expand(
                bad.numel(), -1, -1
            ),
        )
        used_rtn_fallback[bad] = True
    torch.cholesky_inverse(hessians, out=hessians)
    torch.linalg.cholesky(hessians, upper=True, out=hessians)
    return hessians


def column_scale_window(
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    quant_args: QuantizationArgs,
    num_rows: int,
    i1: int,
    i2: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Expand qparams into effective per-row, per-column values for a block.

    Supports tensor, channel, group, tensor-group, and block qparam layouts;
    folds in the optional global scale; and returns tensors consumable by both
    the eager and Triton GPTQ block-update implementations.
    """
    strategy = quant_args.strategy
    block_width = i2 - i1
    has_zp = zero_point is not None and not quant_args.symmetric

    if strategy == QuantizationStrategy.TENSOR:
        eff = scale.reshape(-1, 1, 1)
        zp = zero_point.reshape(-1, 1, 1) if has_zp else None
    elif strategy == QuantizationStrategy.CHANNEL:
        eff = scale[..., :, 0:1]
        zp = zero_point[..., :, 0:1] if has_zp else None
    elif strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        idx = g_idx[..., i1:i2].long()
        eff = torch.gather(
            scale, -1, idx.unsqueeze(-2).expand(*scale.shape[:-1], block_width)
        )
        zp = (
            torch.gather(
                zero_point,
                -1,
                idx.unsqueeze(-2).expand(*zero_point.shape[:-1], block_width),
            )
            if has_zp
            else None
        )
    elif strategy == QuantizationStrategy.BLOCK:
        block_height, _ = quant_args.block_structure
        row_idx = torch.arange(num_rows, device=scale.device) // block_height
        col_idx = g_idx[..., i1:i2].long().unsqueeze(-2)
        eff = torch.gather(scale, -1, col_idx.expand(*scale.shape[:-1], block_width))
        row_idx = row_idx.reshape((1,) * (eff.ndim - 2) + (num_rows, 1))
        eff = torch.gather(
            eff,
            -2,
            row_idx.expand(*eff.shape[:-2], num_rows, block_width),
        )
        if has_zp:
            zp = torch.gather(
                zero_point,
                -1,
                col_idx.expand(*zero_point.shape[:-1], block_width),
            )
            zp = torch.gather(
                zp,
                -2,
                row_idx.expand(*zp.shape[:-2], num_rows, block_width),
            )
        else:
            zp = None
    else:
        raise ValueError(f"Unsupported strategy for column scale window: {strategy}")

    if global_scale is not None:
        global_scale = global_scale.reshape(
            *global_scale.shape, *([1] * (eff.ndim - global_scale.ndim))
        )
        eff = eff / global_scale
    eff = eff.expand(*eff.shape[:-2], num_rows, block_width).contiguous()
    if zp is None:
        return eff, None
    zp = zp.to(GPTQ_PRECISION)
    zp = zp.expand(*zp.shape[:-2], num_rows, block_width).contiguous()
    return eff, zp


def get_triton_gptq_config(
    quant_args: QuantizationArgs,
) -> tuple[int, float, float] | None:
    """Map supported qargs to fused-kernel type and numeric code range.

    Return ``None`` when the registered Triton block-update backend cannot
    represent the requested quantization scheme.
    """
    if quant_args.strategy not in (
        QuantizationStrategy.TENSOR,
        QuantizationStrategy.CHANNEL,
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.BLOCK,
    ):
        return None

    if quant_args.type == QuantizationType.INT:
        quant_type = FusedQuantType.INT
    elif quant_args.type == QuantizationType.FLOAT and quant_args.num_bits == 4:
        quant_type = FusedQuantType.FP4_E2M1
    elif quant_args.type == QuantizationType.FLOAT and quant_args.num_bits == 8:
        quant_type = FusedQuantType.FP8_E4M3
    else:
        return None

    q_min, q_max = calculate_range(quant_args, "cpu")
    return quant_type, float(q_min), float(q_max)
