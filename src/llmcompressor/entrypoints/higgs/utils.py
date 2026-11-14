"""
Utility functions for HIGGS mixed-precision quantization.

MSE computation, heuristic alpha calculation, fused layer detection,
and config generation from ILP solutions.
"""

import re
from collections import defaultdict
from typing import Collection, Dict, List

import numpy as np
import torch
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.quantization import (
    QuantizationScheme,
    fake_quantize,
    initialize_module_for_quantization,
)
from loguru import logger

from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    observe,
    update_qparams,
)
from llmcompressor.observers import FusionHandler

__all__ = [
    "compute_layer_mse",
    "compute_fused_layer_mse",
    "compute_heuristic_alphas",
    "generate_config_groups",
]


UNQUANTIZED_SCHEME = "__unquantized__"


# ---------------------------------------------------------------------------
# MSE computation
# ---------------------------------------------------------------------------


def compute_layer_mse(weight, scheme, device=None):
    """Compute MSE between original and fake-quantized weight."""
    return compute_fused_layer_mse({"weight": weight}, scheme, device)["weight"]


def compute_fused_layer_mse(weights, scheme, device=None):
    """Compute MSE for layers that share an NVFP4 global scale."""
    try:
        modules = {}
        for name, weight in weights.items():
            device = device or weight.device
            weight = weight.to(device)
            out_features, in_features = weight.shape
            module = torch.nn.Linear(
                in_features, out_features, bias=False, device="meta"
            )
            module.weight = torch.nn.Parameter(weight, requires_grad=False)
            initialize_module_for_quantization(module, scheme, force_zero_point=False)
            initialize_observer(module, "weight")
            apply_calibration_status(module)
            modules[name] = module

        FusionHandler.fuse(
            [(module.weight_observer, module) for module in modules.values()]
        )
        observe(modules.values(), base_name="weight")
        update_qparams(modules.values(), base_name="weight")

        mse = {}
        for name, module in modules.items():
            scale = module.weight_scale
            zero_point = getattr(module, "weight_zero_point", torch.zeros_like(scale))
            quantized = fake_quantize(
                x=module.weight,
                scale=scale,
                zero_point=zero_point,
                args=scheme.weights,
                global_scale=getattr(module, "weight_global_scale", None),
            )
            mse[name] = torch.mean((module.weight - quantized) ** 2).item()
            freeze_module_quantization(module)

        return mse

    except Exception as e:
        logger.warning(
            f"Failed to compute MSE for scheme {scheme}: {e}. Returning inf."
        )
        return {name: float("inf") for name in weights}


# ---------------------------------------------------------------------------
# Alpha heuristic: alpha = log(size+1) * (1 + depth*0.05) * type_multiplier
# ---------------------------------------------------------------------------

_DEPTH_RE = re.compile(r"(?:layers|layer|h|blocks)\.(\d+)\.")
_TYPE_KEYWORDS = {
    "attention": (["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"], 1.2),
    "embedding": (["embed"], 1.5),
    "mlp": (["mlp", "ffn", "fc", "gate_proj", "up_proj", "down_proj"], 0.9),
}


def compute_heuristic_alphas(
    layer_names: List[str],
    layer_param_counts: Dict[str, int],
) -> Dict[str, float]:
    """Compute importance weights: log(size+1) * (1 + depth*0.05) * type_multiplier."""
    alphas = {}
    for name in layer_names:
        size = layer_param_counts.get(name, 0)
        if size == 0:
            alphas[name] = 1.0
            continue

        depth_match = _DEPTH_RE.search(name)
        depth = int(depth_match.group(1)) if depth_match else 0

        type_mult = 1.0
        lower = name.lower()
        for keywords, mult in _TYPE_KEYWORDS.values():
            if any(kw in lower for kw in keywords):
                type_mult = mult
                break

        alphas[name] = float(np.log(size + 1) * (1 + depth * 0.05) * type_mult)

    if alphas:
        vals = list(alphas.values())
        logger.info(
            f"Heuristic alphas for {len(alphas)} layers: "
            f"range=[{min(vals):.2f}, {max(vals):.2f}], mean={np.mean(vals):.2f}"
        )
    return alphas


# ---------------------------------------------------------------------------
# Config generation from ILP solution
# ---------------------------------------------------------------------------

_EXPERT_RE = re.compile(r"^(.+\.experts)\.(\d+)\.(.+)$")


def _collapse_expert_patterns(
    layer_list: list[str],
    *,
    all_layers: Collection[str] | None = None,
) -> list[str]:
    """Collapse expert layers only when the wildcard remains disjoint."""
    expert_groups: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    all_expert_groups: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    exact_targets = []
    wildcard_targets = []

    if all_layers is None:
        all_layers = layer_list

    for layer in all_layers:
        match = _EXPERT_RE.match(layer)
        if match:
            prefix, index, suffix = (
                match.group(1),
                int(match.group(2)),
                match.group(3),
            )
            all_expert_groups[prefix][suffix].add(index)

    for layer in layer_list:
        m = _EXPERT_RE.match(layer)
        if m:
            prefix, index, suffix = (
                m.group(1),
                int(m.group(2)),
                m.group(3),
            )
            expert_groups[prefix][suffix].add(index)
        else:
            exact_targets.append(f"re:^{re.escape(layer)}$")

    for prefix in sorted(expert_groups):
        escaped = re.escape(prefix)
        wildcard_suffixes = []
        for suffix in sorted(expert_groups[prefix]):
            indices = expert_groups[prefix][suffix]
            complete = indices == all_expert_groups[prefix][suffix]
            if complete:
                wildcard_suffixes.append(suffix)
                continue
            for index in sorted(indices):
                exact_targets.append(
                    f"re:^{escaped}\\.{index}\\.{re.escape(suffix)}$"
                )

        if len(wildcard_suffixes) == 1:
            suffix = wildcard_suffixes[0]
            wildcard_targets.append(
                f"re:^{escaped}\\.\\d+\\.{re.escape(suffix)}$"
            )
        elif wildcard_suffixes:
            alt = "|".join(re.escape(s) for s in wildcard_suffixes)
            wildcard_targets.append(f"re:^{escaped}\\.\\d+\\.({alt})$")

    if exact_targets:
        exact_body = "|".join(
            target.removeprefix("re:^").removesuffix("$")
            for target in exact_targets
        )
        targets = [f"re:^(?:{exact_body})$"]
    else:
        targets = []
    targets.extend(wildcard_targets)

    return targets


def generate_config_groups(
    ilp_solution: Dict[str, str],
    candidate_schemes: Dict[str, QuantizationScheme],
) -> Dict[str, QuantizationScheme]:
    """Convert ILP assignments into config groups with targets and formats."""
    scheme_to_layers: dict[str, list[str]] = defaultdict(list)
    for layer, scheme_name in ilp_solution.items():
        scheme_to_layers[scheme_name].append(layer)

    config_groups = {}
    quantized_groups = [
        item
        for item in sorted(scheme_to_layers.items())
        if item[0] != UNQUANTIZED_SCHEME
    ]
    for idx, (scheme_name, layer_list) in enumerate(quantized_groups):
        base_scheme = candidate_schemes.get(scheme_name)
        if base_scheme is None:
            logger.warning(f"Scheme {scheme_name} not in candidate_schemes, skipping")
            continue

        scheme_dict = base_scheme.model_dump()
        scheme_dict["targets"] = _collapse_expert_patterns(
            sorted(layer_list),
            all_layers=ilp_solution.keys(),
        )

        if base_scheme.format is None:
            dummy = torch.nn.Linear(64, 64, bias=False)
            initialize_module_for_quantization(
                dummy, base_scheme, force_zero_point=False
            )
            scheme_dict["format"] = infer_module_format(type(dummy), base_scheme).value

        group_name = f"group_{idx}"
        config_groups[group_name] = QuantizationScheme(**scheme_dict)
        logger.info(f"Config group '{group_name}': {len(layer_list)} layers")

    unquantized_layers = scheme_to_layers.get(UNQUANTIZED_SCHEME, [])
    if unquantized_layers:
        logger.info(f"Leaving {len(unquantized_layers)} layers unquantized")

    return config_groups
