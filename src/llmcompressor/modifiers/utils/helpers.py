"""
Helper functions for modifier operations and weight management.

Provides utility functions for updating layer weights, managing
global scales for quantization, and handling fused layer operations in
neural network compression workflows. Supports specialized quantization
strategies like NVFP4.
"""

import torch
from compressed_tensors.quantization import QuantizationStrategy, is_attention_module
from compressed_tensors.utils import align_modules, update_parameter_data
from torch.nn import Module

__all__ = ["update_fused_layer_weight_global_scales"]


def update_fused_layer_weight_global_scales(submodule: torch.nn.Module):
    """
    When running NVFP4 quantization, update the global scale
    such that q,k,v layers are treated as one tensor with the same
    global_scale and gate_proj/up_proj layers are treated as one tensor
    with the same global scale. This is requirement currently being set
    by vLLM and may be removed in the future OR potentially make it
    an optional step.

    :param model: model to quantize
    """

    def _is_mlp_module(module: Module):
        return "mlp" in module.__class__.__name__.lower() and (
            hasattr(module, "gate_proj") and hasattr(module, "up_proj")
        )

    def _valid_tensor_group_quant(layer_list: list[Module]):
        """
        Return True if all the linear layers in the layer_list are
        TENSOR_GROUP quantized.
        """
        for layer in layer_list:
            scheme = getattr(layer, "quantization_scheme", None)
            if scheme is None:
                return False

            weight_quant_args = scheme.weights

            if weight_quant_args is None:
                return False

            if weight_quant_args.strategy != QuantizationStrategy.TENSOR_GROUP:
                return False
        return True

    if is_attention_module(submodule):
        # already fused/treated as one layer
        if hasattr(submodule, "qkv_proj"):
            return

        has_standard_qkv = all(
            hasattr(submodule, layer_name)
            for layer_name in ("q_proj", "k_proj", "v_proj")
        )
        if not has_standard_qkv:
            return

        q_proj = getattr(submodule, "q_proj")
        k_proj = getattr(submodule, "k_proj")
        v_proj = getattr(submodule, "v_proj")

        if not _valid_tensor_group_quant([q_proj, v_proj, k_proj]):
            return

        with align_modules([q_proj, v_proj, k_proj]):
            global_scale = torch.min(
                torch.cat(
                    (
                        q_proj.weight_global_scale.data,
                        k_proj.weight_global_scale.data,
                        v_proj.weight_global_scale.data,
                    )
                )
            ).reshape([1])

        update_parameter_data(k_proj, global_scale, "weight_global_scale")
        update_parameter_data(q_proj, global_scale, "weight_global_scale")
        update_parameter_data(v_proj, global_scale, "weight_global_scale")

        del global_scale

    if _is_mlp_module(submodule):
        gate_proj = getattr(submodule, "gate_proj")
        up_proj = getattr(submodule, "up_proj")

        if not _valid_tensor_group_quant([gate_proj, up_proj]):
            return

        with align_modules([gate_proj, up_proj]):
            global_scale = torch.min(
                torch.cat(
                    (
                        gate_proj.weight_global_scale.data,
                        up_proj.weight_global_scale.data,
                    )
                )
            ).reshape([1])

        update_parameter_data(gate_proj, global_scale, "weight_global_scale")
        update_parameter_data(up_proj, global_scale, "weight_global_scale")

        del global_scale
