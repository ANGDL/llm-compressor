import os

import torch
from safetensors.torch import save_file

__all__ = ["KVCacheQParamExporter"]


class KVCacheQParamExporter:
    """Utility class to export KV-cache qparams from a quantized model.

    Collected qparams include k/v scales by default, and optionally k/v
    zero-points for asymmetric quantization.
    """

    _SCALE_ATTRS = ("k_scale", "v_scale")
    _ZERO_POINT_ATTRS = ("k_zero_point", "v_zero_point")

    @classmethod
    def collect_from_model(
        cls,
        model: torch.nn.Module,
        include_zero_point: bool = True,
        move_to_cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        attrs: list[str] = list(cls._SCALE_ATTRS)
        if include_zero_point:
            attrs.extend(cls._ZERO_POINT_ATTRS)

        tensors: dict[str, torch.Tensor] = {}
        for module_name, module in model.named_modules():
            for attr_name in attrs:
                if not hasattr(module, attr_name):
                    continue

                tensor = getattr(module, attr_name)
                if not isinstance(tensor, torch.Tensor):
                    continue

                key = f"{module_name}.{attr_name}" if module_name else attr_name
                value = tensor.detach()
                if move_to_cpu:
                    value = value.cpu()
                tensors[key] = value

        return tensors

    @staticmethod
    def save_tensors(tensors: dict[str, torch.Tensor], save_path: str) -> None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_file(tensors, save_path)

    @classmethod
    def export_from_model(
        cls,
        model: torch.nn.Module,
        save_path: str,
        include_zero_point: bool = True,
        move_to_cpu: bool = True,
    ) -> int:
        tensors = cls.collect_from_model(
            model,
            include_zero_point=include_zero_point,
            move_to_cpu=move_to_cpu,
        )
        if len(tensors) == 0:
            raise RuntimeError("No KV-cache qparam tensors were found on the model")

        cls.save_tensors(tensors, save_path)
        return len(tensors)