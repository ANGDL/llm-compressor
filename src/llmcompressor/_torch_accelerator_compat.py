import types

import torch

__all__ = [
    "accelerator_device",
    "accelerator_device_count",
    "accelerator_get_memory_info",
    "accelerator_is_available",
    "accelerator_max_memory_allocated",
    "accelerator_reset_peak_memory_stats",
    "current_accelerator_type",
    "current_device_index",
    "ensure_torch_accelerator",
]


class _AcceleratorCompat:
    def _current_backend(self) -> str:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if hasattr(torch, "mps") and torch.mps.is_available():
            return "mps"
        return "cpu"

    def _device_index(self, device=None) -> int:
        if device is None:
            return self.current_device_index()
        if isinstance(device, torch.device):
            return device.index or 0
        if isinstance(device, str):
            parsed = torch.device(device)
            return parsed.index or 0
        if isinstance(device, int):
            return device
        return 0

    def is_available(self) -> bool:
        return self._current_backend() != "cpu"

    def current_accelerator(self):
        return types.SimpleNamespace(type=self._current_backend())

    def current_device_index(self) -> int:
        backend = self._current_backend()
        if backend == "cuda":
            return torch.cuda.current_device()
        if backend == "xpu":
            return torch.xpu.current_device()
        return 0

    def device_count(self) -> int:
        backend = self._current_backend()
        if backend == "cuda":
            return torch.cuda.device_count()
        if backend == "xpu":
            return torch.xpu.device_count()
        if backend == "mps":
            return 1
        return 0

    def reset_peak_memory_stats(self, device=None):
        backend = self._current_backend()
        index = self._device_index(device)
        if backend == "cuda":
            torch.cuda.reset_peak_memory_stats(index)
        elif backend == "xpu" and hasattr(torch.xpu, "reset_peak_memory_stats"):
            torch.xpu.reset_peak_memory_stats(index)

    def max_memory_allocated(self, device=None) -> int:
        backend = self._current_backend()
        index = self._device_index(device)
        if backend == "cuda":
            return torch.cuda.max_memory_allocated(index)
        if backend == "xpu" and hasattr(torch.xpu, "max_memory_allocated"):
            return torch.xpu.max_memory_allocated(index)
        if backend == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory()
        return 0

    def get_memory_info(self, device=None):
        backend = self._current_backend()
        index = self._device_index(device)
        if backend == "cuda":
            return torch.cuda.mem_get_info(index)
        if backend == "xpu" and hasattr(torch.xpu, "mem_get_info"):
            return torch.xpu.mem_get_info(index)
        return (0, 0)


def ensure_torch_accelerator():
    if not hasattr(torch, "accelerator"):
        torch.accelerator = _AcceleratorCompat()
    return torch.accelerator


def accelerator_is_available() -> bool:
    return ensure_torch_accelerator().is_available()


def current_accelerator_type() -> str:
    return ensure_torch_accelerator().current_accelerator().type


def current_device_index() -> int:
    return ensure_torch_accelerator().current_device_index()


def accelerator_device_count() -> int:
    return ensure_torch_accelerator().device_count()


def accelerator_reset_peak_memory_stats(device=None):
    return ensure_torch_accelerator().reset_peak_memory_stats(device)


def accelerator_max_memory_allocated(device=None) -> int:
    return ensure_torch_accelerator().max_memory_allocated(device)


def accelerator_get_memory_info(device=None):
    return ensure_torch_accelerator().get_memory_info(device)


def accelerator_device(device_index: int | None = None) -> torch.device:
    index = current_device_index() if device_index is None else device_index
    return torch.device(current_accelerator_type(), index)
