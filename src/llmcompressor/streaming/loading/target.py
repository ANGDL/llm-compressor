"""Load one target's weights for the duration of a context manager."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

import torch
from torch import nn

from llmcompressor.streaming.checkpoint import CheckpointWeightSource
from llmcompressor.streaming.materialization import (
    CastWeightMaterializer,
    WeightMaterializer,
    materialize_weights,
)

_ModelT = TypeVar("_ModelT", bound=nn.Module)


def build_meta_model(
    factory: Callable[..., _ModelT], *args, **kwargs
) -> _ModelT:
    """Construct a module while allocating parameters and buffers on meta."""

    with torch.device("meta"):
        model = factory(*args, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("Meta model factory must return torch.nn.Module")
    return model


def _join_name(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix and name else prefix or name


def _owner_and_name(module: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    owner_name, _, tensor_name = qualified_name.rpartition(".")
    return module.get_submodule(owner_name) if owner_name else module, tensor_name


class TargetWeightLoader:
    """Materialize exactly one model target and restore it to meta on exit."""

    def __init__(
        self,
        model: nn.Module,
        source: CheckpointWeightSource,
        materializer: WeightMaterializer | None = None,
    ):
        self.model = model
        self.source = source
        self.materializer = materializer or CastWeightMaterializer()
        self._active_target: str | None = None

    @contextmanager
    def loaded(
        self,
        target_name: str,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Iterator[nn.Module]:
        """Yield a real target, then release all of its storage on any exit."""

        if self._active_target is not None:
            raise RuntimeError(
                f"Target {self._active_target!r} is already materialized"
            )
        if not dtype.is_floating_point:
            raise TypeError(f"Target computation dtype must be floating, got {dtype}")
        try:
            target = self.model.get_submodule(target_name)
        except AttributeError as error:
            raise ValueError(f"Unknown target module {target_name!r}") from error

        parameter_groups = self._group_tensors(target, parameters=True)
        buffer_groups = self._group_tensors(target, parameters=False)
        self._validate_meta(parameter_groups, "parameter")
        self._validate_meta(buffer_groups, "buffer")

        parameter_sources = self._resolve_sources(target_name, parameter_groups)
        buffer_sources = self._resolve_sources(target_name, buffer_groups)
        self._validate_shapes(parameter_sources, parameter_groups)
        self._validate_shapes(buffer_sources, buffer_groups)

        device = torch.device(device)
        parameter_values = materialize_weights(
            self.source,
            parameter_sources.values(),
            self.materializer,
            target_dtype=dtype,
            device=device,
        )
        buffer_values = self._load_buffers(
            buffer_sources, buffer_groups, dtype=dtype, device=device
        )

        self._active_target = target_name
        installed_parameters = []
        installed_buffers = []
        try:
            installed_parameters = self._install_parameters(
                target, parameter_groups, parameter_sources, parameter_values
            )
            installed_buffers = self._install_buffers(
                target, buffer_groups, buffer_sources, buffer_values
            )
            yield target
        finally:
            self._restore_meta_parameters(target, installed_parameters)
            self._restore_meta_buffers(target, installed_buffers)
            parameter_values.clear()
            buffer_values.clear()
            self._active_target = None

    @staticmethod
    def _group_tensors(
        target: nn.Module, *, parameters: bool
    ) -> list[tuple[torch.Tensor, list[str]]]:
        named = (
            target.named_parameters(recurse=True, remove_duplicate=False)
            if parameters
            else target.named_buffers(recurse=True, remove_duplicate=False)
        )
        groups: dict[int, tuple[torch.Tensor, list[str]]] = {}
        for name, tensor in named:
            identity = id(tensor)
            if identity not in groups:
                groups[identity] = (tensor, [])
            groups[identity][1].append(name)
        return list(groups.values())

    @staticmethod
    def _validate_meta(
        groups: list[tuple[torch.Tensor, list[str]]], kind: str
    ) -> None:
        non_meta = [
            name
            for tensor, names in groups
            if not tensor.is_meta
            for name in names
        ]
        if non_meta:
            raise RuntimeError(
                f"Target {kind}s must be meta before loading; found real tensors: "
                f"{sorted(non_meta)}"
            )

    def _resolve_sources(
        self,
        target_name: str,
        groups: list[tuple[torch.Tensor, list[str]]],
    ) -> dict[int, str]:
        available = set(self.source.tensor_names())
        resolved = {}
        for tensor, aliases in groups:
            candidates = [_join_name(target_name, alias) for alias in aliases]
            matches = [name for name in candidates if name in available]
            if not matches:
                raise KeyError(
                    "No checkpoint tensor matches model tensor aliases "
                    f"{candidates}. Fused or renamed checkpoint tensors require a "
                    "custom mapping and are not supported by TargetWeightLoader."
                )
            resolved[id(tensor)] = matches[0]
        return resolved

    def _validate_shapes(
        self,
        sources: dict[int, str],
        groups: list[tuple[torch.Tensor, list[str]]],
    ) -> None:
        for tensor, _ in groups:
            source_name = sources[id(tensor)]
            source_shape = self.source.metadata(source_name).shape
            if source_shape != tuple(tensor.shape):
                raise ValueError(
                    f"Checkpoint tensor {source_name!r} has shape {source_shape}; "
                    f"model expects {tuple(tensor.shape)}"
                )

    def _load_buffers(
        self,
        sources: dict[int, str],
        groups: list[tuple[torch.Tensor, list[str]]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        floating = []
        non_floating = []
        for tensor, _ in groups:
            source_name = sources[id(tensor)]
            if tensor.dtype.is_floating_point:
                floating.append(source_name)
            else:
                non_floating.append(source_name)
        values = materialize_weights(
            self.source,
            floating,
            self.materializer,
            target_dtype=dtype,
            device=device,
        )
        values.update(self.source.load_tensors(non_floating, device=device))
        for tensor, _ in groups:
            source_name = sources[id(tensor)]
            value = values[source_name]
            expected_dtype = dtype if tensor.dtype.is_floating_point else tensor.dtype
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"Buffer {source_name!r} has dtype {value.dtype}; "
                    f"expected {expected_dtype}"
                )
        return values

    @staticmethod
    def _install_parameters(
        target: nn.Module,
        groups: list[tuple[torch.Tensor, list[str]]],
        sources: dict[int, str],
        values: dict[str, torch.Tensor],
    ) -> list[tuple[list[str], bool, torch.dtype]]:
        installed = []
        for original, aliases in groups:
            parameter = nn.Parameter(
                values[sources[id(original)]], requires_grad=original.requires_grad
            )
            for alias in aliases:
                owner, name = _owner_and_name(target, alias)
                owner._parameters[name] = parameter
            installed.append((aliases, original.requires_grad, original.dtype))
        return installed

    @staticmethod
    def _install_buffers(
        target: nn.Module,
        groups: list[tuple[torch.Tensor, list[str]]],
        sources: dict[int, str],
        values: dict[str, torch.Tensor],
    ) -> list[tuple[list[str], tuple[int, ...], torch.dtype]]:
        installed = []
        for original, aliases in groups:
            value = values[sources[id(original)]]
            for alias in aliases:
                owner, name = _owner_and_name(target, alias)
                owner._buffers[name] = value
            installed.append((aliases, tuple(original.shape), original.dtype))
        return installed

    @staticmethod
    def _restore_meta_parameters(
        target: nn.Module,
        installed: list[tuple[list[str], bool, torch.dtype]],
    ) -> None:
        for aliases, requires_grad, dtype in installed:
            owner, name = _owner_and_name(target, aliases[0])
            value = owner._parameters[name]
            meta = nn.Parameter(
                torch.empty(value.shape, dtype=dtype, device="meta"),
                requires_grad=requires_grad,
            )
            for alias in aliases:
                alias_owner, alias_name = _owner_and_name(target, alias)
                alias_owner._parameters[alias_name] = meta

    @staticmethod
    def _restore_meta_buffers(
        target: nn.Module,
        installed: list[tuple[list[str], tuple[int, ...], torch.dtype]],
    ) -> None:
        for aliases, shape, dtype in installed:
            meta = torch.empty(shape, dtype=dtype, device="meta")
            for alias in aliases:
                owner, name = _owner_and_name(target, alias)
                owner._buffers[name] = meta
