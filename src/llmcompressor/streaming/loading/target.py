"""Load one target's weights for the duration of a context manager."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

import torch
from accelerate import init_empty_weights
from torch import nn

from llmcompressor.streaming.checkpoint import CheckpointWeightSource
from llmcompressor.streaming.materialization import (
    CastWeightMaterializer,
    WeightMaterializer,
    materialize_weights,
)

_ModelT = TypeVar("_ModelT", bound=nn.Module)


def build_meta_model(
    factory: Callable[..., _ModelT],
    *args,
    keep_nonpersistent_buffers: bool = False,
    **kwargs,
) -> _ModelT:
    """Construct a module while allocating parameters and buffers on meta."""

    if keep_nonpersistent_buffers:
        # Model-derived buffers such as rotary frequencies are not checkpoint
        # weights, but traced prefix execution needs their real values.
        configs = [
            value
            for value in (*args, *kwargs.values())
            if hasattr(value, "to_dict")
        ]
        previous = [
            (config, getattr(config, "_streaming_meta_init", None))
            for config in configs
        ]
        try:
            for config in configs:
                config._streaming_meta_init = True
            with init_empty_weights(include_buffers=False):
                model = factory(*args, **kwargs)
        finally:
            for config, value in previous:
                if value is None:
                    delattr(config, "_streaming_meta_init")
                else:
                    config._streaming_meta_init = value
        tie_weights = getattr(model, "tie_weights", None)
        if callable(tie_weights):
            tie_weights()
    else:
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
        self._active_targets: set[str] = set()

    @contextmanager
    def loaded(
        self,
        target_name: str,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        allow_missing_state: bool = False,
    ) -> Iterator[nn.Module]:
        """Yield a real target, then release all of its storage on any exit."""

        if target_name in self._active_targets:
            raise RuntimeError(
                f"Target {target_name!r} is already materialized"
            )
        if not dtype.is_floating_point:
            raise TypeError(f"Target computation dtype must be floating, got {dtype}")
        try:
            target = self.model.get_submodule(target_name)
        except AttributeError as error:
            raise ValueError(f"Unknown target module {target_name!r}") from error

        parameter_groups = self._group_tensors(target, parameters=True)
        buffer_groups, runtime_buffer_groups = self._group_buffers(target)
        self._validate_meta(parameter_groups, "parameter")
        self._validate_meta(buffer_groups, "buffer")

        parameter_sources = self._resolve_sources(
            target_name, parameter_groups, allow_missing=allow_missing_state
        )
        buffer_sources = self._resolve_sources(
            target_name, buffer_groups, allow_missing=allow_missing_state
        )
        parameter_groups = [
            group for group in parameter_groups if id(group[0]) in parameter_sources
        ]
        buffer_groups = [
            group for group in buffer_groups if id(group[0]) in buffer_sources
        ]
        auxiliary_parameters = (
            self._materialize_missing_parameters(
                target, device=device, dtype=dtype
            )
            if allow_missing_state
            else []
        )
        self._validate_shapes(parameter_sources, parameter_groups)
        self._validate_shapes(buffer_sources, buffer_groups)

        device = torch.device(device)
        parameter_values = self._load_parameters(
            parameter_sources, parameter_groups, dtype=dtype, device=device
        )
        buffer_values = self._load_buffers(
            buffer_sources, buffer_groups, dtype=dtype, device=device
        )

        self._active_targets.add(target_name)
        installed_parameters = []
        installed_buffers = []
        runtime_buffers = []
        try:
            installed_parameters = self._install_parameters(
                target, parameter_groups, parameter_sources, parameter_values
            )
            installed_buffers = self._install_buffers(
                target, buffer_groups, buffer_sources, buffer_values
            )
            runtime_buffers = self._move_runtime_buffers(
                target, runtime_buffer_groups, device
            )
            reinitialize = getattr(
                self.model, "_reinitialize_non_persistent_buffers", None
            )
            if callable(reinitialize) and any(
                tensor.is_meta for tensor, _ in runtime_buffer_groups
            ):
                reinitialize()
            yield target
        finally:
            self._restore_meta_parameters(target, installed_parameters)
            self._restore_meta_buffers(target, installed_buffers)
            self._restore_runtime_buffers(target, runtime_buffers)
            self._restore_auxiliary_parameters(target, auxiliary_parameters)
            parameter_values.clear()
            buffer_values.clear()
            self._active_targets.remove(target_name)

    @staticmethod
    def _materialize_missing_parameters(
        target: nn.Module, *, device: torch.device, dtype: torch.dtype
    ) -> list[tuple[str, nn.Parameter]]:
        installed = []
        for name, parameter in target.named_parameters(
            recurse=True, remove_duplicate=False
        ):
            if not parameter.is_meta:
                continue
            owner, local_name = _owner_and_name(target, name)
            value_dtype = (
                dtype if parameter.dtype.is_floating_point else parameter.dtype
            )
            owner._parameters[local_name] = nn.Parameter(
                torch.zeros(parameter.shape, dtype=value_dtype, device=device),
                requires_grad=False,
            )
            installed.append((name, parameter))
        return installed

    @staticmethod
    def _restore_auxiliary_parameters(
        target: nn.Module, installed: list[tuple[str, nn.Parameter]]
    ) -> None:
        for name, original in installed:
            owner, local_name = _owner_and_name(target, name)
            owner._parameters[local_name] = original

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
    def _group_buffers(
        target: nn.Module,
    ) -> tuple[
        list[tuple[torch.Tensor, list[str]]],
        list[tuple[torch.Tensor, list[str]]],
    ]:
        persistent = {}
        runtime = {}
        for module_name, module in target.named_modules():
            for name, tensor in module._buffers.items():
                if tensor is None:
                    continue
                qualified = _join_name(module_name, name)
                destination = (
                    runtime
                    if name in module._non_persistent_buffers_set
                    else persistent
                )
                identity = id(tensor)
                if identity not in destination:
                    destination[identity] = (tensor, [])
                destination[identity][1].append(qualified)
        return list(persistent.values()), list(runtime.values())

    @staticmethod
    def _move_runtime_buffers(
        target: nn.Module,
        groups: list[tuple[torch.Tensor, list[str]]],
        device: torch.device,
    ) -> list[tuple[list[str], torch.Tensor]]:
        moved = []
        for original, aliases in groups:
            for alias in aliases:
                owner, name = _owner_and_name(target, alias)
                owner._buffers[name] = (
                    torch.zeros(
                        original.shape, dtype=original.dtype, device=device
                    )
                    if original.is_meta
                    else original.to(device=device)
                )
            moved.append((aliases, original))
        return moved

    @staticmethod
    def _restore_runtime_buffers(
        target: nn.Module,
        moved: list[tuple[list[str], torch.Tensor]],
    ) -> None:
        for aliases, original in moved:
            for alias in aliases:
                owner, name = _owner_and_name(target, alias)
                owner._buffers[name] = original

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
        *,
        allow_missing: bool = False,
    ) -> dict[int, str]:
        available = set(self.source.tensor_names())
        resolved = {}
        for tensor, aliases in groups:
            candidates = [_join_name(target_name, alias) for alias in aliases]
            named = (
                self.model.named_parameters(recurse=True, remove_duplicate=False)
                if isinstance(tensor, nn.Parameter)
                else self.model.named_buffers(recurse=True, remove_duplicate=False)
            )
            candidates.extend(
                name for name, candidate in named if candidate is tensor
            )
            candidates = list(dict.fromkeys(candidates))
            matches = [name for name in candidates if name in available]
            if not matches:
                if allow_missing:
                    continue
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
            metadata = self.source.metadata(source_name)
            source_shape = self.materializer.logical_shape(
                source_name, metadata
            )
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

    def _load_parameters(
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
            expected_dtype = dtype if tensor.dtype.is_floating_point else tensor.dtype
            if values[source_name].dtype != expected_dtype:
                raise ValueError(
                    f"Parameter {source_name!r} has dtype "
                    f"{values[source_name].dtype}; expected {expected_dtype}"
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
            installed.append(
                (
                    aliases,
                    original.requires_grad,
                    original.dtype,
                    tuple(original.shape),
                )
            )
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
        installed: list[
            tuple[list[str], bool, torch.dtype, tuple[int, ...]]
        ],
    ) -> None:
        for aliases, requires_grad, dtype, shape in installed:
            meta = nn.Parameter(
                torch.empty(shape, dtype=dtype, device="meta"),
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
