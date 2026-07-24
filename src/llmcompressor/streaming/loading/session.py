"""Checkpoint-backed weight sessions for traced subgraphs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.streaming.checkpoint import CheckpointWeightSource
from llmcompressor.streaming.materialization import WeightMaterializer

from .target import TargetWeightLoader

__all__ = ["LoadedSubgraph", "SubgraphWeightSession"]


def _contains(parent: str, child: str) -> bool:
    return not parent or child == parent or child.startswith(f"{parent}.")


def _minimal_roots(names: Iterable[str]) -> tuple[str, ...]:
    """Remove duplicate and descendant module names, preserving graph order."""

    roots: list[str] = []
    for name in dict.fromkeys(names):
        if any(_contains(root, name) for root in roots):
            continue
        roots = [root for root in roots if not _contains(name, root)]
        roots.append(name)
    return tuple(roots)


@dataclass
class LoadedSubgraph:
    """The real modules owned by one active subgraph weight session."""

    model: nn.Module
    module_names: tuple[str, ...]

    @property
    def modules(self) -> tuple[nn.Module, ...]:
        return tuple(self.model.get_submodule(name) for name in self.module_names)

    def state_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield persistent, non-meta state without copying it to CPU.

        The checkpoint writer is responsible for consuming one yielded tensor at a
        time. Parameters or qparams added by a modifier while the session is active
        are included automatically.
        """

        seen: set[str] = set()
        for module_name in self.module_names:
            module = self.model.get_submodule(module_name)
            for local_name, parameter in module.named_parameters(
                recurse=True, remove_duplicate=False
            ):
                name = f"{module_name}.{local_name}" if module_name else local_name
                if name not in seen and not parameter.is_meta:
                    seen.add(name)
                    yield name, parameter
            for owner_name, owner in module.named_modules():
                for local_name, buffer in owner._buffers.items():
                    if (
                        buffer is None
                        or buffer.is_meta
                        or local_name in owner._non_persistent_buffers_set
                    ):
                        continue
                    relative = (
                        f"{owner_name}.{local_name}"
                        if owner_name
                        else local_name
                    )
                    name = (
                        f"{module_name}.{relative}" if module_name else relative
                    )
                    if name not in seen:
                        seen.add(name)
                        yield name, buffer

    def state_tensors_under(
        self, module_names: Sequence[str]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield persistent state only for selected descendants."""

        selected = tuple(module_names)
        for name, tensor in self.state_tensors():
            owner, _, _ = name.rpartition(".")
            if any(_contains(root, owner) for root in selected):
                yield name, tensor


class SubgraphWeightSession:
    """Infer and materialize the checkpoint-backed working set of a subgraph.

    The execution unit is independent from source checkpoint shards. The session
    follows ``call_module`` and tensor ``get_attr`` nodes, then merges explicitly
    requested modifier working-set modules. Descendant module names are collapsed
    under their nearest requested ancestor so a parameter is never loaded twice.
    """

    def __init__(
        self,
        model: nn.Module,
        source: CheckpointWeightSource,
        materializer: WeightMaterializer | None = None,
    ):
        self.model = model
        self.source = source
        self.loader = TargetWeightLoader(model, source, materializer)
        self._source_names = set(source.tensor_names())

    def working_set(
        self,
        subgraph: Subgraph,
        *,
        include_modules: Sequence[str] = (),
        exclude_modules: Sequence[str] = (),
    ) -> tuple[str, ...]:
        excluded = tuple(exclude_modules)
        candidates: list[str] = []
        for node in subgraph.graph.nodes:
            name: str | None = None
            if node.op == "call_module":
                name = str(node.target)
            elif node.op == "get_attr":
                attribute_name = str(node.target)
                owner_name, separator, _ = attribute_name.rpartition(".")
                value = self._get_attribute(attribute_name)
                if isinstance(value, torch.Tensor) and value.is_meta:
                    if not separator:
                        raise ValueError(
                            "A root-level checkpoint tensor cannot be loaded as a "
                            f"module working set: {attribute_name!r}"
                        )
                    name = owner_name
            if name is None or any(_contains(item, name) for item in excluded):
                continue
            if self._has_checkpoint_state(name):
                candidates.append(name)

        for name in include_modules:
            if any(_contains(item, name) for item in excluded):
                continue
            try:
                self.model.get_submodule(name)
            except AttributeError as error:
                raise ValueError(
                    f"Unknown modifier working-set module {name!r}"
                ) from error
            if self._has_checkpoint_state(name):
                candidates.append(name)
        return _minimal_roots(candidates)

    @contextmanager
    def loaded(
        self,
        subgraph: Subgraph,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        include_modules: Sequence[str] = (),
        exclude_modules: Sequence[str] = (),
    ) -> Iterator[LoadedSubgraph]:
        module_names = self.working_set(
            subgraph,
            include_modules=include_modules,
            exclude_modules=exclude_modules,
        )
        registered_state = self._registered_state(module_names)
        runtime_attributes = self._runtime_tensor_attributes(module_names)
        with ExitStack() as stack:
            stack.callback(self._restore_registered_state, registered_state)
            stack.callback(
                self._restore_runtime_tensor_attributes, runtime_attributes
            )
            for name in module_names:
                stack.enter_context(
                    self.loader.loaded(
                        name,
                        device=torch.device(device),
                        dtype=dtype,
                        allow_missing_state=True,
                    )
                )
            # Loading one subgraph can require several module contexts. A
            # model-specific buffer initializer invoked by an early context may
            # therefore bind plain tensor attributes to buffers that a later
            # context subsequently replaces. Refresh once more only after the
            # complete working set is resident.
            reinitialize = getattr(
                self.model, "_reinitialize_non_persistent_buffers", None
            )
            if callable(reinitialize):
                reinitialize()
            yield LoadedSubgraph(self.model, module_names)

    def _runtime_tensor_attributes(
        self, module_names: Sequence[str]
    ) -> dict[nn.Module, dict[str, torch.Tensor | None]]:
        """Snapshot unregistered tensor references owned by a working set.

        Some model implementations cache views of non-persistent buffers in plain
        attributes. Model-specific reinitialization may replace those references
        while the subgraph is loaded; restoring them prevents real storage from
        leaking after the working set returns to meta.
        """

        state = {}
        for root_name in module_names:
            for module in self.model.get_submodule(root_name).modules():
                attributes = {
                    name: value
                    for name, value in vars(module).items()
                    if (isinstance(value, torch.Tensor) or value is None)
                    and name not in module._parameters
                    and name not in module._buffers
                }
                if attributes:
                    state[module] = attributes
        return state

    @staticmethod
    def _restore_runtime_tensor_attributes(
        state: dict[nn.Module, dict[str, torch.Tensor | None]],
    ) -> None:
        for module, attributes in state.items():
            for name, value in attributes.items():
                setattr(module, name, value)

    def _has_checkpoint_state(self, module_name: str) -> bool:
        prefix = f"{module_name}." if module_name else ""
        return any(
            name == module_name or name.startswith(prefix)
            for name in self._source_names
        )

    def _registered_state(
        self, module_names: Sequence[str]
    ) -> dict[
        nn.Module,
        tuple[dict[str, nn.Parameter | None], dict[str, torch.Tensor | None]],
    ]:
        state = {}
        for root_name in module_names:
            for module in self.model.get_submodule(root_name).modules():
                state[module] = (
                    dict(module._parameters),
                    dict(module._buffers),
                )
        return state

    @staticmethod
    def _restore_registered_state(
        state: dict[
            nn.Module,
            tuple[
                dict[str, nn.Parameter | None],
                dict[str, torch.Tensor | None],
            ],
        ],
    ) -> None:
        for module, (parameters, buffers) in state.items():
            for name in module._parameters.keys() - parameters.keys():
                del module._parameters[name]
            for name, value in parameters.items():
                module._parameters[name] = value
            for name in module._buffers.keys() - buffers.keys():
                del module._buffers[name]
                module._non_persistent_buffers_set.discard(name)
            for name, value in buffers.items():
                module._buffers[name] = value

    def _get_attribute(self, qualified_name: str) -> Any:
        value: Any = self.model
        try:
            for part in qualified_name.split("."):
                value = getattr(value, part)
        except AttributeError as error:
            raise ValueError(
                f"Unknown get_attr target {qualified_name!r} in traced subgraph"
            ) from error
        return value
