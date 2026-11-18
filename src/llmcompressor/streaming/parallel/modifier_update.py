"""Owner-local modifier update scheduling for streaming expert parallelism."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

__all__ = [
    "OwnerModuleGroup",
    "OwnerUpdateScheduler",
    "partition_modules_by_owner",
]


@dataclass(frozen=True)
class OwnerModuleGroup:
    """Modules updated together on one owner device."""

    owner: torch.device
    modules: tuple[nn.Module, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", torch.device(self.owner))
        object.__setattr__(self, "modules", tuple(self.modules))
        if not self.modules:
            raise ValueError("an owner update group must contain a module")


def partition_modules_by_owner(
    modules: Sequence[nn.Module],
    owner_devices: Sequence[torch.device | str],
    module_devices: Mapping[nn.Module, torch.device | str],
) -> tuple[OwnerModuleGroup, ...]:
    """Partition resident modules without moving them between owners."""

    devices = tuple(torch.device(device) for device in owner_devices)
    if not devices:
        raise ValueError("owner_devices must not be empty")
    if len(set(devices)) != len(devices):
        raise ValueError("owner_devices must be unique")

    grouped: dict[torch.device, list[nn.Module]] = {device: [] for device in devices}
    seen: set[int] = set()
    for module in modules:
        identity = id(module)
        if identity in seen:
            continue
        seen.add(identity)
        owner = module_devices.get(module)
        if owner is None:
            raise ValueError(
                f"module {type(module).__qualname__} has no declared owner device"
            )
        owner = torch.device(owner)
        if owner not in grouped:
            raise ValueError(
                f"module owner {owner} is outside the declared owner devices"
            )
        grouped[owner].append(module)

    return tuple(
        OwnerModuleGroup(owner, tuple(grouped[owner]))
        for owner in devices
        if grouped[owner]
    )


class OwnerUpdateScheduler:
    """Run independent owner updates concurrently and join once at the end."""

    def __init__(self, *, max_workers: int | None = None) -> None:
        if max_workers is not None and (
            isinstance(max_workers, bool) or max_workers <= 0
        ):
            raise ValueError("max_workers must be positive or None")
        self.max_workers = max_workers

    @staticmethod
    def _run_owner(
        group: OwnerModuleGroup,
        update: Callable[[torch.device, tuple[nn.Module, ...]], Any],
    ) -> Any:
        if group.owner.type == "cuda":
            with torch.cuda.device(group.owner):
                return update(group.owner, group.modules)
        return update(group.owner, group.modules)

    def run(
        self,
        groups: Sequence[OwnerModuleGroup],
        *,
        update: Callable[[torch.device, tuple[nn.Module, ...]], Any],
    ) -> tuple[tuple[torch.device, Any], ...]:
        """Run all owner-local updates and wait only for this update boundary."""

        ordered = tuple(groups)
        if not ordered:
            return ()
        if len({group.owner for group in ordered}) != len(ordered):
            raise ValueError("owner update groups must have unique owners")
        if len(ordered) == 1:
            group = ordered[0]
            return ((group.owner, self._run_owner(group, update)),)

        results: dict[torch.device, Any] = {}
        workers = self.max_workers or len(ordered)
        futures: dict[Future[Any], OwnerModuleGroup] = {}
        with ThreadPoolExecutor(
            max_workers=min(workers, len(ordered)),
            thread_name_prefix="ep-owner-update",
        ) as executor:
            for group in ordered:
                future = executor.submit(self._run_owner, group, update)
                futures[future] = group
            try:
                for future in as_completed(futures):
                    group = futures[future]
                    results[group.owner] = future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

        return tuple((group.owner, results[group.owner]) for group in ordered)
