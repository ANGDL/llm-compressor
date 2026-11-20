"""Activation-aware expert placement and bounded resident-window scheduling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from math import ceil, isfinite
from threading import Event
from typing import Any, TypeVar

from ..artifacts import fingerprint_json
from .placement import PlacementEntry, PlacementPlan

__all__ = [
    "AsyncResidentWindowScheduler",
    "ExpertCostProfile",
    "LoadBalancedPlacement",
    "OwnerMemoryProfile",
    "ResidentWindow",
    "build_fixed_owner_placement",
    "build_load_balanced_placement",
    "plan_resident_windows",
]

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class ExpertCostProfile:
    """Static and activation-aware cost of one logical expert."""

    storage_bytes: int
    route_tokens: int = 0
    token_work_units: float = 1.0
    persistent_state_bytes: int = 0
    update_work_units: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "storage_bytes",
            "route_tokens",
            "persistent_state_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("token_work_units", "update_work_units"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name == "token_work_units" and value == 0):
                raise ValueError(f"{name} must be positive")

    @property
    def routed_work_units(self) -> float:
        return float(self.route_tokens) * float(self.token_work_units)

    @property
    def total_resident_bytes(self) -> int:
        return self.storage_bytes + self.persistent_state_bytes


@dataclass(frozen=True)
class OwnerMemoryProfile:
    """Memory and dispatch limits for one owner rank."""

    usable_bytes: int
    fixed_bytes: int = 0
    update_bytes: int = 0
    dispatch_bytes_per_token: int = 0
    max_dispatch_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "usable_bytes",
            "fixed_bytes",
            "update_bytes",
            "dispatch_bytes_per_token",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_dispatch_tokens is not None and (
            isinstance(self.max_dispatch_tokens, bool)
            or not isinstance(self.max_dispatch_tokens, int)
            or self.max_dispatch_tokens <= 0
        ):
            raise ValueError("max_dispatch_tokens must be positive or None")
        if self.fixed_bytes + self.update_bytes > self.usable_bytes:
            raise ValueError("fixed and update bytes exceed usable_bytes")

    @property
    def resident_budget_bytes(self) -> int:
        return self.usable_bytes - self.fixed_bytes - self.update_bytes

    def dispatch_capacity(self, resident_bytes: int) -> int:
        if resident_bytes > self.resident_budget_bytes:
            raise MemoryError("resident window exceeds owner memory budget")
        available = self.usable_bytes - self.fixed_bytes - resident_bytes
        if self.dispatch_bytes_per_token:
            capacity = available // self.dispatch_bytes_per_token
        else:
            capacity = self.max_dispatch_tokens or (1 << 62)
        if self.max_dispatch_tokens is not None:
            capacity = min(capacity, self.max_dispatch_tokens)
        return max(0, capacity)


@dataclass(frozen=True)
class LoadBalancedPlacement:
    """A stable mapping that balances weighted routed work and memory."""

    ep_size: int
    owners: tuple[tuple[str, int], ...]
    owner_work_units: tuple[float, ...]
    owner_route_tokens: tuple[int, ...]
    owner_storage_bytes: tuple[int, ...]
    owner_persistent_bytes: tuple[int, ...]
    policy: str = "activation-aware-v1"

    def __post_init__(self) -> None:
        if self.ep_size <= 0:
            raise ValueError("ep_size must be positive")
        if len(self.owner_work_units) != self.ep_size:
            raise ValueError("owner_work_units must match ep_size")
        if len(self.owner_route_tokens) != self.ep_size:
            raise ValueError("owner_route_tokens must match ep_size")
        if len(self.owner_storage_bytes) != self.ep_size:
            raise ValueError("owner_storage_bytes must match ep_size")
        if len(self.owner_persistent_bytes) != self.ep_size:
            raise ValueError("owner_persistent_bytes must match ep_size")
        if any(
            owner < 0 or owner >= self.ep_size for _name, owner in self.owners
        ):
            raise ValueError("placement owner exceeds ep_size")
        names = [name for name, _owner in self.owners]
        if len(names) != len(set(names)):
            raise ValueError("placement expert names must be unique")

    @property
    def owner_by_name(self) -> dict[str, int]:
        return dict(self.owners)

    @property
    def placement(self) -> PlacementPlan:
        return PlacementPlan(
            tuple(
                PlacementEntry(logical_name=name, owner_rank=owner)
                for name, owner in self.owners
            ),
            ep_size=self.ep_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "ep_size": self.ep_size,
            "owners": [list(item) for item in self.owners],
            "owner_work_units": list(self.owner_work_units),
            "owner_route_tokens": list(self.owner_route_tokens),
            "owner_storage_bytes": list(self.owner_storage_bytes),
            "owner_persistent_bytes": list(self.owner_persistent_bytes),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True)
class ResidentWindow:
    """One bounded set of experts that may be resident together."""

    owner_rank: int
    window_index: int
    expert_names: tuple[str, ...]
    resident_bytes: int
    route_tokens: int
    work_units: float
    max_dispatch_tokens: int
    activation_batches: int

    def __post_init__(self) -> None:
        if self.owner_rank < 0 or self.window_index < 0:
            raise ValueError("window and owner ranks must be non-negative")
        if not self.expert_names:
            raise ValueError("a resident window must contain an expert")
        if self.resident_bytes < 0 or self.route_tokens < 0:
            raise ValueError("window byte/token counts must be non-negative")
        if self.work_units < 0:
            raise ValueError("window work_units must be non-negative")
        if self.max_dispatch_tokens < 0 or self.activation_batches < 0:
            raise ValueError("window dispatch counts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_rank": self.owner_rank,
            "window_index": self.window_index,
            "expert_names": list(self.expert_names),
            "resident_bytes": self.resident_bytes,
            "route_tokens": self.route_tokens,
            "work_units": self.work_units,
            "max_dispatch_tokens": self.max_dispatch_tokens,
            "activation_batches": self.activation_batches,
        }


def build_load_balanced_placement(
    expert_costs: Mapping[str, ExpertCostProfile],
    owner_count: int,
    *,
    compute_rates: Sequence[float] | None = None,
    owner_memory: Sequence[OwnerMemoryProfile] | None = None,
    policy: str = "activation-aware-v1",
) -> LoadBalancedPlacement:
    """Place experts by normalized routed work under hard memory limits."""

    if owner_count <= 0:
        raise ValueError("owner_count must be positive")
    if not expert_costs:
        raise ValueError("expert_costs must not be empty")
    if not policy:
        raise ValueError("policy must not be empty")
    rates = (
        tuple(1.0 for _ in range(owner_count))
        if compute_rates is None
        else tuple(float(rate) for rate in compute_rates)
    )
    if len(rates) != owner_count or any(
        not isfinite(rate) or rate <= 0 for rate in rates
    ):
        raise ValueError("compute_rates must contain positive finite values")
    if owner_memory is not None and len(owner_memory) != owner_count:
        raise ValueError("owner_memory must match owner_count")

    owner_work = [0.0] * owner_count
    owner_tokens = [0] * owner_count
    owner_storage = [0] * owner_count
    owner_persistent = [0] * owner_count
    owners: dict[str, int] = {}
    ordered = sorted(
        expert_costs,
        key=lambda name: (
            -expert_costs[name].routed_work_units,
            -expert_costs[name].total_resident_bytes,
            name,
        ),
    )
    for name in ordered:
        cost = expert_costs[name]
        candidates = []
        for rank in range(owner_count):
            if owner_memory is not None:
                required = (
                    owner_memory[rank].fixed_bytes
                    + owner_memory[rank].update_bytes
                    + owner_storage[rank]
                    + cost.storage_bytes
                    + owner_persistent[rank]
                    + cost.persistent_state_bytes
                    + (
                        owner_memory[rank].dispatch_bytes_per_token
                        if owner_tokens[rank] or cost.route_tokens
                        else 0
                    )
                )
                if required > owner_memory[rank].usable_bytes:
                    continue
            new_work = owner_work[rank] + cost.routed_work_units
            normalized = new_work / rates[rank]
            other_max = max(
                (
                    owner_work[other] / rates[other]
                    for other in range(owner_count)
                    if other != rank
                ),
                default=0.0,
            )
            candidates.append(
                (
                    max(normalized, other_max),
                    normalized,
                    owner_storage[rank] + cost.storage_bytes,
                    owner_persistent[rank] + cost.persistent_state_bytes,
                    rank,
                )
            )
        if not candidates:
            raise MemoryError(
                f"expert {name!r} cannot fit on any owner under the declared "
                "resident memory limits"
            )
        rank = min(candidates)[-1]
        owners[name] = rank
        owner_work[rank] += cost.routed_work_units
        owner_tokens[rank] += cost.route_tokens
        owner_storage[rank] += cost.storage_bytes
        owner_persistent[rank] += cost.persistent_state_bytes

    return LoadBalancedPlacement(
        ep_size=owner_count,
        owners=tuple(sorted(owners.items())),
        owner_work_units=tuple(owner_work),
        owner_route_tokens=tuple(owner_tokens),
        owner_storage_bytes=tuple(owner_storage),
        owner_persistent_bytes=tuple(owner_persistent),
        policy=policy,
    )


def build_fixed_owner_placement(
    expert_costs: Mapping[str, ExpertCostProfile],
    owner_count: int,
    *,
    owner_memory: Sequence[OwnerMemoryProfile],
    compute_rates: Sequence[float] | None = None,
) -> LoadBalancedPlacement:
    """Assign experts once while treating owner residency as a hard limit."""

    return build_load_balanced_placement(
        expert_costs,
        owner_count,
        compute_rates=compute_rates,
        owner_memory=owner_memory,
        policy="fixed-owner-v1",
    )


def plan_resident_windows(
    placement: LoadBalancedPlacement,
    expert_costs: Mapping[str, ExpertCostProfile],
    owner_memory: Sequence[OwnerMemoryProfile],
    *,
    staging_bytes: Mapping[int, int] | None = None,
) -> tuple[ResidentWindow, ...]:
    """Pack each owner's experts into memory-safe resident windows."""

    if len(owner_memory) != placement.ep_size:
        raise ValueError("owner_memory must match placement.ep_size")
    owner_names: dict[int, list[str]] = defaultdict(list)
    for name, owner in placement.owners:
        if name not in expert_costs:
            raise KeyError(name)
        owner_names[owner].append(name)

    windows: list[ResidentWindow] = []
    for owner in range(placement.ep_size):
        staging = (staging_bytes or {}).get(owner, 0)
        limits = owner_memory[owner]
        if staging:
            limits = replace(limits, fixed_bytes=limits.fixed_bytes + staging)
        names = sorted(
            owner_names[owner],
            key=lambda name: (
                -expert_costs[name].routed_work_units,
                -expert_costs[name].total_resident_bytes,
                name,
            ),
        )
        current: list[str] = []
        current_bytes = 0
        current_storage = 0
        current_tokens = 0
        current_work = 0.0
        window_index = 0

        def emit() -> None:
            nonlocal current, current_bytes, current_storage, current_tokens
            nonlocal current_work
            if not current:
                return
            capacity = limits.dispatch_capacity(current_bytes)
            if current_tokens and capacity <= 0:
                raise MemoryError(
                    f"resident window {owner}:{window_index} has no dispatch "
                    "capacity for its routed tokens"
                )
            max_tokens = min(capacity, current_tokens) if current_tokens else 0
            batches = sum(
                max(1, ceil(expert_costs[name].route_tokens / max_tokens))
                if max_tokens else 1
                for name in current
            )
            windows.append(
                ResidentWindow(
                    owner_rank=owner,
                    window_index=window_index,
                    expert_names=tuple(current),
                    resident_bytes=current_bytes,
                    route_tokens=current_tokens,
                    work_units=current_work,
                    max_dispatch_tokens=max_tokens,
                    activation_batches=batches,
                )
            )
            current = []
            current_bytes = 0
            current_storage = 0
            current_tokens = 0
            current_work = 0.0

        for name in names:
            cost = expert_costs[name]
            minimum_dispatch = (
                limits.dispatch_bytes_per_token if cost.route_tokens else 0
            )
            if (
                cost.total_resident_bytes + minimum_dispatch
                > limits.resident_budget_bytes
                or (staging and cost.storage_bytes > staging)
            ):
                raise MemoryError(
                    f"expert {name!r} cannot fit in an owner resident window"
                )
            minimum_dispatch = (
                limits.dispatch_bytes_per_token
                if current_tokens + cost.route_tokens else 0
            )
            would_exceed = (
                current
                and (
                    current_bytes + cost.total_resident_bytes + minimum_dispatch
                    > limits.resident_budget_bytes
                    or (staging and current_storage + cost.storage_bytes > staging)
                )
            )
            if would_exceed:
                emit()
                window_index += 1
            current.append(name)
            current_bytes += cost.total_resident_bytes
            current_storage += cost.storage_bytes
            current_tokens += cost.route_tokens
            current_work += cost.routed_work_units
        emit()
    return tuple(windows)


def plan_staging_bytes(windows, costs, owner_memory, enabled_owners):
    """Find space for the next window's weights on owners that require paging."""
    by_owner = defaultdict(list)
    for window in windows:
        if window.owner_rank in enabled_owners:
            by_owner[window.owner_rank].append(window)
    staging = {}
    for owner, owner_windows in by_owner.items():
        if len(owner_windows) < 2:
            continue
        limits = owner_memory[owner]
        experts = [
            costs[name] for window in owner_windows for name in window.expert_names
        ]
        largest_expert = max(expert.storage_bytes for expert in experts)
        active_minimum = max(
            expert.total_resident_bytes
            + (limits.dispatch_bytes_per_token if expert.route_tokens else 0)
            for expert in experts
        )
        largest_window = max(
            sum(costs[name].storage_bytes for name in window.expert_names)
            for window in owner_windows
        )
        candidate = min(
            limits.resident_budget_bytes - active_minimum,
            max(largest_expert, limits.resident_budget_bytes // 2),
            largest_window,
        )
        if candidate and candidate >= largest_expert:
            staging[owner] = candidate
    return staging


class AsyncResidentWindowScheduler:
    """Run owner-local window pipelines concurrently with bounded prefetch."""

    def __init__(self, *, max_prefetch: int = 1) -> None:
        if isinstance(max_prefetch, bool) or max_prefetch < 0:
            raise ValueError("max_prefetch must be a non-negative integer")
        self.max_prefetch = max_prefetch

    def run(
        self,
        windows: Iterable[ResidentWindow],
        *,
        prefetch: Callable[[ResidentWindow], T],
        compute: Callable[[ResidentWindow, T], U],
        update: Callable[[ResidentWindow, T, U], Any] | None = None,
        release: Callable[[ResidentWindow, T, U], Any] | None = None,
        discard: Callable[[T], Any] | None = None,
    ) -> tuple[tuple[ResidentWindow, U], ...]:
        """Run one bounded pipeline per owner and return results in plan order."""

        ordered = tuple(windows)
        if not ordered:
            return ()
        by_owner: dict[int, list[ResidentWindow]] = defaultdict(list)
        for window in ordered:
            by_owner[window.owner_rank].append(window)
        for owner_windows in by_owner.values():
            owner_windows.sort(key=lambda window: window.window_index)

        results: list[tuple[ResidentWindow, U]] = []
        canceled = Event()

        def run_owner(owner_windows: list[ResidentWindow]):
            owner_results = []
            with ThreadPoolExecutor(
                max_workers=max(1, self.max_prefetch),
                thread_name_prefix=f"ep-prefetch-{owner_windows[0].owner_rank}",
            ) as owner_prefetch:
                prefetched: Future[T] | None = None
                try:
                    for index, window in enumerate(owner_windows):
                        if canceled.is_set():
                            break
                        if prefetched is None:
                            prefetched = owner_prefetch.submit(prefetch, window)
                        current = prefetched
                        prefetched = None
                        state = current.result()
                        if self.max_prefetch and index + 1 < len(owner_windows):
                            prefetched = owner_prefetch.submit(
                                prefetch, owner_windows[index + 1]
                            )
                        try:
                            result = compute(window, state)
                            if update is not None:
                                update(window, state, result)
                            if release is not None:
                                release(window, state, result)
                            owner_results.append((window, result))
                        finally:
                            if discard is not None:
                                discard(state)
                except BaseException:
                    canceled.set()
                    raise
                finally:
                    if prefetched is not None:
                        if not prefetched.cancel():
                            try:
                                state = prefetched.result()
                            except BaseException:
                                if not canceled.is_set():
                                    raise
                            else:
                                if discard is not None:
                                    discard(state)
            return owner_results

        with ThreadPoolExecutor(
            max_workers=len(by_owner),
            thread_name_prefix="ep-window",
        ) as owner_workers:
            futures = [
                owner_workers.submit(run_owner, owner_windows)
                for owner_windows in by_owner.values()
            ]
            try:
                for future in futures:
                    results.extend(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        return tuple(
            sorted(results, key=lambda item: (item[0].owner_rank, item[0].window_index))
        )
