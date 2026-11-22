"""Bounded CPU route data shared by the owners of one expert region."""

from types import MappingProxyType

import torch
from torch.fx import Node

from llmcompressor.modeling.moe.routing import group_expert_routes

from ..loading.host_memory import HostMemoryError


def _grouped_route_bytes(weights, indices, num_experts):
    if (
        not isinstance(weights, torch.Tensor)
        or not isinstance(indices, torch.Tensor)
        or weights.device.type != "cpu"
        or indices.device.type != "cpu"
        or weights.layout != torch.strided
        or indices.layout != torch.strided
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.ndim != 2
        or weights.shape != indices.shape
        or type(num_experts) is not int
        or num_experts < 1
    ):
        return None
    routes = indices.numel()
    retained = routes * (8 + weights.element_size()) + 64 * (num_experts + 1) + 1024
    workspace = routes * (48 + weights.element_size()) + 64 * (num_experts + 1)
    return retained, workspace


_ROUTE_ESTIMATORS = {group_expert_routes: _grouped_route_bytes}


class ExpertRouteCache:
    """Prepare whitelisted route results before launching any owner threads."""

    def __init__(self, host_budget, label, *, max_bytes=64 * 2**20):
        self._budget = host_budget
        self._label = label
        self._max_bytes = max_bytes
        self._limit = 0
        self._used = 0
        self._results = {}
        self._reservations = []
        self._prepared = 0
        self._skipped = 0

    def __enter__(self):
        available = self._budget.effective_available_bytes()
        self._limit = min(
            self._max_bytes,
            available // 32,
        )
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self._results.clear()
        for reservation in self._reservations:
            reservation.close()
        self._reservations.clear()
        self._used = 0

    def batch(self, batch):
        return self._results.get(batch)

    def prepare(self, batch, value, nodes):
        if self._used >= self._limit:
            self._skipped += 1
            return
        prepared = {}
        for node in nodes:
            args = tuple(
                value[str(arg.target)] if isinstance(arg, Node) else arg
                for arg in node.args
            )
            estimate = _ROUTE_ESTIMATORS[node.target](*args)
            if estimate is None:
                continue
            retained, workspace = estimate
            if retained > self._limit - self._used:
                continue
            if retained + workspace > self._budget.effective_available_bytes() // 32:
                continue
            try:
                reservation = self._budget.reserve(
                    f"{self._label} batch {batch} grouped routes", retained + workspace
                )
            except HostMemoryError:
                continue
            try:
                prepared[node] = node.target(*args)
                reservation.commit(retained)
                reservation.release_reserved(workspace)
            except BaseException:
                reservation.close()
                raise
            self._reservations.append(reservation)
            self._used += retained
        if prepared:
            self._results[batch] = MappingProxyType(prepared)
            self._prepared += 1
        else:
            self._skipped += 1

    def report(self) -> str:
        """Summarize route-cache coverage without touching cached tensors."""

        return (
            f"batches_cached={self._prepared} | batches_skipped={self._skipped} | "
            f"bytes={self._used:,} | limit={self._limit:,}"
        )


def is_cacheable_route(node):
    return (
        node.op == "call_function"
        and node.target in _ROUTE_ESTIMATORS
        and len(node.args) == 3
        and not node.kwargs
        and all(isinstance(arg, Node) for arg in node.args[:2])
        and type(node.args[2]) is int
        and all(source.op == "placeholder" for source in node.all_input_nodes)
    )
