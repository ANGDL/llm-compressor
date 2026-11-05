"""Execution topology and ownership contracts for streaming PTQ."""

from .context import ParallelContext, StatisticsScope
from .placement import PlacementEntry, PlacementPlan

__all__ = [
    "ParallelContext",
    "PlacementEntry",
    "PlacementPlan",
    "StatisticsScope",
]
