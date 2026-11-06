from __future__ import annotations

import pytest
import torch
from torch import nn

from llmcompressor.streaming.admission import (
    DeviceMemoryBudget,
    DeviceMemoryError,
    estimate_modifier_memory,
)
from llmcompressor.streaming.modifier_contract import ModifierContractError


class EstimatedModifier:
    requires_calibration_data = True

    def estimate_sequential_memory(self, module, num_tokens):
        return module.weight.shape[0] + num_tokens, 7


class StaticModifier:
    requires_calibration_data = True
    memory_estimate = (13, 17)


class UnknownModifier:
    requires_calibration_data = True


def test_modifier_memory_aggregates_held_and_scratch():
    modules = [nn.Linear(4, 3, device="meta"), nn.Linear(4, 5, device="meta")]

    estimate = estimate_modifier_memory(
        [EstimatedModifier(), StaticModifier()],
        modules,
        num_tokens=11,
    )

    assert estimate.held_bytes == 3 + 11 + 5 + 11 + 13
    assert estimate.scratch_bytes == 7 + 7 + 17
    assert estimate.peak_bytes == estimate.held_bytes + estimate.scratch_bytes


def test_modifier_memory_rejects_unknown_calibration_workspace():
    with pytest.raises(ModifierContractError, match="bounded memory estimate"):
        estimate_modifier_memory(
            [UnknownModifier()],
            [nn.Linear(4, 3, device="meta")],
            num_tokens=1,
        )


def test_device_budget_applies_safety_reserve_and_releases_reservation():
    budget = DeviceMemoryBudget(available_bytes=1_000, safety_reserve_fraction=0.1)

    reservation = budget.reserve("target-a", 800, device="cpu")
    with pytest.raises(DeviceMemoryError, match="target-b"):
        budget.reserve("target-b", 101, device="cpu")

    assert reservation.reserved_bytes == 800
    reservation.close()
    released = budget.reserve("target-c", 900, device="cpu")
    released.close()


def test_device_budget_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        DeviceMemoryBudget(available_bytes=-1)
    with pytest.raises(ValueError, match="required_bytes"):
        DeviceMemoryBudget(available_bytes=1_000).reserve(
            "target", -1, device="cpu"
        )
