from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmcompressor.core import create_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier
from llmcompressor.streaming.admission import (
    DeviceMemoryBudget,
    DeviceMemoryError,
    DiskSpaceBudget,
    DiskSpaceError,
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
    assert estimate.scratch_bytes == 7 + 17
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


def test_dynamic_estimate_is_not_replaced_by_identity_probe():
    class ShapeModifier:
        requires_calibration_data = True

        def estimate_sequential_memory(self, module, _num_tokens):
            weight = getattr(module, "weight", None)
            return (0, 0) if weight is None else (weight.numel(), 42)

    estimate = estimate_modifier_memory(
        (modifier for modifier in [ShapeModifier()]),
        [nn.Linear(4, 3, device="meta")],
        num_tokens=5,
    )

    assert estimate.held_bytes == 12
    assert estimate.scratch_bytes == 42


@pytest.mark.parametrize("quantizer_class", [QuantizationModifier, GPTQModifier])
def test_real_quantizer_estimates_on_initialized_meta_modules(quantizer_class):
    model = nn.Sequential(nn.Linear(8, 4, bias=False, device="meta"))
    recipe = [
        quantizer_class(scheme="W8A8", targets=["Linear"]),
    ]
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(model=model, recipe=recipe, start=-1)
        estimate = estimate_modifier_memory(
            session.lifecycle.recipe.modifiers, model.modules(), num_tokens=16
        )

    assert estimate.scratch_bytes > 0
    if quantizer_class is GPTQModifier:
            gptq = next(
                detail
                for detail in estimate.details
                if detail["modifier_type"].endswith("GPTQModifier")
            )
            assert gptq["held_bytes"] == 8 * 8 * torch.float32.itemsize


def test_imatrix_weight_observer_scratch_ignores_token_count():
    model = nn.Sequential(nn.Linear(128, 64, bias=False, device="meta"))
    recipe = [
        QuantizationModifier(
            scheme="W8A8",
            targets=["Linear"],
            weight_observer="imatrix_mse",
        )
    ]
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(model=model, recipe=recipe, start=-1)
        small = estimate_modifier_memory(
            session.lifecycle.recipe.modifiers,
            model.modules(),
            num_tokens=16,
            max_batch_tokens=16,
        )
        large = estimate_modifier_memory(
            session.lifecycle.recipe.modifiers,
            model.modules(),
            num_tokens=10_000_000,
            max_batch_tokens=1_600_000,
        )

    assert small.scratch_bytes == large.scratch_bytes


def test_device_reservations_do_not_compete_across_devices():
    budget = DeviceMemoryBudget(available_bytes=1_000, safety_reserve_fraction=0)
    with budget.reserve("first", 1_000, device="cuda:0"):
        with budget.reserve("other-device", 1_000, device="cuda:1"):
            with pytest.raises(DeviceMemoryError):
                budget.reserve("same-device", 1, device="cuda:0")


def test_zero_available_budget_and_unknown_accelerator_are_explicit():
    with pytest.raises(DeviceMemoryError):
        DeviceMemoryBudget(available_bytes=0).reserve("target", 1, device="cpu")
    with pytest.raises(DeviceMemoryError, match="Cannot determine"):
        DeviceMemoryBudget(available_bytes=lambda _device: None).reserve(
            "target", 1, device="cuda:0"
        )


def test_disk_budget_uses_work_filesystem_and_safety_reserve(tmp_path):
    paths = []

    def available(path):
        paths.append(path)
        return 1_000

    budget = DiskSpaceBudget(
        tmp_path / "new" / "work", available_bytes=available,
        safety_reserve_fraction=0.1,
    )
    budget.check("target", 900)
    with pytest.raises(DiskSpaceError, match="target"):
        budget.check("target", 901)
    assert paths == [tmp_path, tmp_path]


def test_awq_offloaded_cache_is_charged_to_host_budget():
    parent = nn.Identity()
    balance = nn.Linear(8, 4, device="meta")
    modifier = AWQModifier(offload_device=torch.device("cpu"))
    modifier._mappings_by_parent = {
        parent: [SimpleNamespace(balance_layers=[balance])]
    }

    estimate = estimate_modifier_memory([modifier], [parent], num_tokens=20)

    assert estimate.held_bytes == 0
    assert estimate.host_bytes == 20 * 8 * torch.float32.itemsize
    assert estimate.scratch_bytes > 0


@pytest.mark.parametrize("estimate", [(1.5, 2), (-1, 0), (1,), "unknown"])
def test_invalid_dynamic_estimates_fail_with_contract_error(estimate):
    class InvalidModifier:
        requires_calibration_data = True

        def estimate_sequential_memory(self, module, _num_tokens):
            return (0, 0) if isinstance(module, nn.Identity) else estimate

    with pytest.raises(ModifierContractError, match="non-negative"):
        estimate_modifier_memory(
            [InvalidModifier()], [nn.Linear(4, 3, device="meta")], num_tokens=1
        )
