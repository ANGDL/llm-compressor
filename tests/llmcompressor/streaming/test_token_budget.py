import pytest

from llmcompressor.streaming.admission import DeviceMemoryBudget, DeviceMemoryError
from llmcompressor.streaming.parallel.token_budget import (
    select_token_capacity,
    validate_token_budget,
)


@pytest.mark.parametrize("usable, expected", [(136, 6), (149, 7), (500, 20)])
def test_capacity_search_respects_nonlinear_workspace_and_owner_memory(
    usable, expected,
):
    capacity = select_token_capacity(
        owner="expert owner", usable_bytes=usable,
        required_bytes=lambda tokens: 100 + tokens**2,
        maximum_tokens=20, minimum_tokens=2,
    )
    assert capacity == expected


def test_capacity_search_limits_unknown_cpu_memory_to_one_boundary():
    assert select_token_capacity(
        owner="CPU test", usable_bytes=None,
        required_bytes=lambda tokens: tokens**2,
        maximum_tokens=100, minimum_tokens=7,
    ) == 7


@pytest.mark.parametrize("usable, maximum", [(103, 20), (1_000, 1)])
def test_capacity_search_rejects_minimum_that_cannot_fit(usable, maximum):
    with pytest.raises(DeviceMemoryError, match="minimum|requested"):
        select_token_capacity(
            owner="expert owner", usable_bytes=usable,
            required_bytes=lambda tokens: 100 + tokens**2,
            maximum_tokens=maximum, minimum_tokens=2,
        )


def test_capacity_admission_deducts_reserve_and_outstanding_owner_reservations():
    budget = DeviceMemoryBudget(available_bytes=1_000, safety_reserve_fraction=0.1)
    with budget.reserve("prefetch", 200, device="cpu"):
        assert budget.usable_bytes("cpu") == 700
    assert budget.usable_bytes("cpu") == 900


@pytest.mark.parametrize("value", ["auto", 1, 8192])
def test_explicit_token_budget_accepts_auto_or_positive_integer(value):
    validate_token_budget(value)


@pytest.mark.parametrize("value", [0, -1, True, None, "8192", "invalid", 1.5])
def test_explicit_token_budget_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        validate_token_budget(value)
