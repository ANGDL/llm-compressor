from concurrent.futures import ThreadPoolExecutor

import pytest

from llmcompressor.streaming.loading.host_memory import (
    HostMemoryBudget,
    HostMemoryError,
)


def _budget(available=1_000, cgroup=None, safety=100):
    return HostMemoryBudget(
        safety_reserve_bytes=safety,
        system_available=lambda: available,
        cgroup_available=lambda: cgroup,
    )


def test_budget_uses_smaller_system_and_cgroup_availability():
    budget = _budget(available=1_000, cgroup=700, safety=100)

    assert budget.effective_available_bytes() == 600


def test_budget_subtracts_reserved_but_not_committed_bytes_twice():
    budget = _budget()
    reservation = budget.reserve("next target", 400)
    assert budget.effective_available_bytes() == 500

    reservation.commit(250)

    assert reservation.reserved_bytes == 150
    assert reservation.committed_bytes == 250
    assert budget.effective_available_bytes() == 750


def test_insufficient_capacity_fails_before_reservation_is_created():
    budget = _budget()

    with pytest.raises(HostMemoryError, match="requested"):
        budget.reserve("too large", 901)

    assert budget.effective_available_bytes() == 900


def test_reservation_transitions_and_idempotent_close():
    budget = _budget()
    reservation = budget.reserve("target", 600)

    reservation.commit(400)
    reservation.uncommit(100)
    reservation.release_reserved(200)
    reservation.release_committed(200)

    assert reservation.reserved_bytes == 100
    assert reservation.committed_bytes == 100
    reservation.close()
    reservation.close()
    assert reservation.closed
    assert budget.effective_available_bytes() == 900


@pytest.mark.parametrize(
    "operation",
    [
        lambda item: item.commit(101),
        lambda item: item.uncommit(1),
        lambda item: item.release_reserved(101),
        lambda item: item.release_committed(1),
    ],
)
def test_reservation_rejects_underflow(operation):
    reservation = _budget().reserve("target", 100)

    with pytest.raises(ValueError):
        operation(reservation)


def test_reservation_transitions_are_thread_safe():
    reservation = _budget(available=10_000, safety=0).reserve("target", 4_000)

    def cycle():
        for _ in range(100):
            reservation.commit(1)
            reservation.uncommit(1)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _index: cycle(), range(8)))

    assert reservation.reserved_bytes == 4_000
    assert reservation.committed_bytes == 0
