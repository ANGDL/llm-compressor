from __future__ import annotations

import pytest

from llmcompressor.streaming.modifier_contract import (
    ModifierContractError,
    build_modifier_contracts,
    calibration_outputs_reusable,
)


class ReusableModifier:
    calibration_outputs_reusable = True
    requires_calibration_data = True

    def estimate_sequential_memory(self, _module, _num_tokens):
        return 12, 34


class NonReusableModifier:
    calibration_outputs_reusable = False


def test_modifier_contract_requires_explicit_reuse_for_every_modifier():
    assert calibration_outputs_reusable([ReusableModifier()])
    assert not calibration_outputs_reusable(
        [ReusableModifier(), NonReusableModifier()]
    )


def test_modifier_contract_records_memory_and_execution_promises():
    contract = build_modifier_contracts([ReusableModifier()])[0]

    assert contract.requires_calibration_data
    assert contract.memory_estimate == (12, 34)
    assert contract.propagation_mode == "sequential-compatible"
    assert contract.recovery_state == "replay-subgraph"
    assert contract.to_dict()["output_ownership"] == "pipeline"


def test_modifier_contract_rejects_invalid_memory_declaration():
    class InvalidModifier:
        def estimate_sequential_memory(self, _module, _num_tokens):
            return 1

    with pytest.raises(ModifierContractError, match="must return"):
        build_modifier_contracts([InvalidModifier()])
