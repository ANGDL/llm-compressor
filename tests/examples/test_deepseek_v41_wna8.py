import importlib.util
from pathlib import Path

import torch

from llmcompressor.streaming.execution import streaming_input_contract

MODULE_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "streaming_oneshot"
    / "deepseek_v41_wNa8.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("deepseek_v41_wna8", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)


def calibration_row(height, width):
    return {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.ones(3, dtype=torch.bool),
        "image_mask": torch.tensor([False, True, False]),
        "image_token_types": torch.tensor([-1, 0, -1]),
        "image_patches": torch.zeros(2, 2, 3, dtype=torch.bfloat16),
        "image_height": height,
        "image_width": width,
    }


def test_collator_keeps_image_geometry_as_variable_tensor_metadata():
    first = MODULE.data_collator([calibration_row(48, 32)])
    second = MODULE.data_collator([calibration_row(34, 45)])

    for batch in (first, second):
        for key in ("image_height", "image_width"):
            assert batch[key].shape == ()
            assert batch[key].dtype == torch.int64

    assert streaming_input_contract(first) == streaming_input_contract(second)
