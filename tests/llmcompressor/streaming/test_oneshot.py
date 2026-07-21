from __future__ import annotations

import torch

from llmcompressor.streaming import streaming_oneshot
from tests.llmcompressor.streaming.test_quantize import TwoLayer, prepare, scheme


def test_streaming_oneshot_runs_three_stages_and_resumes(tmp_path):
    checkpoint, _, _, inputs = prepare(tmp_path)
    work = tmp_path / "work"
    output = tmp_path / "output"
    calls = []

    def batches():
        calls.append("dataset")
        return iter(inputs)

    result = streaming_oneshot(
        model_factory=TwoLayer,
        checkpoint=checkpoint,
        output_dir=output,
        work_dir=work,
        calibration_batches=batches,
        targets=("layers.0", "layers.1"),
        recipe={"GPTQModifier": {}},
        dataset_fingerprint="c" * 64,
        schemes={"layers.0": scheme(), "layers.1": scheme()},
        algorithms=("gptq",),
        target_dtype=torch.float32,
        validate_config=False,
    )
    assert result == output
    assert (output / "FINALIZED").is_file()
    assert calls == ["dataset"]

    resumed = streaming_oneshot(
        model_factory=TwoLayer,
        checkpoint=checkpoint,
        output_dir=output,
        work_dir=work,
        calibration_batches=batches,
        targets=("layers.0", "layers.1"),
        recipe={"GPTQModifier": {}},
        dataset_fingerprint="c" * 64,
        schemes={"layers.0": scheme(), "layers.1": scheme()},
        algorithms=("gptq",),
        target_dtype=torch.float32,
        validate_config=False,
    )
    assert resumed == output
    assert calls == ["dataset"]
