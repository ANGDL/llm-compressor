from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from llmcompressor.modifiers.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_gptq_statistics,
)
from llmcompressor.observers.imatrix import (
    accumulate_imatrix_statistics,
    make_empty_imatrix_statistics,
)
from llmcompressor.streaming import (
    ArtifactStore,
    CalibrationInfo,
    GPTQStatisticsCollector,
    IMatrixStatisticsCollector,
    MaterializerInfo,
    RecipeInfo,
    SequentialInfo,
    SoftwareInfo,
    StatisticsCollectorGroup,
    StreamingRunManifest,
    TargetStatisticsMetadata,
    fingerprint_checkpoint,
    fingerprint_json,
)


class TwoLinearTarget(nn.Module):
    def __init__(self, features=4):
        super().__init__()
        self.first = nn.Linear(features, features)
        self.second = nn.Linear(features, features, bias=False)

    def forward(self, inputs):
        return self.second(torch.tanh(self.first(inputs)))


def clone_weights(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_weights_equal(model, expected):
    assert model.state_dict().keys() == expected.keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])


def direct_gptq_statistics(inputs):
    flattened = inputs.reshape(-1, inputs.shape[-1]).float()
    return 2 * flattened.T @ flattened, torch.tensor(inputs.shape[0])


def direct_imatrix_statistics(inputs):
    flattened = inputs.reshape(-1, inputs.shape[-1]).float()
    return flattened.square().sum(dim=0), torch.tensor(flattened.shape[0])


def test_gptq_matches_direct_hessian_and_does_not_change_weights():
    torch.manual_seed(1)
    module = nn.Linear(4, 3)
    weights = clone_weights(module)
    inputs = torch.randn(2, 5, 4)
    collector = GPTQStatisticsCollector({"proj": module})

    with collector:
        module(inputs)
    statistics = collector.export()
    expected_hessian, expected_count = direct_gptq_statistics(inputs)

    torch.testing.assert_close(statistics["proj.gptq_hessian"], expected_hessian)
    assert torch.equal(statistics["proj.gptq_num_samples"], expected_count)
    assert_weights_equal(module, weights)


def test_streaming_gptq_matches_shared_oneshot_primitives():
    module = nn.Linear(4, 3)
    inputs = torch.randn(2, 5, 4)
    collector = GPTQStatisticsCollector({"proj": module})
    shared_hessian, shared_count = make_empty_gptq_statistics(module)

    with collector:
        module(inputs)
    shared_hessian, shared_count = accumulate_hessian(
        inputs, module, shared_hessian, shared_count
    )
    statistics = collector.export()

    assert torch.equal(statistics["proj.gptq_hessian"], shared_hessian)
    assert torch.equal(statistics["proj.gptq_num_samples"], shared_count)


def test_imatrix_matches_direct_sum_and_count():
    module = nn.Linear(4, 3)
    inputs = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]]
    )
    collector = IMatrixStatisticsCollector({"proj": module})

    with collector:
        module(inputs)
    statistics = collector.export()
    expected_sum, expected_count = direct_imatrix_statistics(inputs)

    assert torch.equal(statistics["proj.imatrix_sum"], expected_sum)
    assert torch.equal(statistics["proj.imatrix_count"], expected_count)


def test_streaming_imatrix_matches_shared_oneshot_primitive():
    module = nn.Linear(4, 3)
    inputs = torch.randn(2, 5, 4, dtype=torch.bfloat16)
    collector = IMatrixStatisticsCollector({"proj": module})
    shared_sum, shared_count = make_empty_imatrix_statistics(4)

    with collector:
        module(inputs.float())
    shared_sum, shared_count = accumulate_imatrix_statistics(
        inputs.float(), shared_sum, shared_count
    )
    statistics = collector.export()

    assert torch.equal(statistics["proj.imatrix_sum"], shared_sum)
    assert torch.equal(statistics["proj.imatrix_count"], shared_count)


@pytest.mark.parametrize(
    "collector_type",
    [GPTQStatisticsCollector, IMatrixStatisticsCollector],
)
def test_split_batches_equal_single_batch(collector_type):
    torch.manual_seed(4)
    inputs = torch.randn(4, 3, 4)
    whole_module = nn.Linear(4, 4)
    split_module = nn.Linear(4, 4)
    split_module.load_state_dict(whole_module.state_dict())
    whole = collector_type({"proj": whole_module})
    split = collector_type({"proj": split_module})

    with whole:
        whole_module(inputs)
    with split:
        split_module(inputs[:2])
        split_module(inputs[2:])

    whole_stats = whole.export()
    split_stats = split.export()
    assert whole_stats.keys() == split_stats.keys()
    for name in whole_stats:
        torch.testing.assert_close(whole_stats[name], split_stats[name])


def test_gptq_and_imatrix_collect_together_for_multiple_modules():
    torch.manual_seed(2)
    model = TwoLinearTarget()
    weights = clone_weights(model)
    modules = {"first": model.first, "second": model.second}
    group = StatisticsCollectorGroup(
        [GPTQStatisticsCollector(modules), IMatrixStatisticsCollector(modules)]
    )

    with group:
        model(torch.randn(2, 3, 4))
    statistics = group.export()

    assert set(statistics) == {
        f"{module}.{algorithm}_{statistic}"
        for module in modules
        for algorithm, statistics_for_algorithm in (
            ("gptq", ("hessian", "num_samples")),
            ("imatrix", ("sum", "count")),
        )
        for statistic in statistics_for_algorithm
    }
    assert_weights_equal(model, weights)


def test_detach_stops_collection_and_export_is_an_isolated_snapshot():
    module = nn.Linear(4, 4)
    collector = IMatrixStatisticsCollector({"proj": module})
    collector.attach()
    module(torch.ones(1, 2, 4))
    collector.detach()
    before = collector.export()

    module(torch.full((1, 2, 4), 1000.0))
    after = collector.export()
    before["proj.imatrix_sum"].add_(100)

    assert torch.equal(
        after["proj.imatrix_sum"], torch.full((4,), 2.0)
    )
    assert after["proj.imatrix_count"].item() == 2


def test_context_exception_removes_hooks():
    module = nn.Linear(4, 4)
    collector = GPTQStatisticsCollector({"proj": module})

    with pytest.raises(RuntimeError, match="calibration failed"):
        with collector:
            module(torch.ones(1, 2, 4))
            raise RuntimeError("calibration failed")

    assert not collector.attached
    before = collector.export()
    module(torch.full((1, 2, 4), 1000.0))
    after = collector.export()
    assert torch.equal(
        before["proj.gptq_hessian"], after["proj.gptq_hessian"]
    )


def test_collector_allocates_statistics_only_on_first_forward():
    module = nn.Linear(1024, 4)
    collector = GPTQStatisticsCollector({"proj": module})

    collector.attach()
    assert collector._hessians == {}
    collector.detach()


def test_group_cleans_already_attached_collectors_when_attach_fails():
    good_module = nn.Linear(4, 4)
    bad_module = nn.Identity()
    good = GPTQStatisticsCollector({"good": good_module})
    bad = GPTQStatisticsCollector({"bad": bad_module})
    group = StatisticsCollectorGroup([good, bad])

    with pytest.raises(TypeError, match="does not have a weight"):
        group.attach()

    assert not group.attached
    assert not good.attached
    assert len(good_module._forward_pre_hooks) == 0


def make_store(tmp_path, tensor_names):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "tiny"}), encoding="utf-8"
    )
    save_file({"weight": torch.ones(1)}, checkpoint / "model.safetensors")
    manifest = StreamingRunManifest(
        source=fingerprint_checkpoint(checkpoint),
        recipe=RecipeInfo(fingerprint_json({"GPTQModifier": {}})),
        calibration=CalibrationInfo("dataset-v1", 1),
        sequential=SequentialInfo(("Linear",)),
        materializer=MaterializerInfo("cast", fingerprint_json({})),
        software=SoftwareInfo.from_versions({"torch": torch.__version__}),
    )
    store = ArtifactStore(tmp_path / "artifacts")
    store.initialize(
        manifest, normalized_recipe={"GPTQModifier": {}}, targets=["proj"]
    )
    metadata = TargetStatisticsMetadata(
        target_name="proj",
        target_index=0,
        algorithms=("gptq", "imatrix"),
        tensor_names=tuple(sorted(tensor_names)),
        source_tensor_fingerprints=(("proj.weight", "digest"),),
        completed=False,
    )
    return store, metadata


def test_export_persists_and_loads_in_a_fresh_process(tmp_path):
    module = nn.Linear(4, 4)
    modules = {"proj": module}
    group = StatisticsCollectorGroup(
        [GPTQStatisticsCollector(modules), IMatrixStatisticsCollector(modules)]
    )
    with group:
        module(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))
    statistics = group.export()
    store, metadata = make_store(tmp_path, statistics)
    store.commit_target(metadata, statistics)

    script = """
import json
import sys
from llmcompressor.streaming import ArtifactStore
stats = ArtifactStore(sys.argv[1]).load_target_statistics(0)
summary = {
    name: [list(tensor.shape), str(tensor.dtype), float(tensor.sum())]
    for name, tensor in stats.items()
}
print(json.dumps(summary, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(store.root)],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(result.stdout.strip().splitlines()[-1])

    assert set(loaded) == set(statistics)
    for name, tensor in statistics.items():
        assert loaded[name] == [
            list(tensor.shape),
            str(tensor.dtype),
            float(tensor.sum()),
        ]
