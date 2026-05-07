from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import torch

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.basic.pipeline import BasicPipeline
from llmcompressor.pipelines.sequential.pipeline import SequentialPipeline


def _make_fake_session(modifier_names=()):
    modifiers = [type(name, (), {})() for name in modifier_names]
    return SimpleNamespace(
        state=SimpleNamespace(
            loss_masks=None,
            current_batch_idx=None,
            sequential_prefetch=False,
        ),
        lifecycle=SimpleNamespace(recipe=SimpleNamespace(modifiers=modifiers)),
    )


class _BasicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids):
        self.calls += 1
        return input_ids


class _FakeSubgraph:
    def __init__(self, input_names, consumed_names, output_name, transform):
        self.input_names = input_names
        self.consumed_names = consumed_names
        self._output_name = output_name
        self._transform = transform

    def forward(self, _model, **inputs):
        return {self._output_name: self._transform(inputs)}

    def submodules(self, model):
        return [model]


class _FakeActivations:
    def __init__(self, batches):
        self._store = {idx: dict(batch) for idx, batch in enumerate(batches)}

    @classmethod
    def from_dataloader(cls, dataloader, *_args):
        return cls(list(dataloader))

    def iter(self, input_names):
        for idx in range(len(self._store)):
            yield {name: self._store[idx][name] for name in input_names}

    def iter_prefetch(self, input_names):
        yield from self.iter(input_names)

    def update(self, batch_idx, output):
        self._store[batch_idx].update(output)

    def delete(self, batch_idx, names):
        for name in names:
            self._store[batch_idx].pop(name, None)

    def fetch(self, batch_idx, names):
        return {name: self._store[batch_idx].get(name) for name in names}


def test_basic_pipeline_qac_true_keeps_quantization_enabled(monkeypatch):
    from llmcompressor.pipelines.basic import pipeline as basic_pipeline

    disable_quantization_calls = []

    @contextmanager
    def _record_disable_quantization(_model):
        disable_quantization_calls.append(True)
        yield

    monkeypatch.setattr(basic_pipeline, "active_session", lambda: _make_fake_session())
    monkeypatch.setattr(basic_pipeline, "dispatch_model", lambda _model: None)
    monkeypatch.setattr(basic_pipeline, "get_execution_device", lambda _model: "cpu")
    monkeypatch.setattr(basic_pipeline, "tensors_to_device", lambda batch, _device: batch)
    monkeypatch.setattr(basic_pipeline, "calibration_forward_context", lambda _model: nullcontext())
    monkeypatch.setattr(basic_pipeline, "DisableQuantization", _record_disable_quantization)
    monkeypatch.setattr(
        basic_pipeline.LifecycleCallbacks,
        "calibration_epoch_start",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        basic_pipeline.LifecycleCallbacks,
        "calibration_epoch_end",
        lambda **_kwargs: None,
    )

    model = _BasicModel()
    dataloader = [{"input_ids": torch.tensor([1])}]
    dataset_args = SimpleNamespace(
        use_loss_mask=False,
        quantization_aware_calibration=True,
    )

    BasicPipeline()(model, dataloader, dataset_args)

    assert model.calls == 1
    assert disable_quantization_calls == []


def test_basic_pipeline_qac_false_disables_quantization(monkeypatch):
    from llmcompressor.pipelines.basic import pipeline as basic_pipeline

    disable_quantization_calls = []

    @contextmanager
    def _record_disable_quantization(_model):
        disable_quantization_calls.append(True)
        yield

    monkeypatch.setattr(basic_pipeline, "active_session", lambda: _make_fake_session())
    monkeypatch.setattr(basic_pipeline, "dispatch_model", lambda _model: None)
    monkeypatch.setattr(basic_pipeline, "get_execution_device", lambda _model: "cpu")
    monkeypatch.setattr(basic_pipeline, "tensors_to_device", lambda batch, _device: batch)
    monkeypatch.setattr(basic_pipeline, "calibration_forward_context", lambda _model: nullcontext())
    monkeypatch.setattr(basic_pipeline, "DisableQuantization", _record_disable_quantization)
    monkeypatch.setattr(
        basic_pipeline.LifecycleCallbacks,
        "calibration_epoch_start",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        basic_pipeline.LifecycleCallbacks,
        "calibration_epoch_end",
        lambda **_kwargs: None,
    )

    model = _BasicModel()
    dataloader = [{"input_ids": torch.tensor([1])}]
    dataset_args = SimpleNamespace(
        use_loss_mask=False,
        quantization_aware_calibration=False,
    )

    BasicPipeline()(model, dataloader, dataset_args)

    assert disable_quantization_calls == [True]


def test_sequential_pipeline_uses_propagate_error_without_flipping_qac(monkeypatch):
    from llmcompressor.pipelines.sequential import pipeline as sequential_pipeline

    hook_state = {"hooks_disabled": False}

    @contextmanager
    def _disable_hooks():
        hook_state["hooks_disabled"] = True
        try:
            yield
        finally:
            hook_state["hooks_disabled"] = False

    def _run_case(propagate_error, quantization_aware_calibration):
        disable_quantization_calls = []
        second_subgraph_inputs = []

        @contextmanager
        def _record_disable_quantization(_model):
            disable_quantization_calls.append(True)
            yield

        subgraphs = [
            _FakeSubgraph(
                ["x"],
                ["x"],
                "y",
                lambda inputs: torch.tensor(
                    [10 if hook_state["hooks_disabled"] else 1],
                    dtype=inputs["x"].dtype,
                ),
            ),
            _FakeSubgraph(
                ["y"],
                ["y"],
                "z",
                lambda inputs: second_subgraph_inputs.append(int(inputs["y"].item()))
                or inputs["y"],
            ),
        ]

        monkeypatch.setattr(
            sequential_pipeline,
            "active_session",
            lambda: _make_fake_session(),
        )
        monkeypatch.setattr(
            sequential_pipeline,
            "set_onload_device",
            lambda _model, _device: None,
        )
        monkeypatch.setattr(
            sequential_pipeline,
            "infer_sequential_targets",
            lambda _model, _targets: ["DummyLayer"],
        )
        monkeypatch.setattr(
            sequential_pipeline,
            "trace_subgraphs",
            lambda *_args, **_kwargs: subgraphs,
        )
        monkeypatch.setattr(
            sequential_pipeline,
            "calibration_forward_context",
            lambda _model: nullcontext(),
        )
        monkeypatch.setattr(
            sequential_pipeline,
            "DisableQuantization",
            _record_disable_quantization,
        )
        monkeypatch.setattr(
            sequential_pipeline,
            "disable_offloading",
            lambda: nullcontext(),
        )
        monkeypatch.setattr(
            sequential_pipeline,
            "get_main_device",
            lambda: "cpu",
        )
        monkeypatch.setattr(
            sequential_pipeline.IntermediatesCache,
            "from_dataloader",
            _FakeActivations.from_dataloader,
        )
        monkeypatch.setattr(
            sequential_pipeline.HooksMixin,
            "disable_hooks",
            lambda: _disable_hooks(),
        )
        monkeypatch.setattr(
            sequential_pipeline.LifecycleCallbacks,
            "calibration_epoch_start",
            lambda **_kwargs: None,
        )
        monkeypatch.setattr(
            sequential_pipeline.LifecycleCallbacks,
            "sequential_epoch_end",
            lambda _modules, **_kwargs: None,
        )
        monkeypatch.setattr(
            sequential_pipeline.LifecycleCallbacks,
            "calibration_epoch_end",
            lambda **_kwargs: None,
        )

        dataset_args = SimpleNamespace(
            sequential_offload_device="cpu",
            sequential_targets=None,
            tracing_ignore=None,
            sequential_targets_per_subgraph=1,
            quantization_aware_calibration=quantization_aware_calibration,
            use_loss_mask=False,
            sequential_prefetch=False,
            propagate_error=propagate_error,
        )
        dataloader = [{"x": torch.tensor([0])}]

        SequentialPipeline()(torch.nn.Identity(), dataloader, dataset_args)

        return disable_quantization_calls, second_subgraph_inputs

    disable_calls, propagated_inputs = _run_case(
        propagate_error=True,
        quantization_aware_calibration=True,
    )
    assert disable_calls == []
    assert propagated_inputs == [10, 10]

    disable_calls, propagated_inputs = _run_case(
        propagate_error=False,
        quantization_aware_calibration=False,
    )
    assert disable_calls == [True]
    assert propagated_inputs == [1]


def test_dataset_arguments_keep_propagate_error_default_true():
    assert DatasetArguments.__dataclass_fields__["propagate_error"].default is True