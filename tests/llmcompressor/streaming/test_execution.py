import operator
from dataclasses import dataclass, replace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.fx import Graph

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.sequential.helpers import Subgraph, trace_consumed_names
from llmcompressor.pipelines.sequential.plan import SequentialExecutionPlan
from llmcompressor.recipe import Recipe
from llmcompressor.streaming import (
    ArtifactCompatibilityError,
    CastWeightMaterializer,
    SafetensorsWeightSource,
    StreamingDTypePolicy,
    TracedBoundaryAdapter,
    build_meta_model,
    describe_streaming_execution,
    fingerprint_streaming_value,
    fingerprint_streaming_values,
)
from llmcompressor.streaming.checkpoint.writer import _Transaction
from llmcompressor.streaming.execution import (
    fingerprint_execution_graph,
    freeze_calibration_inputs,
    streaming_input_contract,
)
from llmcompressor.streaming.loading import SubgraphWeightSession, TargetWeightLoader
from llmcompressor.streaming.pipeline import run_subgraph_streaming_pipeline


@dataclass
class _CalibrationMetadata:
    frames: int
    gain: float


class BranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.vision = nn.Linear(4, 4).to(torch.bfloat16)
        self.projector = nn.Linear(4, 4).to(torch.bfloat16)
        self.audio = nn.Linear(4, 4).to(torch.bfloat16)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.mtp = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 8, bias=False)
        self.head.weight = self.embed.weight
        self.aux_bias = nn.Parameter(torch.randn(4))

    def streaming_execution_metadata(self):
        return {
            "modalities": ("text", "image", "audio"),
            "branches": {
                "decoder": ("prefix", "layers.0", "layers.1"),
                "mtp": ("layers.0", "mtp"),
            },
            "runtime_dependencies": [{"stage": "mtp", "parameters": ("aux_bias",)}],
        }


def _subgraph(graph, inputs, output):
    graph.output(output)
    return Subgraph(graph=graph, input_names=set(inputs), consumed_names=set())


def _adapter(checkpoint):
    model = build_meta_model(BranchModel)
    source = SafetensorsWeightSource(checkpoint)
    policy = StreamingDTypePolicy.for_quantized_modules(
        torch.float32, ("layers.0", "layers.1", "mtp")
    )
    prefix = Graph()
    tokens = prefix.placeholder("input_ids")
    pixels = prefix.placeholder("pixel_values")
    audio = prefix.placeholder("audio_values")
    text = prefix.call_module("embed", (tokens,))
    image = prefix.call_module("projector", (prefix.call_module("vision", (pixels,)),))
    image = prefix.call_method("mean", (image, 1))
    image = prefix.call_method("unsqueeze", (image, 1))
    image = prefix.call_method("float", (image,))
    audio = prefix.call_module("audio", (audio,))
    audio = prefix.call_method("mean", (audio, 1))
    audio = prefix.call_method("unsqueeze", (audio, 1))
    audio = prefix.call_method("float", (audio,))
    hidden = prefix.call_function(operator.add, (text, image))
    hidden = prefix.call_function(operator.add, (hidden, audio))
    prefix = _subgraph(
        prefix,
        ("input_ids", "pixel_values", "audio_values"),
        {"hidden": hidden, "image": image},
    )
    first = Graph()
    hidden = first.placeholder("hidden")
    hidden = first.call_module("layers.0", (hidden,))
    first = _subgraph(first, ("hidden",), {"first": hidden})
    second = Graph()
    hidden = second.placeholder("first")
    hidden = second.call_module("layers.1", (hidden,))
    second = _subgraph(second, ("first",), {"second": hidden})
    third = Graph()
    hidden = third.placeholder("second")
    skip = third.placeholder("first")
    image = third.placeholder("image")
    hidden = third.call_function(operator.add, (hidden, skip))
    hidden = third.call_function(operator.add, (hidden, image))

    def auxiliary_bias(value):
        return value + model.aux_bias

    hidden = third.call_function(auxiliary_bias, (hidden,))
    hidden = third.call_module("norm", (hidden,))
    hidden = third.call_module("mtp", (hidden,))
    logits = third.call_module("head", (hidden,))
    third = _subgraph(third, ("second", "first", "image"), {"logits": logits})
    subgraphs = (first, second, third)
    trace_consumed_names(subgraphs)
    plan = SequentialExecutionPlan(
        subgraphs=(prefix, *subgraphs),
        target_names=("layers.0", "layers.1", "mtp"),
        target_subgraph_indices=(1, 2, 3),
    )
    return TracedBoundaryAdapter(
        model,
        plan,
        TargetWeightLoader(model, source, dtype_policy=policy),
        SubgraphWeightSession(model, source, dtype_policy=policy),
        torch.device("cpu"),
        torch.float32,
        subgraphs,
        execution=describe_streaming_execution(model, plan.target_names),
    )


def _fixture(tmp_path):
    torch.manual_seed(41)
    reference = BranchModel()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "tiny"}')
    save_file(
        {
            name: value.detach().clone()
            for name, value in reference.state_dict().items()
        },
        checkpoint / "model.safetensors",
    )
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.randn(1, length, 4, dtype=torch.bfloat16),
            "audio_values": torch.randn(1, length + 1, 4, dtype=torch.bfloat16),
        }
        for length in (2, 5)
    ]
    return checkpoint, reference, batches


def _run(adapter, checkpoint, work, batches):
    return run_subgraph_streaming_pipeline(
        adapter=adapter,
        checkpoint=checkpoint,
        work_dir=work,
        calibration_batches=batches,
        recipe=Recipe.create_instance(
            QuantizationModifier(
                config_groups={
                    "weights": {
                        "targets": ["layers.0", "layers.1", "mtp"],
                        "weights": {"num_bits": 8, "strategy": "channel"},
                    }
                }
            )
        ),
        dataset_fingerprint="b" * 64,
        materializer=CastWeightMaterializer(),
        device=torch.device("cpu"),
        target_dtype=torch.float32,
        num_samples=2,
        max_seq_length=3,
        seed=41,
        checkpoint_progress=True,
    )


def test_multimodal_skip_mtp_runtime_parameter_and_shared_head(tmp_path):
    checkpoint, reference, batches = _fixture(tmp_path)
    adapter = _adapter(checkpoint)
    boundaries = list(adapter.calibration_boundaries(batches))
    assert ("head.weight", "embed.weight") in adapter.execution.shared_tensors
    for batch, boundary in zip(batches, boundaries):
        image = (
            reference.projector(reference.vision(batch["pixel_values"]))
            .mean(1)
            .unsqueeze(1)
            .float()
        )
        audio = reference.audio(batch["audio_values"]).mean(1).unsqueeze(1).float()
        hidden = reference.embed(batch["input_ids"]) + image + audio
        first = reference.layers[0](hidden)
        second = reference.layers[1](first)
        expected = reference.head(
            reference.mtp(reference.norm(second + first + image + reference.aux_bias))
        )
        value = boundary
        for index, name in enumerate(adapter.targets):
            with adapter.weight_session.loaded_plan(
                adapter.plan_target(index), device="cpu", dtype=torch.float32
            ):
                target = adapter.model.get_submodule(name)
                target._streaming_target_name = name
                subgraph = adapter.target_subgraphs[index]
                output = subgraph.forward(
                    adapter.model, **{key: value[key] for key in subgraph.input_names}
                )
                value = {**value, **output}
                for key in subgraph.consumed_names:
                    value.pop(key, None)
                if index == 1:
                    assert "first" in value and "image" in value
        torch.testing.assert_close(value["logits"], expected)
    assert all(parameter.is_meta for parameter in adapter.model.parameters())


@pytest.mark.parametrize(
    "interruption", ["before_prefix", "after_prefix", "after_first_target"]
)
def test_prefix_transaction_resume_avoids_reencoding_and_preserves_static_dtype(
    tmp_path, monkeypatch, interruption
):
    checkpoint, reference, batches = _fixture(tmp_path)
    work = tmp_path / "work"
    original = _Transaction.commit

    def fail_after_prefix(transaction):
        if (
            interruption == "before_prefix"
            and transaction.transaction_id == "prefix-00000"
        ):
            raise RuntimeError("injected prefix interruption")
        original(transaction)
        selected = (
            "subgraph-00000" if interruption == "after_first_target" else "prefix-00000"
        )
        if transaction.transaction_id == selected:
            raise RuntimeError("injected prefix interruption")

    with monkeypatch.context() as patch:
        patch.setattr(_Transaction, "commit", fail_after_prefix)
        with pytest.raises(RuntimeError, match="prefix interruption"):
            _run(_adapter(checkpoint), checkpoint, work, batches)
    recovered = _adapter(checkpoint)

    def unexpected(*_args):
        raise AssertionError("Committed prefix must not execute again")

    if interruption != "before_prefix":
        monkeypatch.setattr(recovered, "calibration_boundaries", unexpected)
    _artifacts, staging, _fingerprint = _run(recovered, checkpoint, work, batches)
    actual = {}
    for shard in (staging / "shards").glob("*.safetensors"):
        actual.update(load_file(shard))
    for name, tensor in reference.state_dict().items():
        if name.startswith(("layers.", "mtp.")):
            continue
        assert actual[name].dtype == tensor.dtype
        torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0)
    assert (staging / "retirements/prefix-00000.json").is_file()
    assert not (staging / "transactions/prefix-00000/boundaries").exists()
    _artifacts, baseline, _fingerprint = _run(
        _adapter(checkpoint), checkpoint, tmp_path / "baseline", batches
    )
    expected = {}
    for shard in (baseline / "shards").glob("*.safetensors"):
        expected.update(load_file(shard))
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_committed_prefix_corruption_is_not_recomputed(tmp_path, monkeypatch):
    checkpoint, _reference, batches = _fixture(tmp_path)
    work = tmp_path / "work"
    original = _Transaction.commit

    def interrupt(transaction):
        original(transaction)
        if transaction.transaction_id == "prefix-00000":
            raise RuntimeError("prefix interruption")

    with monkeypatch.context() as patch:
        patch.setattr(_Transaction, "commit", interrupt)
        with pytest.raises(RuntimeError, match="interruption"):
            _run(_adapter(checkpoint), checkpoint, work, batches)
    payload = next((work / "staging/transactions/prefix-00000/boundaries").glob("*.pt"))
    data = bytearray(payload.read_bytes())
    data[-1] ^= 1
    payload.write_bytes(data)
    recovered = _adapter(checkpoint)
    monkeypatch.setattr(
        recovered,
        "calibration_boundaries",
        lambda *_args: pytest.fail("committed corruption must not reencode"),
    )
    with pytest.raises(RuntimeError, match="corrupt"):
        _run(recovered, checkpoint, work, batches)


def test_root_get_attr_loads_only_the_direct_parameter_and_current_target(tmp_path):
    checkpoint, _reference, _batches = _fixture(tmp_path)
    adapter = _adapter(checkpoint)
    graph = Graph()
    value = graph.placeholder("value")
    bias = graph.get_attr("aux_bias")
    value = graph.call_function(operator.add, (value, bias))
    value = graph.call_module("mtp", (value,))
    subgraph = _subgraph(graph, ("value",), {"value": value})
    plan = adapter.weight_session.plan(subgraph)
    with adapter.weight_session.loaded_plan(plan, device="cpu", dtype=torch.float32):
        assert not adapter.model.aux_bias.is_meta
        assert not adapter.model.mtp.weight.is_meta
        assert adapter.model.vision.weight.is_meta
        assert adapter.model.layers[0].weight.is_meta
    assert all(parameter.is_meta for parameter in adapter.model.parameters())


def test_runtime_dependency_cannot_reload_a_previous_quantized_target(tmp_path):
    checkpoint, _reference, _batches = _fixture(tmp_path)
    adapter = _adapter(checkpoint)
    adapter.execution = replace(
        adapter.execution, runtime_dependencies=(("mtp", ("layers.0",), ()),)
    )
    with pytest.raises(ValueError, match="propagated boundary"):
        adapter.plan_target(2)


@pytest.mark.parametrize("change", ["image", "audio", "processor", "topology"])
def test_resume_rejects_changed_nontext_identity(tmp_path, monkeypatch, change):
    checkpoint, _reference, batches = _fixture(tmp_path)
    work = tmp_path / "work"
    original = _Transaction.commit

    def interrupt(transaction):
        original(transaction)
        if transaction.transaction_id == "prefix-00000":
            raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(_Transaction, "commit", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            _run(_adapter(checkpoint), checkpoint, work, batches)
    adapter = _adapter(checkpoint)
    if change in ("image", "audio"):
        key = "pixel_values" if change == "image" else "audio_values"
        batches[-1][key].add_(1)
    elif change == "processor":
        adapter.execution = adapter.execution.with_inputs(
            fingerprint_streaming_values(enumerate(batches)),
            {"resize": 1024},
        )
    else:
        adapter.execution = replace(adapter.execution, graph_fingerprint="f" * 64)
    with pytest.raises(ArtifactCompatibilityError):
        _run(adapter, checkpoint, work, batches)


def test_fingerprint_is_framed_and_supports_scalar_bfloat16(tmp_path):
    assert fingerprint_streaming_value([1, 23]) != fingerprint_streaming_value([12, 3])
    assert fingerprint_streaming_value([1, [2]]) != fingerprint_streaming_value(
        [[1], 2]
    )
    value = torch.tensor(1.5, dtype=torch.bfloat16)
    assert fingerprint_streaming_value(value) == fingerprint_streaming_value(
        value.clone()
    )
    assert fingerprint_streaming_value(value) != fingerprint_streaming_value(
        value.float()
    )
    frozen = freeze_calibration_inputs(
        [{"pixels": torch.ones(1, length, 4)} for length in (2, 9)], tmp_path
    )
    assert len(frozen) == 2
    with pytest.raises(ValueError, match="branch changed"):
        freeze_calibration_inputs(
            [{"pixels": value, "use_audio": flag} for flag in (True, False)], tmp_path
        )


def test_graph_identity_ignores_placeholder_order_but_binds_operations():
    def graph(order, operation):
        result = Graph()
        inputs = {name: result.placeholder(name) for name in order}
        output = result.call_function(operation, (inputs["first"], inputs["second"]))
        return _subgraph(result, order, {"output": output})

    first = fingerprint_execution_graph([graph(("first", "second"), operator.add)])
    reordered = fingerprint_execution_graph([graph(("second", "first"), operator.add)])
    changed = fingerprint_execution_graph([graph(("first", "second"), operator.mul)])
    assert first == reordered
    assert first != changed


def test_numeric_metadata_varies_without_changing_native_types(tmp_path):
    batches = [
        {
            "pixels": torch.ones(1, length, 4),
            "metadata": {"size": [length, length + 1], "gain": gain},
            "auxiliary": _CalibrationMetadata(length, gain),
            "enabled": True,
            "mode": "image",
        }
        for length, gain in ((2, 0.5), (9, 1.25))
    ]
    frozen = freeze_calibration_inputs(batches, tmp_path)
    assert streaming_input_contract(batches[0]) == streaming_input_contract(batches[1])
    assert fingerprint_streaming_value(batches[0]) != fingerprint_streaming_value(
        batches[1]
    )
    for original, replayed in zip(batches, frozen):
        assert replayed["metadata"] == original["metadata"]
        assert replayed["auxiliary"] == original["auxiliary"]
        assert type(replayed["metadata"]["size"][0]) is int
        assert type(replayed["metadata"]["gain"]) is float


@pytest.mark.parametrize("changed", [False, 1, 1.0, None, "other", [1, 2]])
def test_input_contract_preserves_branch_flags_and_structure(tmp_path, changed):
    with pytest.raises(ValueError, match="branch changed"):
        freeze_calibration_inputs(
            [{"metadata": True}, {"metadata": changed}], tmp_path
        )


def test_numeric_input_contract_preserves_type():
    assert streaming_input_contract(1) != streaming_input_contract(1.0)
    assert streaming_input_contract(1) != streaming_input_contract(True)
    assert streaming_input_contract(1, specialize_numeric=True) != (
        streaming_input_contract(2, specialize_numeric=True)
    )


def test_transformers_llava_decoder_only_export_and_reload(tmp_path):
    from torch.utils.data import DataLoader
    from transformers import (
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
    )

    from llmcompressor.streaming import streaming_oneshot

    config = LlavaConfig(
        vision_config=CLIPVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
            projection_dim=8,
        ).to_dict(),
        text_config=LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        ).to_dict(),
        image_token_index=31,
        image_seq_length=4,
    )
    torch.manual_seed(19)
    reference = LlavaForConditionalGeneration(config).eval()
    checkpoint = tmp_path / "llava"
    reference.save_pretrained(checkpoint, safe_serialization=True)
    dataset = DataLoader(
        [
            {
                "input_ids": torch.tensor([1, 31, 31, 31, 31, token]),
                "attention_mask": torch.ones(6, dtype=torch.long),
                "pixel_values": torch.randn(3, 8, 8),
            }
            for token in (2, 3)
        ],
        batch_size=1,
    )
    output = streaming_oneshot(
        model=checkpoint,
        model_factory=LlavaForConditionalGeneration,
        dataset=dataset,
        dataset_fingerprint="llava-images",
        processor_identity={"processor": "tiny-clip", "image_size": 8},
        recipe=QuantizationModifier(
            config_groups={
                "decoder": {
                    "targets": [
                        "re:.*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
                    ],
                    "weights": {"num_bits": 8, "strategy": "channel"},
                }
            }
        ),
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        num_calibration_samples=2,
        max_seq_length=6,
        target_dtype=torch.float32,
        device="cpu",
        checkpoint_progress=True,
    )
    state = {}
    for shard in output.glob("*.safetensors"):
        state.update(load_file(shard))
    for name, value in load_file(checkpoint / "model.safetensors").items():
        if "vision_tower" in name or "multi_modal_projector" in name:
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)
    loaded = LlavaForConditionalGeneration.from_pretrained(
        output, dtype=torch.float32, local_files_only=True
    ).eval()
    with torch.no_grad():
        batch = next(iter(dataset))
        actual = loaded(**batch).logits
        expected = reference(**batch).logits
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.01)
