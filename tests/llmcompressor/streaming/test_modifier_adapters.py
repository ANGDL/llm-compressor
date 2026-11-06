from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from safetensors import safe_open
from torch.utils.data import DataLoader
from transformers import Qwen3Config, Qwen3ForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import QuIPModifier, SpinQuantModifier
from llmcompressor.recipe import Recipe
from llmcompressor.streaming import streaming_oneshot
from llmcompressor.streaming.admission import estimate_modifier_memory
from llmcompressor.streaming.modifier_adapters import (
    AutoRoundStreamingAdapter,
    get_modifier_adapter,
    target_rng_scope,
)
from llmcompressor.streaming.modifier_contract import (
    ModifierContractError,
    build_modifier_contracts,
)
from llmcompressor.streaming.pretrained import _recipe_quantizer


def _data(count=4):
    return DataLoader([
        {
            "input_ids": torch.tensor([1, 2, 3, index + 4]),
            "attention_mask": torch.ones(4, dtype=torch.long),
        }
        for index in range(count)
    ], batch_size=1)


def _checkpoint(tmp_path, dtype=torch.float32):
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=32, hidden_size=64, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, max_position_embeddings=32, tie_word_embeddings=True,
    )
    checkpoint = tmp_path / "checkpoint"
    Qwen3ForCausalLM(config).to(dtype).save_pretrained(
        checkpoint, safe_serialization=True
    )
    return checkpoint


def _rtn():
    return QuantizationModifier(
        config_groups={"local": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4, "type": "int", "strategy": "group",
                "group_size": 8, "symmetric": True,
            },
        }},
        ignore=["lm_head"],
    )


def _recipe(kind):
    if kind == "qad":
        return [
            _rtn(),
            QADModifier(num_epochs=1, lr=1e-3, validation_fraction=0.25),
        ]
    if kind == "autosmooth":
        return [
            AutoSmoothModifier(
                mappings=[{
                    "smooth_layer": "re:.*input_layernorm",
                    "balance_layers": [
                        "re:.*self_attn.q_proj", "re:.*self_attn.k_proj",
                        "re:.*self_attn.v_proj",
                    ],
                }],
                n_grid=2, duo_scaling=False,
            ),
            _rtn(),
        ]
    autoround = pytest.importorskip("llmcompressor.modifiers.autoround")
    return [
        autoround.AutoRoundModifier(
            config_groups={"local": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4, "type": "int", "strategy": "group",
                    "group_size": 8, "symmetric": True,
                },
            }},
            ignore=["lm_head"], iters=2, batch_size=1,
            enable_torch_compile=False,
        )
    ]


def _run(
    checkpoint, tmp_path, kind, name, *, checkpoint_progress=False,
    recipe=None, dataloader=None, device="cpu", dtype=torch.float32,
):
    return streaming_oneshot(
        model=checkpoint, dataset=_data() if dataloader is None else dataloader,
        dataset_fingerprint="e5-fixed-batches",
        recipe=_recipe(kind) if recipe is None else recipe,
        output_dir=tmp_path / name,
        work_dir=tmp_path / f"{name}-work", device=device,
        target_dtype=dtype, num_calibration_samples=4,
        max_seq_length=4, seed=17, checkpoint_progress=checkpoint_progress,
    )


def _tensors(output):
    result = {}
    for shard in Path(output).glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as reader:
            result.update({name: reader.get_tensor(name) for name in reader.keys()})
    return result


def test_e5_contracts_fingerprint_algorithm_ownership():
    qad = build_modifier_contracts([QADModifier()])[0]
    smooth = build_modifier_contracts([AutoSmoothModifier()])[0]

    assert qad.streaming_adapter == "qad-local-training-v1"
    assert qad.propagation_mode == "post-update-quantized"
    assert not qad.calibration_outputs_reusable
    assert smooth.modifies_norms
    assert smooth.output_ownership == "pipeline"
    assert "streaming_adapter" in qad.to_dict()


@pytest.mark.parametrize("modifier", [SpinQuantModifier(), QuIPModifier()])
def test_global_and_online_transforms_fail_with_explicit_scope(modifier):
    recipe = Recipe.create_instance([modifier, _rtn()])
    with pytest.raises(ModifierContractError, match="norm fusion"):
        _recipe_quantizer(recipe)


@pytest.mark.parametrize("kind", ["qad", "autosmooth"])
def test_e5_rejects_modifier_reordering(kind):
    with pytest.raises(ValueError, match="requires|subsequent"):
        recipe = Recipe.create_instance(list(reversed(_recipe(kind))))
        _recipe_quantizer(recipe)


def test_target_rng_is_replayable_and_preserves_caller_state():
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    samples = []
    for _attempt in range(2):
        with target_rng_scope(17, 1, torch.device("cpu")):
            samples.append((random.random(), np.random.rand(), torch.rand(1).item()))

    assert samples[0] == samples[1]
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_target_rng_restores_all_initialized_cuda_generators_on_failure():
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible accelerator is unavailable")
    torch.cuda.init()
    states = [generator.get_state() for generator in torch.cuda.default_generators]
    with pytest.raises(RuntimeError, match="algorithm failed"):
        with target_rng_scope(29, 1, torch.device("cuda:0")):
            torch.manual_seed(1234)
            raise RuntimeError("algorithm failed")
    for generator, expected in zip(torch.cuda.default_generators, states):
        assert torch.equal(generator.get_state(), expected)


@pytest.mark.parametrize("kind", ["qad", "autosmooth", "autoround"])
def test_e5_executes_two_targets_and_reloads_checkpoint(tmp_path, monkeypatch, kind):
    searched = []
    trained = {}
    original_search = AutoSmoothModifier._compute_best_scale

    def record_search(modifier, mapping, outputs):
        scales = original_search(modifier, mapping, outputs)
        searched.append((mapping.smooth_name, scales.detach().cpu().clone()))
        return scales

    monkeypatch.setattr(AutoSmoothModifier, "_compute_best_scale", record_search)
    original_after_update = AutoRoundStreamingAdapter.after_update

    def record_trained(adapter, modules):
        original_after_update(adapter, modules)
        for module in modules:
            name = getattr(module, "_tmp_name", "")
            if name.endswith("self_attn.q_proj"):
                trained[name] = module.weight.detach().cpu().clone()

    monkeypatch.setattr(AutoRoundStreamingAdapter, "after_update", record_trained)
    checkpoint = _checkpoint(tmp_path)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    output = _run(checkpoint, tmp_path, kind, "output")
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == workspace
    tensors = _tensors(output)

    assert (output / "FINALIZED").is_file()
    for index in range(2):
        prefix = f"model.layers.{index}.self_attn.q_proj"
        assert tensors[f"{prefix}.weight_scale"].isfinite().all()
        assert tensors[f"{prefix}.weight_scale"].numel() > 0
    model = Qwen3ForCausalLM.from_pretrained(
        output, local_files_only=True, dtype=torch.float32
    ).eval()
    with torch.no_grad():
        assert model(**next(iter(_data()))).logits.isfinite().all()
    if kind == "autoround":
        assert len(trained) == 2
        for name, weight in trained.items():
            torch.testing.assert_close(
                model.get_submodule(name).weight, weight, rtol=3e-3, atol=3e-5
            )
    saved_recipe = yaml.safe_load((output / "recipe.yaml").read_text())
    assert saved_recipe
    if kind == "autosmooth":
        assert [name for name, _scales in searched] == [
            f"model.layers.{index}.input_layernorm" for index in range(2)
        ]
        for name, scales in searched:
            torch.testing.assert_close(tensors[f"{name}.weight"], 1 / scales)


@pytest.mark.parametrize("kind", ["qad", "autosmooth"])
def test_e5_matches_sequential_oneshot(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(
        "llmcompressor.pipelines.sequential.pipeline.resolve_execution_device",
        lambda: torch.device("cpu"),
    )
    checkpoint = _checkpoint(tmp_path)
    baseline = tmp_path / "baseline"
    oneshot(
        model=str(checkpoint), dataset=_data(), recipe=_recipe(kind),
        output_dir=baseline, num_calibration_samples=4, max_seq_length=4,
        precision="float32", pipeline="sequential",
        sequential_targets=["Qwen3DecoderLayer"],
    )
    streamed = _run(checkpoint, tmp_path, kind, "streamed")

    expected = _tensors(baseline)
    actual = _tensors(streamed)
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(
            actual[name], expected[name], rtol=3e-3, atol=3e-5, msg=name
        )


@pytest.mark.parametrize("kind", ["qad", "autosmooth", "autoround"])
@pytest.mark.parametrize("interruption", ["after_update", "after_commit"])
def test_e5_resume_after_committed_target_matches_uninterrupted(
    tmp_path, monkeypatch, kind, interruption
):
    import llmcompressor.streaming.pipeline as pipeline
    from llmcompressor.streaming.checkpoint.writer import _Transaction

    checkpoint = _checkpoint(tmp_path)
    expected = _run(
        checkpoint, tmp_path, kind, "baseline", checkpoint_progress=True
    )
    original_commit = _Transaction.commit

    def fail_after_commit(transaction):
        original_commit(transaction)
        if transaction.transaction_id == "subgraph-00000":
            raise RuntimeError("e5 injected interruption after commit")

    original_update = pipeline._modifier_update
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    def fail_after_update(*args, **kwargs):
        original_update(*args, **kwargs)
        raise RuntimeError("e5 injected interruption after update")

    with monkeypatch.context() as patch:
        if interruption == "after_commit":
            patch.setattr(_Transaction, "commit", fail_after_commit)
        else:
            patch.setattr(pipeline, "_modifier_update", fail_after_update)
        with pytest.raises(RuntimeError, match="e5 injected interruption"):
            _run(checkpoint, tmp_path, kind, "resumed", checkpoint_progress=True)
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == workspace
    actual = _run(checkpoint, tmp_path, kind, "resumed", checkpoint_progress=True)
    expected_tensors = _tensors(expected)
    actual_tensors = _tensors(actual)
    assert actual_tensors.keys() == expected_tensors.keys()
    for name in expected_tensors:
        torch.testing.assert_close(
            actual_tensors[name], expected_tensors[name], rtol=0, atol=0, msg=name
        )


def test_qad_adapter_reports_training_and_cpu_cache_memory():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, device="meta"))
    quantizer = _rtn()
    quantizer.initialize_quantization(model)
    qad = QADModifier()
    qad._seq_target_names = {model: "root"}
    estimate = estimate_modifier_memory([qad], model.modules(), num_tokens=16)

    assert estimate.held_bytes > 0
    assert estimate.scratch_bytes > 0
    assert estimate.host_bytes > 0
    assert get_modifier_adapter(qad).minimum_batches == 2


def test_qad_rejects_insufficient_validation_data(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(ModifierContractError, match="at least two batches"):
        _run(checkpoint, tmp_path, "qad", "output", dataloader=_data(1))
    assert not (tmp_path / "output" / "FINALIZED").exists()


def test_autosmooth_rejects_cross_target_mapping(tmp_path):
    recipe = [
        AutoSmoothModifier(mappings=[{
            "smooth_layer": "model.layers.0.input_layernorm",
            "balance_layers": ["model.layers.1.self_attn.q_proj"],
        }]),
        _rtn(),
    ]
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(ModifierContractError, match="crosses.*target boundary"):
        _run(checkpoint, tmp_path, "autosmooth", "output", recipe=recipe)
    assert not (tmp_path / "output" / "FINALIZED").exists()


def test_autoround_rejects_external_layer_skipping():
    modifier = _recipe("autoround")[0]
    model = torch.nn.Sequential(torch.nn.Linear(16, 16, device="meta"))
    modifier.initialize_quantization(model)
    with pytest.raises(ModifierContractError, match="divisible by 32"):
        get_modifier_adapter(modifier).validate_targets(model, ["0"])


def test_autoround_rejects_missing_output_quantization():
    modifier = _recipe("autoround")[0]
    model = torch.nn.Sequential(torch.nn.Linear(32, 32))
    modifier.initialize_quantization(model)
    adapter = get_modifier_adapter(modifier)
    adapter.before_target(list(model.modules()))
    del model[0].quantization_scheme
    with pytest.raises(ModifierContractError, match="unexpectedly skipped"):
        adapter.after_update(list(model.modules()))


def test_autoround_int8_checkpoint_reloads(tmp_path):
    recipe = _recipe("autoround")
    recipe[0].config_groups["local"].weights.num_bits = 8
    output = _run(
        _checkpoint(tmp_path), tmp_path, "autoround", "output", recipe=recipe
    )
    tensors = _tensors(output)
    assert tensors["model.layers.0.self_attn.q_proj.weight_scale"].isfinite().all()
    model = Qwen3ForCausalLM.from_pretrained(
        output, local_files_only=True, dtype=torch.float32
    ).eval()
    scheme = model.model.layers[0].self_attn.q_proj.quantization_scheme
    assert scheme.weights.num_bits == 8
    with torch.no_grad():
        assert model(**next(iter(_data()))).logits.isfinite().all()


@pytest.mark.parametrize("kind", ["qad", "autosmooth", "autoround"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_e5_executes_on_accelerator(tmp_path, kind, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible accelerator is unavailable")
    checkpoint = _checkpoint(tmp_path, dtype)
    output = _run(
        checkpoint, tmp_path, kind, "output", device="cuda:0", dtype=dtype
    )
    assert (output / "FINALIZED").is_file()
    tensors = _tensors(output)
    assert tensors["model.layers.1.self_attn.q_proj.weight_scale"].isfinite().all()
    assert tensors["model.layers.0.input_layernorm.weight"].dtype == dtype
    model = Qwen3ForCausalLM.from_pretrained(
        output, local_files_only=True, dtype=dtype
    ).eval()
    with torch.no_grad():
        assert model(**next(iter(_data()))).logits.isfinite().all()


@pytest.mark.parametrize("kind", ["qad", "autosmooth"])
def test_e5_rejects_shared_parameters_outside_target(tmp_path, kind):
    checkpoint = _checkpoint(tmp_path)
    model = Qwen3ForCausalLM.from_pretrained(checkpoint, local_files_only=True)
    model.model.layers[1].self_attn.q_proj.weight = (
        model.model.layers[0].self_attn.q_proj.weight
    )
    recipe = _recipe(kind)
    quantizer = next(
        modifier for modifier in recipe if type(modifier) is QuantizationModifier
    )
    quantizer.initialize_quantization(model)
    modifier = next(
        modifier for modifier in recipe if get_modifier_adapter(modifier) is not None
    )
    targets = ["model.layers.0", "model.layers.1"]
    if kind == "qad":
        modifier._seq_target_names = {
            model.get_submodule(name): name for name in targets
        }
    else:
        modifier._set_resolved_mappings(model)
    with pytest.raises(ModifierContractError, match="shared outside target"):
        get_modifier_adapter(modifier).validate_targets(model, targets)
