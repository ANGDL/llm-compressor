from __future__ import annotations

import json
from contextlib import contextmanager

import pytest
import torch
import yaml
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors import safe_open
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.pruning import (
    SparseGPTModifier,
    WandaPruningModifier,
)
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier, SmoothQuantModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.streaming import streaming_oneshot
from llmcompressor.streaming.loading import (
    SubgraphPrefetcher,
    SubgraphWeightSession,
)
from llmcompressor.streaming.output import prepare_quantized_tensor_for_save
from llmcompressor.streaming.pipeline import (
    _empty_device_cache,
    _stage_logger,
    _write_loaded_target_direct,
)


def _checkpoint_tensors(path):
    tensors = {}
    for shard in path.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as file:
            tensors.update(
                {name: file.get_tensor(name) for name in file.keys()}
            )
    return tensors


def _calibration_data():
    return DataLoader(
        [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.ones(4, dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([4, 3, 2, 1]),
                "attention_mask": torch.ones(4, dtype=torch.long),
            },
        ],
        batch_size=1,
    )


def _imatrix_recipe():
    return [
        IMatrixGatherer(
            ignore=["lm_head"], attach_by_initialize=False
        ),
        QuantizationModifier(
            scheme="W8A8",
            targets=["Linear"],
            weight_observer="imatrix_mse",
            ignore=["lm_head"],
        ),
    ]


def test_direct_target_write_does_not_retain_tensor(tmp_path):
    import gc
    import weakref

    class Loaded:
        model = torch.nn.Module()

        def state_tensors_under(self, _module_names):
            tensor = torch.ones(8)
            references.append(weakref.ref(tensor))
            yield "layer.weight", tensor

    class Writer:
        def write_shard(self, _shard_id, tensors, **_kwargs):
            assert set(tensors) == {"layer.weight"}

    references = []
    written = _write_loaded_target_direct(Writer(), Loaded(), "layer", "target-0")
    gc.collect()

    assert written == {"layer.weight"}
    assert references[0]() is None


def test_direct_target_write_reserves_before_w4a8_cpu_snapshot(monkeypatch):
    class Loaded:
        model = torch.nn.Module()
        model.layer = torch.nn.Linear(4, 2, bias=False)
        model.layer.quantization_scheme = QuantizationScheme(
            targets=["layer"],
            weights=QuantizationArgs(
                num_bits=4,
                type="int",
                strategy="channel",
                symmetric=True,
            ),
            input_activations=QuantizationArgs(
                num_bits=8,
                type="int",
                strategy="token",
                dynamic=True,
            ),
        )

        def state_tensors_under(self, _module_names):
            yield "layer.weight", torch.tensor(
                [[-8, -1, 0, 7], [1, 2, 3, 4]], dtype=torch.int8
            )

    class Writer:
        @staticmethod
        def estimate_snapshot_bytes(tensors):
            return sum(tensor.nbytes for tensor in tensors.values())

        def write_shard(self, _shard_id, tensors, **kwargs):
            assert tensors["layer.weight"].device.type == "cpu"
            assert kwargs["owned_cpu_tensors"] == {"layer.weight"}
            assert kwargs["reservation"] is reservation
            captured.update(
                {name: value.clone() for name, value in tensors.items()}
            )
            assert kwargs["quantized_modules"] == {"layer": "int-quantized"}

    class Reservation:
        def close(self):
            pass

    class Budget:
        def reserve(self, owner, required):
            assert owner == "checkpoint shard target-0"
            assert required == 12
            events.append("reserve")
            return reservation

    captured = {}
    events = []
    reservation = Reservation()
    original_prepare = prepare_quantized_tensor_for_save

    def record_prepare(*args, **kwargs):
        assert events == ["reserve"]
        events.append("pack")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        "llmcompressor.streaming.pipeline.prepare_quantized_tensor_for_save",
        record_prepare,
    )
    written = _write_loaded_target_direct(
        Writer(),
        Loaded(),
        "layer",
        "target-0",
        host_budget=Budget(),
    )

    assert written == {"layer.weight"}
    assert events == ["reserve", "pack"]
    assert captured["layer.weight"].shape == (2, 2)
    assert captured["layer.weight"].tolist() == [[-8, 112], [33, 67]]


@pytest.mark.parametrize("async_save", [False, True])
def test_pretrained_streaming_packs_w4a8_final_shard(tmp_path, async_save):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    checkpoint = tmp_path / "checkpoint"
    Qwen3ForCausalLM(config).save_pretrained(
        checkpoint, safe_serialization=True
    )
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            strategy="channel",
            symmetric=True,
            observer="imatrix_mse",
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type="int",
            strategy="token",
            dynamic=True,
        ),
    )

    output = streaming_oneshot(
        model=checkpoint,
        dataset=_calibration_data(),
        dataset_fingerprint="7" * 64,
        recipe=[
            IMatrixGatherer(ignore=["lm_head"]),
            QuantizationModifier(
                config_groups={"group_0": scheme}, ignore=["lm_head"]
            ),
        ],
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        num_calibration_samples=2,
        max_seq_length=4,
        target_dtype=torch.float32,
        async_save=async_save,
    )

    tensors = _checkpoint_tensors(output)
    weight = tensors["model.layers.0.self_attn.q_proj.weight"]
    assert weight.dtype is torch.int8
    assert weight.shape == (8, 4)
    saved_config = json.loads((output / "config.json").read_text())
    assert saved_config["quantization_config"]["format"] == "int-quantized"
    assert not (tmp_path / "work/staging/transactions").exists()


def test_empty_device_cache_dispatches_to_execution_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))

    _empty_device_cache(torch.device("cuda:0"))
    _empty_device_cache(torch.device("cpu"))

    assert calls == ["cuda"]


def test_stage_logger_only_monitors_execution_device():
    model = torch.nn.Module()

    assert _stage_logger(model, "cuda-stage", torch.device("cuda:3")).device_ids == (
        3,
    )
    assert _stage_logger(model, "cpu-stage", torch.device("cpu")).device_ids == ()


def test_pretrained_streaming_writes_each_subgraph_as_final_shard(
    tmp_path, monkeypatch
):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    torch.manual_seed(7)
    checkpoint = tmp_path / "checkpoint"
    Qwen3ForCausalLM(config).save_pretrained(
        checkpoint, safe_serialization=True
    )
    dataset = DataLoader(
        [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.ones(4, dtype=torch.long),
            }
        ],
        batch_size=1,
    )
    monkeypatch.setattr(
        "llmcompressor.streaming.pipeline.StreamingCheckpointWriter."
        "assemble_shards",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default streaming path must not assemble shards")
        ),
    )
    active_installation = False
    installed_devices = []
    original_installed = SubgraphWeightSession.installed

    @contextmanager
    def record_installation(session, prepared):
        nonlocal active_installation
        assert not active_installation
        with original_installed(session, prepared) as loaded:
            active_installation = True
            try:
                if loaded.module_names:
                    installed_devices.append(prepared.device)
                yield loaded
            finally:
                active_installation = False

    monkeypatch.setattr(SubgraphWeightSession, "installed", record_installation)
    output = streaming_oneshot(
        model=checkpoint,
        dataset=dataset,
        dataset_fingerprint="d" * 64,
        recipe=[
            IMatrixGatherer(ignore=["lm_head"]),
            QuantizationModifier(
                scheme="W8A8",
                targets=["Linear"],
                weight_observer="imatrix_mse",
                ignore=["lm_head"],
            ),
        ],
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        num_calibration_samples=1,
        max_seq_length=4,
        target_dtype=torch.float32,
        async_save=True,
    )

    assert (output / "FINALIZED").is_file()
    saved_recipe = yaml.safe_load((output / "recipe.yaml").read_text())
    saved_modifiers = saved_recipe["default_stage"]["default_modifiers"]
    assert set(saved_modifiers) == {
        "IMatrixGatherer",
        "QuantizationModifier",
    }
    staging = tmp_path / "work" / "staging"
    assert not (staging / "transactions").exists()
    assert not list(staging.rglob("*.bin"))
    output_shards = {path.name for path in output.glob("*.safetensors")}
    assert "model-subgraph-00000.safetensors" in output_shards
    assert "model-subgraph-00001.safetensors" in output_shards
    assert not list(
        (tmp_path / "work" / "artifacts" / "statistics").glob(
            "target-*"
        )
    )
    assert not (tmp_path / "work" / "boundaries").exists()
    assert not staging.exists()
    assert not (tmp_path / "work" / "publish").exists()
    assert installed_devices
    assert set(installed_devices) == {torch.device("cpu")}

    resumed = streaming_oneshot(
        model=checkpoint,
        dataset=dataset,
        dataset_fingerprint="d" * 64,
        recipe=[
            IMatrixGatherer(ignore=["lm_head"]),
            QuantizationModifier(
                scheme="W8A8",
                targets=["Linear"],
                weight_observer="imatrix_mse",
                ignore=["lm_head"],
            ),
        ],
        output_dir=output,
        work_dir=tmp_path / "work",
        num_calibration_samples=1,
        max_seq_length=4,
        target_dtype=torch.float32,
    )
    assert resumed == output


def test_pretrained_streaming_overwrites_output_only_when_requested(tmp_path):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    checkpoint = tmp_path / "checkpoint"
    Qwen3ForCausalLM(config).save_pretrained(
        checkpoint, safe_serialization=True
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "old-file").write_text("old")

    result = streaming_oneshot(
        model=checkpoint,
        dataset=_calibration_data(),
        dataset_fingerprint="8" * 64,
        recipe=_imatrix_recipe(),
        output_dir=output,
        work_dir=tmp_path / "work",
        num_calibration_samples=1,
        max_seq_length=4,
        target_dtype=torch.float32,
        overwrite_output=True,
    )

    assert result == output
    assert (output / "FINALIZED").is_file()
    assert not (output / "old-file").exists()
    assert not (tmp_path / "work" / "publish").exists()
    assert not (tmp_path / "work" / "replaced-output").exists()


def test_checkpoint_progress_persists_boundaries_and_prefetches(tmp_path, monkeypatch):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    checkpoint = tmp_path / "checkpoint"
    Qwen3ForCausalLM(config).save_pretrained(
        checkpoint, safe_serialization=True
    )
    submitted = []
    original_submit = SubgraphPrefetcher.submit

    def record_submit(prefetcher, plan, **kwargs):
        submitted.append(plan.module_names)
        return original_submit(prefetcher, plan, **kwargs)

    monkeypatch.setattr(SubgraphPrefetcher, "submit", record_submit)

    streaming_oneshot(
        model=checkpoint,
        dataset=_calibration_data(),
        dataset_fingerprint="9" * 64,
        recipe=_imatrix_recipe(),
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        num_calibration_samples=2,
        max_seq_length=4,
        target_dtype=torch.float32,
        checkpoint_progress=True,
    )

    assert submitted
    transaction = (
        tmp_path / "work/staging/transactions/subgraph-00000/metadata.json"
    )
    assert '"boundary": {' in transaction.read_text()


def test_streaming_gptq_matches_sequential_oneshot(tmp_path, monkeypatch):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    torch.manual_seed(11)
    checkpoint = tmp_path / "checkpoint"
    Qwen3ForCausalLM(config).save_pretrained(
        checkpoint, safe_serialization=True
    )
    recipe = GPTQModifier(
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 8,
                },
            }
        },
        ignore=["lm_head"],
        actorder=None,
        block_size=8,
    )

    baseline = tmp_path / "baseline"
    oneshot(
        model=str(checkpoint),
        dataset=_calibration_data(),
        recipe=recipe,
        output_dir=baseline,
        num_calibration_samples=2,
        max_seq_length=4,
        precision="float32",
        pipeline="sequential",
        sequential_targets=["Qwen3DecoderLayer"],
    )

    remaining_statistics = []
    original = GPTQModifier.compress_module_list

    def record_release(self, modules):
        result = original(self, modules)
        remaining_statistics.append(
            (len(self._hessians), len(self._num_samples))
        )
        return result

    monkeypatch.setattr(GPTQModifier, "compress_module_list", record_release)
    streamed = streaming_oneshot(
        model=checkpoint,
        dataset=_calibration_data(),
        dataset_fingerprint="e" * 64,
        recipe=GPTQModifier(
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 8,
                    },
                }
            },
            ignore=["lm_head"],
            actorder=None,
            block_size=8,
        ),
        output_dir=tmp_path / "streamed",
        work_dir=tmp_path / "work",
        num_calibration_samples=2,
        max_seq_length=4,
        target_dtype=torch.float32,
    )

    expected = _checkpoint_tensors(baseline)
    actual = _checkpoint_tensors(streamed)
    assert set(actual) == set(expected)
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name
    assert remaining_statistics
    assert all(counts == (0, 0) for counts in remaining_statistics)


def test_streaming_imatrix_rtn_matches_sequential_oneshot(tmp_path, monkeypatch):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    torch.manual_seed(13)
    checkpoint = tmp_path / "checkpoint"
    Qwen3ForCausalLM(config).save_pretrained(
        checkpoint, safe_serialization=True
    )

    baseline = tmp_path / "baseline"
    oneshot(
        model=str(checkpoint),
        dataset=_calibration_data(),
        recipe=_imatrix_recipe(),
        output_dir=baseline,
        num_calibration_samples=2,
        max_seq_length=4,
        precision="float32",
        pipeline="sequential",
        sequential_targets=["Qwen3DecoderLayer"],
    )

    quantization_states = []
    original_forward = Subgraph.forward

    def record_quantization_state(subgraph, model, *args, **kwargs):
        states = tuple(
            module.quantization_enabled
            for module in subgraph.submodules(model)
            if hasattr(module, "quantization_scheme")
        )
        if states:
            quantization_states.append(states)
        return original_forward(subgraph, model, *args, **kwargs)

    monkeypatch.setattr(Subgraph, "forward", record_quantization_state)
    streamed = streaming_oneshot(
        model=checkpoint,
        dataset=_calibration_data(),
        dataset_fingerprint="f" * 64,
        recipe=_imatrix_recipe(),
        output_dir=tmp_path / "streamed",
        work_dir=tmp_path / "work",
        num_calibration_samples=2,
        max_seq_length=4,
        target_dtype=torch.float32,
    )

    # Calibration and propagation should both bypass fake-quant QDQ, matching
    # the ordinary sequential pipeline. The latter propagates algorithmic
    # in-place weight changes (for example GPTQ), not simulated RTN outputs.
    assert quantization_states
    assert all(
        enabled is False
        for states in quantization_states
        for enabled in states
    )

    expected = _checkpoint_tensors(baseline)
    actual = _checkpoint_tensors(streamed)
    assert set(actual) == set(expected)
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


def _assert_streaming_matches_oneshot(
    tmp_path, *, transform, quantizer, fingerprint
):
    transform_name = type(transform()).__name__
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    torch.manual_seed(17)
    checkpoint = tmp_path / "checkpoint"
    Qwen3ForCausalLM(config).save_pretrained(
        checkpoint, safe_serialization=True
    )

    baseline = tmp_path / "baseline"
    oneshot(
        model=str(checkpoint),
        dataset=_calibration_data(),
        recipe=[transform(), quantizer()],
        output_dir=baseline,
        num_calibration_samples=2,
        max_seq_length=4,
        precision="float32",
        pipeline="sequential",
        sequential_targets=["Qwen3DecoderLayer"],
    )
    streamed = streaming_oneshot(
        model=checkpoint,
        dataset=_calibration_data(),
        dataset_fingerprint=fingerprint * 64,
        recipe=[transform(), quantizer()],
        output_dir=tmp_path / "streamed",
        work_dir=tmp_path / "work",
        num_calibration_samples=2,
        max_seq_length=4,
        target_dtype=torch.float32,
    )

    expected = _checkpoint_tensors(baseline)
    actual = _checkpoint_tensors(streamed)
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].dtype == expected[name].dtype, name
        if (
            expected[name].dtype.is_floating_point
            and transform_name != "AWQModifier"
        ):
            torch.testing.assert_close(
                actual[name],
                expected[name],
                rtol=3e-3,
                atol=3e-5,
                msg=lambda message: f"{name}: {message}",
            )

    sample = next(iter(_calibration_data()))
    expected_model = AutoModelForCausalLM.from_pretrained(
        baseline, local_files_only=True, dtype=torch.float32
    ).eval()
    actual_model = AutoModelForCausalLM.from_pretrained(
        streamed, local_files_only=True, dtype=torch.float32
    ).eval()
    with torch.no_grad():
        expected_logits = expected_model(**sample).logits
        actual_logits = actual_model(**sample).logits
    torch.testing.assert_close(
        actual_logits, expected_logits, rtol=2e-2, atol=2e-3
    )


def test_streaming_smoothquant_matches_sequential_oneshot(tmp_path):
    _assert_streaming_matches_oneshot(
        tmp_path,
        transform=lambda: SmoothQuantModifier(smoothing_strength=0.5),
        quantizer=lambda: QuantizationModifier(
            scheme="W8A8", ignore=["lm_head"]
        ),
        fingerprint="6",
    )


def test_streaming_awq_matches_sequential_oneshot(tmp_path):
    _assert_streaming_matches_oneshot(
        tmp_path,
        transform=lambda: AWQModifier(n_grid=3, duo_scaling=False),
        quantizer=lambda: QuantizationModifier(
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 8,
                    },
                }
            },
            ignore=["lm_head"],
        ),
        fingerprint="7",
    )


def test_streaming_sparsegpt_composes_with_quantization(tmp_path):
    _assert_streaming_matches_oneshot(
        tmp_path,
        transform=lambda: SparseGPTModifier(
            sparsity=0.5,
            targets=["Linear"],
            ignore=["re:.*lm_head"],
        ),
        quantizer=lambda: QuantizationModifier(
            scheme="W8A8", ignore=["lm_head"]
        ),
        fingerprint="4",
    )


def test_streaming_wanda_composes_with_quantization(tmp_path):
    _assert_streaming_matches_oneshot(
        tmp_path,
        transform=lambda: WandaPruningModifier(
            sparsity=0.5,
            targets=["Linear"],
            ignore=["re:.*lm_head"],
        ),
        quantizer=lambda: QuantizationModifier(
            scheme="W8A8", ignore=["lm_head"]
        ),
        fingerprint="5",
    )
