from __future__ import annotations

import gc
import weakref

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.streaming import (
    KimiK3WeightMaterializer,
    KimiK3WeightSource,
    SafetensorsWeightSource,
    StreamingDTypePolicy,
    TargetWeightLoader,
    build_meta_model,
)


class TinyBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.register_buffer("step", torch.tensor(3, dtype=torch.int64))

    def forward(self, inputs):
        return self.norm(self.proj(inputs)) + self.step


class TinyModel(nn.Module):
    def __init__(self, hidden_size: int = 4, layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, hidden_size)
        self.layers = nn.ModuleList(
            [TinyBlock(hidden_size) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, 8, bias=False)


class MixedBufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        shared = torch.arange(4, dtype=torch.float32)
        self.register_buffer("checkpoint_state", shared)
        self.register_buffer("checkpoint_alias", shared)
        self.register_buffer(
            "runtime_state",
            torch.arange(4, dtype=torch.float32) + 10,
            persistent=False,
        )


class RemoteStyleBlock(nn.Module):
    """Fixture standing in for a checkpoint-defined modeling class."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(hidden_size))
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, inputs):
        return self.projection(inputs) * self.gate


class KdaModel(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.layer = nn.Module()
        self.layer.self_attn = nn.Module()
        self.layer.self_attn.A_log = nn.Parameter(torch.empty(num_heads))


class ResidentLinear(nn.Linear):
    def forward(self, inputs):
        assert not self.weight.is_meta
        return super().forward(inputs)


class ResidentExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = ResidentLinear(4, 4, bias=False)
        self.w2 = ResidentLinear(4, 4, bias=False)
        self.w3 = ResidentLinear(4, 4, bias=False)

    def forward(self, inputs):
        return self.w2(self.w1(inputs) + self.w3(inputs))


class ResidentExpertModel(nn.Module):
    def __init__(self, expert_count: int = 2):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        layer = nn.Module()
        layer.gate = nn.Linear(4, 4, bias=False)
        layer.block_sparse_moe = nn.Module()
        layer.block_sparse_moe.experts = nn.ModuleList(
            [ResidentExpert() for _ in range(expert_count)]
        )
        self.language_model.model.layers = nn.ModuleList([layer])


def _write_checkpoint(model: nn.Module, path, *, overrides=None):
    tensors = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in model.state_dict().items()
    }
    tensors.update(overrides or {})
    save_file(tensors, path)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch.manual_seed(7)
    model = TinyModel()
    path = tmp_path / "model.safetensors"
    _write_checkpoint(model, path)
    return model, path


def _assert_all_meta(module):
    assert all(tensor.is_meta for tensor in module.parameters())
    assert all(tensor.is_meta for tensor in module.buffers())


def _real_storage_bytes(module):
    tensors = [*module.parameters(), *module.buffers()]
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in tensors
        if not tensor.is_meta
    )


def test_build_meta_model_does_not_allocate_real_weights():
    model = build_meta_model(TinyModel, hidden_size=16, layers=20)

    _assert_all_meta(model)
    assert len(model.layers) == 20


def test_build_meta_model_keeps_only_nonpersistent_buffers_real(tmp_path):
    reference = MixedBufferModel()
    path = tmp_path / "model.safetensors"
    _write_checkpoint(reference, path)

    model = build_meta_model(
        MixedBufferModel,
        keep_nonpersistent_buffers=True,
    )

    assert all(parameter.is_meta for parameter in model.parameters())
    assert model.checkpoint_state.is_meta
    assert model.checkpoint_alias is model.checkpoint_state
    assert not model.runtime_state.is_meta
    assert torch.equal(
        model.runtime_state,
        torch.arange(4, dtype=torch.float32) + 10,
    )

    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))
    plan = loader.plan("")
    assert len(plan.buffer_groups) == 1
    assert len(plan.runtime_buffer_groups) == 1


def test_tiny_llama_decoder_target_runs_and_unloads(tmp_path):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(11)
    reference = LlamaForCausalLM(config)
    path = tmp_path / "model.safetensors"
    _write_checkpoint(reference, path)
    model = build_meta_model(LlamaForCausalLM, config)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))
    inputs = torch.randn(1, 5, config.hidden_size)
    position_ids = torch.arange(5).unsqueeze(0)
    position_embeddings = reference.model.rotary_emb(inputs, position_ids)
    expected = reference.model.layers[0](
        inputs, position_embeddings=position_embeddings
    )

    with loader.loaded(
        "model.layers.0",
        device=torch.device("cpu"),
        dtype=torch.float32,
    ) as layer:
        actual = layer(inputs, position_embeddings=position_embeddings)
        assert torch.allclose(actual, expected)
        _assert_all_meta(model.model.layers[1])

    _assert_all_meta(model)


def test_loads_one_target_for_forward_then_restores_meta(tiny_checkpoint):
    reference, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))
    inputs = torch.randn(2, 4)
    expected = reference.layers[0](inputs)

    with loader.loaded(
        "layers.0", device=torch.device("cpu"), dtype=torch.float32
    ) as layer:
        assert all(not parameter.is_meta for parameter in layer.parameters())
        assert all(not buffer.is_meta for buffer in layer.buffers())
        assert torch.allclose(layer(inputs), expected)
        _assert_all_meta(model.layers[1])
        loaded_parameter = next(layer.parameters())
        reference_to_parameter = weakref.ref(loaded_parameter)

    del loaded_parameter
    gc.collect()
    _assert_all_meta(model)
    assert reference_to_parameter() is None
    assert next(model.layers[0].parameters()).dtype == torch.float32


def test_mixed_dtype_policy_preserves_non_quantized_fp32_state(tmp_path):
    class MixedDTypeTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.attn_sink = nn.Parameter(torch.arange(4, dtype=torch.float32))
            self.register_buffer(
                "control_scale", torch.tensor(0.5, dtype=torch.float32)
            )

    reference = MixedDTypeTarget()
    path = tmp_path / "mixed-dtype.safetensors"
    _write_checkpoint(reference, path)
    model = build_meta_model(MixedDTypeTarget)
    policy = StreamingDTypePolicy.for_quantized_modules(
        torch.bfloat16, ("proj",)
    )
    loader = TargetWeightLoader(
        model,
        SafetensorsWeightSource(path),
        dtype_policy=policy,
    )

    with loader.loaded("", device=torch.device("cpu"), dtype=torch.bfloat16):
        assert model.proj.weight.dtype == torch.bfloat16
        assert model.proj.bias.dtype == torch.bfloat16
        assert model.attn_sink.dtype == torch.float32
        assert model.control_scale.dtype == torch.float32
        assert (
            model.proj(torch.ones(1, 4, dtype=torch.bfloat16)).dtype
            == torch.bfloat16
        )

    _assert_all_meta(model)


def test_target_loader_rejects_accelerator_transfer(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    with pytest.raises(ValueError, match="PreparedSubgraphWeights.move_to"):
        with loader.loaded("layers.0", device=torch.device("cuda")):
            pass

    _assert_all_meta(model)


def test_prepare_does_not_mutate_model_before_install(tiny_checkpoint):
    reference, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))
    original = model.layers[0].proj.weight

    plan = loader.plan("layers.0")
    prepared = loader.materialize_cpu(plan, dtype=torch.float32)

    assert prepared.parameter_values
    assert prepared.buffer_values
    assert model.layers[0].proj.weight is original
    _assert_all_meta(model)
    with loader.installed(prepared):
        assert not prepared.parameter_values
        assert not prepared.buffer_values
        assert prepared.nbytes == 0
        assert not model.layers[0].proj.weight.is_meta
        assert torch.equal(
            model.layers[0].proj.weight, reference.layers[0].proj.weight
        )

    _assert_all_meta(model)


def test_plan_uses_local_names_without_scanning_the_full_model(
    tiny_checkpoint, monkeypatch
):
    _, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    def fail_global_scan(*_args, **_kwargs):
        raise AssertionError("local checkpoint names must not scan the full model")

    monkeypatch.setattr(model, "named_parameters", fail_global_scan)
    monkeypatch.setattr(model, "named_buffers", fail_global_scan)

    plan = loader.plan("layers.0")

    assert set(plan.parameter_sources.values()) == {
        "layers.0.proj.weight",
        "layers.0.proj.bias",
        "layers.0.norm.weight",
        "layers.0.norm.bias",
    }
    assert set(plan.buffer_sources.values()) == {"layers.0.step"}


def test_plan_scans_global_aliases_once_for_omitted_tied_weight(
    tmp_path, monkeypatch
):
    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.lm_head = nn.Linear(4, 8, bias=False)
            self.lm_head.weight = self.embed.weight

    reference = TiedModel()
    path = tmp_path / "model.safetensors"
    save_file({"embed.weight": reference.embed.weight.detach().clone()}, path)
    model = build_meta_model(TiedModel)
    original_named_parameters = model.named_parameters
    scans = 0

    def record_scan(*args, **kwargs):
        nonlocal scans
        scans += 1
        return original_named_parameters(*args, **kwargs)

    monkeypatch.setattr(model, "named_parameters", record_scan)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    plan = loader.plan("lm_head")

    assert set(plan.parameter_sources.values()) == {"embed.weight"}
    assert scans == 1


def test_plan_uses_declared_tied_weight_source_without_shared_identity(tmp_path):
    class DeclaredTiedModel(nn.Module):
        _tied_weights_keys = {"alias.weight": "canonical.weight"}

        def __init__(self):
            super().__init__()
            self.canonical = nn.Linear(4, 4, bias=False)
            self.alias = nn.Linear(4, 4, bias=False)

    reference = DeclaredTiedModel()
    path = tmp_path / "model.safetensors"
    save_file(
        {"canonical.weight": reference.canonical.weight.detach().clone()}, path
    )
    model = build_meta_model(DeclaredTiedModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    plan = loader.plan("alias")

    assert set(plan.parameter_sources.values()) == {"canonical.weight"}
    with loader.loaded("alias", device=torch.device("cpu"), dtype=torch.float32):
        assert torch.equal(model.alias.weight, reference.canonical.weight)

    _assert_all_meta(model)


def test_install_rejects_stale_prepared_plan(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))
    prepared = loader.materialize_cpu(
        loader.plan("layers.0"), dtype=torch.float32
    )
    model.layers[0].proj.weight = nn.Parameter(
        torch.empty_like(model.layers[0].proj.weight, device="meta")
    )

    with pytest.raises(RuntimeError, match="stale"):
        with loader.installed(prepared):
            pass

    assert not prepared.parameter_values


def test_materialized_storage_does_not_grow_with_model_depth(tmp_path):
    active_bytes = []
    for layer_count in (2, 20):
        reference = TinyModel(hidden_size=32, layers=layer_count)
        path = tmp_path / f"model-{layer_count}.safetensors"
        _write_checkpoint(reference, path)
        del reference
        model = build_meta_model(TinyModel, hidden_size=32, layers=layer_count)
        loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

        with loader.loaded(
            "layers.0", device=torch.device("cpu"), dtype=torch.float32
        ):
            active_bytes.append(_real_storage_bytes(model))
            assert _real_storage_bytes(model.layers[1]) == 0

        assert _real_storage_bytes(model) == 0

    assert active_bytes[0] == active_bytes[1]


def test_exception_exit_restores_target_to_meta(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    with pytest.raises(RuntimeError, match="forward failed"):
        with loader.loaded(
            "layers.0", device=torch.device("cpu"), dtype=torch.float32
        ):
            raise RuntimeError("forward failed")

    _assert_all_meta(model)
    with loader.loaded(
        "layers.1", device=torch.device("cpu"), dtype=torch.float32
    ):
        assert not next(model.layers[1].parameters()).is_meta


@pytest.mark.parametrize(
    "target_name", ["embed_tokens", "layers.0", "final_norm", "lm_head"]
)
def test_supports_model_boundary_target_types(tiny_checkpoint, target_name):
    _, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    with loader.loaded(
        target_name, device=torch.device("cpu"), dtype=torch.float32
    ) as target:
        assert all(not parameter.is_meta for parameter in target.parameters())

    _assert_all_meta(model)


def test_tied_parameters_load_once_and_remain_tied(tmp_path, monkeypatch):
    class TiedTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(4, 4, bias=False)
            self.right = nn.Linear(4, 4, bias=False)
            self.right.weight = self.left.weight

    reference = TiedTarget()
    path = tmp_path / "model.safetensors"
    save_file({"left.weight": reference.left.weight.detach().clone()}, path)
    model = build_meta_model(TiedTarget)
    source = SafetensorsWeightSource(path)
    loaded_names = []
    original_iter = source.iter_tensor_groups_cpu

    def recording_iter(groups):
        groups = tuple(tuple(group) for group in groups)
        loaded_names.extend(name for group in groups for name in group)
        return original_iter(groups)

    monkeypatch.setattr(source, "iter_tensor_groups_cpu", recording_iter)
    loader = TargetWeightLoader(model, source)

    with loader.loaded("", device=torch.device("cpu"), dtype=torch.float32):
        assert model.left.weight is model.right.weight
        assert loaded_names.count("left.weight") == 1

    assert model.left.weight is model.right.weight
    _assert_all_meta(model)


def test_remote_style_model_target_can_run(tmp_path):
    torch.manual_seed(3)
    reference = RemoteStyleBlock(4)
    path = tmp_path / "model.safetensors"
    _write_checkpoint(reference, path)
    model = build_meta_model(RemoteStyleBlock, 4)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))
    inputs = torch.randn(2, 4)

    with loader.loaded("", device=torch.device("cpu"), dtype=torch.float32):
        assert torch.allclose(model(inputs), reference(inputs))

    _assert_all_meta(model)


def test_rejects_unknown_target_and_missing_weight(tiny_checkpoint, tmp_path):
    _, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    with pytest.raises(ValueError, match="Unknown target"):
        with loader.loaded("layers.99", device=torch.device("cpu")):
            pass

    incomplete = tmp_path / "incomplete.safetensors"
    save_file({"layers.0.proj.weight": torch.ones(4, 4)}, incomplete)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(incomplete))
    with pytest.raises(KeyError, match="layers.0.proj.bias"):
        with loader.loaded("layers.0", device=torch.device("cpu")):
            pass


def test_rejects_shape_mismatch_before_installing(tiny_checkpoint, tmp_path):
    reference, _ = tiny_checkpoint
    path = tmp_path / "wrong-shape.safetensors"
    _write_checkpoint(
        reference, path, overrides={"layers.0.proj.weight": torch.ones(3, 4)}
    )
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    with pytest.raises(ValueError, match="model expects"):
        with loader.loaded(
            "layers.0", device=torch.device("cpu"), dtype=torch.float32
        ):
            pass

    _assert_all_meta(model)


def test_kimi_k3_kda_a_log_uses_configured_head_count(tmp_path):
    path = tmp_path / "model.safetensors"
    source = torch.arange(128, dtype=torch.float32)
    save_file({"layer.self_attn.A_log": source}, path)
    model = build_meta_model(KdaModel, num_heads=96)
    loader = TargetWeightLoader(
        model,
        SafetensorsWeightSource(path),
        materializer=KimiK3WeightMaterializer(kda_num_heads=96),
    )

    with loader.loaded("layer", device=torch.device("cpu"), dtype=torch.float32):
        assert torch.equal(model.layer.self_attn.A_log, source[:96])

    _assert_all_meta(model)


def test_kimi_k3_routed_experts_reside_with_complete_parent(tmp_path):
    path = tmp_path / "model.safetensors"
    prefix = "language_model.model.layers.0"
    values = {f"{prefix}.gate.weight": torch.eye(4)}
    for expert_index in range(2):
        for projection in ("w1", "w2", "w3"):
            name = (
                f"{prefix}.block_sparse_moe.experts.{expert_index}."
                f"{projection}"
            )
            values[f"{name}.weight_packed"] = torch.full(
                (4, 2), 0x22 + expert_index, dtype=torch.uint8
            )
            values[f"{name}.weight_scale"] = torch.full(
                (4, 1), 127, dtype=torch.uint8
            )
    save_file(values, path)
    model = build_meta_model(ResidentExpertModel)
    loader = TargetWeightLoader(
        model,
        KimiK3WeightSource(path),
        materializer=KimiK3WeightMaterializer(fp4_block_size=4),
    )

    with loader.loaded(
        prefix,
        device=torch.device("cpu"),
        dtype=torch.float32,
        allow_missing_state=True,
    ):
        layer = model.language_model.model.layers[0]
        assert all(
            not parameter.is_meta
            for parameter in layer.block_sparse_moe.experts.parameters()
        )

    _assert_all_meta(model)


def test_kimi_k3_plan_includes_all_896_routed_experts(tmp_path):
    path = tmp_path / "model.safetensors"
    prefix = "language_model.model.layers.0"
    values = {f"{prefix}.gate.weight": torch.eye(4)}
    for expert_index in range(896):
        for projection in ("w1", "w2", "w3"):
            name = (
                f"{prefix}.block_sparse_moe.experts.{expert_index}."
                f"{projection}"
            )
            values[f"{name}.weight_packed"] = torch.full(
                (4, 2), 0x22, dtype=torch.uint8
            )
            values[f"{name}.weight_scale"] = torch.full(
                (4, 1), 127, dtype=torch.uint8
            )
    save_file(values, path)
    model = build_meta_model(ResidentExpertModel, expert_count=896)
    loader = TargetWeightLoader(
        model,
        KimiK3WeightSource(path),
        materializer=KimiK3WeightMaterializer(fp4_block_size=4),
    )

    plan = loader.plan(prefix)

    assert len(plan.parameter_sources) == 1 + 896 * 3
    with loader.loaded(prefix, device=torch.device("cpu"), dtype=torch.float32):
        experts = model.language_model.model.layers[0].block_sparse_moe.experts
        assert len(experts) == 896
        assert all(not expert.w1.weight.is_meta for expert in experts)

    _assert_all_meta(model)


def test_missing_source_state_is_not_initialized_as_a_placeholder(tmp_path):
    class PartialModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded = nn.Linear(4, 3, bias=False)
            self.missing = nn.Linear(4, 5, bias=False)

    path = tmp_path / "model.safetensors"
    save_file({"loaded.weight": torch.ones(3, 4)}, path)
    model = build_meta_model(PartialModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))
    with pytest.raises(KeyError, match="missing model state"):
        loader.plan("", allow_missing_state=True)

    _assert_all_meta(model)


def test_rejects_fused_checkpoint_tensor_with_clear_error(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"fused_qkv": torch.ones(12, 4)}, path)
    model = build_meta_model(nn.Linear, 4, 4)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    with pytest.raises(KeyError, match="Fused or renamed checkpoint tensors"):
        with loader.loaded("", device=torch.device("cpu")):
            pass
