import ast
import inspect
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.utils import generic as transformers_generic
from transformers.utils import import_utils
from transformers.utils.output_capturing import OutputRecorder

from llmcompressor.modeling.kimi_k3 import (
    CalibrationKimiK3SparseMoeBlock,
    KimiK3VisionConfig,
    _patch_flash_attn_varlen_func,
    _patch_transformers_v5_for_checkpoint_code,
    patch_kimi_k3_transformers_compat,
)
from llmcompressor.modeling.kimi_k3.modeling_kimi_k3 import MoonViT3dPretrainedModel
from llmcompressor.modeling.kimi_k3.modeling_kimi_linear import (
    _fla_reference_kda_adapter,
    naive_chunk_kda,
    naive_kda_gate,
    naive_kda_lowerbound_gate,
    naive_recurrent_kda,
)
from llmcompressor.pipelines.sequential.ast_utils.auto_wrapper import AutoWrapper


def _transformers_kda_gate(raw_g, A_log, dt_bias, lower_bound=None):
    """Mirror Transformers' forget-gate calculation for independent parity tests."""
    heads, head_dim = raw_g.shape[-2:]
    g = raw_g.float() + dt_bias.float().view(1, 1, heads, head_dim)
    decay_rate = A_log.float().view(1, 1, heads, 1).exp()
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(decay_rate * g)
    g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))
    return -decay_rate * g_softplus


def test_kimi_k3_vision_config_preserves_attention_implementation():
    assert KimiK3VisionConfig()._attn_implementation == "flash_attention_2"
    assert (
        KimiK3VisionConfig(_attn_implementation="eager")._attn_implementation
        == "eager"
    )
    assert MoonViT3dPretrainedModel._supports_sdpa is False


def test_kimi_k3_fla_reference_adapter_preserves_fused_contract():
    captured = {}

    def chunk_kda(**kwargs):
        captured.update(kwargs)
        q = kwargs["q"]
        state = q.new_zeros(
            q.shape[0], q.shape[2], q.shape[3], 5
        )
        return q, state

    def recurrent_kda(**kwargs):
        raise AssertionError("non-packed chunk input must use FLA naive chunk KDA")

    def gate(g, A_log, dt_bias, output_dtype):
        captured["gate_args"] = (A_log, dt_bias, output_dtype)
        return g + 1

    def lowerbound_gate(g, A_log, dt_bias, lower_bound, output_dtype):
        captured["lowerbound_gate_args"] = (
            A_log,
            dt_bias,
            lower_bound,
            output_dtype,
        )
        return g + 2

    q = torch.randn(1, 6, 2, 4)
    g = torch.randn_like(q)
    beta = torch.randn(1, 6, 2).sigmoid()
    A_log = torch.ones(2)
    dt_bias = torch.zeros(8)
    output, state = _fla_reference_kda_adapter(
        chunk_kda,
        recurrent_kda,
        gate,
        lowerbound_gate,
        use_chunk=True,
    )(
        q=q,
        k=q,
        v=q,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        transpose_state_layout=True,
    )

    assert captured["chunk_size"] == 2
    assert captured["scale"] == 4**-0.5
    assert captured["lowerbound_gate_args"] == (
        A_log,
        dt_bias,
        -5.0,
        torch.float32,
    )
    torch.testing.assert_close(captured["beta"], beta)
    torch.testing.assert_close(
        captured["g"], g + 2
    )
    torch.testing.assert_close(
        captured["q"].square().sum(dim=-1),
        torch.ones_like(captured["q"][..., 0]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert output is captured["q"]
    assert state.shape == (1, 2, 5, 4)


def test_kimi_k3_fla_reference_adapter_uses_recurrent_for_packed_sequences():
    calls = []

    def chunk_kda(**kwargs):
        raise AssertionError("packed input must not use FLA naive chunk KDA")

    def recurrent_kda(**kwargs):
        calls.append(kwargs)
        q = kwargs["q"]
        state = q.new_full((1, q.shape[2], q.shape[3], 5), len(calls))
        return q, state

    def unused_gate(*args, **kwargs):
        raise AssertionError("gate preprocessing was not requested")

    q = torch.randn(1, 7, 2, 4)
    output, state = _fla_reference_kda_adapter(
        chunk_kda,
        recurrent_kda,
        unused_gate,
        unused_gate,
        use_chunk=True,
    )(
        q=q,
        k=q,
        v=q,
        g=q,
        beta=torch.randn(1, 7, 2),
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 3, 7]),
        transpose_state_layout=True,
    )

    assert [call["q"].shape[1] for call in calls] == [3, 4]
    assert output.shape == q.shape
    assert state.shape == (2, 2, 5, 4)


@pytest.mark.skipif(naive_chunk_kda is None, reason="FLA is not installed")
@pytest.mark.parametrize(
    ("sequence_length", "lower_bound", "dtype"),
    [
        (65, -5.0, torch.float32),
        (128, -5.0, torch.float32),
        (65, -5.0, torch.bfloat16),
        (64, None, torch.float32),
    ],
)
def test_kimi_k3_fla_reference_adapter_matches_transformers_native(
    sequence_length, lower_bound, dtype
):
    from transformers.models.glm5_next import modeling_glm5_next as glm5

    torch.manual_seed(sequence_length + (0 if lower_bound is None else 1000))
    batch, heads, key_dim, value_dim = 2, 3, 8, 5
    q = torch.randn(batch, sequence_length, heads, key_dim, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence_length, heads, value_dim, dtype=dtype)
    raw_g = torch.randn_like(q)
    beta_logits = torch.randn(batch, sequence_length, heads)
    # This is the caller contract from both original K3 and Transformers:
    # beta is projected as logits and sigmoid is applied before KDA receives it.
    beta = beta_logits.sigmoid()
    A_log = torch.randn(heads, dtype=torch.float32)
    dt_bias = torch.randn(heads * key_dim, dtype=torch.float32)
    gate_function = (
        naive_kda_lowerbound_gate if lower_bound is not None else naive_kda_gate
    )
    fla_gate = gate_function(
        raw_g,
        A_log,
        dt_bias,
        **({"lower_bound": lower_bound} if lower_bound is not None else {}),
        output_dtype=torch.float32,
    )
    native_gate = _transformers_kda_gate(raw_g, A_log, dt_bias, lower_bound)
    torch.testing.assert_close(fla_gate, native_gate, atol=2e-6, rtol=2e-6)
    native_initial_state = torch.randn(
        batch, heads, key_dim, value_dim, dtype=torch.float32
    )
    adapter_initial_state = native_initial_state.transpose(-1, -2).contiguous()
    adapter = _fla_reference_kda_adapter(
        naive_chunk_kda,
        naive_recurrent_kda,
        naive_kda_gate,
        naive_kda_lowerbound_gate,
        use_chunk=True,
    )
    actual, actual_state = adapter(
        q=q,
        k=k,
        v=v,
        # Pass raw gate logits and request the adapter's gate preprocessing.
        # Passing `g` here would reproduce the old, insufficient parity test.
        g=raw_g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        output_final_state=True,
        initial_state=adapter_initial_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        transpose_state_layout=True,
    )
    native_recurrent = inspect.unwrap(glm5.recurrent_kimi_delta_attention)
    expected, expected_state = native_recurrent(
        q,
        k,
        v,
        native_gate,
        beta,
        native_initial_state,
        True,
        use_qk_l2norm_in_kernel=True,
    )
    atol = 2e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)
    torch.testing.assert_close(
        actual_state.transpose(-1, -2), expected_state, atol=atol, rtol=atol
    )

    if sequence_length in (64, 128):
        native_chunk = inspect.unwrap(glm5.chunk_kimi_delta_attention)
        expected, expected_state = native_chunk(
            q,
            k,
            v,
            native_gate,
            beta,
            chunk_size=64,
            initial_state=native_initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(
            actual_state.transpose(-1, -2), expected_state, atol=3e-5, rtol=3e-5
        )


@pytest.mark.skipif(naive_recurrent_kda is None, reason="FLA is not installed")
def test_kimi_k3_fla_reference_adapter_matches_packed_native():
    from transformers.models.glm5_next import modeling_glm5_next as glm5

    torch.manual_seed(20261012)
    heads, key_dim, value_dim = 3, 8, 5
    q = torch.randn(1, 7, heads, key_dim)
    k = torch.randn_like(q)
    v = torch.randn(1, 7, heads, value_dim)
    raw_g = torch.randn_like(q)
    beta = torch.randn(1, 7, heads).sigmoid()
    A_log = torch.randn(heads)
    dt_bias = torch.randn(heads * key_dim)
    native_gate = _transformers_kda_gate(raw_g, A_log, dt_bias, -5.0)
    native_initial_state = torch.randn(2, heads, key_dim, value_dim)
    adapter_initial_state = native_initial_state.transpose(-1, -2).contiguous()
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    adapter = _fla_reference_kda_adapter(
        naive_chunk_kda,
        naive_recurrent_kda,
        naive_kda_gate,
        naive_kda_lowerbound_gate,
        use_chunk=True,
    )
    actual, actual_state = adapter(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=adapter_initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
        transpose_state_layout=True,
    )

    native_recurrent = inspect.unwrap(glm5.recurrent_kimi_delta_attention)
    outputs, states = [], []
    for index, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        output, state = native_recurrent(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            native_gate[:, start:end],
            beta[:, start:end],
            native_initial_state[index : index + 1],
            True,
            use_qk_l2norm_in_kernel=True,
        )
        outputs.append(output)
        states.append(state)

    torch.testing.assert_close(
        actual, torch.cat(outputs, dim=1), atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        actual_state.transpose(-1, -2),
        torch.cat(states, dim=0),
        atol=2e-5,
        rtol=2e-5,
    )


def test_kimi_k3_transformers_output_recorder_compat():
    with patch.object(transformers_generic, "OutputRecorder", create=True):
        del transformers_generic.OutputRecorder

        patch_kimi_k3_transformers_compat()
        patch_kimi_k3_transformers_compat()

        assert transformers_generic.OutputRecorder is OutputRecorder


def test_kimi_k3_transformers_tie_weights_compat():
    class CheckpointModel:
        def tie_weights(self):
            self.called = True

    original_tie_weights = CheckpointModel.tie_weights
    patch_kimi_k3_transformers_compat(CheckpointModel)
    patched_tie_weights = CheckpointModel.tie_weights
    patch_kimi_k3_transformers_compat(CheckpointModel)

    model = CheckpointModel()
    model.tie_weights(missing_keys={"lm_head.weight"}, recompute_mapping=False)

    assert model.called
    assert patched_tie_weights is CheckpointModel.tie_weights
    assert patched_tie_weights is not original_tie_weights


def test_kimi_k3_transformers_generation_mixin_compat():
    class CheckpointModel(PreTrainedModel):
        config_class = PretrainedConfig

        def tie_weights(self):
            pass

    compatible = patch_kimi_k3_transformers_compat(CheckpointModel)

    assert compatible is patch_kimi_k3_transformers_compat(CheckpointModel)
    assert compatible is not CheckpointModel
    assert issubclass(compatible, GenerationMixin)


def test_kimi_k3_transformers_v5_remote_model_compat():
    calls = []

    def create_causal_mask(*, inputs_embeds, attention_mask, past_key_values):
        calls.append((inputs_embeds, attention_mask, past_key_values))
        return "mask"

    class KimiLinearModel:
        def __init__(self, config):
            config._attn_implementation = "flash_attention_2"
            self._use_flash_attention_2 = True

    linear_module = ModuleType("test_kimi_k3_linear_module")
    linear_module.create_causal_mask = create_causal_mask
    linear_module.KimiLinearModel = KimiLinearModel
    linear_lm_class = type(
        "KimiLinearForCausalLM",
        (),
        {"__module__": linear_module.__name__},
    )
    outer_module = ModuleType("test_kimi_k3_outer_module")
    outer_module.KimiLinearForCausalLM = linear_lm_class
    vision_model_class = type(
        "MoonViT3dPretrainedModel",
        (),
        {"_supports_sdpa": True},
    )
    outer_module.MoonViT3dPretrainedModel = vision_model_class

    class CheckpointModel:
        def tie_weights(self):
            pass

    CheckpointModel.__module__ = outer_module.__name__
    with patch.dict(
        sys.modules,
        {
            linear_module.__name__: linear_module,
            outer_module.__name__: outer_module,
        },
    ):
        patch_kimi_k3_transformers_compat(CheckpointModel)
        patch_kimi_k3_transformers_compat(CheckpointModel)

        config = type("Config", (), {"_attn_implementation": "eager"})()
        model = KimiLinearModel(config)
        result = linear_module.create_causal_mask(
            input_embeds="embeds",
            attention_mask="attention",
            cache_position="position",
            past_key_values="cache",
        )
        tree = ast.parse(
            "def forward(input_embeds, attention_mask):\n"
            "    return create_causal_mask(\n"
            "        input_embeds=input_embeds,\n"
            "        attention_mask=attention_mask,\n"
            "        cache_position=None,\n"
            "        past_key_values=None,\n"
            "    )\n"
        )
        wrapped_source = ast.unparse(
            AutoWrapper(
                {"create_causal_mask": linear_module.create_causal_mask},
                ["create_causal_mask"],
            ).auto_wrap(tree)
        )

    assert result == "mask"
    assert calls == [("embeds", "attention", "cache")]
    assert linear_module.create_causal_mask.__name__ == "create_causal_mask"
    assert "def wrapped_0" in wrapped_source
    assert "return wrapped_0(attention_mask, input_embeds)" in wrapped_source
    assert vision_model_class._supports_sdpa is False
    assert not hasattr(CheckpointModel, "_streaming_prefix_module_dependencies")
    assert config._attn_implementation == "eager"
    assert model._use_flash_attention_2 is False


def test_kimi_k3_flash_attn_compat_drops_unsupported_deterministic_keyword():
    calls = []

    def old_flash_attn(*args, **kwargs):
        calls.append((args, kwargs))
        return "output"

    module = ModuleType("flash_attn")
    module.flash_attn_varlen_func = old_flash_attn
    with patch.dict("sys.modules", {"flash_attn": module}):
        _patch_flash_attn_varlen_func()
        _patch_flash_attn_varlen_func()
        assert module.flash_attn_varlen_func(1, deterministic=True) == "output"

    assert calls == [((1,), {})]


def test_kimi_k3_transformers_v5_flash_attn_dispatch_compat():
    with (
        patch.object(import_utils, "is_torch_fx_available", create=True),
        patch.object(
            PreTrainedModel,
            "_kimi_k3_fa2_compat_patched",
            False,
            create=True,
        ),
        patch.object(PreTrainedModel, "_flash_attn_can_dispatch") as dispatch,
    ):
        del import_utils.is_torch_fx_available
        _patch_transformers_v5_for_checkpoint_code()
        _patch_transformers_v5_for_checkpoint_code()

        model = type(
            "CheckpointK3Model",
            (),
            {
                "_supports_flash_attn": False,
                "_supports_flash_attn_2": True,
            },
        )()
        PreTrainedModel._flash_attn_can_dispatch(model, 2, True)

        assert import_utils.is_torch_fx_available() is hasattr(torch, "fx")
        dispatch.assert_called_once_with(model, 2, True)
        assert model._supports_flash_attn is False


class _FixedGate(nn.Module):
    def forward(self, hidden_states):
        device = hidden_states.device
        return (
            torch.tensor([[0], [1]], device=device),
            torch.ones((2, 1), dtype=torch.float32, device=device),
        )


class _DummyKimiK3Moe(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_shared_experts": 1})()
        self.hidden_dim = 2
        self.num_experts = 2
        self.top_k = 1
        self.use_latent_moe = True
        self.gate = _FixedGate()
        self.experts = nn.ModuleList(
            [nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)]
        )
        self.shared_experts = nn.Linear(2, 2, bias=False)
        self.routed_expert_down_proj = nn.Linear(2, 2, bias=False)
        self.routed_expert_norm = nn.Identity()
        self.routed_expert_up_proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.experts[0].weight.copy_(torch.eye(2))
            self.experts[1].weight.copy_(2 * torch.eye(2))
            self.shared_experts.weight.copy_(0.5 * torch.eye(2))
            self.routed_expert_down_proj.weight.copy_(torch.eye(2))
            self.routed_expert_up_proj.weight.copy_(torch.eye(2))
        self.to(dtype=torch.bfloat16)


def _run(calibrate_all_experts: bool):
    replacement = CalibrationKimiK3SparseMoeBlock(
        _DummyKimiK3Moe(),
        config=None,
        calibrate_all_experts=calibrate_all_experts,
    )
    seen_tokens = [0, 0]
    hooks = [
        expert.register_forward_pre_hook(
            lambda _module, inputs, index=index: seen_tokens.__setitem__(
                index, inputs[0].shape[0]
            )
        )
        for index, expert in enumerate(replacement.experts)
    ]
    try:
        output = replacement(
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16)
        )
    finally:
        for hook in hooks:
            hook.remove()
    return output, seen_tokens


def test_kimi_k3_calibration_preserves_routed_and_shared_output():
    all_expert_output, all_expert_tokens = _run(True)
    routed_output, routed_tokens = _run(False)
    expected = torch.tensor([[[1.5, 3.0], [7.5, 10.0]]], dtype=torch.bfloat16)

    assert torch.equal(all_expert_output, expected)
    assert torch.equal(routed_output, expected)
    assert all_expert_tokens == [2, 2]
    assert routed_tokens == [1, 1]
