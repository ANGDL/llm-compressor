"""Global controls for optional Triton-backed implementations."""

from __future__ import annotations

import math

import torch

from llmcompressor._triton_env import is_triton_disabled


def _fla_reference_kda_adapter(
    chunk_function,
    recurrent_function,
    gate_function,
    lowerbound_gate_function,
    *,
    use_chunk: bool,
):
    """Adapt FLA reference KDA functions to the public FLA kernel contract."""

    def adapted(
        q,
        k,
        v,
        g,
        beta,
        *,
        A_log=None,
        dt_bias=None,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        lower_bound=None,
        cu_seqlens=None,
        transpose_state_layout=False,
        **kwargs,
    ):
        del kwargs
        if use_gate_in_kernel:
            if A_log is None:
                raise ValueError("A_log is required when use_gate_in_kernel=True")
            if lower_bound is None:
                g = gate_function(g, A_log, dt_bias, output_dtype=torch.float32)
            else:
                g = lowerbound_gate_function(
                    g,
                    A_log,
                    dt_bias,
                    lower_bound=lower_bound,
                    output_dtype=torch.float32,
                )
        if use_qk_l2norm_in_kernel:
            q_float = q.float()
            k_float = k.float()
            q = q_float / torch.sqrt(
                q_float.square().sum(-1, keepdim=True) + 1e-6
            )
            k = k_float / torch.sqrt(
                k_float.square().sum(-1, keepdim=True) + 1e-6
            )
        # Match both Transformers native KDA and FLA's documented default.
        if scale is None:
            scale = q.shape[-1] ** -0.5
        if transpose_state_layout and initial_state is not None:
            initial_state = initial_state.transpose(-1, -2)

        if cu_seqlens is not None and len(cu_seqlens) > 2:
            if q.shape[0] != 1:
                raise ValueError("Packed FLA KDA inputs must have batch size 1")
            boundaries = cu_seqlens.tolist()
            outputs, states = [], []
            for index, (start, end) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                sequence_state = (
                    None if initial_state is None else initial_state[index : index + 1]
                )
                output, state = recurrent_function(
                    q=q[:, start:end],
                    k=k[:, start:end],
                    v=v[:, start:end],
                    g=g[:, start:end],
                    beta=beta[:, start:end],
                    scale=scale,
                    initial_state=sequence_state,
                    output_final_state=output_final_state,
                )
                outputs.append(output)
                if output_final_state:
                    states.append(state)
            output = torch.cat(outputs, dim=1)
            state = torch.cat(states, dim=0) if output_final_state else None
        elif use_chunk:
            chunk_size = math.gcd(q.shape[1], 64)
            output, state = chunk_function(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
            )
        else:
            output, state = recurrent_function(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )
        if transpose_state_layout and state is not None:
            state = state.transpose(-1, -2)
        return output, state

    return adapted


def enable_fla_reference_kda() -> None:
    """Replace FLA Triton KDA entrypoints with FLA's own reference kernels.

    This must run before Transformers imports a model whose KDA functions are
    decorated with ``use_kernel_func_from_hub_with_fallback``. The decorator
    then freezes these FLA-compatible adapters instead of binding a native
    Transformers implementation.
    """

    try:
        import fla
        import fla.ops.kda as kda
        from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
        from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
    except Exception:
        return

    chunk_reference = _fla_reference_kda_adapter(
        naive_chunk_kda,
        naive_recurrent_kda,
        naive_kda_gate,
        naive_kda_lowerbound_gate,
        use_chunk=True,
    )
    recurrent_reference = _fla_reference_kda_adapter(
        naive_chunk_kda,
        naive_recurrent_kda,
        naive_kda_gate,
        naive_kda_lowerbound_gate,
        use_chunk=False,
    )
    kda.chunk_kda = chunk_reference
    kda.fused_recurrent_kda = recurrent_reference
    fla.chunk_kda = chunk_reference
    fla.fused_recurrent_kda = recurrent_reference


def disable_fla_kda() -> None:
    """Backward-compatible alias for the FLA reference KDA switch."""

    enable_fla_reference_kda()


def disable_triton_backends() -> None:
    """Disable model-level Triton entrypoints when the global switch is set."""

    if is_triton_disabled():
        enable_fla_reference_kda()
