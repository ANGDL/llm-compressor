"""DeepSeek-V4 DSpark model used by streaming calibration.

The 0731 checkpoint replaces the original one-layer MTP block with three
``mtp.*`` DSpark decoder blocks.  This module is deliberately separate from
the stable original model implementation so the regular DeepSeek-V4 oneshot
path keeps its old MTP contract.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from .config import ModelConfig
from .kernels import sparse_attention
from .model import (
    Attention,
    Block,
    DeepseekV4NativeForCausalLM,
    DeepseekV4NativePreTrainedModel,
    Embedding,
    Linear,
    ParallelHead,
    RMSNorm,
    get_compress_topk_idxs,
    set_dtype,
)


@torch.fx.wrap
def _window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int
) -> torch.Tensor:
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [
                torch.arange(start_pos + 1, window_size),
                torch.arange(0, start_pos + 1),
            ]
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size)
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@torch.fx.wrap
def _compress_topk_idxs(
    ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int
) -> torch.Tensor:
    return get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)


def _apply_rotary_3d(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    rotated = torch.view_as_real(
        x_complex * freqs.view(1, x_complex.size(1), x_complex.size(-1))
    ).flatten(-2)
    x.copy_(rotated)
    return x


def _apply_rotary_4d(
    x: torch.Tensor, freqs: torch.Tensor, *, inverse: bool = False
) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    rotated = torch.view_as_real(
        x_complex * freqs.view(1, x_complex.size(1), 1, x_complex.size(-1))
    ).flatten(-2)
    x.copy_(rotated)
    return x


def _dense_dspark_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Dense equivalent of DSV4 sparse attention for one draft block.

    DSpark attends to the last sliding-window context and every token in its
    draft block.  Keeping this implementation tensor-only makes it traceable
    by the streaming FX tracer while preserving the reference kernel's sink
    normalization and BF16 value accumulation.
    """

    scores = torch.einsum("bshd,btd->bsht", q.float(), kv.float())
    scores = scores * softmax_scale
    scores_max = scores.amax(dim=-1, keepdim=True).clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max)
    exp_scores_bf16 = exp_scores.bfloat16()
    output = torch.einsum(
        "bsht,btd->bshd", exp_scores_bf16.float(), kv.float()
    )
    sink = attn_sink.view(1, 1, -1, 1)
    denominator = exp_scores.sum(dim=-1, keepdim=True) + torch.exp(
        sink - scores_max
    )
    return (output / denominator).to(kv.dtype)


class TraceFriendlyAttention(Attention):
    """Original DSV4 attention with FX-safe fixed-rank RoPE operations."""

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        kv_cache = self.kv_cache
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(
            bsz, seqlen, self.n_local_heads, self.head_dim
        )
        q = q * torch.rsqrt(
            q.float().square().mean(-1, keepdim=True) + self.eps
        ).to(q.dtype)
        _apply_rotary_4d(q[..., -self.rope_head_dim :], freqs_cis)
        kv = self.kv_norm(self.wkv(x))
        _apply_rotary_3d(kv[..., -self.rope_head_dim :], freqs_cis)
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = kv_cache[:, self.window_size :]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        topk_idxs = _window_topk_idxs(
            self.window_size, bsz, seqlen, start_pos
        ).to(x.device).int()
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else self.window_size
            if self.indexer is not None:
                compress_topk = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk = _compress_topk_idxs(
                    self.compress_ratio, bsz, seqlen, start_pos, offset
                ).to(x.device)
            topk_idxs = torch.cat((topk_idxs, compress_topk.int()), dim=-1)
        if start_pos == 0:
            if seqlen <= self.window_size:
                kv_cache[:bsz, :seqlen] = kv.float()
            else:
                cutoff = seqlen % self.window_size
                left, right = kv[:, -self.window_size :].float().split(
                    [self.window_size - cutoff, cutoff], dim=1
                )
                kv_cache[:bsz, cutoff : self.window_size] = left
                kv_cache[:bsz, :cutoff] = right
            if self.compress_ratio:
                compressed = self.compressor(x, start_pos)
                attn_kv = torch.cat((kv, compressed), dim=1)
            else:
                attn_kv = kv
        else:
            kv_cache[:bsz, start_pos % self.window_size] = kv[:, 0].float()
            if self.compress_ratio:
                self.compressor(x, start_pos)
            attn_kv = kv_cache[:bsz]
        output = sparse_attention(
            q, attn_kv, self.attn_sink, topk_idxs, self.softmax_scale
        ).to(x.dtype)
        _apply_rotary_4d(
            output[..., -self.rope_head_dim :], freqs_cis, inverse=True
        )
        output = output.view(bsz, seqlen, self.n_local_groups, -1)
        all_inputs = output.reshape(-1, output.size(-1))
        all_outputs = self.wo_a(all_inputs).view(
            bsz * seqlen, self.n_local_groups, -1
        )
        output = torch.stack(
            [
                all_outputs[
                    :, group, group * self.o_lora_rank : (group + 1) * self.o_lora_rank
                ]
                for group in range(self.n_local_groups)
            ],
            dim=1,
        ).view(bsz, seqlen, self.n_local_groups, self.o_lora_rank)
        return self.wo_b(output.flatten(2).to(x.dtype))


class TraceFriendlyBlock(Block):
    """Main decoder block that carries DSpark context across boundaries."""

    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__(layer_id, args)
        self.attn = TraceFriendlyAttention(layer_id, args)
        self._target_flags = tuple(
            float(layer_id == target_id)
            for target_id in args.dspark_target_layer_ids
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: Optional[torch.Tensor],
        main_hiddens: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = super().forward(x, start_pos, input_ids)
        current_hidden = x.mean(dim=2)
        main_hiddens = tuple(
            current_hidden * flag + previous * (1.0 - flag)
            for flag, previous in zip(self._target_flags, main_hiddens)
        )
        return x, main_hiddens


class DSparkAttention(TraceFriendlyAttention):
    """Non-causal attention over context plus one speculative block."""

    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__(layer_id, args)
        if self.compress_ratio:
            raise ValueError("DeepSeek-V4 DSpark layers must not be compressed")
        self.register_buffer(
            "draft_offsets",
            torch.arange(args.dspark_block_size, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, main_x: torch.Tensor) -> torch.Tensor:
        block_size = x.shape[1]
        context_positions = self.freqs_cis[: main_x.shape[1]]
        draft_positions = self.draft_offsets + main_x.shape[1]
        draft_freqs = self.freqs_cis[draft_positions]
        main_kv = self.kv_norm(self.wkv(main_x))
        _apply_rotary_3d(main_kv[..., -self.rope_head_dim :], context_positions)

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(
            x.shape[0], block_size, self.n_local_heads, self.head_dim
        )
        q = q * torch.rsqrt(
            q.float().square().mean(-1, keepdim=True) + self.eps
        ).to(q.dtype)
        _apply_rotary_4d(q[..., -self.rope_head_dim :], draft_freqs)

        kv = self.kv_norm(self.wkv(x))
        _apply_rotary_3d(kv[..., -self.rope_head_dim :], draft_freqs)
        context = main_kv[:, -self.window_size :]
        output = _dense_dspark_attention(
            q,
            torch.cat((context, kv), dim=1),
            self.attn_sink,
            self.softmax_scale,
        )
        _apply_rotary_4d(
            output[..., -self.rope_head_dim :], draft_freqs, inverse=True
        )
        output = output.view(x.shape[0], block_size, self.n_local_groups, -1)
        all_inputs = output.reshape(-1, output.size(-1))
        all_outputs = self.wo_a(all_inputs).view(
            x.shape[0] * block_size, self.n_local_groups, -1
        )
        output = torch.stack(
            [
                all_outputs[
                    :, group, group * self.o_lora_rank : (group + 1) * self.o_lora_rank
                ]
                for group in range(self.n_local_groups)
            ],
            dim=1,
        )
        return self.wo_b(output.reshape(x.shape[0], block_size, -1))


class DSparkBlock(Block):
    """One checkpointed DSpark decoder block under ``mtp.<index>``."""

    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__(layer_id, args)
        self.attn = DSparkAttention(layer_id, args)
        self.stage_id = layer_id - args.n_layers
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        if self.stage_id == 0:
            if not args.dspark_target_layer_ids:
                raise ValueError("DSpark requires dspark_target_layer_ids")
            self.main_proj = Linear(
                args.dim * len(args.dspark_target_layer_ids), args.dim, bias=False
            )
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if self.stage_id == args.n_mtp_layers - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(
                args.vocab_size, args.dspark_markov_rank
            )
            self.confidence_head = DSparkConfidenceHead(
                args.dim + args.dspark_markov_rank
            )
            with set_dtype(torch.float32):
                self.hc_head_fn = nn.Parameter(
                    torch.empty(self.hc_mult, self.hc_mult * args.dim)
                )
                self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult))
                self.hc_head_scale = nn.Parameter(torch.empty(1))

    def forward_embed(
        self, main_hidden: torch.Tensor, input_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        main_x = self.main_norm(self.main_proj(main_hidden))
        x = input_embeds
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x

    def forward(
        self,
        x: torch.Tensor | tuple[torch.Tensor, ...],
        input_ids: Optional[torch.Tensor],
        main_x: Optional[torch.Tensor] = None,
        embedding_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.stage_id == 0:
            if input_ids is None or embedding_weight is None:
                raise ValueError(
                    "The first DSpark block requires input ids and embedding"
                )
            if not isinstance(x, tuple):
                raise TypeError("The first DSpark block requires target hidden states")
            first = input_ids[:, -1:]
            noise = first[:, :1].expand(-1, self.block_size - 1) * 0
            draft_ids = torch.cat((first, noise + self.noise_token_id), dim=1)
            input_embeds = F.embedding(draft_ids, embedding_weight)
            x, main_x = self.forward_embed(torch.cat(x, dim=-1), input_embeds)
            input_ids = draft_ids
        elif main_x is None or not isinstance(x, torch.Tensor):
            raise ValueError("DSpark decoder blocks require hidden and context states")

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(x, main_x)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        if self.stage_id == 0:
            return x, main_x, input_ids
        return x


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = Embedding(vocab_size, rank)
        self.markov_w2 = ParallelHead(vocab_size, rank)


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = Linear(input_dim, 1, bias=False)


class DSparkTransformer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        if not args.is_dspark:
            raise ValueError("DSparkTransformer requires dspark_block_size > 0")
        self.max_seq_len = args.max_seq_len
        self.hc_mult = args.hc_mult
        self._config_target_layer_ids = tuple(args.dspark_target_layer_ids)
        invalid_targets = set(self._config_target_layer_ids) - set(
            range(args.n_layers)
        )
        if invalid_targets:
            raise ValueError(
                f"DSpark target layer ids are out of range: {sorted(invalid_targets)}"
            )
        self.embed = Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            [TraceFriendlyBlock(layer_id, args) for layer_id in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = ParallelHead(
            args.vocab_size, args.dim, args.norm_eps, args.hc_eps
        )
        self.mtp = nn.ModuleList(
            [
                DSparkBlock(args.n_layers + layer_id, args)
                for layer_id in range(args.n_mtp_layers)
            ]
        )
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(
                torch.empty(args.hc_mult, args.hc_mult * args.dim)
            )
            self.hc_head_base = nn.Parameter(torch.empty(args.hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed(input_ids)
        hidden_states = hidden_states.unsqueeze(2).repeat(
            1, 1, self.hc_mult, 1
        )
        current_hidden = hidden_states.mean(dim=2)
        main_hiddens = tuple(
            current_hidden * 0 for _ in self.config_target_layer_ids
        )
        for layer in self.layers:
            hidden_states, main_hiddens = layer(
                hidden_states, 0, input_ids, main_hiddens
            )

        x, main_x, draft_ids = self.mtp[0](
            main_hiddens,
            input_ids,
            embedding_weight=self.embed.weight,
        )
        for layer in self.mtp[1:]:
            x = layer(x, draft_ids, main_x)
        dependency = torch.zeros_like(x[..., :1]).sum()
        logits = self.lm_head(
            hidden_states + dependency,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.norm,
        )
        # Keep DSpark execution in the traced dependency chain. The zero tensor
        # is shape-independent and cannot propagate NaNs from the draft path.
        return logits

    @property
    def config_target_layer_ids(self) -> tuple[int, ...]:
        return self._config_target_layer_ids


class DeepseekV4DSparkForCausalLM(
    DeepseekV4NativePreTrainedModel, GenerationMixin
):
    """Transformers wrapper for the 0731 DeepSeek-V4 DSpark checkpoint."""

    config: ModelConfig
    _no_split_modules = ["TraceFriendlyBlock", "DSparkBlock"]
    _tied_weights_keys = {}
    _remap_state_dict_for_saving = staticmethod(
        DeepseekV4NativeForCausalLM._remap_state_dict_for_saving
    )

    def __init__(self, config: ModelConfig):
        if not config.is_dspark:
            raise ValueError(
                "DeepseekV4DSparkForCausalLM requires dspark_block_size > 0"
            )
        super().__init__(config)
        self.model = DSparkTransformer(config)
        self.model._config_target_layer_ids = tuple(
            config.dspark_target_layer_ids
        )
        self.vocab_size = config.vocab_size
        self.save_raw_format = False
        self.register_load_state_dict_pre_hook(
            self._remap_checkpoint_keys_for_loading
        )
        self._register_state_dict_hook(self._remap_state_dict_for_saving)
        self.post_init()

    def get_expanded_tied_weights_keys(
        self, all_submodels: bool = False
    ) -> dict[str, str]:
        return {}

    def get_input_embeddings(self):
        return self.model.embed

    def get_output_embeddings(self):
        return self.model.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("DeepSeek-V4 DSpark requires input_ids")
        logits = self.model(input_ids=input_ids)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:]
        elif isinstance(logits_to_keep, torch.Tensor):
            logits = logits[:, logits_to_keep]
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


__all__ = ["DeepseekV4DSparkForCausalLM"]
