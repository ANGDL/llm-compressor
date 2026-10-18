"""Native V4.1 complete-sequence calibration and cached reference inference.

CSA2 state is passed through block outputs instead of process-global variables.
DSpark is teacher forced explicitly; it is not assumed to run during generation.
"""

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from llmcompressor.modeling.deepseekv4.kernels import hc_split, sparse_attention
from llmcompressor.modeling.deepseekv4.model import RMSNorm, precompute_freqs_cis

from .cache import DeepseekV41Cache
from .config import DeepseekV41Config, ModelConfig
from .engram import Engram, EngramEmbedding, EngramHash
from .quantization import Linear, fp4_qdq, fp8_qdq
from .vision import Aligner, Vision


@lru_cache(maxsize=8)
def _frequency_buffer(args, device):
    # The V4 helper's cache does not include device. Loading on meta must not
    # poison the real runtime frequencies used after from_pretrained.
    with torch.device(device):
        return precompute_freqs_cis.__wrapped__(*args)


def _rope(x, freqs, inverse=False):
    value = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    batch = freqs.shape[0] if freqs.ndim == 3 else 1
    shape = [batch, x.shape[1]] + [1] * (x.ndim - 3) + [value.shape[-1]]
    rotation = freqs.conj() if inverse else freqs
    return torch.view_as_real(value * rotation.view(shape)).flatten(-2).to(x.dtype)


def _rotate_tail(x, freqs, rope_dim, inverse=False):
    return torch.cat(
        (x[..., :-rope_dim], _rope(x[..., -rope_dim:], freqs, inverse)), dim=-1
    )


def _stable_topk(scores, count):
    """Resolve equal scores by position, independent of future masked columns."""
    indices = scores.argsort(dim=-1, descending=True, stable=True)[..., :count]
    return scores.gather(-1, indices), indices


def _candidate_mask(scores, lengths, topk_blocks, block_size):
    width = scores.shape[-1]
    if width == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    blocks = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    blocks = blocks.unflatten(-1, (-1, block_size)).amax(-1)
    last = (lengths - 1) // block_size
    blocks = blocks.masked_fill(
        torch.arange(blocks.shape[-1], device=scores.device) == last, torch.inf
    )
    values, indices = _stable_topk(blocks, min(topk_blocks, blocks.shape[-1]))
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(
        -1, indices, values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Compressor(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.ratio = config.compress_ratios[layer_id]
        self.wkv = Linear(
            config.dim,
            config.head_dim,
            dtype=torch.float32 if self.ratio > 1 else torch.bfloat16,
        )
        self.norm = RMSNorm(config.head_dim, config.norm_eps)
        self.wgate = (
            Linear(config.dim, config.head_dim, dtype=torch.float32)
            if self.ratio > 1
            else None
        )

    def forward(self, x, mask, source=None):
        if source is not None:
            return self.forward_cached(x, mask, source.state)
        ratio = self.ratio
        if ratio == 1:
            return self.norm(self.wkv(x)), mask
        length = x.shape[1] - x.shape[1] % ratio
        values = self.wkv(x[:, :length].float()).unflatten(1, (-1, ratio))
        scores = self.wgate(x[:, :length].float()).unflatten(1, (-1, ratio))
        valid = mask[:, :length].unflatten(1, (-1, ratio))
        scores = scores.masked_fill(~valid.unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = scores.softmax(2).masked_fill(~valid.unsqueeze(-1), 0)
        pooled = (values * weights).sum(2)
        return self.norm(pooled.to(x.dtype)), valid.any(-1)

    def forward_cached(self, x, mask, state):
        """Project only new tokens and carry at most ratio-1 projected tokens."""
        if self.ratio == 1:
            return self.forward(x, mask)
        values, scores = self.wkv(x.float()), self.wgate(x.float())
        if "tail_values" in state:
            values = torch.cat((state["tail_values"].to(x.device), values), dim=1)
            scores = torch.cat((state["tail_scores"].to(x.device), scores), dim=1)
            mask = torch.cat((state["tail_mask"].to(x.device), mask), dim=1)
        length = values.shape[1] - values.shape[1] % self.ratio
        for key, value in (
            ("tail_values", values),
            ("tail_scores", scores),
            ("tail_mask", mask),
        ):
            state[key] = value[:, length:].detach().cpu().clone()
        values = values[:, :length].unflatten(1, (-1, self.ratio))
        scores = scores[:, :length].unflatten(1, (-1, self.ratio))
        valid = mask[:, :length].unflatten(1, (-1, self.ratio))
        scores = scores.masked_fill(~valid.unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = scores.softmax(2).masked_fill(~valid.unsqueeze(-1), 0)
        return self.norm((values * weights).sum(2).to(x.dtype)), valid.any(-1)


class Indexer(nn.Module):
    def __init__(self, config, layer_id, source_fp8):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.owns_k = layer_id in config.kv_source_layer_ids
        self.wq_b = Linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            source_fp8=source_fp8,
        )
        self.weights_proj = Linear(config.dim, config.index_n_heads)
        if self.owns_k:
            self.wk = Linear(config.head_dim, config.index_head_dim)
            self.k_norm = RMSNorm(config.index_head_dim, config.norm_eps)

    def forward(
        self,
        x,
        qr,
        latent,
        index_k,
        candidates,
        freqs,
        compressed_mask,
        start_pos=None,
        source=None,
        group_freqs=None,
    ):
        if start_pos is not None:
            return self.forward_cached(
                x, qr, latent, candidates, freqs, start_pos, source, group_freqs
            )
        config = self.config
        ratio = config.compress_ratios[self.layer_id]
        if self.owns_k:
            index_k = self.k_norm(self.wk(latent))
            index_k = _rotate_tail(
                index_k,
                freqs[: latent.shape[1] * ratio : ratio],
                config.rope_head_dim,
            )
            index_k = fp4_qdq(index_k)
        q = self.wq_b(qr).unflatten(-1, (config.index_n_heads, config.index_head_dim))
        q = fp4_qdq(_rotate_tail(q, freqs, config.rope_head_dim))
        weights = self.weights_proj(x) * (
            config.index_head_dim**-0.5 * config.index_n_heads**-0.5
        )
        scores = torch.einsum("bshd,btd->bsht", q, index_k)
        scores = (scores.relu() * weights.unsqueeze(-1)).sum(2)
        lengths = ((torch.arange(x.shape[1], device=x.device) + 1) // ratio).view(
            1, -1, 1
        )
        positions = torch.arange(index_k.shape[1], device=x.device)
        valid = (positions < lengths) & compressed_mask.unsqueeze(1)
        scores = scores.masked_fill(~valid, -torch.inf)
        if self.layer_id == config.candidate_source_layer_id:
            candidates = _candidate_mask(
                scores,
                lengths,
                config.candidate_topk_blocks,
                config.candidate_block_size,
            )
        elif 0 <= config.candidate_source_layer_id < self.layer_id:
            scores = scores.masked_fill(~candidates, -torch.inf)
        topk = min(config.index_topk, index_k.shape[1])
        values, indices = _stable_topk(scores, topk)
        indices = torch.where(values > -torch.inf, indices, -1).sort(-1).values
        return index_k, indices.int(), candidates

    def forward_cached(
        self, x, qr, latent, candidates, freqs, start_pos, source, group_freqs
    ):
        config = self.config
        if self.owns_k:
            new_index = fp4_qdq(
                _rotate_tail(
                    self.k_norm(self.wk(latent)), group_freqs, config.rope_head_dim
                )
            )
            DeepseekV41Cache.append(source.state, "index_k", new_index)
        index_k = source.state["index_k"]
        mask = source.state["compressed_mask"]
        q = self.wq_b(qr).unflatten(-1, (config.index_n_heads, config.index_head_dim))
        q = fp4_qdq(_rotate_tail(q, freqs, config.rope_head_dim))
        weights = self.weights_proj(x) * (
            config.index_head_dim**-0.5 * config.index_n_heads**-0.5
        )
        scores = torch.einsum("bshd,btd->bsht", q, index_k.to(x.device))
        scores = (scores.relu() * weights.unsqueeze(-1)).sum(2)
        positions = start_pos + torch.arange(x.shape[1], device=x.device)
        lengths = ((positions + 1) // config.compress_ratios[self.layer_id]).view(
            1, -1, 1
        )
        valid = (torch.arange(index_k.shape[1], device=x.device) < lengths) & mask.to(
            x.device
        ).unsqueeze(1)
        scores = scores.masked_fill(~valid, -torch.inf)
        if self.layer_id == config.candidate_source_layer_id:
            candidates = _candidate_mask(
                scores,
                lengths,
                config.candidate_topk_blocks,
                config.candidate_block_size,
            )
        elif 0 <= config.candidate_source_layer_id < self.layer_id:
            scores = scores.masked_fill(~candidates, -torch.inf)
        values, indices = _stable_topk(scores, min(config.index_topk, index_k.shape[1]))
        indices = torch.where(values > -torch.inf, indices, -1).sort(-1).values
        return index_k, indices.int(), candidates


class Attention(nn.Module):
    # The shared full-position table stays on CPU during streaming. Forward
    # transfers only the positions needed by the current chunk/token.
    _streaming_cpu_buffers = ("freqs_cis",)

    def __init__(self, config, layer_id, source_fp8):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.ratio = config.compress_ratios[layer_id]
        self.attn_sink = nn.Parameter(torch.zeros(config.n_heads, dtype=torch.float32))
        self.wq_a = Linear(config.dim, config.q_lora_rank, source_fp8=source_fp8)
        self.q_norm = RMSNorm(config.q_lora_rank, config.norm_eps)
        self.wq_b = Linear(
            config.q_lora_rank,
            config.n_heads * config.head_dim,
            source_fp8=source_fp8,
        )
        self.wkv = Linear(config.dim, config.head_dim, source_fp8=source_fp8)
        self.kv_norm = RMSNorm(config.head_dim, config.norm_eps)
        self.wo_a = Linear(
            config.n_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
        )
        self.wo_b = Linear(
            config.o_groups * config.o_lora_rank,
            config.dim,
            source_fp8=source_fp8,
        )
        self.compressor = (
            Compressor(config, layer_id)
            if layer_id in config.kv_source_layer_ids
            else None
        )
        self.indexer = (
            Indexer(config, layer_id, source_fp8)
            if layer_id in config.index_source_layer_ids
            else None
        )
        self._freqs_args = (
            config.rope_head_dim,
            config.max_seq_len,
            config.original_seq_len if self.ratio else 0,
            config.compress_rope_theta if self.ratio else config.rope_theta,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
        self.register_buffer(
            "freqs_cis",
            _frequency_buffer(self._freqs_args, str(torch.get_default_device())),
            persistent=False,
        )

    def reset_runtime_buffers(self):
        device = (
            self.wkv.weight.device if self.freqs_cis.is_meta else self.freqs_cis.device
        )
        self.freqs_cis = _frequency_buffer(self._freqs_args, str(device))

    def project_query(self, x, freqs):
        config = self.config
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (config.n_heads, config.head_dim))
        return qr, _rotate_tail(q, freqs, config.rope_head_dim)

    def project_output(self, output, freqs):
        config = self.config
        output = _rotate_tail(output, freqs, config.rope_head_dim, inverse=True)
        output = output.unflatten(
            -2, (config.o_groups, config.n_heads // config.o_groups)
        ).flatten(-2)
        output = fp8_qdq(output)
        # Invoke the Linear so PTQ hooks also see this grouped projection.
        all_outputs = self.wo_a(output).unflatten(
            -1, (config.o_groups, config.o_lora_rank)
        )
        output = torch.stack(
            [all_outputs[..., group, group, :] for group in range(config.o_groups)],
            dim=-2,
        )
        return self.wo_b(output.flatten(-2))

    def forward(self, x, mask, shared, cache=None, start_pos=0):
        if cache is not None:
            return self.forward_cached(x, mask, shared, cache, start_pos)
        config = self.config
        freqs = self.freqs_cis[: x.shape[1]].to(x.device)
        qr, q = self.project_query(x, freqs)
        window_kv = fp8_qdq(
            _rotate_tail(self.kv_norm(self.wkv(x)), freqs, config.rope_head_dim)
        )
        positions = torch.arange(x.shape[1], device=x.device)
        window = positions[:, None] - config.window_size + 1
        window = window + torch.arange(config.window_size, device=x.device)
        window = window.unsqueeze(0).expand(x.shape[0], -1, -1)
        valid_window = (window >= 0) & mask[:, window[0].clamp_min(0)]
        indices = window.masked_fill(~valid_window, -1).int()
        kv = window_kv
        if self.ratio:
            compressed, index_k, topk, candidates, compressed_mask = shared
            latent = None
            if self.compressor is not None:
                latent, compressed_mask = self.compressor(x, mask)
            if self.indexer is not None:
                index_k, topk, candidates = self.indexer(
                    x, qr, latent, index_k, candidates, freqs, compressed_mask
                )
            if latent is not None:
                compressed = _rotate_tail(
                    latent,
                    freqs[: latent.shape[1] * self.ratio : self.ratio],
                    config.rope_head_dim,
                )
                compressed = fp4_qdq(compressed, 16, e4m3_scale=True)
            shared = (compressed, index_k, topk, candidates, compressed_mask)
            kv = torch.cat((window_kv, compressed), dim=1)
            sparse = torch.where(topk >= 0, topk + x.shape[1], -1)
            indices = torch.cat((indices, sparse.int()), dim=-1)
        output = sparse_attention(q, kv, self.attn_sink, indices, config.head_dim**-0.5)
        return self.project_output(output, freqs), shared

    def forward_cached(self, x, mask, shared, cache, start_pos):
        config, state = self.config, cache.states[self.layer_id]
        freqs = self.freqs_cis[start_pos : start_pos + x.shape[1]].to(x.device)
        qr, q = self.project_query(x, freqs)
        current_kv = fp8_qdq(
            _rotate_tail(self.kv_norm(self.wkv(x)), freqs, config.rope_head_dim)
        )
        window_kv = current_kv
        if "window" in state:
            window_kv = torch.cat((state["window"].to(x.device), current_kv), dim=1)
        window_start = start_pos - (window_kv.shape[1] - x.shape[1])
        positions = start_pos + torch.arange(x.shape[1], device=x.device)
        indices = (
            positions[:, None]
            - config.window_size
            + 1
            + torch.arange(config.window_size, device=x.device)
        )
        indices = indices - window_start
        indices = indices.masked_fill(indices < 0, -1).unsqueeze(0).int()
        keep = max(0, config.window_size - 1)
        state["window"] = (
            window_kv[:, -keep:].detach().cpu().clone()
            if keep
            else window_kv[:, :0].detach().cpu()
        )
        kv = window_kv
        if self.ratio:
            source, _, topk, candidates, _ = shared
            latent = group_freqs = None
            if self.compressor is not None:
                source = cache.sources[self.layer_id]
                old_length = (
                    state["compressed"].shape[1] if "compressed" in state else 0
                )
                latent, new_mask = self.compressor(x, mask, source=source)
                group_positions = (
                    old_length + torch.arange(latent.shape[1], device=x.device)
                ) * self.ratio
                group_freqs = self.freqs_cis[
                    group_positions.to(self.freqs_cis.device)
                ].to(x.device)
                new_compressed = fp4_qdq(
                    _rotate_tail(latent, group_freqs, config.rope_head_dim),
                    16,
                    e4m3_scale=True,
                )
                cache.append(state, "compressed", new_compressed)
                cache.append(state, "compressed_mask", new_mask)
            compressed = source.state["compressed"]
            if self.indexer is not None:
                _, topk, candidates = self.indexer(
                    x,
                    qr,
                    latent,
                    None,
                    candidates,
                    freqs,
                    None,
                    start_pos=start_pos,
                    source=source,
                    group_freqs=group_freqs,
                )
            shared = (source, None, topk, candidates, None)
            kv = torch.cat((window_kv, compressed.to(x.device)), dim=1)
            sparse = torch.where(topk >= 0, topk + window_kv.shape[1], -1)
            indices = torch.cat((indices, sparse.int()), dim=-1)
        output = sparse_attention(q, kv, self.attn_sink, indices, config.head_dim**-0.5)
        return self.project_output(output, freqs), shared


class Gate(nn.Module):
    def __init__(self, config, *, draft=False, vision=True):
        super().__init__()
        self.config = config
        self.topk = (
            config.dspark_num_experts_per_tok if draft else config.num_experts_per_tok
        )
        count = config.dspark_n_routed_experts if draft else config.n_routed_experts
        self.weight = nn.Parameter(torch.empty(count, config.dim))
        self.bias = nn.Parameter(torch.zeros(count, dtype=torch.float32))
        self.bias_vl = (
            nn.Parameter(torch.zeros(count, dtype=torch.float32)) if vision else None
        )

    def forward(self, x, image_mask=None):
        config = self.config
        scores = F.linear(x.float(), self.weight.float()) / config.gate_temp
        if config.score_func == "softmax":
            scores = scores.softmax(-1)
        elif config.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        indices = (scores + bias).topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if config.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * config.route_scale, indices


class Expert(nn.Module):
    def __init__(self, config, source_fp8):
        super().__init__()
        self.limit = config.swiglu_limit
        self.w1 = Linear(config.dim, config.moe_inter_dim, source_fp8=source_fp8)
        self.w2 = Linear(config.moe_inter_dim, config.dim, source_fp8=source_fp8)
        self.w3 = Linear(config.dim, config.moe_inter_dim, source_fp8=source_fp8)

    def forward(self, x, weights=None):
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.limit > 0:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(-self.limit, self.limit)
        value = F.silu(gate) * up
        if weights is not None:
            value = value * weights
        return self.w2(value.to(x.dtype))


class DeepseekV41MoE(nn.Module):
    def __init__(self, config, source_fp8, *, draft=False, vision=True):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.gate = Gate(config, draft=draft, vision=vision)
        self.n_routed_experts = self.gate.weight.shape[0]
        self.experts = nn.ModuleList(
            [Expert(config, source_fp8) for _ in range(self.n_routed_experts)]
        )
        self.shared_experts = Expert(config, source_fp8)
        self.calibrate_all_experts = False
        self.requires_all_experts_for_calibration = draft

    def forward(self, x, image_mask=None):
        shape = x.shape
        x = x.reshape(-1, self.dim)
        mask = None if image_mask is None else image_mask.flatten()
        weights, indices = self.gate(x, mask)
        output = torch.zeros_like(x, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == expert_id)
            if self.calibrate_all_experts:
                # Unrouted inputs observe each expert with unit weight. Routed
                # rows retain their actual pre-w2 routing weights.
                all_weights = x.new_ones((x.shape[0], 1), dtype=weights.dtype)
                all_weights[rows, 0] = weights[rows, slots]
                values = expert(x, all_weights)
                output.index_add_(0, rows, values[rows].float())
            elif rows.numel():
                values = expert(x[rows], weights[rows, slots, None])
                output.index_add_(0, rows, values.float())
        return (output + self.shared_experts(x)).to(x.dtype).reshape(shape)


class DeepseekV41Block(nn.Module):
    def __init__(
        self, config, layer_id, source_fp8, *, vision=True, engram_checkpoint=None
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.attn = Attention(config, layer_id, source_fp8)
        self._target_flags = tuple(
            layer_id == target for target in config.dspark_target_layer_ids
        )
        self.ffn = DeepseekV41MoE(
            config, source_fp8, draft=layer_id >= config.n_layers, vision=vision
        )
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.engram = (
            Engram(
                config,
                layer_id,
                checkpoint=engram_checkpoint,
                source_fp8=source_fp8,
            )
            if layer_id in config.engram_layer_ids
            else None
        )
        for kind in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{kind}_fn",
                nn.Parameter(
                    torch.empty(
                        (2 + config.hc_mult) * config.hc_mult,
                        config.hc_mult * config.dim,
                        dtype=torch.float32,
                    )
                ),
            )
            self.register_parameter(
                f"hc_{kind}_base",
                nn.Parameter(
                    torch.zeros(
                        (2 + config.hc_mult) * config.hc_mult, dtype=torch.float32
                    )
                ),
            )
            self.register_parameter(
                f"hc_{kind}_scale", nn.Parameter(torch.ones(3, dtype=torch.float32))
            )

    def mixes(self, x, kind):
        flat = x.flatten(2).float()
        rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.config.norm_eps)
        value = F.linear(flat, getattr(self, f"hc_{kind}_fn").float()) * rms
        return hc_split(
            value,
            getattr(self, f"hc_{kind}_scale").float(),
            getattr(self, f"hc_{kind}_base").float(),
            self.config.hc_mult,
            self.config.hc_sinkhorn_iters,
            self.config.hc_eps,
        )

    @staticmethod
    def pre(x, mix):
        return (x.float() * mix.unsqueeze(-1)).sum(2).to(x.dtype)

    @staticmethod
    def post(x, residual, post, comb):
        return (
            post.unsqueeze(-1) * x.unsqueeze(-2)
            + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(2)
        ).to(x.dtype)

    def forward(
        self,
        x,
        pre_mix,
        shared,
        mask,
        image_mask,
        hashes,
        targets=(),
        cache=None,
        start_pos=0,
    ):
        if self.engram is not None:
            valid = mask & ~image_mask
            x = self.engram(x, hashes[:, :, self.engram.hash_index], valid)
        current = x.mean(2) if any(self._target_flags) else None
        targets = tuple(
            current if flag else previous
            for flag, previous in zip(self._target_flags, targets)
        )
        attn_pre, attn_post, attn_comb = self.mixes(x, "attn")
        attn_input = self.attn_norm(self.pre(x, pre_mix))
        value, shared = (
            self.attn(attn_input, mask, shared)
            if cache is None
            else self.attn(attn_input, mask, shared, cache=cache, start_pos=start_pos)
        )
        x = self.post(value, x, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = self.mixes(x, "ffn")
        value = self.ffn(self.ffn_norm(self.pre(x, attn_pre)), image_mask)
        return self.post(value, x, ffn_post, ffn_comb), ffn_pre, shared, targets


class Head(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab, dim, dtype=torch.float32))

    def forward(self, x):
        return F.linear(x.float(), self.weight.float())


class DSparkAttention(Attention):
    def forward(self, x, main_x, main_mask, context_positions):
        config = self.config
        main_freqs = self.freqs_cis[
            context_positions.clamp_min(0).to(self.freqs_cis.device)
        ].to(x.device)
        draft_positions = context_positions[:, -1:] + 1
        draft_positions = draft_positions + torch.arange(x.shape[1], device=x.device)
        draft_freqs = self.freqs_cis[draft_positions.to(self.freqs_cis.device)].to(
            x.device
        )
        _, q = self.project_query(x, draft_freqs)
        main_kv = fp8_qdq(
            _rotate_tail(
                self.kv_norm(self.wkv(main_x)), main_freqs, config.rope_head_dim
            )
        )
        draft_kv = fp8_qdq(
            _rotate_tail(self.kv_norm(self.wkv(x)), draft_freqs, config.rope_head_dim)
        )
        kv = torch.cat((main_kv, draft_kv), dim=1)
        # A draft block is bidirectional, unlike backbone sliding attention.
        valid = torch.cat(
            (main_mask, torch.ones_like(x[..., 0], dtype=torch.bool)), dim=1
        )
        indices = torch.arange(kv.shape[1], device=x.device)
        indices = indices.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
        indices = indices.masked_fill(~valid.unsqueeze(1), -1).int()
        output = sparse_attention(q, kv, self.attn_sink, indices, config.head_dim**-0.5)
        return self.project_output(output, draft_freqs)


class DSparkMarkovHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.dspark_markov_rank)
        self.head = Head(config.dspark_markov_rank, config.vocab_size)

    def forward(self, input_ids):
        embedded = self.embed(input_ids)
        return self.head(embedded), embedded


class DSparkConfidenceHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = Linear(
            config.dim + config.dspark_markov_rank, 1, dtype=torch.float32
        )

    def forward(self, hidden, markov):
        return self.proj(torch.cat((hidden, markov), dim=-1).float()).squeeze(-1)


@torch.fx.wrap
def _draft_inputs(input_ids, mask, targets, window_size, block_size, noise_id):
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    lengths = mask.long().sum(-1)
    # Hold out as many real draft tokens as possible, keeping one seed token.
    seeds = (lengths - block_size - 1).clamp_min(0)
    # Right-padded samples retain their original token positions.
    valid_positions = positions.expand_as(input_ids).masked_fill(~mask, -1)
    order = valid_positions.sort(-1).values
    selected = order.gather(1, (input_ids.shape[1] - lengths + seeds)[:, None])
    seeds = selected.squeeze(1).clamp_min(0)
    context_positions = seeds[:, None] - window_size
    context_positions = (
        context_positions + torch.arange(window_size, device=input_ids.device)[None, :]
    )
    context_indices = context_positions.clamp_min(0)
    context_mask = mask.gather(1, context_indices) & (context_positions >= 0)
    context = torch.cat(
        [
            hidden.gather(
                1, context_indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            )
            for hidden in targets
        ],
        dim=-1,
    )
    teacher_positions = (
        seeds[:, None] + torch.arange(block_size, device=input_ids.device)[None, :]
    )
    teacher_valid = teacher_positions < input_ids.shape[1]
    teacher_indices = teacher_positions.clamp(max=input_ids.shape[1] - 1)
    teacher_valid &= mask.gather(1, teacher_indices)
    teacher = input_ids.gather(1, teacher_indices)
    teacher = torch.where(teacher_valid, teacher, input_ids.gather(1, seeds[:, None]))
    draft_ids = torch.full_like(teacher, noise_id)
    draft_ids[:, 0] = teacher[:, 0]
    return context, context_mask, draft_ids, teacher, context_positions


class DeepseekV41DSparkBlock(DeepseekV41Block):
    def __init__(self, config, stage, source_fp8, *, vision=True):
        super().__init__(config, config.n_layers + stage, source_fp8, vision=vision)
        self.attn = DSparkAttention(config, config.n_layers + stage, source_fp8)
        self.stage = stage
        if stage == 0:
            self.main_proj = Linear(
                config.dim * len(config.dspark_target_layer_ids),
                config.dim,
                source_fp8=source_fp8,
            )
            self.main_norm = RMSNorm(config.dim, config.norm_eps)
        if stage == config.num_nextn_predict_layers - 1:
            self.norm = RMSNorm(config.dim, config.norm_eps)
            self.markov_head = DSparkMarkovHead(config)
            self.confidence_head = DSparkConfidenceHead(config)

    def forward(self, x, pre_mix, main_x, main_mask, context_positions):
        if self.stage == 0:
            main_x = self.main_norm(self.main_proj(main_x))
        attn_pre, attn_post, attn_comb = self.mixes(x, "attn")
        value = self.attn(
            self.attn_norm(self.pre(x, pre_mix)), main_x, main_mask, context_positions
        )
        x = self.post(value, x, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = self.mixes(x, "ffn")
        value = self.ffn(self.ffn_norm(self.pre(x, attn_pre)))
        return self.post(value, x, ffn_post, ffn_comb), ffn_pre, main_x


class DSparkInputAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window_size = config.window_size
        self.block_size = config.dspark_block_size
        self.noise_id = config.dspark_noise_token_id

    def forward(self, input_ids, mask, targets):
        return _draft_inputs(
            input_ids,
            mask,
            targets,
            self.window_size,
            self.block_size,
            self.noise_id,
        )


class Transformer(nn.Module):
    def __init__(
        self, config, *, tokenizer=None, engram_token_map=None, engram_checkpoint=None
    ):
        super().__init__()
        text = config.text_config
        self.config = config
        source_fp8 = bool(config.source_quantization_config) and (
            text.simulate_source_quantization
        )
        self.embed = nn.Embedding(text.vocab_size, text.dim)
        self.engram_hash = (
            EngramHash(text, tokenizer=tokenizer, token_map=engram_token_map)
            if text.engram_layer_ids
            else None
        )
        vision = bool(config.vision_config.num_hidden_layers)
        self.layers = nn.ModuleList(
            [
                DeepseekV41Block(
                    text,
                    index,
                    source_fp8,
                    vision=vision,
                    engram_checkpoint=engram_checkpoint,
                )
                for index in range(text.n_layers)
            ]
        )
        self.norm = RMSNorm(text.dim, text.norm_eps)
        self.lm_head = Head(text.dim, text.vocab_size)
        self.draft_input = DSparkInputAdapter(text)
        self.mtp = nn.ModuleList(
            [
                DeepseekV41DSparkBlock(text, stage, source_fp8, vision=vision)
                for stage in range(text.num_nextn_predict_layers)
            ]
        )
        self.vision = Vision(config.vision_config) if vision else None
        if vision:
            self.aligner = Aligner(config.vision_config, text)
            self.image_start = nn.Parameter(torch.empty(text.dim))
            self.image_end = nn.Parameter(torch.empty(text.dim))
            self.image_newline = nn.Parameter(torch.empty(text.dim))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        inputs_embeds=None,
        image_mask=None,
        cache=None,
        logits_to_keep=0,
    ):
        if cache is not None:
            return self.forward_cached(
                input_ids,
                attention_mask,
                inputs_embeds,
                image_mask,
                cache,
                logits_to_keep,
            )
        text = self.config.text_config
        mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        image_mask = torch.zeros_like(mask) if image_mask is None else image_mask.bool()
        hashes = (
            self.engram_hash(input_ids, mask & ~image_mask)
            if self.engram_hash is not None
            else None
        )
        h = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        h = h.unsqueeze(2).repeat(1, 1, text.hc_mult, 1)
        pre = torch.zeros_like(h[..., 0], dtype=torch.float32)
        # Build the initial one-hot without an FX-invisible in-place write.
        pre = (
            pre
            + F.one_hot(torch.zeros_like(input_ids), num_classes=text.hc_mult).float()
        )
        shared = (None, None, None, None, None)
        targets = (None,) * len(text.dspark_target_layer_ids)
        for layer in self.layers:
            h, pre, shared, targets = layer(
                h, pre, shared, mask, image_mask, hashes, targets
            )
        logits = self.lm_head(self.norm(DeepseekV41Block.pre(h, pre)))
        if self.mtp and self.config.calibrate_dspark:
            main_x, main_mask, draft_ids, teacher, context_positions = self.draft_input(
                input_ids, mask, targets
            )
            draft = self.embed(draft_ids).unsqueeze(2).repeat(1, 1, text.hc_mult, 1)
            draft_pre = F.one_hot(
                torch.zeros_like(draft_ids), num_classes=text.hc_mult
            ).float()
            for layer in self.mtp:
                draft, draft_pre, main_x = layer(
                    draft, draft_pre, main_x, main_mask, context_positions
                )
            hidden = DeepseekV41Block.pre(draft, draft_pre)
            draft_logits = self.lm_head(self.mtp[-1].norm(hidden))
            markov_logits, embedded = self.mtp[-1].markov_head(teacher)
            confidence = self.mtp[-1].confidence_head(hidden, embedded)
            # Keep every draft head in the graph without changing LM logits.
            dependency = (
                torch.zeros_like(draft_logits + markov_logits).sum()
                + torch.zeros_like(confidence).sum()
            )
            logits = logits + dependency
        return logits

    def forward_cached(
        self, input_ids, mask, inputs_embeds, image_mask, cache, logits_to_keep
    ):
        """Use the same blocks while keeping calibration tracing stateless."""
        text = self.config.text_config
        image_mask = torch.zeros_like(mask) if image_mask is None else image_mask.bool()
        hash_ids, hash_valid = input_ids, mask & ~image_mask
        if cache.history_ids is not None:
            hash_ids = torch.cat(
                (cache.history_ids.to(input_ids.device), hash_ids), dim=1
            )
            hash_valid = torch.cat(
                (cache.history_valid.to(input_ids.device), hash_valid), dim=1
            )
        hashes = (
            self.engram_hash(hash_ids, hash_valid)[:, -input_ids.shape[1] :]
            if self.engram_hash is not None
            else None
        )
        hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = hidden.unsqueeze(2).repeat(1, 1, text.hc_mult, 1)
        pre = F.one_hot(torch.zeros_like(input_ids), num_classes=text.hc_mult).float()
        shared = (None, None, None, None, None)
        targets = (None,) * len(text.dspark_target_layer_ids)
        for layer in self.layers:
            hidden, pre, shared, targets = layer(
                hidden,
                pre,
                shared,
                mask,
                image_mask,
                hashes,
                targets,
                cache=cache,
                start_pos=cache.seen_tokens,
            )
        hidden = self.norm(DeepseekV41Block.pre(hidden, pre))
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:]
        return self.lm_head(hidden)


class DeepseekV41NativeForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ModelConfig
    base_model_prefix = "model"
    _no_split_modules = [
        "DeepseekV41Block",
        "DeepseekV41DSparkBlock",
        "DeepseekV41VisionBlock",
    ]
    _tied_weights_keys = {}
    _supports_sdpa = True
    _skip_keys_device_placement = ["past_key_values"]

    @classmethod
    def _supports_default_dynamic_cache(cls):
        # CSA2 has shared, compressed state and Engram history; a standard KV
        # DynamicCache cannot describe it. Generation still uses HF's loop.
        return False

    def __init__(
        self, config, *, tokenizer=None, engram_token_map=None, engram_checkpoint=None
    ):
        super().__init__(config)
        self.model = Transformer(
            config,
            tokenizer=tokenizer,
            engram_token_map=engram_token_map,
            engram_checkpoint=engram_checkpoint,
        )
        self.vocab_size = config.text_config.vocab_size
        self.post_init()

    def _init_weights(self, module):
        std = self.config.text_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding, Head, EngramEmbedding)):
            if module.weight is not None:
                nn.init.normal_(module.weight, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, Attention):
            module.reset_runtime_buffers()
            nn.init.zeros_(module.attn_sink)
        elif isinstance(module, EngramHash):
            module.reset_runtime_buffers()
        elif isinstance(module, DeepseekV41Block):
            for name, parameter in module.named_parameters(recurse=False):
                if name.endswith("_fn"):
                    nn.init.normal_(parameter, std=std)
                elif name.endswith("_scale"):
                    nn.init.ones_(parameter)
                else:
                    nn.init.zeros_(parameter)
        elif isinstance(module, Gate):
            nn.init.normal_(module.weight, std=std)
            nn.init.zeros_(module.bias)
            if module.bias_vl is not None:
                nn.init.zeros_(module.bias_vl)
        elif isinstance(module, Transformer) and module.vision is not None:
            for name in ("image_start", "image_end", "image_newline"):
                nn.init.normal_(getattr(module, name), std=std)

    def get_input_embeddings(self):
        return self.model.embed

    def get_output_embeddings(self):
        return self.model.lm_head

    def encode_image(self, patches, height, width):
        patches = patches.to(dtype=self.model.vision.patch_embed.proj.weight.dtype)
        return self.model.aligner(
            self.model.vision(patches, height, width), height, width
        )

    def _forward_cached(
        self,
        input_ids,
        attention_mask,
        inputs_embeds,
        image_mask,
        cache,
        position_ids,
        logits_to_keep,
    ):
        if cache is None:
            cache = DeepseekV41Cache(self.config.text_config)
        if not isinstance(cache, DeepseekV41Cache):
            raise TypeError("V4.1 requires DeepseekV41Cache")
        if attention_mask is not None:
            expected_length = cache.seen_tokens + input_ids.shape[1]
            if attention_mask.shape[-1] not in (
                input_ids.shape[1],
                expected_length,
            ):
                raise ValueError("V4.1 attention_mask length disagrees with cache")
            if not attention_mask.bool().all():
                raise ValueError("V4.1 cached inference requires unpadded inputs")
            attention_mask = attention_mask[:, -input_ids.shape[1] :]
        if position_ids is not None:
            expected_positions = torch.arange(
                cache.seen_tokens,
                cache.seen_tokens + input_ids.shape[1],
                device=input_ids.device,
            ).unsqueeze(0)
            if not torch.equal(position_ids, expected_positions):
                raise ValueError("V4.1 position_ids disagree with committed cache")
        mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        cache.begin(input_ids, mask)
        try:
            logits = self.model(
                input_ids,
                mask,
                inputs_embeds,
                image_mask,
                cache=cache,
                logits_to_keep=logits_to_keep,
            )
            valid = mask if image_mask is None else mask & ~image_mask.bool()
            cache.commit(input_ids, valid)
            return CausalLMOutputWithPast(logits=logits, past_key_values=cache)
        except BaseException:
            cache._streaming_failed = True
            cache._active = False
            raise

    def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, **kwargs):
        prepared = super().prepare_inputs_for_generation(
            input_ids, inputs_embeds=inputs_embeds, **kwargs
        )
        if prepared.get("input_ids") is None:
            length = prepared["inputs_embeds"].shape[1]
            prepared["input_ids"] = input_ids[:, -length:]
        cache = prepared.get("past_key_values")
        if cache is not None and cache.get_seq_length() > 0:
            # Image embeddings belong to prefill. Newly generated tokens are text.
            if prepared.get("image_mask") is not None:
                prepared["image_mask"] = torch.zeros_like(
                    prepared["input_ids"], dtype=torch.bool
                )
        return prepared

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        image_mask=None,
        labels=None,
        logits_to_keep=0,
        past_key_values=None,
        use_cache=False,
        position_ids=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError(
                "V4.1 requires input_ids for deterministic Engram addressing"
            )
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if use_cache:
            if labels is not None:
                raise ValueError("Use full-sequence forward for training labels")
            return self._forward_cached(
                input_ids,
                attention_mask,
                inputs_embeds,
                image_mask,
                past_key_values,
                position_ids,
                logits_to_keep,
            )
        logits = self.model(input_ids, attention_mask, inputs_embeds, image_mask)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:]
        loss = (
            self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size)
            if labels is not None
            else None
        )
        return CausalLMOutputWithPast(logits=logits, loss=loss)


class DeepseekV41ForCausalLM(DeepseekV41NativeForCausalLM):
    """Original model_type bridge used only while Transformers lacks V4.1."""

    config_class = DeepseekV41Config


class DeepseekV41NativeForConditionalGeneration(DeepseekV41NativeForCausalLM):
    """Multimodal entry point using encode_image and prepared image embeddings."""


class DeepseekV41ForConditionalGeneration(DeepseekV41NativeForConditionalGeneration):
    config_class = DeepseekV41Config
