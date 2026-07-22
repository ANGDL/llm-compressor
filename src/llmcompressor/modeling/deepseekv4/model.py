import math
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from .config import ModelConfig
from .kernels import hc_split, sparse_attention

default_dtype = torch.float32
block_size = 128


def _runtime_buffer_device(args: ModelConfig) -> torch.device | None:
    return (
        torch.device("meta")
        if getattr(args, "_streaming_meta_init", False)
        else None
    )


@contextmanager
def set_dtype(dtype: torch.dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


class Embedding(nn.Embedding):
    pass


def _dequantize_block_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if scale is None:
        return weight.float()
    out_block = max(weight.shape[0] // scale.shape[0], 1)
    in_block = max(weight.shape[1] // scale.shape[1], 1)
    view = weight.float().view(scale.shape[0], out_block, scale.shape[1], in_block)
    view = view * scale[:, None, :, None].float()
    return view.reshape(weight.shape[0], weight.shape[1])


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    if scale is not None:
        weight = _dequantize_block_weight(weight, scale)
    return F.linear(x.float(), weight.float(), None if bias is None else bias.float()).to(x.dtype)


Linear = nn.Linear

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(input_dtype)


@lru_cache(2)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_idx, max_idx, dim):
        if min_idx == max_idx:
            max_idx += 1e-3
        return torch.clamp(
            (torch.arange(dim, dtype=torch.float32) - min_idx) / (max_idx - min_idx),
            0,
            1,
        )

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    y = x
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    x_rot = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    y.copy_(x_rot)
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    try:
        from compressed_tensors.transform.utils.hadamard import (
            deterministic_hadamard_matrix,
        )

        hidden_size = x.size(-1)
        hadamard = deterministic_hadamard_matrix(hidden_size, x.dtype, x.device)
        return F.linear(x, hadamard) * hidden_size**-0.5
    except Exception:
        return x


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def hc_split_softmax(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return hc_split(mixes, hc_scale, hc_base, hc_mult, 0, hc_eps)


class Compressor(nn.Module):
    def __init__(self, args: ModelConfig, compress_ratio: int = 4, head_dim: Optional[int] = None, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim or args.head_dim
        self.overlap = compress_ratio == 4
        self.compress_ratio = compress_ratio
        self.rotate = rotate
        overlap_factor = 1 + self.overlap
        self.ape = nn.Parameter(torch.empty(compress_ratio, overlap_factor * self.head_dim, dtype=torch.float32))
        self.wkv = Linear(self.dim, overlap_factor * self.head_dim, bias=False)
        self.wgate = Linear(self.dim, overlap_factor * self.head_dim, bias=False)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.rope_head_dim = args.rope_head_dim
        self.kv_cache: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None
        self.register_buffer(
            "kv_state",
            torch.zeros(
                args.max_batch_size,
                overlap_factor * compress_ratio,
                overlap_factor * self.head_dim,
                dtype=torch.float32,
                device=_runtime_buffer_device(args),
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (args.max_batch_size, overlap_factor * compress_ratio, overlap_factor * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
                device=_runtime_buffer_device(args),
            ),
            persistent=False,
        )

    def overlap_transform(self, tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        bsz, seqlen, _, _ = tensor.size()
        ratio = self.compress_ratio
        dim = self.head_dim
        new_tensor = tensor.new_full((bsz, seqlen, 2 * ratio, dim), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, dim:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :dim]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        if self.compress_ratio <= 0:
            return None
        assert self.kv_cache is not None
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        overlap = self.overlap
        dim = self.head_dim
        kv_state = cast(torch.Tensor, self.kv_state)
        score_state = cast(torch.Tensor, self.score_state)
        kv = self.wkv(x).float()
        score = self.wgate(x).float()
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                score_state[:bsz, :ratio] = score[:, cutoff - ratio : cutoff] + self.ape
            if remainder > 0:
                kv, kv_state[:bsz, offset : offset + remainder] = kv.split([cutoff, remainder], dim=1)
                score_state[:bsz, offset : offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            if not should_compress:
                return None
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0.0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
            compressed = self.norm(kv.to(x.dtype))
            chunk_count = cutoff // ratio
            freqs_cis = cast(torch.Tensor, self.freqs_cis)[:cutoff:ratio]
            if self.rope_head_dim > 0:
                apply_rotary_emb(compressed[..., -self.rope_head_dim :], freqs_cis)
            cast(torch.Tensor, self.kv_cache)[:bsz, :chunk_count] = compressed.float()
            return compressed

        should_compress = (start_pos + 1) % ratio == 0
        score = score + self.ape[start_pos % ratio]
        if overlap:
            kv_state[:bsz, ratio + start_pos % ratio] = kv[:, 0]
            score_state[:bsz, ratio + start_pos % ratio] = score[:, 0]
            if should_compress:
                kv_live = torch.cat([kv_state[:bsz, :ratio, :dim], kv_state[:bsz, ratio:, dim:]], dim=1)
                score_live = torch.cat([score_state[:bsz, :ratio, :dim], score_state[:bsz, ratio:, dim:]], dim=1)
                kv = (kv_live * score_live.softmax(dim=1)).sum(dim=1, keepdim=True)
                kv_state[:bsz, :ratio] = kv_state[:bsz, ratio:]
                score_state[:bsz, :ratio] = score_state[:bsz, ratio:]
        else:
            slot = start_pos % ratio
            kv_state[:bsz, slot, :dim] = kv[:, 0, :dim]
            score_state[:bsz, slot, :dim] = score[:, 0, :dim]
            if should_compress:
                kv = (kv_state[:bsz, :, :dim] * score_state[:bsz, :, :dim].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        compressed = self.norm(kv.to(x.dtype))
        freqs_cis = cast(torch.Tensor, self.freqs_cis)[start_pos + 1 - ratio].unsqueeze(0)
        if self.rope_head_dim > 0:
            apply_rotary_emb(compressed[..., -self.rope_head_dim :], freqs_cis)
        cast(torch.Tensor, self.kv_cache)[:bsz, start_pos // ratio] = compressed[:, 0].float()
        return compressed


class Indexer(nn.Module):
    def __init__(self, args: ModelConfig, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = Linear(args.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len // compress_ratio,
                self.head_dim,
                dtype=torch.float32,
                device=_runtime_buffer_device(args),
            ),
            persistent=False,
        )
        self.freqs_cis: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.shape
        assert self.freqs_cis is not None
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        ratio = self.compress_ratio
        end_pos = start_pos + seqlen
        q = self.wq_b(qr).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        if self.rope_head_dim > 0:
            apply_rotary_emb(q[..., -self.rope_head_dim :], freqs_cis)
            q = rotate_activation(q)
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = cast(torch.Tensor, self.kv_cache)
            self.compressor.freqs_cis = self.freqs_cis
        _ = self.compressor(x, start_pos)
        compressed_len = end_pos // ratio
        if compressed_len == 0:
            return torch.full((bsz, seqlen, 0), -1, dtype=torch.long, device=x.device)
        weights = self.weights_proj(x).float() * (self.softmax_scale * self.n_heads**-0.5)
        score = torch.einsum(
            "bshd,btd->bsht",
            q.float(),
            cast(torch.Tensor, self.kv_cache)[:bsz, :compressed_len].float(),
        )
        score = (score.relu() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(compressed_len, device=x.device).repeat(seqlen, 1) >= (
                torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            score = score + torch.where(mask, float("-inf"), 0.0)
        topk = min(self.index_topk, compressed_len)
        indices = score.topk(topk, dim=-1).indices
        if start_pos == 0:
            mask = indices >= (torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio)
            indices = torch.where(mask, torch.full_like(indices, -1), indices + offset)
        else:
            indices = indices + offset
        return indices


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = args.o_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id] if layer_id < len(args.compress_ratios) else 0
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, args.norm_eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, args.norm_eps)
        self.wo_a = Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            bias=False,
        )
        self.wo_b = Linear(self.n_groups * args.o_lora_rank, self.dim, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            self.indexer = Indexer(args, self.compress_ratio) if self.compress_ratio == 4 else None
        else:
            self.compressor = None
            self.indexer = None
        kv_cache_size = self.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        if self.compress_ratio:
            original_seq_len = args.original_seq_len
            rope_theta = args.compress_rope_theta
        else:
            original_seq_len = 0
            rope_theta = args.rope_theta
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                args.max_batch_size,
                kv_cache_size,
                self.head_dim,
                dtype=torch.float32,
                device=_runtime_buffer_device(args),
            ),
            persistent=False,
        )
        # `precompute_freqs_cis` is `@lru_cache`-d, so layers with matching
        # hyperparameters would otherwise share a single storage. When transformers
        # 5.x instantiates the model under `torch.device("meta")` and then moves
        # non-persistent buffers off meta via `named_non_persistent_buffers`
        # (default `remove_duplicate=True`), only the first alias is moved and the
        # rest stay on meta — which later crashes `set_onload_device` /
        # `offload_module` with "Cannot copy out of meta tensor". Clone here to
        # give every Attention its own storage.
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.rope_head_dim,
                args.max_seq_len,
                original_seq_len,
                rope_theta,
                args.rope_factor,
                args.beta_fast,
                args.beta_slow,
            ).clone().to(device=_runtime_buffer_device(args)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        freqs_cis = cast(torch.Tensor, self.freqs_cis)[start_pos : start_pos + seqlen]
        kv_cache = cast(torch.Tensor, self.kv_cache)
        win = self.window_size
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps).to(q.dtype)
        if self.rope_head_dim > 0:
            apply_rotary_emb(q[..., -self.rope_head_dim :], freqs_cis)

        kv = self.kv_norm(self.wkv(x))
        if self.rope_head_dim > 0:
            apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)

        if self.compress_ratio and self.compressor is not None and self.compressor.kv_cache is None:
            self.compressor.kv_cache = kv_cache[:, win:]
            self.compressor.freqs_cis = cast(torch.Tensor, self.freqs_cis)
            if self.indexer is not None:
                self.indexer.freqs_cis = cast(torch.Tensor, self.freqs_cis)

        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device).int()
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(self.compress_ratio, bsz, seqlen, start_pos, offset).to(x.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs.int()], dim=-1)

        if self.compress_ratio == 0:
            if start_pos == 0:
                if seqlen <= win:
                    kv_cache[:bsz, :seqlen] = kv.float()
                else:
                    cutoff = seqlen % win
                    left, right = kv[:, -win:].float().split([win - cutoff, cutoff], dim=1)
                    kv_cache[:bsz, cutoff:win] = left
                    kv_cache[:bsz, :cutoff] = right
                attn_kv = kv
            else:
                kv_cache[:bsz, start_pos % win] = kv[:, 0].float()
                attn_kv = kv_cache[:bsz]
        else:
            if start_pos == 0:
                if seqlen <= win:
                    kv_cache[:bsz, :seqlen] = kv.float()
                else:
                    cutoff = seqlen % win
                    left, right = kv[:, -win:].float().split([win - cutoff, cutoff], dim=1)
                    kv_cache[:bsz, cutoff:win] = left
                    kv_cache[:bsz, :cutoff] = right
                kv_compress = self.compressor(x, start_pos) if self.compressor is not None else None
                attn_kv = torch.cat([kv, kv_compress], dim=1) if kv_compress is not None else kv
            else:
                kv_cache[:bsz, start_pos % win] = kv[:, 0].float()
                if self.compressor is not None:
                    self.compressor(x, start_pos)
                attn_kv = kv_cache[:bsz]

        output = sparse_attention(q, attn_kv, self.attn_sink, topk_idxs, self.softmax_scale).to(x.dtype)
        if self.rope_head_dim > 0:
            apply_rotary_emb(output[..., -self.rope_head_dim :], freqs_cis, inverse=True)
        output = output.view(bsz, seqlen, self.n_local_groups, -1)
        # Call wo_a through forward() so quantization hooks can observe activations.
        # wo_a is a block-diagonal grouped projection: each group's input (per_group_dim)
        # maps to its own o_lora_rank outputs via the corresponding weight rows.
        per_group_dim = output.size(-1)
        all_inputs = output.reshape(-1, per_group_dim)
        all_outputs = self.wo_a(all_inputs)
        all_outputs = all_outputs.view(bsz * seqlen, self.n_local_groups, -1)
        output = torch.stack([
            all_outputs[:, g, g * self.o_lora_rank : (g + 1) * self.o_lora_rank]
            for g in range(self.n_local_groups)
        ], dim=1).view(bsz, seqlen, self.n_local_groups, self.o_lora_rank)
        return self.wo_b(output.flatten(2).to(x.dtype))


class Gate(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=torch.float32))
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.arange(args.vocab_size, dtype=torch.int64).unsqueeze(-1).repeat(1, self.topk) % args.n_routed_experts,
                requires_grad=False,
            )
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            original_scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            original_scores = scores.sigmoid()
        else:
            original_scores = F.softplus(scores).sqrt()
        routed_scores = original_scores if self.bias is None else original_scores + self.bias
        if self.hash and input_ids is not None:
            indices = self.tid2eid[input_ids]
        else:
            indices = routed_scores.topk(self.topk, dim=-1).indices
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, bias=False)
        self.w2 = Linear(inter_dim, dim, bias=False)
        self.w3 = Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.swiglu_limit)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        hidden = F.silu(gate) * up
        if weights is not None:
            hidden = hidden * weights
        return self.w2(hidden.to(x.dtype))


class DeepseekV4MoE(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.gate = Gate(layer_id, args)
        self.experts = nn.ModuleList(
            [
                Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
                for index in range(args.n_routed_experts)
            ]
        )
        self.shared_experts = Expert(args.dim, args.moe_inter_dim)

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        orig_shape = x.shape
        x = x.view(-1, self.dim)
        flat_ids = None if input_ids is None else input_ids.flatten()
        weights, indices = self.gate(x, flat_ids)
        output = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for expert_index in range(self.n_routed_experts):
            if counts[expert_index] == 0:
                continue
            idx, top = torch.where(indices == expert_index)
            expert = self.experts[expert_index]
            output[idx] += expert(x[idx], weights[idx, top, None])
        output += self.shared_experts(x)
        return output.to(x.dtype).view(orig_shape)


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = DeepseekV4MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        orig_shape = x.shape
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, hc_fn.float()) * rsqrt
        pre, post, comb = hc_split(
            mixes,
            hc_scale.float(),
            hc_base.float(),
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        reduced = torch.sum(pre.unsqueeze(-1) * flat.view(orig_shape), dim=2)
        return reduced.to(x.dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        mixed = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return mixed.to(x.dtype)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class ParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.dim, dtype=torch.float32))

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        orig_shape = x.shape
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + self.hc_eps
        return torch.sum(pre.unsqueeze(-1) * flat.view(orig_shape), dim=2).to(x.dtype)

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm):
        hidden = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = F.linear(norm(hidden).float(), self.weight.float())
        return logits


class MTPBlock(Block):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__(layer_id, args)
        self.e_proj = Linear(args.dim, args.dim, bias=False)
        self.h_proj = Linear(args.dim, args.dim, bias=False)
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.embed: Optional[Embedding] = None
        self.lm_head: Optional[ParallelHead] = None
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(args.hc_mult, args.hc_mult * args.dim))
            self.hc_head_base = nn.Parameter(torch.empty(args.hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        assert self.embed is not None and self.lm_head is not None
        assert input_ids is not None
        e = self.enorm(self.embed(input_ids))
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        return self.lm_head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)


class Transformer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.hc_mult = args.hc_mult
        self.embed = Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(layer_id, args) for layer_id in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        self.mtp = nn.ModuleList([MTPBlock(args.n_layers + layer_id, args) for layer_id in range(args.n_mtp_layers)])
        for mtp_block in self.mtp:
            mtp_block.embed = self.embed
            mtp_block.lm_head = self.lm_head
        self._dynamic_tied_weights_keys = {}
        for layer_id in range(args.n_mtp_layers):
            self._dynamic_tied_weights_keys[f"mtp.{layer_id}.embed.weight"] = "embed.weight"
            self._dynamic_tied_weights_keys[f"mtp.{layer_id}.lm_head.weight"] = "lm_head.weight"
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(args.hc_mult, args.hc_mult * args.dim))
            self.hc_head_base = nn.Parameter(torch.empty(args.hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        hidden_states = self.embed(input_ids)
        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, start_pos, input_ids)
        logits = self.lm_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        for mtp_block in self.mtp:
            mtp_block(hidden_states, start_pos, input_ids)
        return logits


class DeepseekV4NativePreTrainedModel(PreTrainedModel):
    config_class = ModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block", "MTPBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _keys_to_ignore_on_load_unexpected = [r".*zero_point$", r".*\.scale$"]

    @staticmethod
    def _remap_checkpoint_keys_for_loading(module, state_dict, prefix, *args):
        if prefix:
            return

        has_model_prefix = any(key.startswith("model.") for key in state_dict)
        roots = ("embed.", "layers.", "norm.", "head.", "lm_head.", "mtp.", "hc_head_")
        updates = {}
        removals = []

        if not has_model_prefix:
            for key in list(state_dict.keys()):
                if key.startswith(roots):
                    updates[f"model.{key}"] = state_dict[key]
                    removals.append(key)

        for key in removals:
            del state_dict[key]
        state_dict.update(updates)

        legacy_updates = {}
        legacy_removals = []
        for key in list(state_dict.keys()):
            if key.startswith("model.head."):
                legacy_updates[key.replace("model.head.", "model.lm_head.", 1)] = state_dict[key]
                legacy_removals.append(key)

        for key in legacy_removals:
            del state_dict[key]
        state_dict.update(legacy_updates)

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, (Linear, Embedding, ParallelHead)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(cast(torch.Tensor, module.bias))
            if getattr(module, "scale", None) is not None:
                torch.nn.init.ones_(cast(torch.Tensor, module.scale))
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, (Gate, Attention, Block, MTPBlock)):
            for name, parameter in module.named_parameters(recurse=False):
                if parameter is None:
                    continue
                if "scale" in name:
                    torch.nn.init.ones_(parameter)
                elif "base" in name or "bias" in name or "attn_sink" in name:
                    torch.nn.init.zeros_(parameter)
                else:
                    try:
                        torch.nn.init.normal_(parameter, mean=0.0, std=std)
                    except Exception:
                        pass

    def _move_missing_keys_from_meta_to_device(
        self, missing_keys, device_map, device_mesh, hf_quantizer
    ):
        # Under meta-device init (e.g. from_pretrained with ``max_memory=`` set,
        # which engages accelerate low-mem loading), transformers materializes
        # non-persistent buffers with ``torch.empty_like`` — i.e. *uninitialized*
        # memory — discarding the values computed in ``__init__``. In particular
        # ``freqs_cis`` (computed via ``precompute_freqs_cis``) becomes NaN/garbage
        # and is read by ``apply_rotary_emb`` before being overwritten, which
        # corrupts every activation and makes imatrix importance non-finite
        # (every module then falls back to uniform MSE). Restore the computed /
        # constant-init buffers here on their materialized device.
        super()._move_missing_keys_from_meta_to_device(
            missing_keys, device_map, device_mesh, hf_quantizer
        )
        self._reinitialize_non_persistent_buffers()

    def _reinitialize_non_persistent_buffers(self):
        """Recompute ``freqs_cis`` and re-zero kv cache/state buffers in place.

        Idempotent and safe to call on a normally-initialized model (values are
        just overwritten with the same content). Must run after the buffers have
        been moved off meta by the parent loader.
        """
        # ``precompute_freqs_cis`` is ``@lru_cache``-d and may still hold the
        # meta tensors produced during ``__init__``; clear so recomputation
        # happens on a real device.
        precompute_freqs_cis.cache_clear()
        args = self.config
        for module in self.modules():
            if isinstance(module, Attention):
                compress_ratio = module.compress_ratio
                original_seq_len = args.original_seq_len if compress_ratio else 0
                rope_theta = args.compress_rope_theta if compress_ratio else args.rope_theta
                freqs_cis = precompute_freqs_cis(
                    module.rope_head_dim,
                    args.max_seq_len,
                    original_seq_len,
                    rope_theta,
                    args.rope_factor,
                    args.beta_fast,
                    args.beta_slow,
                )
                module._buffers["freqs_cis"] = freqs_cis.clone().to(module.freqs_cis.device)
                kv_cache_size = module.window_size + (
                    args.max_seq_len // compress_ratio if compress_ratio else 0
                )
                module._buffers["kv_cache"] = torch.zeros(
                    args.max_batch_size,
                    kv_cache_size,
                    module.head_dim,
                    dtype=torch.float32,
                    device=module.kv_cache.device,
                )
            elif isinstance(module, Compressor):
                overlap_factor = 1 + int(module.overlap)
                shape = (
                    args.max_batch_size,
                    overlap_factor * module.compress_ratio,
                    overlap_factor * module.head_dim,
                )
                module._buffers["kv_state"] = torch.zeros(
                    *shape, dtype=torch.float32, device=module.kv_state.device
                )
                module._buffers["score_state"] = torch.full(
                    shape,
                    float("-inf"),
                    dtype=torch.float32,
                    device=module.score_state.device,
                )
            elif isinstance(module, Indexer):
                module._buffers["kv_cache"] = torch.zeros(
                    args.max_batch_size,
                    args.max_seq_len // module.compress_ratio,
                    module.head_dim,
                    dtype=torch.float32,
                    device=module.kv_cache.device,
                )


class DeepseekV4NativeForCausalLM(DeepseekV4NativePreTrainedModel, GenerationMixin):
    config: ModelConfig
    _tied_weights_keys = {
        "model.mtp.0.embed.weight": "model.embed.weight",
        "model.mtp.0.lm_head.weight": "model.lm_head.weight",
    }

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.register_load_state_dict_pre_hook(self._remap_checkpoint_keys_for_loading)
        self._register_state_dict_hook(self._remap_state_dict_for_saving)
        self.save_raw_format = False
        self.model = Transformer(config)
        self.vocab_size = config.vocab_size
        self.post_init()

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        # MTP embed/lm_head are tied to main embed/lm_head regardless of
        # tie_word_embeddings (which only controls the main lm_head<->embed tie).
        tied = {}
        for i in range(getattr(self.config, "n_mtp_layers", 1)):
            tied[f"model.mtp.{i}.embed.weight"] = "model.embed.weight"
            tied[f"model.mtp.{i}.lm_head.weight"] = "model.lm_head.weight"
        return tied

    @staticmethod
    def _remap_state_dict_for_saving(module, state_dict, prefix, local_metadata):
        if not getattr(module, "save_raw_format", False):
            return
        remapped = {}
        for key, value in list(state_dict.items()):
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            new_key = new_key.replace(".weight_scale", ".scale")
            new_key = new_key.replace("lm_head.", "head.")
            remapped[new_key] = value
        state_dict.clear()
        state_dict.update(remapped)
        if hasattr(module, "_tied_weights_keys") and module._tied_weights_keys:
            new_tied = {}
            for k, v in module._tied_weights_keys.items():
                nk = k.removeprefix("model.").replace("lm_head.", "head.")
                nv = v.removeprefix("model.").replace("lm_head.", "head.")
                new_tied[nk] = nv
            module._tied_weights_keys = new_tied

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
        logits = self.model(input_ids=input_ids)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else logits_to_keep
        if isinstance(slice_indices, slice):
            logits = logits[:, slice_indices, :]
        elif isinstance(slice_indices, torch.Tensor):
            logits = logits[:, slice_indices, :]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
