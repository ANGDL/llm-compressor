"""Deterministic Engram addressing and bounded row materialization."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from .quantization import Linear


def compressed_token_map(tokenizer):
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalize = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    mapping, keys = [], {}
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            key = normalize.normalize_str(text) or text
        mapping.append(keys.setdefault(key, len(keys)))
    return mapping


def _next_prime(start, seen):
    value = start + 1
    while True:
        prime = value >= 2 and (
            value == 2
            or (
                value % 2 and all(value % d for d in range(3, math.isqrt(value) + 1, 2))
            )
        )
        if prime and value not in seen:
            seen.add(value)
            return value
        value += 1


@torch.fx.wrap
def _hash_ids(input_ids, valid, token_map, multipliers, primes, offsets, pad_id):
    compressed = token_map[input_ids]
    compressed = torch.where(valid, compressed, -1)
    batch, length = input_ids.shape
    tokens, blocked = [], torch.zeros_like(valid)
    for shift in range(multipliers.shape[-1]):
        if shift == 0:
            source = compressed
        elif shift < length:
            source = F.pad(compressed[:, :-shift], (shift, 0), value=-1)
        else:
            source = torch.full_like(compressed, -1)
        blocked = blocked | (source == -1)
        tokens.append(torch.where(blocked, pad_id, source))
    products = torch.stack(tokens, dim=-1).unsqueeze(2) * multipliers
    rolling, hashes = products[..., 0], []
    for order in range(1, multipliers.shape[-1]):
        rolling = torch.bitwise_xor(rolling, products[..., order])
        hashes.append(rolling.unsqueeze(-1) % primes[:, order - 1])
    return torch.cat(hashes, dim=-1) + offsets


class EngramHash(nn.Module):
    def __init__(self, config, *, tokenizer=None, token_map=None):
        super().__init__()
        if token_map is None:
            if tokenizer is None:
                raise ValueError("Engram needs the checkpoint tokenizer or token_map")
            token_map = compressed_token_map(tokenizer)
        vocab_size = len(set(token_map))
        if (
            len(token_map) != config.vocab_size
            or vocab_size != config.engram_compressed_vocab_size
        ):
            raise ValueError("Engram compressed token map disagrees with configuration")
        self.pad_id = token_map[config.engram_pad_token_id]
        primes, multipliers, offsets, seen = [], [], [], set()
        bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
        for index, layer_id in enumerate(config.engram_layer_ids):
            per_layer = []
            for _ in range(config.engram_max_ngram_size - 1):
                current, sizes = config.engram_vocab_size - 1, []
                for _ in range(config.engram_n_heads):
                    current = _next_prime(current, seen)
                    sizes.append(current)
                per_layer.append(sizes)
            flat = [size for group in per_layer for size in group]
            if sum(flat) != config.engram_num_embeddings[index]:
                raise ValueError("Engram table size disagrees with its prime buckets")
            primes.append(per_layer)
            offsets.append(np.cumsum([0, *flat[:-1]]))
            generator = np.random.default_rng(10007 * layer_id)
            values = generator.integers(
                0, bound, size=(config.engram_max_ngram_size,), dtype=np.int64
            )
            multipliers.append(values * 2 + 1)
        self._hash_arrays = {
            "token_map": np.asarray(token_map, dtype=np.int64),
            "multipliers": np.asarray(multipliers, dtype=np.int64),
            "primes": np.asarray(primes, dtype=np.int64),
            "offsets": np.asarray(offsets, dtype=np.int64),
        }
        for name, array in self._hash_arrays.items():
            self.register_buffer(name, torch.tensor(array), persistent=False)

    def reset_runtime_buffers(self):
        # Transformers materializes nonpersistent meta buffers with empty_like;
        # reconstruct their deterministic values during missing-state init.
        for name, array in self._hash_arrays.items():
            device = getattr(self, name).device
            self._buffers[name] = torch.tensor(array, device=device)

    def forward(self, input_ids, valid):
        return _hash_ids(
            input_ids,
            valid,
            self.token_map.to(input_ids.device),
            self.multipliers.to(input_ids.device),
            self.primes.to(input_ids.device),
            self.offsets.to(input_ids.device),
            self.pad_id,
        )


class CheckpointEngramLookup:
    """Gather only requested BF16 rows or FP8 bytes and their row scales."""

    def __init__(self, checkpoint, layer_id, shape):
        self.root = Path(checkpoint).expanduser().resolve()
        index = json.loads((self.root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        prefix = f"layers.{layer_id}.engram.embed."
        if f"{prefix}weight" not in index:
            prefix = f"model.{prefix}"
        self.names = (f"{prefix}weight",)
        if f"{prefix}scale" in index:
            self.names += (f"{prefix}scale",)
        self.paths = tuple((self.root / index[name]).resolve() for name in self.names)
        if any(not path.is_relative_to(self.root) for path in self.paths):
            raise ValueError("Engram shard escapes checkpoint directory")
        self.shape = tuple(shape)

    def __call__(self, indices):
        flat = indices.detach().to("cpu").reshape(-1)
        unique, inverse = flat.unique(sorted=True, return_inverse=True)
        values = []
        for path, name in zip(self.paths, self.names):
            with safe_open(path, framework="pt", device="cpu") as shard:
                # get_tensor is an mmap view: byte gathering happens before float().
                tensor = shard.get_tensor(name)
                expected = (
                    self.shape
                    if name.endswith(".weight")
                    else (self.shape[0], self.shape[1] // 32)
                )
                if tuple(tensor.shape) != expected:
                    raise ValueError(f"Invalid Engram shape for {name}")
                if name.endswith(".weight") and tensor.dtype in {
                    torch.bfloat16,
                    torch.float16,
                    torch.float32,
                }:
                    if len(self.names) != 1:
                        raise ValueError(
                            "Unquantized Engram weights must not have scales"
                        )
                    rows = tensor.index_select(0, unique).index_select(0, inverse)
                    return rows.reshape(*indices.shape, self.shape[1]).to(
                        device=indices.device, dtype=torch.bfloat16
                    )
                dtype = (
                    torch.float8_e4m3fn
                    if name.endswith(".weight")
                    else torch.float8_e8m0fnu
                )
                if tensor.dtype not in {dtype, torch.uint8}:
                    raise TypeError(f"Expected source FP8 Engram tensor: {name}")
                rows = tensor.view(torch.uint8).index_select(0, unique)
                values.append(rows.view(dtype).float())
        if len(values) != 2:
            raise ValueError("FP8 Engram weights require row scales")
        weight, scale = values
        decoded = weight.unflatten(-1, (-1, 32)) * scale.unsqueeze(-1)
        result = decoded.flatten(-2).index_select(0, inverse)
        return result.reshape(*indices.shape, self.shape[1]).to(
            device=indices.device, dtype=torch.bfloat16
        )


class EngramEmbedding(nn.Module):
    def __init__(self, rows, dim, *, checkpoint=None, layer_id=None):
        super().__init__()
        self.lookup = None
        if checkpoint is None:
            self.weight = nn.Parameter(torch.empty(rows, dim))
        else:
            self.register_parameter("weight", None)
            self.lookup = CheckpointEngramLookup(checkpoint, layer_id, (rows, dim))

    def forward(self, indices):
        if self.lookup is not None:
            return self.lookup(indices)
        return F.embedding(indices, self.weight)


class Engram(nn.Module):
    def __init__(self, config, layer_id, *, checkpoint=None):
        super().__init__()
        self.config = config
        self.hash_index = config.engram_layer_ids.index(layer_id)
        self.embed = EngramEmbedding(
            config.engram_num_embeddings[self.hash_index],
            config.engram_head_dim,
            checkpoint=checkpoint,
            layer_id=layer_id,
        )
        width = (
            (config.engram_max_ngram_size - 1)
            * config.engram_n_heads
            * config.engram_head_dim
        )
        self.wkv = Linear(width, config.dim * (config.hc_mult + 1))
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.dim))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.dim))

    def forward(self, x, hashes, valid):
        config = self.config
        embedding = self.embed(hashes).flatten(-2)
        # Checkpoint rows are BF16 even during FP32 calibration. PTQ wraps the
        # nn.Linear class forward, so match dtypes before entering that wrapper.
        kv = self.wkv(embedding.to(self.wkv.weight.dtype)).to(embedding.dtype)
        key, value = kv.split([config.hc_mult * config.dim, config.dim], dim=-1)
        key = key.float().unflatten(-1, (config.hc_mult, config.dim))
        hidden = x.float()
        rstd = torch.rsqrt(hidden.square().mean(-1) + config.norm_eps)
        rstd = rstd * torch.rsqrt(key.square().mean(-1) + config.norm_eps)
        dot = (
            (hidden * self.q_weight.float() * self.k_weight.float() * key).sum(-1)
            * rstd
            * config.dim**-0.5
        )
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        gate = gate.masked_fill(~valid.unsqueeze(-1), 0)
        return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
