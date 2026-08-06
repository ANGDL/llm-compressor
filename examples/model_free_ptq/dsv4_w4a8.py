"""
Convert DeepSeek-V4-Flash (FP4 + FP8 mixed) -> W4A8
(INT4 experts + INT8 projectors, per-channel symmetric).

Based on ``convert_dsv4_to_w4a8.py`` (raw safetensors reader, FP4/FP8
dequantization, INT4 packing, multi-GPU shard fan-out). The converter supports
both the preview MTP layout (``mtp.e_proj``/``mtp.h_proj``) and the 0731 DSpark
layout (``mtp.0.main_proj`` plus three MTP blocks).

Source checkpoint layout (DeepSeek native naming):
  - FP8 weights : F8_E4M3 weight + F8_E8M0 block-128x128 scale
                  (attn wq_a/wq_b/wkv/wo_b, ffn.shared_experts.w*,
                   mtp.* counterparts, mtp.e_proj/h_proj/main_proj,
                   attn.indexer.wq_b)
  - FP4 experts : packed e2m1 stored as int8 (2 nibbles/byte) + F8_E8M0
                  block-32 scale; logical in_dim = 2 * stored in_dim
                  (layers/mtp ffn.experts.N.w1/w2/w3)
  - kept as-is  : norms, ffn.gate.weight/bias/tid2eid, embed, head,
                  hc_* params, attn_sink, attn.compressor.*,
                  attn.indexer.compressor.*/weights_proj, hc_head_*

Target scheme (compatible with the KLX ``w4a8_int4`` loader — see
``python/sglang/srt/layers/quantization/w4a8_int4.py``):
  - MoE experts (FP4 e2m1)           -> per-channel symmetric INT4, packed
    2 nibbles/byte into int8, shape (N, K/2).  Scale is F32 per-output-row
    [N, 1], stored as ``<name>.scale``.  The loader pre-multiplies the scale
    by 7 (``process_weights_after_loading``), so we store ``abs_max / 8``
    with the quantized range clamped to [-8, 7] (matches the GLM/KiMi
    w4a8 convention).
  - Attn / shared / e_proj / h_proj / main_proj (FP8) -> per-channel symmetric
    INT8 weight (torch.int8) with F32 per-output-channel scale named
    ``<name>.scale``.
    The W8A8 loader pre-multiplies by 127, so we store ``abs_max / 127``.
  - wo_a (FP8)                       -> BF16 (kept, not quantized; the DSV4
    loader's manual einsum path consumes BF16 wo_a).
  - DSpark markov/confidence heads   -> copied through byte-for-byte.
  - Everything else                  -> copied through byte-for-byte.

Runtime W4A8_W8_PATTERN:
  The ``w4a8_int4`` loader uses the ``W4A8_W8_PATTERN`` env var to decide
  which non-expert Linear layers fall back to W8A8 (INT8) instead of W4A8
  (INT4).  This script always stores non-expert projectors as INT8 and
  experts as INT4, so the server should set ``W4A8_W8_PATTERN`` to match
  every quantized projector name (i.e. all non-expert Linear layers use
  W8A8, only MoE experts use W4A8).  See the DSV4 w4a8 server scripts under
  ``klx/server/ds_v4/w4a8/``.

Usage:
  python convert_flash_to_w4a8.py \
      --model-id /ssd3/models/DeepSeek-V4-Flash \
      --save-dir /ssd3/models/DeepSeek-V4-Flash-W4A8 \
      [--gpus 0,1,2,3,4,5,6,7] [--device cuda] [--overwrite]

Performance:
  The FP4 dequant + per-channel requant dominates runtime. Two changes give
  the speedup (inherited from convert_dsv4_to_w4a8.py):
    1. --device cuda  : run all dequant/quant math on a GPU.
    2. --gpus a,b,c   : convert N shards concurrently, one process per GPU
       (ProcessPoolExecutor with 'spawn' — forked CUDA contexts crash).
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import torch
from safetensors.torch import save_file
from tqdm import tqdm

# ---------------------------------------------------------------------------
# FP4 e2m1 lookup table (signed).  Index = 4-bit nibble.
# Matches inference/convert.py and convert_flash_to_int8.py.
# ---------------------------------------------------------------------------

FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

FP8_BLOCK = 128  # block-wise scale tile for FP8 e4m3 weights
FP4_BLOCK = 32  # block-wise scale tile for packed FP4 experts (along logical in_dim)

# safetensors dtype string -> torch dtype
_ST_DTYPE = {
    "F8_E4M3": torch.float8_e4m3fn,
    "I8": torch.int8,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F16": torch.float16,
    "I64": torch.int64,
    "I32": torch.int32,
}


# ---------------------------------------------------------------------------
# Raw safetensors reader
#
# The KLX xpytorch build ships a safetensors_rust that fails to parse some of
# these headers ("InvalidHeaderDeserialization"), so we read the header and the
# raw tensor bytes ourselves and build tensors with torch.frombuffer.  F8_E8M0
# is not a torch dtype here, so it is returned as a uint8 exponent tensor.
# ---------------------------------------------------------------------------


def _read_st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def _load_state_dict_raw(path):
    """Return {name: (kind, tensor)} where kind is the safetensors dtype string.

    F8_E8M0 tensors are returned as ("E8M0", uint8_tensor) since torch has no
    e8m0 dtype on this stack; the raw byte is the IEEE-754 biased exponent.
    """
    header, base = _read_st_header(path)
    out = {}
    with open(path, "rb") as f:
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = meta["dtype"]
            start, end = meta["data_offsets"]
            f.seek(base + start)
            buf = bytearray(f.read(end - start))
            shape = meta["shape"]
            if dtype == "F8_E8M0":
                t = torch.frombuffer(buf, dtype=torch.uint8).clone()
                out[name] = ("E8M0", t.view(shape) if shape else t)
                continue
            torch_dtype = _ST_DTYPE.get(dtype)
            if torch_dtype is None:
                raise ValueError(f"Unsupported source dtype {dtype} for tensor {name}")
            t = torch.frombuffer(buf, dtype=torch_dtype).clone()
            out[name] = (dtype, t.view(shape) if shape else t)
    return header, out


def _e8m0_to_f32(exp_u8: torch.Tensor) -> torch.Tensor:
    """E8M0 stores a biased power-of-two exponent: value = 2^(byte - 127)."""
    return torch.pow(2.0, exp_u8.to(torch.float32) - 127.0)


# ---------------------------------------------------------------------------
# Dequantization (source FP8 / FP4 -> float32)
# ---------------------------------------------------------------------------


def _dequant_fp8_blockwise(
    weight_e4m3: torch.Tensor, scale_e8m0: torch.Tensor, device="cpu"
) -> torch.Tensor:
    """FP8 e4m3 weight [N, K] with E8M0 128x128 block scale -> float32 [N, K]."""
    w = weight_e4m3.to(device=device, dtype=torch.float32)
    n, k = w.shape
    assert (
        n % FP8_BLOCK == 0 and k % FP8_BLOCK == 0
    ), f"FP8 weight {tuple(w.shape)} not divisible by {FP8_BLOCK}"
    sf = _e8m0_to_f32(scale_e8m0.to(device))
    bn, bk = n // FP8_BLOCK, k // FP8_BLOCK
    assert tuple(sf.shape) == (
        bn,
        bk,
    ), f"FP8 scale shape {tuple(sf.shape)} != expected {(bn, bk)} for weight {tuple(w.shape)}"
    w = w.view(bn, FP8_BLOCK, bk, FP8_BLOCK)
    w = w * sf[:, None, :, None]
    return w.reshape(n, k)


def _dequant_fp4_expert(
    weight_i8: torch.Tensor, scale_e8m0: torch.Tensor, device="cpu"
) -> torch.Tensor:
    """Packed FP4 e2m1 expert (int8, 2 nibbles/byte) [N, K/2] with E8M0 block-32
    scale [N, K/32] -> float32 [N, K] (logical in_dim K = 2 * stored)."""
    x = weight_i8.to(device).view(torch.uint8)
    low = (x & 0x0F).long()
    high = ((x >> 4) & 0x0F).long()
    table = FP4_TABLE.to(x.device)
    unpacked = torch.stack([table[low], table[high]], dim=-1).flatten(1)  # [N, K]
    sf = _e8m0_to_f32(scale_e8m0.to(device))  # [N, K/32]
    n, k = unpacked.shape
    assert (
        sf.shape[0] == n and sf.shape[1] == k // FP4_BLOCK
    ), f"FP4 expert scale {tuple(sf.shape)} incompatible with logical weight {tuple(unpacked.shape)}"
    scales = sf.repeat_interleave(FP4_BLOCK, dim=1)  # [N, K]
    return unpacked * scales


# ---------------------------------------------------------------------------
# Requantization (float32 -> per-channel symmetric INT4 / INT8)
#
# INT4 convention (matches klx/quant_scripts/GLM/quant_w4a8.py):
#   q_max = 8, scale = abs_max / 8, q = round(w / scale).clamp(-8, 7)
#   Two INT4 values packed into one int8 byte (low nibble = first element).
#
# INT8 convention (same as convert_flash_to_int8.py):
#   q_max = 127, scale = abs_max / 127, q = round(w / scale).clamp(-127, 127)
# ---------------------------------------------------------------------------


def _pack_int4_to_int8(weight_int4: torch.Tensor) -> torch.Tensor:
    """Pack two signed INT4 values into one int8 byte.

    Compatible with the ``unpack_int4_pairs`` convention used by the KLX
    w4a8_int4 loader: first element (index 0) is the low nibble, second
    (index 1) is the high nibble.

    Args:
        weight_int4: INT4 tensor of shape (out_features, in_features)
                     with values in [-8, 7].

    Returns:
        packed_weight: INT8 tensor of shape (out_features, in_features // 2)
    """
    weight_int4 = weight_int4.clamp(-8, 7).to(torch.int8)

    out_features, in_features = weight_int4.shape
    if in_features % 2 != 0:
        weight_int4 = torch.cat(
            [weight_int4, torch.zeros(out_features, 1, dtype=torch.int8)], dim=1
        )
        in_features += 1

    weight_reshaped = weight_int4.reshape(out_features, in_features // 2, 2)
    low_nibbles = weight_reshaped[:, :, 0]
    high_nibbles = weight_reshaped[:, :, 1]
    packed_weight = (high_nibbles << 4) | (low_nibbles & 0x0F)
    return packed_weight.to(torch.int8)


def _per_channel_quantize_int4(weight_f32: torch.Tensor):
    """Per-channel (per output row) symmetric INT4.

    Returns (packed_int4_weight int8 [N, K/2], f32 scale [N, 1]).
    Scale is ``abs_max / 8`` (the w4a8_int4 loader pre-multiplies by 7; the
    7/8 ratio is the same convention used by the GLM/KiMi w4a8 scripts).
    """
    assert weight_f32.dim() == 2, f"expected 2D weight, got {tuple(weight_f32.shape)}"
    w = weight_f32.to(torch.float32)
    abs_max = w.abs().amax(dim=1, keepdim=True)  # [N, 1]
    q_max = 8.0
    scale = (abs_max / q_max).clamp_min(1e-12)
    q = (w / scale).round().clamp(-q_max, q_max - 1).to(torch.int8)
    packed = _pack_int4_to_int8(q)
    return packed, scale.to(torch.float32)


def _per_channel_quantize_int8(weight_f32: torch.Tensor):
    """Per-channel (per output row) symmetric INT8.

    Returns (int8 weight [N, K], f32 scale [N, 1]).
    Scale is ``abs_max / 127`` (the W8A8 loader pre-multiplies by 127).
    """
    assert weight_f32.dim() == 2, f"expected 2D weight, got {tuple(weight_f32.shape)}"
    w = weight_f32.to(torch.float32)
    abs_max = w.abs().amax(dim=1, keepdim=True)  # [N, 1]
    q_max = 127.0
    scale = (abs_max / q_max).clamp_min(1e-12)
    q = (w / scale).round().clamp(-q_max, q_max).to(torch.int8)
    return q, scale.to(torch.float32)


# ---------------------------------------------------------------------------
# Per-tensor classification
# ---------------------------------------------------------------------------

# FP8 e4m3 projector weights that become per-channel INT8.
_FP8_INT8_PROJ_SUFFIXES = (
    ".attn.wq_a.weight",
    ".attn.wq_b.weight",
    ".attn.wkv.weight",
    ".attn.wo_b.weight",
    ".attn.indexer.wq_b.weight",
    ".ffn.shared_experts.w1.weight",
    ".ffn.shared_experts.w2.weight",
    ".ffn.shared_experts.w3.weight",
    ".e_proj.weight",  # mtp.<k>.e_proj.weight (MTP embedding projector)
    ".h_proj.weight",  # mtp.<k>.h_proj.weight (MTP hidden projector)
    ".main_proj.weight",  # mtp.0.main_proj.weight (0731 DSpark projector)
)

# FP8 e4m3 weights that are dequantized to BF16 and left BF16 (not quantized).
_FP8_TO_BF16_SUFFIXES = (".attn.wo_a.weight",)


def _is_fp4_expert_weight(name: str) -> bool:
    """layers.<l>.ffn.experts.<e>.w{1,2,3}.weight or mtp.<k>.ffn.experts.<e>.w*.weight."""
    if ".ffn.experts." not in name:
        return False
    return (
        name.endswith(".w1.weight")
        or name.endswith(".w2.weight")
        or name.endswith(".w3.weight")
    )


def _classify(name: str) -> str:
    """Return one of: 'fp4_expert_int4', 'fp8_int8', 'fp8_bf16', 'copy'."""
    if _is_fp4_expert_weight(name):
        return "fp4_expert_int4"
    if any(name.endswith(s) for s in _FP8_TO_BF16_SUFFIXES):
        return "fp8_bf16"
    if any(name.endswith(s) for s in _FP8_INT8_PROJ_SUFFIXES):
        return "fp8_int8"
    return "copy"


def _scale_name(weight_name: str) -> str:
    """Output scale tensor name. DSV4 native naming uses '<module>.scale'."""
    assert weight_name.endswith(".weight")
    return weight_name[: -len(".weight")] + ".scale"


# ---------------------------------------------------------------------------
# Per-file conversion
# ---------------------------------------------------------------------------


def convert_single_file(args):
    """Convert one safetensors shard. Returns (filename, stats dict).

    ``device`` selects where the dequant/requant math runs. On GPU the FP4
    unpack + E8M0 scaling + abs-max/round all stay on-device; every produced
    tensor is moved to CPU exactly once (``.cpu()`` at save).
    """
    input_path, output_path, skip_existing, device = args
    fname = os.path.basename(input_path)
    if skip_existing and os.path.exists(output_path):
        return fname, {"skipped": True}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    _, state = _load_state_dict_raw(input_path)

    # Index scale tensors (E8M0) by the weight they belong to so we can pair them.
    scale_for = {}
    for name in state:
        if name.endswith(".scale"):
            scale_for[name[: -len(".scale")] + ".weight"] = name

    out = {}
    stats = {"int4_expert": 0, "int8": 0, "bf16": 0, "copy": 0}

    for name, (kind, tensor) in state.items():
        if name.endswith(".scale"):
            continue

        action = _classify(name)

        if action == "fp4_expert_int4":
            scale_name = scale_for.get(name)
            assert scale_name is not None, f"missing FP4 scale for {name}"
            _, scale_t = state[scale_name]
            w_f32 = _dequant_fp4_expert(tensor, scale_t, device)
            q, sc = _per_channel_quantize_int4(w_f32)
            out[name] = q
            out[_scale_name(name)] = sc
            stats["int4_expert"] += 1

        elif action == "fp8_int8":
            scale_name = scale_for.get(name)
            assert scale_name is not None, f"missing FP8 scale for {name}"
            _, scale_t = state[scale_name]
            w_f32 = _dequant_fp8_blockwise(tensor, scale_t, device)
            q, sc = _per_channel_quantize_int8(w_f32)
            out[name] = q
            out[_scale_name(name)] = sc
            stats["int8"] += 1

        elif action == "fp8_bf16":
            scale_name = scale_for.get(name)
            assert scale_name is not None, f"missing FP8 scale for {name}"
            _, scale_t = state[scale_name]
            w_bf16 = _dequant_fp8_blockwise(tensor, scale_t, device).to(torch.bfloat16)
            out[name] = w_bf16
            stats["bf16"] += 1

        else:  # copy through unchanged (norms, gate, embed, head, hc_*, markov, etc.)
            if kind == "E8M0":
                out[name] = tensor.to(torch.uint8)
            else:
                out[name] = tensor
            stats["copy"] += 1

    save_file({k: v.cpu().contiguous() for k, v in out.items()}, output_path)
    return fname, stats


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def _copy_aux_files(input_dir: str, output_dir: str):
    """Copy config/tokenizer/etc. (everything but safetensors + the index)."""
    for fname in os.listdir(input_dir):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        src = os.path.join(input_dir, fname)
        if not os.path.isfile(src):
            continue
        shutil.copy2(src, os.path.join(output_dir, fname))


def _write_w4a8_config(output_dir: str):
    """Rewrite config.json: replace the FP8 quantization_config with a
    w4a8-int4 block compatible with the KLX ``w4a8_int4`` loader.

    Two config groups:
      - group_0: INT8 per-channel weight + INT8 dynamic per-token activation
        for attention/shared/e_proj/h_proj/main_proj projectors.
      - group_1: INT4 per-channel weight + INT8 dynamic per-token activation
        for MoE experts.
    """
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "memoryless",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                },
                "output_activations": None,
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
                "targets": ["Linear"],
            },
            "group_1": {
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "memoryless",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                },
                "output_activations": None,
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 4,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
                "targets": ["^.*ffn\\.experts.*$"],
            },
        },
        "format": "int-quantized",
        "ignore": [
            "lm_head",
            "re:.*confidence_head\\.proj$",
            "re:.*markov_head\\.(markov_w1|markov_w2)$",
        ],
        "quant_method": "compressed-tensors",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)


def _write_index(output_dir: str):
    """Regenerate model.safetensors.index.json from the converted shards."""
    weight_map = {}
    total_size = 0
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".safetensors"):
            continue
        header, _ = _read_st_header(os.path.join(output_dir, fname))
        for key, m in header.items():
            if key == "__metadata__":
                continue
            weight_map[key] = fname
            start, end = m["data_offsets"]
            total_size += end - start
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(
        os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(index, f, indent=2, ensure_ascii=False, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek-V4-Flash FP4+FP8 -> W4A8 (INT4 experts + INT8 projectors)."
    )
    parser.add_argument(
        "--model-id", required=True, help="Source Flash FP4+FP8 checkpoint dir."
    )
    parser.add_argument(
        "--save-dir", required=True, help="Destination W4A8 checkpoint dir."
    )
    parser.add_argument(
        "--files", nargs="+", default=None, help="Specific shard filenames to convert."
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Where to run dequant/quant math. 'cuda' (default) is faster; "
        "'cpu' forces the CPU path.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids to fan shards across (e.g. '0,1,2,3'). "
        "Default: every visible GPU when --device cuda. One worker process per id.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of worker processes. Default: #gpus (cuda) or "
        "min(8, #files) (cpu).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rewrite shards even if they exist."
    )
    return parser.parse_args()


def _resolve_devices(args):
    """Return the list of per-worker device strings, one entry per worker."""
    if args.device == "cpu":
        return ["cpu"]
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but torch.cuda.is_available() is False. "
            "Re-run with --device cpu or fix the GPU runtime."
        )
    if args.gpus:
        ids = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    else:
        ids = list(range(torch.cuda.device_count()))
    if not ids:
        raise RuntimeError("No GPUs selected for --device cuda.")
    return [f"cuda:{i}" for i in ids]


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.model_id)
    output_dir = os.path.abspath(args.save_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input checkpoint not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if args.files:
        shards = [os.path.join(input_dir, f) for f in args.files]
    else:
        shards = sorted(glob(os.path.join(input_dir, "*.safetensors")))
    shards = [p for p in shards if os.path.isfile(p)]
    if not shards:
        print("No safetensors shards found.")
        return

    devices = _resolve_devices(args)
    tasks = [
        (
            p,
            os.path.join(output_dir, os.path.basename(p)),
            not args.overwrite,
            devices[i % len(devices)],
        )
        for i, p in enumerate(shards)
    ]
    if args.workers is not None:
        max_workers = max(1, args.workers)
    elif args.device == "cuda":
        max_workers = len(devices)
    else:
        max_workers = min(8, len(tasks))

    print("Convert DeepSeek-V4-Flash FP4+FP8 -> W4A8 (INT4 experts + INT8 projectors)")
    print("  experts (FP4 e2m1)            -> INT4 per-channel packed + .scale")
    print(
        "  attn/shared/e_proj/h_proj/main_proj(FP8)-> "
        "INT8 per-channel weight + .scale"
    )
    print("  wo_a (FP8)                    -> BF16 (kept, not quantized)")
    print("  norms/gate/embed/head/hc/...  -> copied unchanged")
    print(f"  source: {input_dir}")
    print(f"  target: {output_dir}")
    print(f"  device: {args.device}  devices={devices}")
    print(f"  workers: {max_workers} (processes)")

    totals = {"int4_expert": 0, "int8": 0, "bf16": 0, "copy": 0}
    mp_ctx = mp.get_context("spawn") if args.device == "cuda" else None
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as ex:
        futs = {ex.submit(convert_single_file, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(tasks), desc="Converting"):
            try:
                _fname, stats = fut.result()
                if not stats.get("skipped"):
                    for k in totals:
                        totals[k] += stats.get(k, 0)
            except Exception as e:  # noqa: BLE001
                t = futs[fut]
                print(f"ERROR converting {os.path.basename(t[0])}: {e}")
                raise

    print("Copying auxiliary files (config/tokenizer/...)")
    _copy_aux_files(input_dir, output_dir)
    print("Writing W4A8 quantization_config into config.json")
    _write_w4a8_config(output_dir)
    print("Regenerating model.safetensors.index.json")
    _write_index(output_dir)

    print(
        f"\nDone. INT4 experts={totals['int4_expert']}, "
        f"INT8 proj/attn={totals['int8']}, "
        f"BF16 wo_a={totals['bf16']}, copied={totals['copy']}"
    )
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
    print("done")
