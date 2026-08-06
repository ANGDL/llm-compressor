"""
Convert DeepSeek-V4-Flash (FP4 + FP8 mixed) -> INT8 per-channel.

Source checkpoint layout (DeepSeek native naming, verified from the HF release):
  - FP8 weights : F8_E4M3 weight + F8_E8M0 block-128x128 scale
                  (attn wq_a/wq_b/wkv/wo_b, ffn.shared_experts.w*,
                   mtp.* counterparts, mtp.e_proj/h_proj, attn.indexer.wq_b)
  - FP4 experts : packed e2m1 stored as int8 (2 nibbles/byte) + F8_E8M0
                  block-32 scale; logical in_dim = 2 * stored in_dim
                  (layers/mtp ffn.experts.N.w1/w2/w3)
  - kept as-is  : norms, ffn.gate.weight/bias/tid2eid, embed, head,
                  hc_* params, attn_sink, attn.compressor.*,
                  attn.indexer.compressor.*/weights_proj, hc_head_*

Target scheme (matches the reference DeepSeek-V4-Flash-INT8 release and the
KLX w8a8_int8 / w4a8 loaders):
  - Every quantized projector + every MoE expert -> per-channel symmetric INT8
    weight (torch.int8, shape unchanged after FP4 unpack) with an F32
    per-output-channel scale named "<name>.scale" of shape [out_features, 1].
  - wo_a is dequantized to BF16 and kept BF16 (DSV4 loader's manual einsum path
    consumes BF16 wo_a; it is NOT quantized in the reference INT8 model).
  - Everything else is copied through byte-for-byte (dtype preserved).

INT8 here is *per-channel*, not block-wise.

Usage:
  python convert_flash_to_int8.py \
      --model-id /ssd3/models/DeepSeek-V4-Flash \
      --save-dir /ssd3/models/DeepSeek-V4-Flash-INT8 \
      [--gpus 0,1,2,3,4,5,6,7] [--device cuda] [--overwrite]

Performance:
  The FP4 dequant + per-channel INT8 requant dominates runtime (~28 s/shard on
  CPU vs ~4 s on a GPU). Two changes give the speedup:
    1. --device cuda  : run all dequant/quant math on a GPU (FP4 nibble unpack,
       E8M0 scaling, abs-max + round) instead of CPU tensor ops.
    2. --gpus a,b,c   : convert N shards concurrently, one process per GPU, so
       the 46 shards fan out across every device (ProcessPoolExecutor, NOT
       threads — the per-shard work is pure Python/torch and is GIL-bound, so a
       ThreadPoolExecutor serializes it).
  Defaults: device=cuda using every visible GPU. Pass --device cpu to force the
  original CPU path.
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

# float8_e2m1 lookup (signed). Index = 4-bit nibble. Matches inference/convert.py.
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

# safetensors dtype string -> torch dtype (for the formats we read directly).
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
# raw tensor bytes ourselves and build tensors with torch.frombuffer. F8_E8M0 is
# not a torch dtype here, so it is returned as a uint8 exponent tensor with a
# sentinel marker.
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
    # low nibble is the first logical element, high nibble the second (matches
    # inference/convert.py and the GLM/KiMi int4 pack/unpack convention).
    unpacked = torch.stack([table[low], table[high]], dim=-1).flatten(1)  # [N, K]
    sf = _e8m0_to_f32(scale_e8m0.to(device))  # [N, K/32]
    n, k = unpacked.shape
    assert (
        sf.shape[0] == n and sf.shape[1] == k // FP4_BLOCK
    ), f"FP4 expert scale {tuple(sf.shape)} incompatible with logical weight {tuple(unpacked.shape)}"
    scales = sf.repeat_interleave(FP4_BLOCK, dim=1)  # [N, K]
    return unpacked * scales


# ---------------------------------------------------------------------------
# Requantization (float32 -> per-channel symmetric INT8)
# ---------------------------------------------------------------------------


def _per_channel_quantize_int8(weight_f32: torch.Tensor):
    """Per-channel (per output row) symmetric INT8. Returns (int8 weight, f32 scale[N,1]).

    Keeps the result on whatever device ``weight_f32`` lives on; the caller moves
    it back to CPU once (a single D2H copy per tensor) before save.
    """
    assert weight_f32.dim() == 2, f"expected 2D weight, got {tuple(weight_f32.shape)}"
    w = weight_f32.to(torch.float32)
    abs_max = w.abs().amax(dim=1, keepdim=True)  # [N, 1]
    qmax = 127.0
    scale = (abs_max / qmax).clamp_min(1e-12)  # avoid div-by-zero on all-zero rows
    q = (w / scale).round().clamp(-qmax - 1, qmax).to(torch.int8)
    return q, scale.to(torch.float32)


# ---------------------------------------------------------------------------
# Per-tensor classification
#
# Decide, from the DeepSeek-native tensor name, what to do with each tensor.
# ---------------------------------------------------------------------------

# FP8 e4m3 projector weights that become per-channel INT8.
# (Suffixes are matched against the leaf "<module>.weight".)
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
    """Return one of: 'fp4_expert', 'fp8_int8', 'fp8_bf16', 'copy'."""
    if _is_fp4_expert_weight(name):
        return "fp4_expert"
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
    tensor is moved to CPU exactly once (``.cpu()`` at save), so the only host
    <-> device traffic is one D2H per output tensor.
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
    stats = {"int8": 0, "int8_expert": 0, "bf16": 0, "copy": 0}

    for name, (kind, tensor) in state.items():
        # Scale tensors are consumed together with their weight; skip standalone.
        if name.endswith(".scale"):
            continue

        action = _classify(name)

        if action == "fp4_expert":
            scale_name = scale_for.get(name)
            assert scale_name is not None, f"missing FP4 scale for {name}"
            _, scale_t = state[scale_name]
            w_f32 = _dequant_fp4_expert(tensor, scale_t, device)
            q, sc = _per_channel_quantize_int8(w_f32)
            out[name] = q
            out[_scale_name(name)] = sc
            stats["int8_expert"] += 1

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
                # An E8M0 tensor that is NOT a recognised quant scale — preserve
                # it as its raw uint8 exponent bytes so nothing is silently lost.
                out[name] = tensor.to(torch.uint8)
            else:
                out[name] = tensor
            stats["copy"] += 1

    # Single host transfer + contiguity pass right before serialization. When
    # device == "cpu" this .cpu() is a no-op; on GPU it is the one D2H copy.
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


def _write_int8_config(output_dir: str):
    """Rewrite config.json: replace the FP8 quantization_config with a
    compressed-tensors int-quantized (channel weight / dynamic-token act) block,
    matching the reference DeepSeek-V4-Flash-INT8 release."""
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
            }
        },
        "format": "int-quantized",
        "ignore": ["lm_head"],
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
        description="Convert DeepSeek-V4-Flash FP4+FP8 -> per-channel INT8."
    )
    parser.add_argument(
        "--model-id", required=True, help="Source Flash FP4+FP8 checkpoint dir."
    )
    parser.add_argument(
        "--save-dir", required=True, help="Destination INT8 checkpoint dir."
    )
    parser.add_argument(
        "--files", nargs="+", default=None, help="Specific shard filenames to convert."
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Where to run dequant/quant math. 'cuda' (default) is ~7x faster; "
        "'cpu' forces the original CPU path.",
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
    """Return the list of per-worker device strings, one entry per worker.

    For cuda: round-robin the requested GPU ids so worker i uses gpus[i % n].
    For cpu: a single 'cpu' device replicated across the worker pool.
    """
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
    # Round-robin a device onto each shard so the pool spreads work evenly.
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

    print("Convert DeepSeek-V4-Flash FP4+FP8 -> per-channel INT8")
    print("  experts (FP4 e2m1)            -> INT8 per-channel weight + .scale")
    print("  attn/shared/e_proj/h_proj(FP8)-> INT8 per-channel weight + .scale")
    print("  wo_a (FP8)                    -> BF16 (kept, not quantized)")
    print("  norms/gate/embed/head/hc/...  -> copied unchanged")
    print(f"  source: {input_dir}")
    print(f"  target: {output_dir}")
    print(f"  device: {args.device}  devices={devices}")
    print(f"  workers: {max_workers} (processes)")

    totals = {"int8": 0, "int8_expert": 0, "bf16": 0, "copy": 0}
    # Processes, not threads: per-shard work is pure Python/torch and GIL-bound,
    # so threads would serialize it. Each process owns its CUDA context/device.
    # 'spawn' is required for CUDA — a forked process inherits a broken CUDA
    # context and crashes on the first device op.
    mp_ctx = mp.get_context("spawn") if args.device == "cuda" else None
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as ex:
        futs = {ex.submit(convert_single_file, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(tasks), desc="Converting"):
            try:
                _fname, stats = fut.result()
                if not stats.get("skipped"):
                    for k in totals:
                        totals[k] += stats.get(k, 0)
            except (
                Exception
            ) as e:  # noqa: BLE001 - surface per-shard failure, keep going
                t = futs[fut]
                print(f"ERROR converting {os.path.basename(t[0])}: {e}")
                raise

    print("Copying auxiliary files (config/tokenizer/...)")
    _copy_aux_files(input_dir, output_dir)
    print("Writing INT8 quantization_config into config.json")
    _write_int8_config(output_dir)
    print("Regenerating model.safetensors.index.json")
    _write_index(output_dir)

    print(
        f"\nDone. INT8 proj/attn={totals['int8']}, INT8 experts={totals['int8_expert']}, "
        f"BF16 wo_a={totals['bf16']}, copied={totals['copy']}"
    )
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
    print("done")
