"""
精度对比脚本：比较 tilelang kernel (reference) 与 torch fallback 实现的数值差异。

用法:
    DEEPSEEK_V4_KERNEL_BACKEND=reference \
    DEEPSEEK_V4_REFERENCE_KERNEL_PATH=/path/to/kernel.py \
    python tools/compare_kernels.py

需要 CUDA GPU 和已安装 tilelang。
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REFERENCE_KERNEL_PATH = os.environ.get(
    "DEEPSEEK_V4_REFERENCE_KERNEL_PATH",
    "/Users/ang/models/DeepSeek-V4-Pro/inference/kernel.py",
)
os.environ["DEEPSEEK_V4_KERNEL_BACKEND"] = "reference"
os.environ["DEEPSEEK_V4_REFERENCE_KERNEL_PATH"] = REFERENCE_KERNEL_PATH

from llmcompressor.modeling.deepseekv4.kernels import (
    sparse_attention,
    hc_split,
    reset_kernel_backend_cache,
    _load_reference_kernel_module,
)


def report(name: str, ref: torch.Tensor, test: torch.Tensor):
    """打印精度对比指标"""
    diff = (ref.float() - test.float()).abs()
    rel = diff / (ref.float().abs().clamp(min=1e-8))
    print(f"  [{name}]")
    print(f"    shape: {list(ref.shape)}")
    print(f"    max abs diff:  {diff.max().item():.6e}")
    print(f"    mean abs diff: {diff.mean().item():.6e}")
    print(f"    max rel diff:  {rel.max().item():.6e}")
    print(f"    mean rel diff: {rel.mean().item():.6e}")
    atol = 1e-3
    rtol = 1e-2
    close = torch.allclose(ref.float(), test.float(), atol=atol, rtol=rtol)
    print(f"    allclose(atol={atol}, rtol={rtol}): {close}")
    print()
    return close


# ============================================================
# 加载 reference kernel module (tilelang)
# ============================================================
reset_kernel_backend_cache()
ref_module = _load_reference_kernel_module()
if ref_module is None:
    print("ERROR: 无法加载 reference kernel module。请检查路径和 tilelang 安装。")
    sys.exit(1)

device = "cuda"
torch.manual_seed(42)
print("=" * 70)
print("DeepSeek-V4 Kernel 精度对比: tilelang (reference) vs torch (fallback)")
print("=" * 70)

all_pass = True

# ============================================================
# 1. sparse_attention
# ============================================================
print("\n" + "=" * 70)
print("1. sparse_attention")
print("=" * 70)

bsz, seqlen, n_heads, head_dim = 2, 4, 8, 64
n_kv = 32
topk = 16

q = torch.randn(bsz, seqlen, n_heads, head_dim, device=device, dtype=torch.bfloat16)
kv = torch.randn(bsz, n_kv, head_dim, device=device, dtype=torch.bfloat16)
attn_sink = torch.randn(n_heads, device=device, dtype=torch.float32)
topk_idxs = torch.randint(0, n_kv, (bsz, seqlen, topk), device=device, dtype=torch.int32)
topk_idxs[:, :, -2:] = -1  # 最后 2 个位置无效
softmax_scale = (1.0 / head_dim) ** 0.5

# tilelang reference
out_ref = ref_module.sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

# torch fallback
os.environ.pop("DEEPSEEK_V4_KERNEL_BACKEND", None)
reset_kernel_backend_cache()
out_torch = sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)

all_pass &= report("sparse_attention output", out_ref, out_torch)

# 恢复 reference backend
os.environ["DEEPSEEK_V4_KERNEL_BACKEND"] = "reference"
reset_kernel_backend_cache()


# ============================================================
# 2. hc_split (Sinkhorn)
# ============================================================
print("=" * 70)
print("2. hc_split (Sinkhorn)")
print("=" * 70)

hc_mult = 4
sinkhorn_iters = 20
hc_eps = 1e-6
mix_dim = (2 + hc_mult) * hc_mult  # = 24
bsz_hc, seqlen_hc = 2, 8

mixes = torch.randn(bsz_hc, seqlen_hc, mix_dim, device=device, dtype=torch.float32)
hc_scale = torch.tensor([1.0, 1.0, 0.5], device=device, dtype=torch.float32)
hc_base = torch.randn(mix_dim, device=device, dtype=torch.float32) * 0.1

# tilelang reference
pre_ref, post_ref, comb_ref = ref_module.hc_split_sinkhorn(
    mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
)

# torch fallback
os.environ.pop("DEEPSEEK_V4_KERNEL_BACKEND", None)
reset_kernel_backend_cache()
pre_torch, post_torch, comb_torch = hc_split(
    mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
)

all_pass &= report("hc_split.pre", pre_ref, pre_torch)
all_pass &= report("hc_split.post", post_ref, post_torch)
all_pass &= report("hc_split.comb", comb_ref, comb_torch)


# ============================================================
# Summary
# ============================================================
print("=" * 70)
if all_pass:
    print("✅ 全部通过。torch fallback 与 tilelang kernel 对齐。")
else:
    print("❌ 存在精度差异，请检查上面标记 allclose=False 的项。")
print("=" * 70)
sys.exit(0 if all_pass else 1)
