# DeepSeek V4 MTP 层量化实现方案

## 概述

DeepSeek V4 模型除了主干的 61 层 decoder 外，还包含 1 层 MTP（Multi-Token Prediction）层用于推测解码加速。HuggingFace transformers 在加载模型时通过 `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` 将 MTP 权重静默丢弃，导致 MTP 层无法参与量化。

本实现通过 `attach_mtp_layer(model, model_path)` 函数，在模型加载后手动读取 MTP 权重并构建完整的 MTP 模块，使其参与 GPTQ/RTN 量化的校准前向传播和 Hessian 累积。

## 原始权重格式与 BF16 转换

### 混合精度格式

DeepSeek V4 原始 checkpoint 使用 FP8 + FP4 混合精度存储：

| 层类型 | 权重 dtype | Scale dtype | Block size | 说明 |
|---|---|---|---|---|
| Attention (wq_a/wq_b/wkv/wo_a/wo_b) | F8_E4M3 | F8_E8M0 | [128, 128] | 标准 FP8 block quant |
| Shared experts (w1/w2/w3) | F8_E4M3 | F8_E8M0 | [128, 128] | 同上 |
| Routed experts (w1/w2/w3) | I8 (packed FP4) | F8_E8M0 | per-row, group=32 | FP4 e2m1fn, 2 values/byte |
| MTP e_proj/h_proj | F8_E4M3 | F8_E8M0 | [128, 128] | MTP 特有投影层 |
| Norm/gate/HC params | BF16/F32 | — | — | 不量化 |

### 反量化公式

**FP8 block dequant**（Attention + Shared experts）：
```
scale_float = 2^(scale_e8m0_byte - 127)           # E8M0 → float32
weight_blocks = weight_fp8.reshape(n_row_blocks, 128, n_col_blocks, 128)
bf16_weight = (weight_blocks * scale_float[..., None, None]).reshape(original_shape)
```

**FP4 dequant**（Routed experts）：
```
FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
low_nibble  = packed_byte & 0x0F
high_nibble = (packed_byte >> 4) & 0x0F
fp4_values  = interleave(FP4_TABLE[low], FP4_TABLE[high])  # 2x expansion
scale_float = 2^(scale_e8m0_byte - 127)                    # per-group (group_size=32)
bf16_weight = fp4_values * scale_per_group
```

### 转换实现

由于 safetensors 0.5.x 不支持 `F8_E8M0` dtype 的反序列化，实现了自定义 `SafetensorsReader` 直接解析二进制 header 和 tensor data。转换按 shard 并行处理，输出标准 BF16 safetensors 文件，同时移除 `.scale` 键和 `quantization_config`。

## 权重命名映射

### 背景

DeepSeek V4 的 safetensors 文件使用**原始内部命名**（如 `attn.wq_a`），而 HuggingFace transformers 中 `DeepseekV4DecoderLayer` 的子模块使用**HF 命名**（如 `self_attn.q_a_proj`）。

MTP 层的加载和保存涉及两次映射：
- **加载时（正向映射）**：从 safetensors 读取原始 key → 转换为 HF key → `load_state_dict()` 到 HF 模块
- **保存时（反向映射）**：`state_dict()` 输出 HF key → 转换回原始 key → `save_pretrained()` 写入 safetensors

这样保证保存的量化模型与原始 checkpoint 格式一致，可被 vLLM/sglang 直接加载。

### 映射表

| 原始 checkpoint key（safetensors 中） | HF 模块 key（load_state_dict 用） | 说明 |
|---|---|---|
| `mtp.0.attn.wq_a.weight` | `decoder.self_attn.q_a_proj.weight` | Q LoRA 下投影 |
| `mtp.0.attn.wq_b.weight` | `decoder.self_attn.q_b_proj.weight` | Q LoRA 上投影 |
| `mtp.0.attn.wkv.weight` | `decoder.self_attn.kv_proj.weight` | 共享 KV 投影 |
| `mtp.0.attn.wo_a.weight` | `decoder.self_attn.o_a_proj.weight` | O LoRA 下投影 (GroupedLinear) |
| `mtp.0.attn.wo_b.weight` | `decoder.self_attn.o_b_proj.weight` | O LoRA 上投影 |
| `mtp.0.attn.q_norm.weight` | `decoder.self_attn.q_a_norm.weight` | |
| `mtp.0.attn.kv_norm.weight` | `decoder.self_attn.kv_norm.weight` | |
| `mtp.0.attn.attn_sink` | `decoder.self_attn.sinks` | 注意力 sink bias (F32) |
| `mtp.0.attn_norm.weight` | `decoder.input_layernorm.weight` | |
| `mtp.0.ffn_norm.weight` | `decoder.post_attention_layernorm.weight` | |
| `mtp.0.ffn.gate.weight` | `decoder.mlp.gate.weight` | MoE 路由器 |
| `mtp.0.ffn.gate.bias` | `decoder.mlp.gate.e_score_correction_bias` | |
| `mtp.0.ffn.experts.{i}.w1.weight` | 融合为 `decoder.mlp.experts.gate_up_proj` | 384 个专家 cat(w1,w3) → [N,2I,H] |
| `mtp.0.ffn.experts.{i}.w3.weight` | ↑ 同上 | |
| `mtp.0.ffn.experts.{i}.w2.weight` | 融合为 `decoder.mlp.experts.down_proj` | stack → [N,H,I] |
| `mtp.0.ffn.shared_experts.w1.weight` | `decoder.mlp.shared_experts.gate_proj.weight` | |
| `mtp.0.ffn.shared_experts.w2.weight` | `decoder.mlp.shared_experts.down_proj.weight` | |
| `mtp.0.ffn.shared_experts.w3.weight` | `decoder.mlp.shared_experts.up_proj.weight` | |
| `mtp.0.hc_attn_fn` | `decoder.attn_hc.fn` | HyperConnection (F32) |
| `mtp.0.hc_attn_base` | `decoder.attn_hc.base` | |
| `mtp.0.hc_attn_scale` | `decoder.attn_hc.scale` | |
| `mtp.0.hc_ffn_fn` | `decoder.ffn_hc.fn` | |
| `mtp.0.hc_ffn_base` | `decoder.ffn_hc.base` | |
| `mtp.0.hc_ffn_scale` | `decoder.ffn_hc.scale` | |
| `mtp.0.hc_head_fn` | `hc_head.hc_fn` | MTP 特有：HyperHead (F32) |
| `mtp.0.hc_head_base` | `hc_head.hc_base` | |
| `mtp.0.hc_head_scale` | `hc_head.hc_scale` | |
| `mtp.0.e_proj.weight` | `e_proj.weight` | MTP 特有：嵌入投影 |
| `mtp.0.h_proj.weight` | `h_proj.weight` | MTP 特有：隐状态投影 |
| `mtp.0.enorm.weight` | `enorm.weight` | |
| `mtp.0.hnorm.weight` | `hnorm.weight` | |
| `mtp.0.norm.weight` | `shared_head_norm.weight` | MTP 特有：输出 RMSNorm |

> HF 模块 key 列中，`decoder.*` 前缀的通过 `self.decoder.load_state_dict()` 加载，其余通过 `self.load_state_dict()` 加载到 `DeepseekV4MTPLayer` 自身。

### 保存时的反向映射

`_patched_state_dict()` 将 HF key 转换回原始格式：

```
model.mtp.decoder.self_attn.q_a_proj.weight  →  mtp.0.attn.wq_a.weight
model.mtp.shared_head_norm.weight            →  mtp.0.norm.weight
model.mtp.hc_head.hc_fn                     →  mtp.0.hc_head_fn
model.mtp.decoder.mlp.shared_experts.gate_proj.weight → mtp.0.ffn.shared_experts.w1.weight
```

量化后，`CalibrationDeepseekV4MoE`（permanent=True）将 fused 3D expert 参数拆解为独立 Linear 层。保存时 routed experts 使用 HF 命名：

```
model.mtp.decoder.mlp.experts.0.gate_proj.weight  →  mtp.0.ffn.experts.0.gate_proj.weight
model.mtp.decoder.mlp.experts.0.up_proj.weight    →  mtp.0.ffn.experts.0.up_proj.weight
model.mtp.decoder.mlp.experts.0.down_proj.weight  →  mtp.0.ffn.experts.0.down_proj.weight
```

### sglang 权重加载兼容性

sglang 的 `load_weights` 方法（见 `sglang/srt/models/deepseek_v4.py`）对 checkpoint key 做如下转换：

```python
# 1. MTP 前缀处理：mtp.0.{rest} → model.layers.{num_hidden_layers}.{rest}
# 2. 通用重命名：
name = name.replace(".attn.", ".self_attn.")
name = name.replace(".ffn.", ".mlp.")
name = name.replace(".w1.", ".gate_proj.")
name = name.replace(".w2.", ".down_proj.")
name = name.replace(".w3.", ".up_proj.")
```

因此 sglang 同时支持两种 routed expert 命名：
- **原始格式**：`mtp.0.ffn.experts.{i}.w1.weight` → 经 `.w1.`→`.gate_proj.` 转换后匹配
- **HF 格式**：`mtp.0.ffn.experts.{i}.gate_proj.weight` → 直接匹配（`.w1.` 规则不触发）

我们的量化输出使用 HF 格式（gate_proj/up_proj/down_proj），与 sglang 完全兼容。

### 关键发现

1. **MTP 前缀**：权重存储在 `mtp.0.*` 下（不是 `model.layers.{N}.*`）
2. **无 Compressor/Indexer**：MTP 的 attention 没有压缩器（对应 `sliding_attention` 类型）
3. **标准 MoE 路由**：MTP 使用 top-k 路由（有 `gate.bias`），不是 hash 路由（无 `tid2eid`）
4. **384 个路由专家 + 共享专家**：与主干层相同
5. **`.scale` 后缀**：FP8/FP4 量化的 scale 信息，BF16 转换后移除
6. **HyperConnection**：MTP decoder 同样使用 mHC（`hc_attn_*`, `hc_ffn_*`），参数必须保持 float32
7. **HyperHead**：MTP 特有的 `hc_head_*` 用于将 `hc_mult` 条并行流折叠为单条

## 架构设计

### MTP 前向传播公式

```
e_hidden = e_proj(enorm(embed_tokens(input_ids)))       # 嵌入分支
h_hidden = h_proj(hnorm(main_hidden_states))            # 主干隐状态分支
mtp_input = (e_hidden + h_hidden)                       # 加法组合
mtp_input = mtp_input.expand(hc_mult)                   # 扩展为 hc_mult 条并行流 [B,S,hc_mult,D]
mtp_out = decoder(mtp_input)                            # DeepseekV4DecoderLayer (含 mHC + MoE)
mtp_out = hc_head(mtp_out)                              # HyperHead 折叠为 [B,S,D]
mtp_out = shared_head_norm(mtp_out)                     # 最终 RMSNorm
```

### 模块结构

```
model.mtp (DeepseekV4MTPLayer)
├── enorm (DeepseekV4RMSNorm)
├── hnorm (DeepseekV4RMSNorm)
├── e_proj (Linear, hidden→hidden, bias=False)
├── h_proj (Linear, hidden→hidden, bias=False)
├── hc_head (DeepseekV4HyperHead)
│   ├── hc_fn [hc_mult, hc_mult*hidden]  (float32)
│   ├── hc_base [hc_mult]                (float32)
│   └── hc_scale [1]                     (float32)
├── shared_head_norm (DeepseekV4RMSNorm)
└── decoder (DeepseekV4DecoderLayer, layer_type=sliding_attention)
    ├── self_attn (DeepseekV4Attention, compressor=None)
    │   ├── q_a_proj, q_a_norm, q_b_proj
    │   ├── kv_proj, kv_norm
    │   ├── o_a_proj, o_b_proj
    │   └── sinks
    ├── mlp (DeepseekV4SparseMoeBlock, type=moe)
    │   ├── gate (DeepseekV4TopKRouter)
    │   ├── experts (DeepseekV4Experts, 384 experts fused)
    │   └── shared_experts (DeepseekV4MLP)
    ├── input_layernorm, post_attention_layernorm
    ├── attn_hc (DeepseekV4HyperConnection)
    └── ffn_hc (DeepseekV4HyperConnection)
```

## 实现细节

### 1. BF16 转换流程（`deepseek_v4_w8a8.py` Module 1）

```python
# 对每个 safetensors shard 并行处理：
for key in shard_keys:
    if has_scale(key):
        if dtype == "F8_E4M3":
            tensor = _dequant_fp8_block(weight, scale, block_size=(128,128))
        elif dtype == "I8":  # packed FP4
            tensor = _dequant_fp4(weight, scale, fp4_block_size=32)
    else:
        tensor = read_as_is(key)  # BF16/F32 pass-through

# 后处理：移除 .scale 键，删除 quantization_config
```

### 2. MTP 权重加载流程

```python
# 1. 从 BF16 safetensors 读取 mtp.0.* 前缀的所有权重
raw = _load_mtp_tensors(model_path, "mtp.0")
raw = _strip_prefix(raw, "mtp.0")

# 2. 分离 MTP 特有权重和 decoder 权重
# 3. 对 decoder 权重执行 key 转换（raw → HF 格式）
# 4. 融合 per-expert 权重为 3D tensor
own_sd, decoder_sd = _convert_and_fuse_weights(raw)

# 5. 加载到模块（HC 参数保持 float32）
self.load_state_dict(own_sd, strict=False)
self.decoder.load_state_dict(decoder_sd, strict=False)
```

### 3. Decoder 构建

MTP decoder 与主干 decoder 的关键区别通过修改 config 实现：

```python
mtp_config = copy.copy(config)
mtp_config.layer_types = ["sliding_attention"]   # 无 compressor
mtp_config.mlp_layer_types = ["moe"]             # 标准 top-k 路由
self.decoder = DeepseekV4DecoderLayer(mtp_config, layer_idx=0)
```

### 4. Forward Patch

通过 `types.MethodType` 绑定新的 forward 方法到模型：
- 只调用 `self.model(...)` 一次（sequential pipeline 的 torch.fx tracing 要求）
- MTP forward 在主干 forward 之后执行
- MTP 的 logits 被丢弃，仅激活值用于 Hessian 累积
- 不传递 attention_mask 给 MTP（避免 mask 格式不兼容问题）
- position_embeddings 使用 dict 格式 `{"main": ..., "compress": ...}`

### 5. State Dict Patch

保存时将 HF 格式的 key 反向转换回 checkpoint 原始格式：
- `mtp.decoder.self_attn.q_a_proj.weight` → `mtp.0.attn.wq_a.weight`
- `mtp.shared_head_norm.weight` → `mtp.0.norm.weight`
- `mtp.hc_head.hc_fn` → `mtp.0.hc_head_fn`

注意：fused experts (`gate_up_proj`, `down_proj`) 在保存时保持 fused 格式，因为量化后的模型通常由 vLLM/sglang 加载，它们支持 fused 格式。

## 使用方式

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modeling.deepseek_v4 import CalibrationDeepseekV4MoE  # 必须导入以注册 MoE 校准模块
from llmcompressor.modeling.deepseek_v4_mtp import attach_mtp_layer

model_id = "/path/to/DeepSeek-V4-Pro-bf16"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map=None)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载并附加 MTP 层
attach_mtp_layer(model, model_id)

# 正常执行量化 — MTP decoder 中的 Linear 层会被量化
oneshot(model=model, dataset=ds, recipe=recipes, ...)
model.save_pretrained(save_dir, save_compressed=True)
```

## 注意事项

1. **HyperConnection 参数保持 float32**：`attn_hc.fn/base/scale` 和 `ffn_hc.fn/base/scale` 在 Sinkhorn 投影中需要 float32 精度，加载时跳过 dtype 转换。`hc_head` 的参数同理。
2. **CalibrationDeepseekV4MoE 必须显式导入**：该类通过 `@MoECalibrationModule.register` 注册，不导入则 MoE 专家不会被拆解为独立 Linear 层，量化无法匹配。
3. **量化 ignore 列表**：`lm_head`、`embed_tokens`、`mlp.gate`（路由器）、`compressor.*`、`indexer.weights_proj` 应加入 ignore。`e_proj` 和 `h_proj` 在原始模型中是被量化的，不应忽略。
4. **内存考虑**：384 个专家的权重融合需要大量临时内存，建议在 CPU 上执行加载后再 dispatch 到 GPU。
5. **Sequential Pipeline 兼容性**：patched forward 遵循 `autowrap_forward` 的约束（函数名为 `forward`、无闭包变量、单次调用 `self.model`）。
6. **Chat Template**：DeepSeek V4 tokenizer 无内置 chat_template，需手动使用 `<｜User｜>`/`<｜Assistant｜>` 格式编码校准数据。
