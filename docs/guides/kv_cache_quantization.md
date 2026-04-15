# KV Cache 量化逻辑与原理

本文面向工程落地，说明 LLM Compressor 中 KV Cache 量化从配置、校准、导出到运行时消费的完整链路，并补充其核心量化原理与精度-性能权衡。

## 为什么要量化 KV Cache

在长上下文推理中，KV Cache 往往比模型权重更快成为显存瓶颈。对 Key/Value 做低比特存储可以显著降低显存占用，并在运行时支持的情况下提升吞吐。

直观上，KV Cache 内存与序列长度近似线性增长：

$$
  ext{Memory}_{KV} \propto B \times L \times H \times D \times 2 \times \text{dtype\_bytes}
$$

其中 $B$ 为 batch，$L$ 为上下文长度，$H$ 为头数，$D$ 为 head dimension，常数 2 对应 K 和 V 两份缓存。

## 量化原理（核心）

### 1) 量化映射

KV Cache 量化本质是把浮点张量 $x$ 映射到离散值 $q$，再在计算时反量化：

$$
q = \text{clip}\left(\text{round}\left(\frac{x}{s}\right) + z\right), \quad \hat{x} = (q-z) \cdot s
$$

- $s$ 是 scale。
- $z$ 是 zero point（对称量化可取 0）。
- $\hat{x}$ 是反量化后的近似值。

### 2) 为什么 K 和 V 要单独统计

Key 与 Value 在数值分布上通常不同，且在不同层、不同头上波动不同。把 K/V 分开做 observer 统计，可减少同一 scale 覆盖两种分布带来的失真。

### 3) 静态与动态

- 静态量化：用校准集先估计好 qparams（scale/zero_point），推理时直接复用。
- 动态量化：推理阶段按输入在线更新或局部更新 qparams。

一般而言：

- 静态路径吞吐更稳定，但依赖校准数据代表性。
- 动态路径适应分布漂移更好，但实现与运行开销更高。

### 4) 误差来源

KV 量化误差会影响注意力分数与上下文聚合：

1. 量化舍入误差（rounding error）。
2. scale 粗糙导致的饱和/截断（clipping error）。
3. 长上下文下误差累积，表现为后段 token 质量下降。

因此实际工程里，常先从 FP8 KV（或更保守配置）起步，再逐步提升压缩率。

## 端到端流程

KV Cache 量化沿用量化框架生命周期：

1. 解析 recipe。
2. 解析 QuantizationConfig（包含 kv_cache_scheme）。
3. 把配置挂载到目标模块。
4. 校准激活统计（包含 K/V 观察器）。
5. 冻结并导出量化参数。
6. 运行时显式启用 KV 量化 dtype。

## 1) 配置入口

在 QuantizationModifier 或 GPTQModifier 下设置 kv_cache_scheme 即可启用。

最小示例：

```yaml
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      kv_cache_scheme:
        num_bits: 8
        type: float
        strategy: tensor
        dynamic: false
        symmetric: true
```

参考示例：

- examples/quantization_kv_cache/llama3_fp8_kv_example.py
- examples/quantization_kv_cache/phi3.5_fp8_kv_example.py

## 2) kv_cache_scheme 如何变成实际目标

QuantizationMixin 在解析配置时，如果检测到 kv_cache_scheme，会把 KV 相关 target 合并进 resolved_targets，使注意力相关模块进入后续初始化与校准流程。

实现位置：

- src/llmcompressor/modifiers/quantization/quantization/mixin.py

重要前提：

- 当前实现依赖注意力模块的命名/结构约定（例如 k_proj、v_proj 风格）。
- 若模型结构不满足约定，KV cache 量化可能无法正确应用。

## 3) Attention 上 Q/K/V observer 初始化

校准开始时，QuantizationMixin.start_calibration 会为目标模块初始化 observer 并注册 hook。

针对注意力模块：

- query 路径注册 q observer。
- kv 路径注册 k observer 与 v observer。

实现位置：

- src/llmcompressor/modifiers/quantization/quantization/mixin.py
- src/llmcompressor/modifiers/quantization/calibration.py

补充说明：

- observer 类型来自 QuantizationArgs.observer。
- 静态模式会把观测结果写回 scale/zero_point。
- weight observer 会被替换为 memoryless 版本以降低内存开销（与 KV 无直接耦合，但在同一校准框架下生效）。

## 4) KV 路径校准 Hook

注意力模块按信号类型注册 hook：

- query hook -> calibrate_query_hook
- key hook -> calibrate_key_hook
- value hook -> calibrate_value_hook

这些 hook 最终调用 calibrate_activations(..., base_name="q|k|v")。

qparams 更新策略：

- dynamic=True 或 DynamicType.LOCAL：跳过静态 qparams 重算。
- 静态模式：按 observer 统计更新 scale（以及可选 zero point）。

实现位置：

- src/llmcompressor/modifiers/quantization/quantization/mixin.py
- src/llmcompressor/modifiers/quantization/calibration.py

## 5) 分布式校准同步

在分布式场景，激活 observer 会在校准边界做跨 rank 同步，包含 q/k/v observer。

流程：

1. all-reduce 统计量。
2. 用全局统计量重算 qparams。
3. 回写模块参数。

实现位置：

- src/llmcompressor/modifiers/quantization/quantization/mixin.py

## 6) 冻结与导出

校准结束后：

- 移除校准 hook。
- detach 并删除 observer。
- quantization_status 置为 frozen。

导出模型时，quantization_config 中会包含 kv_cache_scheme。测试还会验证注意力模块含有 k_scale 与 v_scale。

验证参考：

- tests/llmcompressor/transformers/kv_cache/test_kv_cache.py

## 7) 运行时消费（vLLM）

仅在导出时写入 kv_cache_scheme 还不够，运行时也必须显式启用 KV 低精度缓存类型。

在 vLLM 中，需要设置 kv_cache_dtype=fp8 以让运行时按量化 KV 路径使用标定参数。

参考：

- examples/quantization_kv_cache/README.md

## 8) 与 QuantizationModifier / GPTQModifier 的关系

两者都复用 QuantizationMixin 的 KV 解析与激活校准生命周期。

区别在于：

- QuantizationModifier：偏标准 PTQ/QAT 流程。
- GPTQModifier：在权重量化侧增加 Hessian 驱动的 GPTQ 逻辑，但 KV 激活侧的 observer 与校准机制一致。

实现位置：

- src/llmcompressor/modifiers/quantization/quantization/base.py
- src/llmcompressor/modifiers/gptq/base.py

## 9) 参数选择建议

1. 首次落地优先 FP8 + per-tensor，先确保稳定。
2. 校准集要覆盖真实场景中的长上下文与任务分布。
3. 若出现长文本后段质量下滑，先增加校准样本或改进样本分布，再考虑更保守量化。
4. 先验证显存收益，再验证质量，最后调吞吐。

## 10) 常见问题

1. 模型结构不匹配：不满足注意力命名/结构约定时，KV 量化可能失败。
2. 运行时未开启：未设置 kv_cache_dtype 时，导出的标定参数不会按预期生效。
3. 校准不足：样本过少或分布偏差，会导致 scale 不稳定，出现准确率回退。

## 11) 快速检查清单

1. recipe 中已配置 kv_cache_scheme。
2. 导出后的 config.json 含 quantization_config.kv_cache_scheme。
3. 重新加载后，注意力模块包含 k_scale 与 v_scale。
4. 推理命令显式设置 kv_cache_dtype。

## 12) Qwen3.5 MoE 多模态脚本接入模板

你当前多模态脚本使用的是 GPTQModifier + IMatrixGatherer 组合，接入 KV Cache 量化时，核心是给 GPTQModifier 增加 kv_cache_scheme。

参考脚本：

- examples/multimodal_vision/qwen_3_5_moe_w4a8_int8_imatrix_gptq.py

可直接复用的 recipe 片段（示意）：

```python
from compressed_tensors.quantization.quant_args import (
  QuantizationArgs,
  QuantizationStrategy,
  QuantizationType,
)

kv_cache_args = QuantizationArgs(
  num_bits=8,
  type=QuantizationType.FLOAT,
  strategy=QuantizationStrategy.TENSOR,
  dynamic=False,
  symmetric=True,
)

recipe = [
  IMatrixGatherer(ignore=["lm_head"]),
  GPTQModifier(
    targets="Linear",
    offload_hessians=True,
    config_groups={"group_0": scheme},
    kv_cache_scheme=kv_cache_args,
    ignore=[
      "re:.*lm_head",
      "re:visual.*",
      "re:model.visual.*",
      "re:.*mlp.gate$",
      "re:.*embed_tokens$",
      "re:.*shared_expert_gate$",
      "re:.*linear_attn.*",
    ],
  ),
]
```

运行时要点：

1. 若推理端是 vLLM，需要显式传入 kv_cache_dtype=fp8。
2. ignore 规则不要误伤真正承担 KV 投影的模块，否则会出现“配置了 kv_cache_scheme 但未生效”。
3. 对混合注意力结构（full attention + linear attention）模型，建议先只在 full attention 路径验证，再逐步扩展。

建议验证步骤：

1. 压缩后检查 config.json 是否包含 quantization_config.kv_cache_scheme。
2. 重新加载模型，抽样检查 attention 子模块是否有 k_scale、v_scale。
3. 在固定提示词下对比开启/关闭 kv_cache_dtype 的显存与输出质量。

## 13) CHANNEL 策略兼容实现说明（关键设计）

本节说明为什么 `kv_cache_scheme.strategy=channel` 在上游会报错，以及本仓库如何以最小侵入方式支持它。

### 13.1 问题根因

在 `initialize_quantization` 流程中，`kv_cache_scheme` 会被交给上游 `apply_quantization_config`。上游会在 `_apply_kv_cache_scheme` 内部构造：

```python
QuantizationScheme(input_activations=kv_cache_scheme)
```

而 `compressed_tensors` 当前对 activation strategy 的校验不允许 `CHANNEL`，因此会抛出：

- `NotImplementedError: Using channel strategy is not supported for activation quantization`

### 13.2 设计目标

1. 不改变用户配置接口：用户仍可直接写 `strategy=QuantizationStrategy.CHANNEL`。
2. 不影响其他策略：仅在 `kv_cache_scheme.strategy == CHANNEL` 时走兼容分支。
3. 尽量复用上游默认流程：仅替换会报错的 KV cache 应用步骤。

### 13.3 实现入口

实现位于：

- `src/llmcompressor/modifiers/quantization/quantization/mixin.py`

在 `initialize_quantization` 中：

1. 检测 `config.kv_cache_scheme.strategy == CHANNEL`。
2. 先执行本地 `_apply_channel_kv_cache_scheme(...)`。
3. 将临时配置中的 `kv_cache_scheme` 置空，再调用上游 `apply_quantization_config(model, config)`。

这样做的含义是：

- KV cache 的 CHANNEL 初始化由本地完成。
- 其余线性层/注意力量化流程仍沿用上游逻辑。

### 13.4 为什么还需要静态初始化补丁

仅绕过校验还不够。对于 `dynamic=False` 的 CHANNEL，若直接调用上游注意力初始化，`observed_shape` 含未知序列维（`None`），会在 `torch.empty(expected_shape)` 时触发类型错误。

因此新增 `_initialize_static_channel_attention_qparams(...)`，改为使用注意力几何先验初始化：

- `channels = num_heads * head_dim`
- `q` 用 `num_attn_heads * head_dim`
- `k/v` 用 `num_kv_heads * head_dim`

这避免了把 `None` 带入 scale 形状，同时与 CHANNEL 语义一致（按通道量化）。

### 13.5 兼容分支行为边界

当前分支只在以下条件触发：

1. 存在 `kv_cache_scheme`
2. `strategy == CHANNEL`

其中：

- `dynamic=False`：走自定义静态 qparams 初始化。
- `dynamic=True` / `DynamicType.LOCAL`：复用上游 `initialize_module_for_quantization` 路径。

### 13.6 回归测试

新增用例：

- `tests/llmcompressor/modifiers/quantization/test_kv_cache_calibration.py`
- `test_kv_cache_channel_strategy_is_supported_for_attention_quantization`

验证点：

1. `initialize_quantization` 不再因 CHANNEL 报错。
2. attention 模块保留 `input_activations.strategy == CHANNEL`。
3. 量化状态正确进入 `INITIALIZED`。

### 13.7 后续维护建议

若未来上游 `compressed_tensors` 原生支持 activation CHANNEL（含 KV cache 场景），建议：

1. 先验证上游路径可覆盖当前测试。
2. 再移除本地兼容分支，回归单一路径。
3. 保留回归测试，防止后续版本再次退化。

### 13.8 参考代码索引（可直接跳转）

本实现的主要参考与落地点如下。

- 兼容分支入口（CHANNEL 检测与分流）：
  [src/llmcompressor/modifiers/quantization/quantization/mixin.py](../../src/llmcompressor/modifiers/quantization/quantization/mixin.py#L227)
- CHANNEL 的 KV cache 应用逻辑：
  [src/llmcompressor/modifiers/quantization/quantization/mixin.py](../../src/llmcompressor/modifiers/quantization/quantization/mixin.py#L261)
- 静态 CHANNEL 的 q/k/v qparams 初始化：
  [src/llmcompressor/modifiers/quantization/quantization/mixin.py](../../src/llmcompressor/modifiers/quantization/quantization/mixin.py#L306)
- quantization config 解析入口（`kv_cache_scheme` 来源）：
  [src/llmcompressor/modifiers/quantization/quantization/mixin.py](../../src/llmcompressor/modifiers/quantization/quantization/mixin.py#L441)
- 回归测试（验证 CHANNEL 可初始化）：
  [tests/llmcompressor/modifiers/quantization/test_kv_cache_calibration.py](../../tests/llmcompressor/modifiers/quantization/test_kv_cache_calibration.py#L130)

上游行为参考（依赖包 `compressed_tensors`）：

- `compressed_tensors.quantization.lifecycle.apply._apply_kv_cache_scheme`
- `compressed_tensors.quantization.quant_scheme.QuantizationScheme.validate_model_after`

其中第二个校验函数是报错来源：activation strategy 当前不接受 `CHANNEL`。
