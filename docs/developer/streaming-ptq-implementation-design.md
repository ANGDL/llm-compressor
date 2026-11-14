# Streaming PTQ 实现设计

日期：2026-11-14

状态：当前实现与后续演进的详细设计。代码已经实现的行为以本文“状态”和
“实现位置”为准；仍处于计划中的能力不能被当作已经支持的用户契约。

本文专门记录 Streaming PTQ 如何复用 oneshot Sequential Pipeline 的算法语义，
同时把权重、校准统计、activation 和 checkpoint 输出限制在可计算的资源上界内。
它覆盖普通 dense 模型、MoE、DeepSeek V4/V4.1 等 checkpoint 特化模型，以及
HIGGS、MTP 和 QAD 的兼容边界。

相关文档：

- [Streaming PTQ 逐 Subgraph 闭环计划](streaming-ptq-coding-plan.md)：历史设计、
  迁移步骤和阶段性记录。
- [事务化 Streaming PTQ 计划](transactional-streaming-ptq-plan.md)：事务恢复、
  boundary 回收和未来并行扩展。
- [Developer index](index.md)：开发者文档入口。

## 1. 问题与设计结论

DeepSeek V4.1 GPTQ 的失败不是 checkpoint 权重本身无法放入显存，而是 Hessian
生命周期过长。失败 run `20261112T090344Z_a619a1629881` 在
`model.layers.3` 的 admission 阶段失败：

```text
GPTQ 请求：       119,601,948,696 bytes
安全预留后可用：    92,534,870,477 bytes
```

DSV4.1 的一个 decoder block 包含 384 个 routed experts。对单个 expert，
`w1/w3` 的 Hessian 约为 `5120² * 4` bytes，`w2` 的 Hessian 约为
`2304² * 4` bytes，合计约 221 MiB；384 个 expert 在同一个
`DeepseekV41Block` 中累计约 85 GiB。attention、shared expert、权重和
scratch 加入后，峰值达到约 119.6 GB。

根因是 streaming 把整个 `DeepseekV41Block` 当成一个 sequential target。
GPTQ 必须等这个 target 的 `sequential_epoch_end` 才能压缩并释放 Hessian，
所以所有 routed experts 的 Hessian 同时存活。

设计结论是：

1. **处理单元要缩小到 GPTQ 可结束的 logical target。** 对 DSV4.1，默认
   target policy 应将 routed expert Linear 拆成可独立完成
   `calibration -> sequential_epoch_end -> compression -> cleanup` 的目标。
2. **Hessian 必须是 target-local 资源。** 完成一个 target 后立即释放 Hessian、
   sample count、observer 和临时分解结果；不能把 Hessian 放入全模型 statistics
   artifact。
3. **资源上界由当前 working set 决定。** admission 需要同时计算当前 target
   的物化权重、Hessian、quantization scratch、boundary 和 writer completion
   headroom。
4. **特化模型必须通过 wrapper/materializer/target policy 接入。** 主 pipeline
   不根据模型名称散落判断；DSV4.1 的布局、专家切片和 native MTP ownership
   由特化适配层声明。
5. **不恢复淘汰的暴力 CPU Hessian offload。** CPU cache 只允许作为明确算法
   语义的一部分，例如 QAD 的训练样本缓存；它不能被用来掩盖错误的 target
   粒度，也不能让全模型 Hessian 继续累积。

## 2. 状态约定

本文使用以下状态：

| 状态 | 含义 |
|---|---|
| 已实现 | 代码已存在，至少有针对性测试覆盖 |
| 已验证 | 代码已存在，并有端到端或数值对照证据 |
| 部分实现 | 基础路径可用，但组合矩阵或泛化协议尚未完成 |
| 计划中 | 设计已经确定，代码尚未完成 |
| 不支持 | 当前契约明确拒绝，不能通过隐式 fallback 使用 |

当前状态：

| 能力 | 状态 | 主要实现位置 | 当前边界 |
|---|---|---|---|
| target-local GPTQ Hessian 生命周期 | 已验证 | `streaming/pretrained.py`、`streaming/pipeline.py`、GPTQ adapter | 自动 target grouping 仍未实现 |
| explicit target grouping | 已验证 | `streaming/tracing.py`、`streaming/pipeline.py` | 由 `sequential_targets_per_subgraph` 指定整数 |
| target admission | 已实现 | `streaming/admission.py`、`streaming/pipeline.py` | 估算依赖 modifier 和 materializer 声明 |
| checkpoint-backed weight session | 已验证 | `streaming/loading/*`、`streaming/materialization/*` | 单设备、已知 checkpoint 格式 |
| transaction/resume | 已实现 | `streaming/checkpoint/*`、`streaming/recovery.py` | 完整强杀矩阵仍需扩大 |
| bounded HIGGS provider | 已验证 | `streaming/higgs.py`、`entrypoints/higgs/*` | 尚未成为主 pipeline 自动 stage |
| MTP upstream wrapper | 部分实现 | `streaming/mtp.py`、`streaming/pretrained.py` | 多 MTP、量化 MTP 和完整特化组合矩阵待补 |
| DSV4.1 native MTP 保持 | 已验证 | `streaming/mtp.py`、DeepSeek materializer | native extra-head ownership 仍需更多 checkpoint 覆盖 |
| QAD + Quantization | 已实现 | `streaming/modifier_adapters.py` | 仍需要 CPU cache，weight-only |
| QAD + GPTQ | 已实现 | `streaming/modifier_adapters.py` | numerical parity 和大模型资源验证待补 |
| AutoRound compatibility | 已验证 | `streaming/modifier_adapters.py` | 不支持 target grouping，需单 target |
| observer numerics parity | 计划中 | observer/grid-search/iMatrix | 需适配 upstream `b3faa86f3` |
| `sequential_targets_per_subgraph="auto"` | 计划中 | tracing/admission | 不能只修改 tracer 参数 |
| REAP streaming | 不支持 | — | 结构修改协议尚未实现 |
| multi-GPU streaming | 不支持 | `streaming/pipeline.py` | 当前要求单 participant |
| unknown packed layout | 不支持 | materializer contract | 必须先实现 materializer |

“已实现”只代表当前契约下的代码存在。例如 QAD + GPTQ 已有 adapter，并不代表
QAD 已经具备与 GPTQ 单独运行相同的全部质量和资源验证。

## 3. 总体架构

Streaming PTQ 的主入口是：

```text
streaming_oneshot_from_pretrained()
  -> build_meta_model()
  -> StreamingMTPWrapper.attach()
  -> exact quantization schemes
  -> trace_streaming_boundaries()
  -> run_subgraph_streaming_pipeline()
  -> finalize_streaming_checkpoint()
```

主 pipeline 由以下职责组成：

```text
                   ┌──────────────────────────┐
                   │ pretrained setup         │
                   │ meta model + recipe      │
                   └────────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
  shared tracing          checkpoint source       materializer
  target/subgraph plan    logical tensor metadata  dtype/layout decode
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                ▼
                  ┌──────────────────────────┐
                  │ subgraph pipeline        │
                  │ admission                 │
                  │ weight session            │
                  │ modifier lifecycle        │
                  │ boundary store            │
                  │ writer/transaction       │
                  └────────────┬─────────────┘
                                ▼
                  ┌──────────────────────────┐
                  │ finalizer                 │
                  │ static tensors + index    │
                  │ config + atomic publish   │
                  └──────────────────────────┘
```

主循环不负责解释模型格式和算法数学：

- `CheckpointWeightSource` 负责按 logical name 查询 checkpoint。
- `WeightMaterializer` 负责源格式、packed layout、依赖 tensor 和输出命名。
- tracing 负责真实 forward 图、target 顺序和 boundary 名称。
- `ModifierExecutionAdapter` 负责 modifier 的窄生命周期和资源估算。
- `BoundaryActivationStore` 负责相邻 subgraph 之间的具名输入。
- writer/transaction 负责输出和恢复，不持有完整模型 `state_dict`。

## 4. 一次 subgraph transaction 的生命周期

一个执行单元可以是一个 target，也可以是连续的 target group。它必须满足：
所有 target 在 group 内都能完成自己的 modifier lifecycle，且 group 完成后才
发布下一个 boundary。

### 4.1 状态机

```text
PLANNED
  │ admission 通过
  ▼
RESERVED
  │ prepare/load/install
  ▼
LOADED
  │ calibration forwards
  ▼
OBSERVED
  │ modifier update / sequential_epoch_end
  ▼
UPDATED
  │ propagation 或复用 calibration output
  ▼
PROPAGATED
  │ compress + write pending output
  ▼
PENDING
  │ output、boundary、metadata 完整提交
  ▼
COMMITTED
  │ unload + delete consumed boundary
  ▼
RELEASED
```

异常规则：

- `PLANNED`、`RESERVED`、`LOADED`、`OBSERVED`、`UPDATED` 或 `PROPAGATED`
  失败：释放当前资源，当前 transaction 重做。
- `PENDING` 失败：删除未提交 pending records，当前 transaction 重做。
- `COMMITTED` 但 frontier 尚未更新：扫描 transaction metadata 恢复，不重做。
- committed output 或 checksum 损坏：报告损坏并停止，不能静默覆盖。

### 4.2 实际执行顺序

当前 `run_subgraph_streaming_pipeline()` 的语义顺序是：

```text
1. 确认当前 boundary 存在
2. 清理上次失败留下的 next boundary
3. 规划当前 target/group 的 working set
4. 估算并预留 device、host、disk 资源
5. prepare checkpoint tensor
6. move 到执行设备并 install 到 meta model
7. 初始化 layerwise quantization
8. 使用 DisableQuantization + no_grad 执行 calibration
9. 调用 modifier update / sequential_epoch_end
10. 必要时使用更新后的权重重新 forward，生成 next boundary
11. compress modules
12. 写 output tensor 和 next boundary
13. commit transaction
14. unload weight、释放 Hessian/observer、删除 consumed boundary
15. 进入下一个 target/group
16. 复制 static tensor、assemble shards、finalize
```

对只收集统计、不会修改传播结果的 modifier，可以复用 calibration output。
对 GPTQ、SparseGPT、Wanda、QAD 等会更新权重或依赖更新后权重的 modifier，
必须执行更新后的 propagation，除非 adapter 明确声明了不同的 propagation policy。

## 5. 资源和所有权协议

### 5.1 资源类别

当前执行单元的资源分成六类：

```text
W_materialized  当前 target/group 的 logical 权重和 materializer dependency
W_modifier      modifier 持有的权重副本、Hessian、observer、optimizer/scratch
A_boundary      当前输入 boundary 和待提交 next boundary
S_scratch       forward、kernel、packing、quantization 临时空间
H_completion    output snapshot、CPU 写入和 async writer completion headroom
D_pending       transaction/output/boundary 的磁盘空间
```

device admission 的基本形式：

```text
device_required =
    W_materialized
  + W_modifier.peak
  + S_scratch
  + optional prefetch overlap
```

host admission 的基本形式：

```text
host_required =
    prepare buffers
  + W_modifier.host
  + optional CPU boundary
  + optional pinned H2D staging
  + H_completion
```

disk admission 的基本形式：

```text
disk_required =
    output staging
  + pending boundary records
  + transaction metadata
  + safety margin
```

估算必须在加载当前 target 前完成。估算不足时直接拒绝本次 target，不能先加载
再依赖 OOM 捕获来恢复。

### 5.2 所有权规则

- `WeightSession` 拥有当前 target 的实体参数和 materializer 临时依赖。
- modifier adapter 拥有算法内部状态，但必须在 `after_update` 或 `abort` 后清理。
- boundary store 拥有跨 target 的 activation；pipeline 不保留额外长期引用。
- writer 只拥有已经转移给它的不可变 CPU tensor 或 transaction record。
- finalizer 只读取 committed records 和 source static tensor，不重新执行量化。

当前 target 退出时必须释放：

- 原始和量化前的 device weight；
- Hessian、Cholesky/inverse、sample count；
- observer、activation cache 和 hooks；
- materializer raw/dependency tensors；
- consumed boundary；
- 已完成但未被 prefetch 使用的 reservation。

### 5.3 prefetch 约束

prefetch 只能在 admission 明确有余量时准备下一个 target。它可以占用 host
memory 和 device memory reservation，但不能改变算法的 live working set 语义。
如果 completion headroom、next target 或 pinned transfer 不满足预算，prefetch
必须延期，当前 target 仍能独立完成。

## 6. GPTQ Hessian 设计

### 6.1 目标粒度

GPTQ 的逻辑 target 是能够独立完成 `sequential_epoch_end` 的模块集合。默认
dense 模型可以按 decoder layer 或 Linear 运行；包含大量 routed experts 的模型
必须使用更细粒度的 target policy。

当前入口在检测到 expert collection 且用户没有显式 `sequential_targets` 时，
从 GPTQ 的 `Linear` 语义开始解析 target。随后会把同一个 expert owner
下面的多个量化 Linear 合并为一个 owner target；当调用方没有显式设置
`sequential_targets_per_subgraph` 时，再把最多八个连续 owner target 放入
一个 subgraph：

```python
automatic_gptq_targets = (
    sequential_targets is None
    and isinstance(quantizer, GPTQModifier)
    and _has_expert_collection(meta_model)
)
```

随后 `_quantized_sequential_targets()`：

1. 根据 recipe 找出真正匹配 quantization scheme 的模块；
2. 从 shared `infer_sequential_targets()` 推导可执行 target；
3. 对 attention leaf 保留必要的 attention ancestor；
4. 对路径中带有 expert owner 的模块，向上找到包含多个量化后代的最小
   非 `ModuleList`/`ModuleDict` owner；
5. 删除被 owner 或更小 ancestor 覆盖的重复 target；
6. 检查每个量化模块都被某个 target 覆盖；
7. 将结果交给 shared tracer 和 streaming pipeline，并对自动 GPTQ 使用有界
   target group。

这样处理的意义是让 GPTQ 在一个 bounded expert-owner group 结束时就能进入
`sequential_epoch_end`，而不是等待整个 MoE block 的所有专家，也不会为
大规模 MoE 建立数万条独立 tracing boundary。实际 group 仍经过 admission
检查，物化权重、Hessian 和 scratch 超过设备预算时会在执行前拒绝。

### 6.2 Hessian 生命周期

对每个当前 target/group：

```text
before_target
  -> initialize layerwise calibration
  -> calibration forward: H += XᵀX, count += batch/token count
  -> sequential_epoch_end / modifier update
  -> quantize_weight + error compensation
  -> compress_modules
  -> export compressed weight/qparams
  -> detach hooks
  -> release H, count, decompositions and scratch
```

Hessian 不进入：

- global statistics store；
- next boundary；
- source checkpoint；
- committed transaction；
- prefetch cache。

transaction 只保存量化输出和恢复所需的 boundary。当前 target 在提交前崩溃时，
恢复逻辑重放该 target 的校准；已经提交后，Hessian 已经没有恢复价值。

### 6.3 DSV4.1 资源估算

对一个 `in_features = d` 的 GPTQ Linear：

```text
Hessian bytes ≈ d * d * sizeof(float32)
```

对当前执行单元：

```text
H_total =
    Σ expert_in_features² * 4
  + attention Hessian bytes
  + shared expert Hessian bytes
```

实际 admission 还必须加上：

```text
device_required =
    materialized weights
  + H_total
  + GPTQ factorization/decomposition scratch
  + quantized output/qparams
  + current boundary and forward workspace
```

DSV4.1 的安全做法是让 `H_total` 只包含当前 routed expert 或受控的专家 group。
如果一个专家 Linear 仍然无法通过 admission，应该：

1. 降低 target group 大小；
2. 使用模型适配层提供的按专家切片 materializer；
3. 调整 calibration microbatch/token 数；
4. 明确报告单个 Linear 的 Hessian 下界；

而不是把 Hessian 整体搬到 CPU 后继续保持错误的 target 粒度。

### 6.4 量化后传播

GPTQ 更新当前权重后，如果存在 successor target，pipeline 会：

```text
HooksMixin.disable_hooks()
  + torch.no_grad()
  + propagation mode
  -> forward current target
  -> write next boundary
```

普通 GPTQ propagation 默认关闭 fake quantization，使下一个 target 接收
Sequential Pipeline 语义下的更新后 full-precision output。只有 adapter 明确
声明 `post-update-quantized` 时才保留量化执行路径。传播阶段不能重新累积
Hessian，也不能触发当前 target 的校准 hooks。

## 7. target grouping

### 7.1 显式 grouping

`sequential_targets_per_subgraph` 是正整数，默认值为 `1`。shared tracer 可以
把连续 target 组成 target group；pipeline 对 group 统一：

- admission；
- weight session；
- modifier lifecycle；
- calibration；
- propagation；
- transaction；
- output shard 写入。

group 内 target 的 boundary 不单独发布，只有 group 完成后发布 successor
boundary。这样可以表达跨模块的局部变换，同时保持资源上界可估算。

### 7.2 adapter 兼容性

adapter 通过 `supports_target_groups` 声明是否允许 group：

- GPTQ、Quantization、QAD 等 target-local adapter 默认可以参与 group，但
  group 的 Hessian 总量必须通过 admission；
- AutoRound 当前设置为 `supports_target_groups = False`，必须使用
  `sequential_targets_per_subgraph=1`；
- AutoSmooth 要求 mapping 的所有 owner/smooth/balance layer 落在同一执行
  owner 内，不能跨 target boundary；
- 未声明 working set 的 modifier 不得借助 group 偶然访问 meta parameter。

### 7.3 自动 grouping 尚未实现

`sequential_targets_per_subgraph="auto"` 不能简单转换成一个较大的整数。自动
模式需要：

1. 先用 meta metadata 计算每个 target 的 materialized bytes；
2. 调用所有 modifier 的 `estimate_memory()`；
3. 加入 Hessian、scratch、boundary、writer completion 和 prefetch overlap；
4. 按连续性和模型图约束形成 candidate groups；
5. 对每个 group 做 admission；
6. 以失败即缩小 group、成功即尝试扩大的策略生成最终 plan；
7. 把最终 group plan 写入 run fingerprint。

在这项能力完成前，默认保持 `1`，避免“看似自动”但重新形成 DSV4.1 block
级 Hessian 峰值。

## 8. HIGGS bounded provider

### 8.1 定位

HIGGS 是配置选择层，不是普通 `Modifier`。它先为候选 scheme 计算权重 MSE，
然后在全局 bitwidth 约束下求 ILP，返回普通 `QuantizationConfig`。因此它不能
持有整个模型的权重，也不应被强行塞进当前 GPTQ modifier lifecycle。

公开入口：

```python
from llmcompressor.entrypoints.higgs import get_higgs_config_streaming

config = get_higgs_config_streaming(
    "./model",
    candidate_schemes=["NVFP4A16", "FP8_DYNAMIC"],
    targets="Linear",
    ignore=["lm_head"],
)
```

实现位置：

- `streaming/higgs.py`：bounded source/materializer provider；
- `entrypoints/higgs/__init__.py`：公开入口；
- `entrypoints/higgs/base.py`：复用 MSE collector/converter；
- `entrypoints/higgs/ilp_solver.py`：复用 ILP；
- `tests/llmcompressor/streaming/test_higgs.py`：有界加载、融合和 stacked
  expert 测试。

### 8.2 执行流程

```text
checkpoint metadata scan
  -> logical weight expansion
  -> targets/ignore matching
  -> fused group formation
  -> source.iter_tensor_groups_cpu()
  -> materializer.materialize_cpu()
  -> candidate scheme MSE
  -> retain MSE/parameter/fusion summaries
  -> clear raw/decoded/group values
  -> repeat next group
  -> global ILP
  -> QuantizationConfig
  -> provider.config_fingerprint
```

`StreamingHiggsConfigProvider.collect()` 只保留 collector 需要的摘要。ILP 求解
之前不保留完整 weight、Hessian 或 activation。

provider 将当前逻辑 group 映射为物理 tensor/dependency 请求，并通过
`iter_tensor_groups_cpu()` 复用每个 source shard 的只读 handle。这样只减少重复
打开 shard 的 I/O 开销，不改变内存上界：当前 group 的 raw tensor 在
`materialize_cpu()` 完成后立即失去引用，下一 group 才会继续读取。provider 不会
把完整 shard 或完整 checkpoint 放入长期缓存。

### 8.3 logical weight 与物理 tensor

provider 区分：

```text
physical tensor: checkpoint 中真实存储的 tensor
logical weight: 量化器和 HIGGS 认为独立的二维权重
```

当前支持：

- 普通二维 dense Linear；
- HIGGS fused group；
- 常见三维 stacked expert layout；
- `gate_up_proj`/`w13` 等 fused expert projection 的按 expert/half slice。

同一个 physical tensor 可以被多个 logical weight 读取，但一次 group 内只保留
当前需要的 decoded view。物理 tensor 的 source dependencies 由 materializer
提供，provider 不判断 DSV4.1 或 Kimi 的模型名称。

### 8.4 泛化要求

新模型要接入 HIGGS 时，优先实现：

1. `WeightMaterializer.dependencies()`；
2. `materialize_cpu()`；
3. `output_tensor_name()`/`output_module_name()`；
4. 必要时 source 的 logical slice metadata；
5. provider 的通用 fused/stacked layout 规则无法表达时，增加 materializer
   返回的 logical weight descriptor。

禁止在 HIGGS provider 中增加：

```python
if model_name == "DeepSeekV4":
    ...
```

特化布局属于 source/materializer contract。provider 只消费 logical metadata
和二维 weight view。

### 8.5 HIGGS 与 streaming PTQ 的组合

当前推荐组合：

```text
get_higgs_config_streaming()
  -> 返回 QuantizationConfig
  -> 构造 QuantizationModifier 或 GPTQModifier
  -> streaming_oneshot_from_pretrained()
```

这样 HIGGS 的全局决策和 GPTQ 的 target-local Hessian 生命周期相互独立：
HIGGS 不保存 Hessian，GPTQ 也不需要重新读取 HIGGS 的全模型权重。

DeepSeek V4.1 示例还允许 HIGGS 与 `--quantize-engram-wkv` 组合。provider
临时把 Engram `wkv` 加入 HIGGS ignore，避免它参与通用候选 scheme 的 ILP；
脚本随后把固定的 Engram W8A8 FP32-scale group 合并回 HIGGS 生成的 groups。
因此 HIGGS 的平均 bitwidth 约束只作用于普通二维权重，Engram 的专用存储契约
保持不变。这个环绕式组合逻辑位于示例的模型特化层，通用 HIGGS provider 不
包含 Engram 或 DeepSeek 判断。

尚未完成的是把 provider 自动注册为主 pipeline 的 recipe stage。自动 stage
必须解决 config fingerprint、recipe fingerprint 和恢复语义后才能接入，不能
为了方便在 pipeline 中再次遍历完整模型。

## 9. MTP wrapper 与特化模型兼容

### 9.1 设计目标

MTP 兼容必须满足：

- 已经有 native `model.mtp` 的模型保持原生 ownership；
- config 声明 MTP 但 base model 尚未挂载时，复用 upstream `MtpModel`；
- wrapper 在 meta device 构造，不读取完整 checkpoint；
- shared `embed_tokens`、`shared_head`、`rotary_emb` 不重复注册；
- MTP checkpoint key conversion 进入同一 materializer；
- wrapper-attached upstream MTP 默认从 ordinary quantizer 中跳过 extra state；
- DSV4.1 没有 MTP 时不被强制挂载 MTP，也不改变其 materializer 路径。

### 9.2 当前实现

`StreamingMTPWrapper.attach(meta_model)`：

```text
has model.mtp?
  yes -> return attached=False, preserve native model
  no
    config declares MTP?
      no -> return attached=False
      yes -> get_mtp_config()
             build_mtp_model() on meta
             prepare optional forward globals
             detach shared aliases
             collect streaming weight conversions
             return attached=True
```

当前 wrapper 复用：

- upstream `MtpModel` construction；
- GLM-MoE-DSA adapter；
- Qwen3.5 adapter；
- Transformers conversion mapping。

ordinary quantization setup 会把：

```text
re:^mtp(?:\.|$)
re:^model\.mtp(?:\.|$)
```

加入 quantization config ignore。这样 MTP 不会无意间加入 base model GPTQ
Hessian，也不会与 DSV4.1 的 routed expert target 混在一起。

### 9.3 checkpoint conversion

MTP conversion 必须发生在 source 读取 logical name 之前：

```text
raw checkpoint key
  -> Transformers MTP conversion
  -> streaming materializer logical name
  -> target matching / ignore
  -> load current working set
```

conversion 只能添加 scoped mapping，不得重写整个 source index。映射必须进入
run fingerprint，避免用旧 transaction 恢复到不同的 MTP key layout。

### 9.4 DSV4.1 环绕式兼容规则

DSV4.1 的兼容应通过以下层次完成：

```text
DSV4.1 config/model factory
  -> native DSV4.1 model behavior
  -> DeepSeekV4/V41 WeightMaterializer
  -> streaming target policy
  -> generic pipeline
```

MTP wrapper 只在它检测到 declared MTP 且 native model 没有 `model.mtp` 时生效。
它不能：

- 修改 DSV4.1 的 routed expert naming；
- 把 DSV4.1 native extra-head 当成 upstream MTP；
- 把 DSV4.1 的 attention runtime buffer 当成普通 parameter；
- 为所有模型添加 `model_name` 分支；
- 让 MTP extra state 共享 base model GPTQ Hessian。

仍需补的验证矩阵：

| 模型路径 | 预期 |
|---|---|
| 普通模型，无 MTP | 与引入 wrapper 前一致 |
| upstream MTP 模型 | meta wrapper、conversion、ignore 和 reload 成功 |
| GLM5 类特化 MTP | 复用 upstream adapter，shared alias 不重复 |
| DSV4.1，无 MTP | native materializer/target policy 独立通过 |
| DSV4.1，native extra state | 不被通用 wrapper 接管，state ownership 明确 |
| MTP + GPTQ | base target 的 Hessian 不含 MTP extra state |

### 9.5 MTP 量化边界

当前默认只量化 base model，MTP extra state preserve/ignore。要支持 MTP 自身
量化，必须增加：

- MTP target ownership；
- MTP 专用 working set；
- shared embedding/head 的唯一 owner；
- MTP output delta；
- finalizer 的 config/index 更新；
- MTP 与 base propagation 的顺序定义。

在这些协议完成前，不能因为 wrapper 能构造 MTP 就宣称“支持 MTP 量化”。

## 10. Modifier adapter 通用协议

### 10.1 为什么需要 adapter

主 pipeline 只理解“当前 working set 的生命周期”，不应理解每个 modifier 的
私有字段。adapter 把 upstream/oneshot modifier 的差异压缩成四个声明：

```text
lifecycle          初始化、校准、更新、结束
working set        事件会读取或修改的模块/参数
output delta       新增、修改、删除的 tensor/config
propagation mode   更新后如何产生 successor boundary
```

当前公共接口包括：

- `validate_recipe()`
- `validate_targets()`
- `estimate_memory()`
- `estimate_host_memory()`
- `before_calibration()`
- `before_target()`
- `after_update()`
- `update_context()`
- `restore_target()`
- `abort()`

这些方法不是算法重写入口，而是对已有 modifier lifecycle 的窄包装。

### 10.2 working set

working set 可以大于量化 target，例如：

- AutoSmooth 同时需要 norm、parent 和 balance layers；
- SmoothQuant/AWQ 需要 mapping 内多个 Linear；
- MTP wrapper 需要 shared alias 和 extra state 的 ownership；
- MoE expert adapter 需要当前 expert slice 和 router/shared dependency。

working set 必须：

1. 可由 meta graph 和 recipe 推导；
2. 在 admission 前估算；
3. 由 `WeightSession` 一次性 materialize；
4. 在 transaction 结束时统一导出和释放；
5. 在参数共享跨 target 且无唯一 owner 时拒绝 recipe。

### 10.3 output delta

当前 pipeline 可以把 loaded target 下的 state tensor 写为 output delta。后续需要
把它正式抽象为：

```text
modified tensor records
new tensor records
deleted tensor names
renamed tensor mapping
config patch
```

结构修改型 modifier 只有在 writer/finalizer 支持这些类型后才能进入 streaming。

## 11. QAD 低优先级兼容

### 11.1 当前支持

当前 `QADStreamingAdapter` 支持：

```text
[QuantizationModifier, QADModifier]
[GPTQModifier, QADModifier]
```

生命周期：

```text
quantizer initialize
  -> QAD capture target-local input/output
  -> block-wise grouped training
  -> weight re-observation
  -> fake-quantized/materialized weight update
  -> post-update-quantized propagation
  -> clear cache
```

它保留 upstream QAD 的：

- block-wise training；
- shared calibration data；
- multiple sequential targets per subgraph 的模型能力；
- weight re-observation；
- validation split。

### 11.2 明确限制

当前 QAD 仍要求：

- `offload_device` 为 CPU；
- weight-only quantization；
- 不支持 KV cache quantization；
- 至少两个 calibration batches；
- 当前 target 参数不能被 target 外部共享；
- 与 QAD 不兼容的 quantizer/modifier 顺序直接拒绝。

这里的 CPU cache 是 QAD 训练过程的算法状态，不是 GPTQ Hessian 的逃生通道。
QAD 未来若要去掉 CPU cache，必须重新设计 block-wise optimizer state 的资源
协议，而不是复用 GPTQ Hessian offload。

### 11.3 后续工作

QAD 排在 HIGGS 和 MTP 之后。优先完成：

1. GPTQ + QAD 与 upstream 的 qparams/weight parity；
2. grouped target 的 cache 生命周期；
3. crash recovery 时 QAD block state 的 replay；
4. host/device budget 的实测；
5. 只有在这些完成后再评估非 CPU cache。

## 12. upstream/main 兼容矩阵

以下是当前审计中与 Streaming PTQ 直接相关的 upstream 特性。兼容原则是：
复用 upstream 算法语义，适配 streaming 的 working set、admission 和 transaction，
不把 oneshot 的整模型 offload 机制原样搬回来。

| upstream 提交/特性 | oneshot 改变 | streaming 处理 | 状态 |
|---|---|---|---|
| `b3faa86f3` `Unify MSE and iMatrix observer error numerics` | 统一 FP32/FP64 error reduction、scale rounding、`patience=0` | 继续复用共享 observer/quantizer；保留 streaming bounded chunk search 和硬件 backend dispatch；补数值 parity | 计划中 |
| `176df1cdd` `Improve REAP DDP router diagnostics` | 记录 router logits/probability、top-k boundary gap | REAP 是结构修改型 modifier，不改变 streaming contract；不添加 REAP 特化分支 | 不支持 |
| `179140f73` AutoRound `fp_ref` 预填充 | `IntermediatesCache` 提供 FP16 reference，跳过重复 reference collection | 复用当前 boundary/reference cache；旧 auto-round API 不接受 keyword 时回退原生收集 | 已实现 |
| `f2adfbdaa` `sequential_targets_per_subgraph="auto"` | 根据 node memory 自动分组 | 复用显式 grouping execution unit，但必须接入 target admission 和 Hessian budget | 计划中 |
| `8ea12bd63` sequential offloading | oneshot 按 subgraph 管理 offload/prefetch | 借鉴生命周期和 working set；不恢复全模型 CPU Hessian offload | 部分复用 |
| `68a83df8e` GPTQ explicit `actorder=None` | 保留显式 actorder 语义 | GPTQ adapter 透传配置，并在 admission 使用相同 Hessian 估算 | 已适配 |
| `118c11d01` GPTQ DDP global scale broadcast | 多卡同步 global scale | 当前 streaming 单 participant；未来通过 parallel adapter 接入 | 不支持 |
| `ebc4e7559` HIGGS | 独立生成 mixed precision config | `StreamingHiggsConfigProvider` 按 logical group 收集 bounded MSE，再全局 ILP | 已验证 |

observer 数值适配不能直接覆盖当前分支的全部实现，因为当前分支额外保留：

- `chunk_size` bounded search；
- optional hardware backend；
- streaming admission；
- 特化 materializer 的输入 dtype 和输出 dtype 边界。

正确做法是先把 upstream 的 error dtype/reduction、scale rounding 和 patience
语义移植到共享 observer，再增加 streaming/oneshot 对照测试。

## 13. checkpoint、恢复与发布

### 13.1 transaction 内容

一个 committed subgraph transaction 至少包含：

```text
transaction id
run fingerprint
target/group names
input boundary index
next boundary index（若存在）
output tensor records
quantized module ownership
boundary batch count
shape/dtype/size/checksum metadata
modifier contract fingerprint
```

Hessian、observer 和完整 activation 不属于 durable transaction。

### 13.2 提交协议

```text
write pending output tensor records
write pending next boundary
fsync records and metadata
atomic rename pending -> committed
update resumable frontier
delete consumed boundary
```

当前 direct writer 模式可以直接写 output staging；checkpoint progress 模式使用
`StreamingCheckpointWriter` 和 `SingleParticipantCoordinator`。两种模式共享
logical tensor naming、materializer output conversion 和 finalizer 语义。

### 13.3 恢复

- pending transaction：清理后重做当前 group；
- committed transaction：验证 metadata，恢复 frontier，跳过计算；
- committed 但 output 缺失或 checksum 不符：报损坏；
- stale external next boundary：在当前 group 重做前删除；
- manifest 不匹配：拒绝复用旧 transaction；
- finalizer 已完成且 fingerprint 匹配：直接返回 output。

manifest 至少覆盖 source checkpoint、config、normalized recipe、calibration
dataset、trace plan、target grouping、materializer、dtype policy、MTP
conversion 和 output format。

## 14. 特化模型接入规则

所有模型特化必须遵循以下分层：

```text
model factory/config
  -> shared tracing
  -> model-specific materializer/source
  -> model-specific target/group policy
  -> generic streaming pipeline
  -> generic writer/finalizer + scoped config patch
```

允许特化层声明：

- physical tensor 到 logical weight 的映射；
- source dependencies；
- expert slice；
- target policy；
- native MTP/extra-head ownership；
- output tensor naming；
- runtime buffer 保留规则。

禁止特化层直接修改：

- global boundary store 的提交语义；
- GPTQ Hessian 生命周期；
- writer 的 atomic commit；
- 所有模型共用的 modifier callback 顺序。

新增模型的 preflight 必须报告：

1. quantization target 是否全部被 execution plan 覆盖；
2. 一个 target/group 的最大 materialized bytes；
3. modifier peak bytes，尤其是 Hessian；
4. source 是否支持按 logical weight 读取；
5. 是否有 shared parameter 跨 owner；
6. MTP/extra state 是否 native、upstream 或 absent；
7. output materializer 是否能生成合法 checkpoint。

## 15. 测试与验收矩阵

### 15.1 GPTQ/OOM 核心测试

| 测试 | 验收 |
|---|---|
| 单 Linear Hessian | Hessian/count 与 oneshot 相同 |
| 多 Linear target | 前一个 target 结束后 Hessian 引用和 sample count 清零 |
| synthetic 384-expert MoE | 峰值随当前 expert/group 增长，不随总 expert 数线性累积 |
| DSV4.1 target selection | 无显式 targets 时自动收紧为 Linear，所有 scheme 覆盖 |
| admission failure | 在加载前拒绝，错误包含 materialized/Hessian/scratch/available |
| propagation | 下一 target 输入来自更新后的当前 target |
| reload | compressed checkpoint 可加载并执行有限 forward |

### 15.2 HIGGS

| 测试 | 验收 |
|---|---|
| multi-shard source | 不把完整 shard 读入长期对象 |
| fused group | fused members 只有一个 ILP decision |
| stacked expert | 一次只保留当前 physical/logical group |
| ignored tensor | `lm_head` 等 ignored tensor 不读取 |
| fingerprint | 参数变化产生不同 fingerprint |
| composition | provider config 可被 streaming RTN/GPTQ 消费 |
| specialized layout | 由 materializer 提供 dependency/shape，不在 provider 判断模型名 |

### 15.3 MTP

| 测试 | 验收 |
|---|---|
| no MTP | model tree、target 和 output 与原路径一致 |
| native MTP | wrapper 不重复挂载 |
| upstream MTP | meta construction 和 conversion 成功 |
| shared aliases | `embed_tokens`/`shared_head`/`rotary_emb` 不重复注册 |
| ignore | MTP extra state 不进入普通 quantizer |
| DSV4.1 no MTP | native materializer/target policy 不受影响 |
| MTP + GPTQ | MTP tensor 不进入 base Hessian |

### 15.4 QAD

| 测试 | 验收 |
|---|---|
| Quantization + QAD | recipe、cache、re-observation 和 propagation 成功 |
| GPTQ + QAD | adapter 不破坏 GPTQ Hessian cleanup |
| one batch | 明确拒绝并说明至少需要两个 batch |
| activation quantization | 明确拒绝 weight-only 之外的方案 |
| CPU cache | offload_device 非 CPU 时明确拒绝 |
| grouped target | cache 不跨 transaction 泄漏 |

### 15.5 资源回归

资源测试必须记录：

```text
baseline process RSS / device allocated
materialized weight bytes
modifier held bytes
modifier scratch bytes
Hessian bytes
boundary bytes
writer completion bytes
peak device allocated
```

至少覆盖：

- Qwen3 小模型；
- synthetic MoE；
- DeepSeek V4/V4.1 tiny checkpoint；
- 不同层数但相同 target width；
- 不同 source shard 总大小但相同当前 target；
- interrupted/resumed transaction。

不能只看进程最终 RSS，因为 allocator cache 可能延迟归还；必须同时看 admission
日志、device peak stats 和 target 完成后的对象释放。

## 16. 当前验证基线

上一阶段已经验证：

- HIGGS focused tests；
- QAD adapter tests：`10 passed, 2 skipped`；
- DSV4.1 streaming tests：`6 passed`；
- selected MTP tests；
- ruff；
- `git diff --check`；
- 两层 Qwen3 的 GPTQ/iMatrix 与 sequential oneshot 对照；
- DSV4.1 native path 不被通用 MTP wrapper 接管。

环境限制：

- upstream MTP 全量测试需要本地缺失的 `mocker` fixture；
- QAD upstream chain test 受本地 compressed-tensors Float8 promotion 错误影响。

文档变更的最小检查：

```bash
.venv/bin/ruff check docs/developer/streaming-ptq-implementation-design.md
git diff --check
git status --short
```

如果文档检查工具不接受 Markdown 文件，应改用仓库配置的 Markdown checker，
不能为了通过检查修改无关 Python 文件。

## 17. 实施顺序

### 已完成阶段

1. 修复 GPTQ Hessian 生命周期：target-local cleanup。
2. 增加 explicit target grouping。
3. 适配 AutoRound signature/reference 行为。
4. 实现 bounded HIGGS provider。
5. 实现 upstream MTP wrapper 基础兼容。
6. 扩展 QAD adapter 到 GPTQ + QAD 的基础组合。

### 下一阶段

1. **HIGGS 组合接入**：验证 HIGGS provider 生成的 config 直接进入
   streaming GPTQ/RTN，并把 fingerprint 纳入 run manifest；评估是否需要一个
   recipe-level provider hook。
2. **observer numerics parity**：移植 `b3faa86f3` 的 error dtype/reduction、
   scale rounding、patience 语义，并保留 bounded search。
3. **MTP 完整矩阵**：补 upstream MTP、多 MTP、native DSV4.1 extra state、
   MTP + GPTQ 的 ownership 和 reload 测试。
4. **QAD 完整矩阵**：补 numerical parity、grouped target、recovery 和资源曲线。
5. **自动 grouping**：实现基于 admission 的 `"auto"`，并将 group plan 指纹化。
6. **结构 modifier 协议**：在 writer/finalizer 支持 delete/rename/config patch
   后，重新评估 REAP 等结构修改型 modifier。

每个后续阶段必须保持以下不变量：

```text
处理第 N 个 target 时：
  不保留第 0..N-1 个 target 的实体权重；
  不保留第 0..N-1 个 target 的 Hessian/observer；
  不提前加载第 N+1 个 target，除非 prefetch 通过独立 admission；
  不把特化模型判断散落到主 pipeline；
  不用 CPU offload 掩盖 target 粒度错误。
```

任何实现如果破坏这些不变量，必须先更新资源模型和恢复协议，再进入代码。
