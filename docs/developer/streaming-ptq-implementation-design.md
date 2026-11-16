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

#### 6.3.1 动态 routed MoE 的 tracing 兼容

自动 GPTQ policy 将同一 `Expert` owner 下的 `w1/w2/w3` 作为一个
target，再由 pipeline 把连续 owner 合并成 bounded target group。这个边界
只有在共享 FX graph 中能看到每个 owner 的 `call_module` 时才成立。部分特化
MoE forward 会先从 tensor 计算 active expert id，再用 Python `for`/`zip`
遍历它；对 FX 而言，这会尝试迭代 `Proxy`，在 tracing 阶段直接失败。

实现采用两层兼容策略：

1. 通用 streaming 代码只做 module-tree owner discovery、target grouping、
   admission 和 modifier lifecycle，不读取模型名，也不改 GPTQ 数学；
2. 特化模型在自身 forward 或 calibration wrapper 内提供 FX 可见的绕环。DeepSeek
   V4.1 在 `is_fx_tracing()` 为真时改用固定 `range(num_experts)`，每个循环体
   通过 `torch.where(indices == expert_id)` 得到动态 token rows，再调用对应的
   `Expert` owner；正常执行继续使用 active-expert grouping。

该 tracing 分支必须保持三个不变量：路由权重和 token/slot 顺序与正常路径一致；
shared expert 仍在同一 forward 中执行；每个量化 expert owner 都出现在 target
plan。对应测试需要同时覆盖小模型 FX trace、正常路径输出一致性和 observer/
statistics 输入一致性。不能通过把整个 `experts` collection 设为 leaf 来绕过
问题，因为这会重新把所有 expert Hessian 留到一个 sequential boundary。

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

### 16.1 DSV4.1 GPTQ 端到端资源验证

2026-11-14 的 DSV4.1 多模态 GPTQ 重跑暴露并修复了两个与 Hessian
生命周期正交、但会遮蔽 admission 的通用 working-set 问题：

1. prefix graph 中的 nested `get_attr` 参数（例如
   `model.image_start`）不能把参数 owner `model` 当作完整 module root。
   `SubgraphWeightSession` 现在跳过 nested direct parameter 的 subtree working
   set，并用 parameter-level load plan 恢复它。否则 initial boundary 会错误地
   同时加载 43 个 decoder block。
2. traced prefix 可能跨 boundary 传递 `torch.dtype` 元数据，例如视觉路径中的
   `weight.dtype`。Boundary store 将该值作为不可变标量保存和恢复，不会把它
   当作 activation tensor。

修复后的 attempt `attempt-20261114T140935Z-6a5386e2` 在进入 GPTQ target 前的
initial boundary 峰值为 `4.88/103.1 GB`，prefix 只 materialize 3 个视觉权重
组，每组 259 个 tensor。第一个 bounded target 包含 8 个连续 expert owner，
admission 记录为：

```text
materialized_bytes=1,240,230,656
modifier_held_bytes=2,013,669,552
modifier_scratch_bytes=7,124,025,344
device_required_bytes=10,377,925,552
```

这证明 GPTQ Hessian 已按 owner group 限制在当前 execution unit 内，未恢复
CPU Hessian offload，也没有把整个 DSV4.1 block 作为一个 target。完整量化仍需
等待 1,980 个 bounded groups 和后续 validate/report 阶段完成。

### 16.2 2026-11-15：bounded group 执行时间回归

GPTQ attempt `attempt-20261114T142101Z-65487e63` 于北京时间
`2026-11-14 22:21:04` 启动；`2026-11-15 18:34:55` 检查时仍在运行，
已完成 193/1,980 个 group。不能将该状态记为完整端到端成功。此前的普通
expert group 约需 194–231 秒，包含 attention 的 group 更慢；完整量化按现状
需要数天，不能作为可接受的性能基线。

优先检查 tracing projection 的执行所有权。一个 shared sequential partition
包含多个 target 时，逐 target 从相同 partition 递归收集依赖，可能把先前
target 的 `call_module` 以及带副作用的 routing accumulation 复制到后续
projection。若 group 有 G 个 target，执行各 projection 可能产生 O(G²) 次
target 调用；重复的 attention 和 in-place accumulation 还可能影响结果及
GPTQ observation 次数。增大 group 之前必须消除这种重复。

修复和验收顺序：

1. 用串行 Linear、并行 expert 汇合及 in-place accumulation 的人工 FX graph
   复现；记录每个 target 的真实 forward 次数，不能只比较最终 tensor。
2. 对每个原始 partition 一次性保留所有 target 和后续所需输出，再切分成
   互不重叠的 target execution slices。每个原始 FX 运算只属于一个 slice，
   跨 slice 的值通过 boundary 引用传递；继续复用 shared `partition_graph`
   和 `trace_consumed_names`。
   DSV4.1 tracing 分支的 routing accumulation 还必须显式使用
   `output = output.index_add_(...)` 返回值，将累加顺序表达为 FX 数据依赖；
   否则拓扑分区可以提前执行 shared-expert merge。该调整只作用于 tracing
   兼容分支，并同时验证自然路由和全专家校准的输入与输出。
3. 校准和量化后传播各执行每个 target 一次；Hessian 生命周期仍按 bounded
   group 结束，不改变 GPTQ 数学，不引入 CPU Hessian offload。
4. 用 observer/Hessian sample count、调用次数、输出和 checkpoint 对照验证；
   覆盖 suffix/tuple、dead target、MTP 以及 DSV4.1 routing。
5. 回归通过后替换远端运行，在相同数据和 recipe 下测量真实 group 耗时与
   显存峰值，再决定是否需要按预算自动扩大 group。不能从减少 group 数量
   直接推导加速比，也不能将 boundary 形状启发式估算的 `num_tokens` 当成
   数据集实际 token 数。

#### 16.2.1 grouped working set 必须保留 parameter-level 精度

2026-11-15 复核 tiny native DSV4.1 + GPTQ 时发现，`plan_target()` 已正确将
`get_attr` 和模型声明的直接参数依赖解析为精确参数计划，但 `plan_group()`
把这些计划的 `module_names` 再传入 `plan_modules()`。直接参数计划记录的
owner 名称只用于安装、恢复和导出，不能解释为需要加载 owner 的全部子树。
例如 `layers.0.scale` 的 owner 为 `layers.0`，重新按模块计划会加载该层
所有 expert 和 attention 权重；root parameter 甚至可能扩大为整个模型。

实现方案是在 `SubgraphWeightSession` 中提供多个 subgraph 的统一规划入口：

1. 分别读取各 subgraph 的 `call_module` 工作集并取最小模块根集合。
2. 收集所有 `get_attr` 直接参数和各 target 声明的 runtime parameter；
   仅删除已经被完整模块根覆盖的参数，然后通过既有 `plan_parameters()`
   按 owner 合并、按 tensor identity 处理同 owner 别名。
3. adapter 继续逐 target 验证 runtime dependency，禁止隐式跨 quantized
   target 引用；将验证后的 module/parameter 依赖传给统一规划入口。
4. 单 target 和 group 共用上述实现，保留既有 runtime buffer、缺失状态、
   checkpoint source alias、host/device admission、安装和恢复协议。
   不拼接重复的 `TargetLoadPlan`，也不从 owner 名推测完整模块加载范围。

验收需同时覆盖 root/nested direct parameter、同 owner 多参数和别名、
重复 module dependency、runtime buffer、无关大型 child 保持 meta、实际
物化字节和退出后的 meta 恢复。native GPTQ + DSpark 测试记录每个 module
真实 calibration hook 次数以及输入 batch 对 Hessian sample count 的贡献。
MTP draft 可能增加 batch 维，不能把所有 `_num_samples` 硬编码为样本条数。

已于北京时间 2026-11-15 18:56 通过 dashboard cancel 停止
`attempt-20261115T104904Z-2637c021`，该轮尚在初始化；保留日志及工作目录，
修复通过回归后再用相同原始参数重跑。原 run metadata 中的 git SHA 是创建
run 时的版本，后续验证使用远端工作区增量补丁，必须单独记录代码版本。

#### 16.2.2 本轮回归与端到端运行准备

- 本地定向回归：267 passed、11 skipped；覆盖 tracing、weight session、
  loader、pipeline、pretrained、execution、modifier adapters、HIGGS、
  native DSV4.1。Ruff 与 `git diff --check` 通过。
- 新工作集回归在修复前 root/nested 两种 owner 均错误估算 263,376 bytes
  （当时 fixture 尚未加入 auxiliary module），而精确计划仅需 208 bytes。
  最终 fixture 增加重复声明的 auxiliary module 后，正确值为 288 bytes；
  安装时无关 256×256 child 保持 meta，参数别名、runtime buffer、导出与
  卸载均验证通过。
- native GPTQ 完整测试成功 finalize 5 层主干与 3 层 DSpark/MTP。
  与原生 eager 执行逐模块比较观测 shape 序列，再独立核对 GPTQ sample
  count。DSpark attention 的同一个 `wkv` 会分别处理 main/draft 输入，
  每个样本两次 hook 是模型本身的行为，不能作为重复 projection 判据。
- 远端同一测试集合：272 passed、6 failed。失败均为 AutoRound 的 reference
  output 末维 32 与预测末维 64 不一致。已在独立进程内载入远端 HEAD 的
  修改前 tracing，首个 AutoRound 失败同样复现，未修改磁盘代码。远端
  auto-round 0.15.1 / transformers 5.17.0，本地 auto-round 0.14.2 /
  transformers 5.19.0。该兼容问题单独保留，不能把当前远端集合记为全绿。
  GPTQ、HIGGS、native DSpark 回归均通过。

重跑前还发现存储约束：`/ssd2` 剩余约 262 GiB；源 checkpoint metadata 为
510,286,023,000 bytes。最初按物理 shape 再计算 W4 的 346 GiB 估算错误：
源 expert 已使用 packed FP4，例如 w1 为 `I8 [2304, 2560]`，对应逻辑
shape `[2304, 5120]`，不能在物理 shape 上再次减半。按逻辑 shape、原
WNa8 规则、FP32 channel scale 与保留 Engram 重新计算，产物约为
493,490,084,696 bytes（459.60 GiB），其中：

- routed W4 与 channel scale：279,198,302,208 bytes；
- 其余 W8 与 channel scale：7,411,637,248 bytes；
- 保留 Engram：202,758,032,400 bytes；
- 其余静态状态估算：4,122,112,840 bytes。

该值仍为布局估算，最终以 checkpoint index 为准。磁盘计划必须使用
materializer 的逻辑 shape 和实际输出布局，不能只参考源文件物理 shape。

本轮运行数据迁移至 `/ssd3/model-quality/runs/<run_id>`（迁移前可用约
485 GiB），保留 `/ssd2/model-quality/runs/<run_id>` 为符号链接，使 dashboard、
原参数和 attempt 关系继续可用。量化与原推理容器均已核实挂载 `/ssd2`、
`/ssd3`。迁移在无活动 job 时执行：先完整复制、核对文件和大小，再将原目录
改名保留于 `/ssd2/model-quality/retained`，最后切换符号链接；失败则恢复原路径。
不删除旧 attempt，work/output 仍处于同一实际文件系统，保持 copy-free
publication 约束。迁移后再归档取消轮次的 work，启动相同配置的新 attempt。

迁移已完成，核对 1,001 个文件、29,183,491,238 bytes，原目录保留在
`/ssd2/model-quality/retained/20261112T090344Z_a619a1629881-before-20261115-relocation`。
迁移后 `/ssd3` 剩余约 458 GiB。新轮次：

```text
run_id: 20261112T090344Z_a619a1629881
model_id: deepseek_v41_wNa8-c7c0b93d
attempt_id: attempt-20261115T110755Z-9a853928
quantize_job: job-20261115T110755Z-eb1ab395
inference_job: job-20261115T110755Z-14a3d0d2
report_job: job-20261115T110755Z-3fc3adb9
process_started: 2026-11-15 19:07:59 +08:00
parameters: 原配置，266 samples / max length 8192 / seed 42 / GPTQ WNa8
local_base: c174ca911
remote_base: 0ea24f39f50a692b0894d7e3c5914c517995c33b + 增量补丁
incremental_patch_sha256:
  79e10935f18654fcf714ff1688b6549d41e6c322246de4ff01be44ae105c6a85
tracing.py_sha256:
  3f95ca9753a4d2a825344f3306cbc12646cb84d0e1af6418dcf8ac5925b029a3
loading/session.py_sha256:
  0a589bc63fc57e0d50f1db12936d84384a9169523f64c15a9c1b8961f55051eb
deepseekv41/model.py_sha256:
  e3296b2a293923e7f68ccff2fd71004bc1d621b1496bccb1ef124b1318b2b32c
```

校正容量估算后，将当前 run 下约 27 GiB 的
`work-attempt-20261114T142101Z-65487e63-superseded` 历史工作目录移回
`/ssd2/model-quality/retained/<run_id>-historical-work/`，在 run 的原历史目录
位置保留符号链接。操作仅针对取消轮次，不移动正在执行的 work，不删除旧
日志或历史数据。迁移已完成，核对 656 个文件、28,680,188,144 bytes；
迁移后可用 518,859,063,296 bytes（约 483 GiB），对预计产物留约 23 GiB 余量。

后续 runtime-smoke 必须使用原 plan 的 `zhuang_xsgl_repro_20261102` 和
`/workspace/ds_v41/w4a8/run_server.sh`。本地 manager 默认 inference container
已被其他任务改为 `zhuang_glm53_p_deploytest`，且该容器缺少 repo；本轮通过
环境覆盖 `MQ_INFERENCE_WORKER_CONTAINER` 指向原容器，不修改其他 plan 的默认
容器配置。worker 使用该容器的 python310_torch29_cuda 环境，保持 `sglang`
命令可用。此准备不等于 runtime-smoke 已通过；需要等量化与 validate 完成。

#### 16.2.3 耗时对照与剩余验收

旧轮次完整日志的统计（2026-11-15 19:15 检查）：

| 类型 | 已完成 group | group 耗时中位数 | 范围 |
|---|---:|---:|---:|
| 仅 expert | 189 | 250.94 s | 194.07–431.00 s |
| 含 attention | 4 | 2,820.57 s | 1,713.54–11,966.86 s |

仅 expert 的 calibration 中位数为 50.42 s，modifier update 为 11.26 s，
写 checkpoint 约 0.48 s。含 attention 的 calibration 中位数为 1,294.37 s。
这些是旧实现的实际数据，不能沿用为新轮次 ETA。旧实现 8 个串行 target
共执行 36 次，新实现只执行 8 次，但 group 总耗时还包含 GPTQ update、
activation boundary I/O、量化后传播和权重加载，不能直接承诺 4.5 倍总加速。

新轮次于 `2026-11-15 19:15:15 +08:00` 进入 initial boundary 收集，
初始化包括数据处理、meta 模型、MoE owner discovery、FX tracing 和图分区。
首组与后续普通 group 应分别记录，避免用 attention 重组或初始化时间推算
所有 expert。剩余验收顺序：

1. 首个 bounded group admission、calibration、update、propagation、
   save/unload 全部成功，并记录峰值显存。
2. 至少三个连续普通 expert group 的完成时间、各阶段耗时和资源曲线。
3. 按剩余 attention/expert group 数与各自实测分布推算范围；单独预留 final
   static tensor、validate、runtime-smoke 和 report 的时间。
4. 全部 1,980 groups、最终 checkpoint validate、原推理容器 smoke 和 report
   完成后，才将 GPTQ 大模型 E2E 标记为通过。

当前只确认代码回归通过、存储迁移与新 attempt 启动，完整 GPTQ E2E 仍在运行。

`2026-11-15 19:25 +08:00` 首组已完成校准和 modifier update：

| 首组阶段 | 旧轮次 | 新轮次 |
|---|---:|---:|
| calibration | 987.10 s | 256.51 s |
| modifier update | 15.09 s | 16.82 s |
| activation propagation | 1,190.09 s | 390.29 s |
| group total | 2,230.53 s | 694.98 s |

该比较使用相同配置、相同首组 target 名单。校准阶段约快 3.85 倍，不能直接
当作整组或整个模型的加速比。新轮次 initial boundary 为 221.01 s，
峰值显存 4.88 GB；首组 calibration 峰值 11.71 GB，update 峰值 12.03 GB，
admission 计划为 10,377,925,552 bytes。首组完整完成于 19:30:46，约快
3.21 倍。当前未出现新的 OOM，完整端到端结果仍需继续观察。

#### 16.2.4 只读取 metadata 的 admission 不应复制 activation payload

首个普通 expert group（第 2 组）从旧轮次 431.00 s 降为 328.77 s，只有
约 1.31 倍加速；不能将首组的 3.21 倍推广到整个模型。该组新 calibration
为 70.10 s，update 为 11.68 s，propagation 为 201.51 s；从 start 到
admission 另有约 37.47 s。manifest 中共 42 个含 attention group、1,938
个普通 group，普通组才是总耗时的主要决定因素。

代码复核确认，admission 遍历 266 个 batch 时调用
`boundaries.get(group_start, batch_index)`，只为计算输入 shape 的 token
启发式和 tensor 字节数，但默认 CPU 读取会在 `_move()` 中 clone 整份
CPU activation。该复制没有数值用途，且在每个 group 重复发生。

修复复用现有 `BoundaryActivationStore.get(..., device="meta")`：

- in-memory store 将保存的 tensor 映射为仅含 shape/dtype 的 meta tensor，
  不复制 payload，也不暴露可写的原始 activation；
- disk store 保持现有恢复和错误检查协议；本轮不改变其序列化格式；
- `_boundary_token_count()` 和 `_boundary_nbytes()` 的结果保持一致，实际
  calibration/propagation 仍按原 device 读取真实数据；
- 不修改 GPTQ 算法、target grouping、统计量、recipe 或 recovery identity。

先增加 memory/disk meta 读取与无 clone 回归，在 grouped GPTQ / sequential
逐 tensor 等价测试中验证 admission 确实走 metadata 路径；再同步远端。
当前轮继续收集普通组基线，替换运行时保留该轮全部日志与分片。优化效果以
新轮同组实测为准，不能将 37.47 s 全部直接记为已消除的成本。

metadata 优化的本地回归为 227 passed、5 skipped，涵盖 boundary store、
pipeline、tracing、weight session、pretrained 与 native DSV4.1。memory/disk
meta 读取均验证 shape/dtype/alias 和无 `Tensor.clone()`；group 1/2 的 GPTQ
checkpoint 与 sequential oneshot 逐 tensor 一致，admission 的 meta 读取次数
也验证为每个 group 每个 batch 一次。

远端同组回归为 231 passed、1 failed；唯一失败为
`test_deepseek_v4_preserves_input_ids_and_collects_all_experts[False-False]`
的 reload logits NaN。该参数组独立复跑 3 个 case 全部通过，本轮没有修改
对应 DeepSeek-V4 模型/fixture；如再复现需单独定位，不能隐藏首次失败。
metadata store、grouped GPTQ 与 native DSV4.1/DSpark 的回归均通过。

优化前当前轮三个连续普通 group 实测为 328.77 / 320.02 / 295.11 s，
中位数 320.02 s；该基线仍太慢。若机械地将此普通组中位数和单个 attention
首组 694.98 s 外推到 1,938 + 42 个 group，总量约 180 小时（7.5 天）。
样本有限，attention 差异大，这既不是完工承诺，也不是 metadata 优化后的
ETA；它明确说明“首组快 3.21 倍”并不等于整个模型已达到可接受速度。

下一轮性能工作应按证据排序：

1. 验证 metadata 规划不再发生整份 CPU activation clone，记录同组 start
   到 admission 的耗时。
2. 分开测量 activation H2D、GPU 运算、D2H、CPU snapshot/dedup，而不是只
   用 propagation 总时长推断。评估按 FX liveness 只搬运当前 group 输入，
   对仅透传至后续层/DSpark 的值保留 CPU 缓存。任何实现都要保留 alias、
   in-place routing、CSA2 shared state、MTP context 的语义和快照隔离。
3. 再做预算驱动的自动 group sizing。先校准 GPTQ batch-compression scratch
   模型：第 2 组 admission 为 3,203,145,920 bytes，但实际 update 峰值达
   14,914,841,088 bytes（含 runtime、activation、prefetch 等）。不能据当前
   较小的静态 estimate 直接将固定 8 放大为 64/128，更不能退回 CPU Hessian
   offload。需要分别建模固定工作区、随 group 增长的 Hessian/权重、压缩 batch
   的临时副本、activation 和 prefetch，并在 admission 中保留边界。
4. 自动分组的实际 group 列表、预算策略与版本必须进入 manifest/fingerprint；
   恢复使用已记录的 group 计划，不能因空闲显存变化而静默改变量化数学。
   所有新策略继续与使用相同 group 计划的 sequential oneshot 对照。

#### 16.2.6 2026-11-15 EP 提速实施设计

20:45 +08:00 检查 `attempt-20261115T115518Z-8b92c0b3`：进程已经运行
约 50 分钟，完成 5/1,980 组，第 6 组正在传播，尚无新 OOM。首组为
632.94 s；随后四个普通专家组为 389.31 / 377.60 / 367.21 / 366.04 s，
中位数 372.405 s。普通组校准约 115 s、GPTQ update 约 11 s、传播约
232–256 s。start 到 admission 已降到约 1.1 s，但总速度仍不满足需求。
仅按这些有限样本外推仍约 8–9 天，不能据首组改善承诺全模型加速。
当时卡 0 使用约 23 GiB，其余七张 P800 空闲，主机可用内存约 1.6 TiB，
SSD3 可用约 482 GiB。后续每次设备测试前都需重新检查资源。

**交付范围与旧 EP-B 计划的关系。** 优先交付单进程、多设备、共享输入的
参考专家并行，执行后端标识为 `single-process-expert-parallel-v1`。这是真正的
专家所有权分片：每个专家的权重、observer 和 Hessian 只驻留在其 owner
device，输入 token 只在该专家上观测一次；不复制完整模型，不以数据并行
分片样本，也不使用 CPU Hessian offload。它不是
`transactional-streaming-ptq-plan.md` §3.7/E9 的多进程 EP-B 完成证明；
后者的进程组、故障广播、participant prepare/global commit、强杀与恢复
矩阵继续保留。单进程执行保留当前一个 OS lock、一个 writer 和一个事务参与者，
各设备工作完成并通过校验后才写出统一 checkpoint，不伪造多 rank 提交。

**通用接口与模型边界。**

1. `streaming_oneshot(..., expert_parallel_devices=[...])` 显式启用。
   设备列表必须包含主执行设备、无重复、使用同一可用 CUDA 后端。
   自动 GPTQ MoE target discovery 继续使用最小 expert owner，不识别模型名。
   默认每设备最多八个 owner；全局 group 上限随设备数增长，但逐设备 admission
   仍必须通过。显式 group size 保留调用方的选择。
2. placement 从 FX 可见的独立 expert owner 和模块树推导。标准
   `experts.<id>`、ModuleList/ModuleDict expert collection 都按稳定顺序分配；
   attention、router、shared expert、直接参数和跨层 runtime state 留在主设备。
   非整除专家数与空 owner slot 合法。不能解析为独立 expert 的 fused/共享状态
   在 preflight 明确拒绝跨设备分片，不能静默复制或猜测。
3. native DSV4.1 的 routing、pre-w2 route weight、all-experts、CSA2、
   Engram 和 DSpark/MTP 使用已有 FX 图和模型侧兼容层。EP executor 不包含
   DSV4.1 类型分支，不替换 MTP wrapper，也不改变模型输入输出协议。

**执行路径。** 将一个 group 的互不重叠 target slices 合成同一 FX DAG，
保留原始节点名、外部输入、consumed liveness 和输出边界。独立 expert
`call_module` 交给每设备一个串行 worker；主线程执行其余已就绪节点。
仅数据依赖满足的节点进入 ready queue，禁止把尚未就绪的专家结果堵在串行
队列头。专家输入先从协调设备搬到 owner，输出回协调设备；使用明确的
CUDA 同步边界保证异步执行及 tensor 生命周期正确。初版采用保守同步，
后续实测再优化 peer transfer，不预先承诺线性加速。

同一 expert 内部始终保持 eager 调用顺序和原生 GPTQ hook；不同 expert
允许并发。涉及 in-place/impure 的 FX 节点保持原执行次序，原图未表达的
跨 expert alias 依赖必须拒绝或先由模型 tracing adapter 显式表达。汇合输出
沿原图的 index_add/add 顺序执行，不用一次 all_reduce 改变数学。每个 batch
全部任务结束后，才切换校准/传播 hook 状态或复用输入。

**权重与预算。** 对现有精确 `SubgraphLoadPlan` 按 expert owner 分片，
保留 parameter-level planning，不能将直接参数 owner 重新扩成完整层。
逐设备统计物化权重、Hessian、observer、GPTQ batched scratch、最大 batch
activation 和双向传输工作区。所有设备 admission 在首次物化前完成；
任一设备不满足时失败并释放已取得的 reservation。初版关闭跨 group
prefetch，避免主设备 reservation 误覆盖其他设备。CPU boundary store
仍只有一份，组内每个 batch 只取一次，不给每个 worker 克隆整份历史激活。
加载、安装、observer 初始化、压缩、输出写入和卸载仍复用原有实现。

GPTQ 的多设备 batch key 已包含 execution device，因而可复用原生
`compress_module_list` 的数学和 Hessian pop/释放语义。首版 update 可以
按 owner 顺序压缩，性能指标必须分别报告前向和 update；任何并行 update
都需要独立验证 modifier 私有状态及 observer 生命周期，不能将不同专家的
Hessian reduce 到一起，也不能向各设备广播完整专家权重。

**身份、输出和失败。** manifest/fingerprint 必须包含后端版本、设备顺序、
placement 和实际 group plan。拓扑变化拒绝原地 checkpoint-progress 恢复。
专家参数仍用原逻辑名称写入一个标准 checkpoint，部署不依赖量化设备数。
任一 worker 抛错时等待已提交任务退出，保留第一个异常，不继续发布当前组；
所有设备的加载 context、线程池、reservation 和 hook 必须退出。恢复仍以
现有已提交 group 为边界，不尝试从半个专家集合恢复。

**验收顺序与完成标准。**

- CPU reference：人工 FX 并行分支、真实路由、不同宽度和非整除专家、
  空 token、in-place 汇合、跨 slice liveness、任务异常与卸载；
  用同步 barrier 证明独立专家同时进入 forward，避免把串行多设备叫作并行。
- GPTQ：相同 target group 和输入下对照单设备/oneshot，检查每模块输入、
  Hessian、sample count、量化权重/qparams、清理和标准 checkpoint 重载。
  未经过 scope 适配的默认 distributed process group 继续拒绝。
- native tiny DSV4.1：主干、CSA2/Engram 和 DSpark/MTP 完整路径，
  同时覆盖 routed 与 all-experts，不能只测标准 Transformers MoE。
- 目标设备：先在空闲卡上运行 2/4 卡 tiny E2E 和资源失败用例，
  再决定何时取消当前单卡基线。原 run 大模型用原数据和 recipe，
  只增加 EP 参数；记录每设备峰值、同层普通组吞吐和包含 attention 的组。
- GPTQ 全部量化、validate、原推理容器 smoke、report 完成前保持 E2E 未完成。
  HIGGS 组合必须单独运行，不能用 GPTQ 或 tiny 的通过代替。

**2026-11-15 21:14 实施与验收记录。** 参考 EP 已接入 pretrained API、
DSV4.1 示例、精确加载、逐设备 admission、FX DAG 调度和原 writer。placement
按 FX target 顺序在全部 routed expert owner 间稳定轮转，不能在每个小
collection 重新从 owner 0 开始导致非整除模型的负载集中。GPTQ 压缩继续
使用原实现，按设备兼容 batch 顺序执行；目前未宣称 update 自身并发。

- 本地完整相关集合：193 passed、7 skipped；随后新增 native checkpoint
  parity 与 admission 回归 2 passed。GPU 用例在本地跳过。
- 目标 P800 的最终定向集合：22 passed。包括实际 2/4 卡 GPTQ Hessian、
  sample count、权重/qparams 对照，逐设备安装与完整 meta 卸载，异常退出，
  预算不足的 reservation 回收，native 5 层主干与 3 层 DSpark/MTP。
- 同 group=32 的 native 单卡与 4 卡完整 checkpoint 逐 tensor 对照通过，
  包含 Engram 与 MTP；FP32/FP16/BF16 使用 rtol=1e-6、atol=1e-7，
  packed/整数/保留低精度状态逐字节一致，标准 streaming reload logits
  有限且调用后权重全部恢复 meta。它不是原大模型 runtime-smoke 的替代。
- 首次设备集合有一个测试错误：被 ignore 的 shared expert 后代未真正
  用正则排除，未校准的空 qparam buffer 被错误拿来比较。已修正 ignore
  为后代范围，Hessian、计数及参与量化的专家比较不放宽；最终集合全通过。

原单卡轮次于 21:14:12 通过 API cancel，停止前完成 10/1,980 组，最后
一组 358.87 s。确认所属 PID 1919006 退出后，将 work 原地保留为
`work-attempt-20261115T115518Z-8b92c0b3-single-device-baseline`，没有删除
日志或分片。21:15 检查八张卡均无活动进程。

全量 8 卡计划的保守预估为 240 GPU-hours（这是资源预留估计，不是实测
ETA）。启动时被 web ledger 的 80 GPU-hours 容量拒绝；来源是
`ci/model_quality/web/plans.py` 的 `LaunchPolicy.gpu_hour_capacity` /
`MODEL_QUALITY_GPU_HOUR_CAPACITY`，不能降低全量估算来规避。第一次预览
在 ledger admission 前已经被 consume；重试返回 already launched，
但该 run 下没有 quantize job，不能将其记为全量运行已启动。

在该限制内先执行**有硬时限的性能验证**：使用原始 266 samples、8192
长度和 WNa8 recipe，预留 4.2 GPU-hours，八卡 quantize 窗口最多 1,800 秒，
仅量化/验证，不自动启动推理或上传。独立 watchdog 在窗口结束时，只终止
argv 同时匹配该新 run 和示例脚本的量化进程；先 TERM、再 KILL，并保留
日志与当前分片。该窗口不是完整大模型 E2E，超时退出应归类为主动预算停止，
不能归类为算法失败。窗口预算和 guard 状态写入 run 目录。全量运行继续
受 80 GPU-hours 约束，需要用该窗口实测重新估算，必要时由用户调整容量。

性能窗口已通过正常 preview/start API 启动，21:24 检查为 RUNNING，
实际 script PID 1957189，仍在 meta 模型、MoE owner discovery 与 tracing
初始化；没有完成 group，因此尚无八卡吞吐或 ETA。部署快照和代码指纹随
run 保留，manager/job 中的 `git_sha=a619...` 仍是 manager 的旧修订标识，
不能用它替代实际执行文件的指纹。

```text
run_id: 20261115T130907Z_dsv41_gptq_ep8
model_id: deepseek_v41_wNa8-15dc114c
attempt_id: attempt-20261115T132043Z-589901c1
quantize_job: job-20261115T132127Z-190dbad9
report_job: job-20261115T132127Z-58416c11
plan_hash: 870c0e28ea67b38e694a941a867a70cb36331c3790d7c911873adee4e40e6bf0
script_started: 2026-11-15 21:21:30 +08:00
guard_armed: 2026-11-15 21:20:45 +08:00
guard_deadline: 2026-11-15 21:50:45 +08:00
actual_root: /ssd3/model-quality/runs/20261115T130907Z_dsv41_gptq_ep8
dashboard_root: /ssd2/model-quality/runs/20261115T130907Z_dsv41_gptq_ep8
free_bytes_before_start: 515970236416
parallel/expert.py_sha256:
  7300c00d350c610867ea818dba2fa1313fd3c0869b6f0b16457dba9d723dd816
pipeline.py_sha256:
  8d60e3fa4e3e962e48e76ec29cb34faf77db59c6e1e5c674b6d0216135429472
```

**2026-11-15 用户授权提高预算后的执行约束与正确性结论。**

用户明确允许提高上限。部署配置的 `MQ_GPU_HOUR_CAPACITY` 已设为 240，
通过 `scripts/model-quality-web-remote restart` / `up` 使服务的
`MODEL_QUALITY_GPU_HOUR_CAPACITY=240` 生效；不修改代码中的全局默认值。
原 DSV4.1 推理容器 `zhuang_xsgl_repro_20261102` 加入 allowlist，同时保留
既有 GLM 容器配置。启动管理命令时显式指定 DSV4.1 inference worker，
已运行的量化进程不受 Web 服务重启影响。

完整计划使用 240 GPU-hours 的资源预留和请求上限、8 个设备、原始
266 samples / 8192 sequence length / seed 42 与 GPTQ WNa8 recipe，
包含 quantize、validate、原推理容器 runtime-smoke 与 report。
240 GPU-hours 是 admission 的预留估计，除以八卡为 30 小时，不能据此
宣布量化会在 30 小时内完成。性能窗口的 4.2 GPU-hours reservation 必须
先释放，随后才申请完整 240；不能伪造较低估计或将 30 分钟窗口静默变为
无限运行。原性能窗口没有启用 checkpoint-progress，不能把未提交的部分
分片拼入新完整运行；使用新 run，并保留窗口日志、工作目录与代码快照。

当前正确性证据和边界如下：

| 检查层次 | 已有证据 | 能支持的结论 |
| --- | --- | --- |
| GPTQ 算法 | 复用原生 `GPTQModifier`、`sequential_epoch_end`、`compress_module_list`，没有实现另一套 Hessian 归约/更新公式 | 资源管理与执行设备调整不主动替换 GPTQ 数学 |
| streaming 与 oneshot | 本地回归包含 `test_streaming_gptq_matches_sequential_oneshot[1/2]` 及 grouped-target oneshot 对照 | 已测 toy 模型和相同 sequential 计划下结果对齐 |
| EP 统计与压缩 | CPU 和真实 P800 2/4 卡逐模块 Hessian、sample count、权重/qparams 对照，以及更新后统计释放 | 已测独立 expert 的输入归属、统计设备和释放顺序正确 |
| 特化 DSV4.1 | native tiny 主干、Engram、CSA2、3 层 DSpark/MTP，固定 group=32 的单卡/四卡 checkpoint 对照与重载 logits finite | 已测特化路径和完整 checkpoint 在该输入与分组下保持一致 |
| 原大模型 | 八卡窗口已完成首组校准与 GPTQ update，尚未完成完整 checkpoint 和原推理容器验证 | 仍属于运行中，不能宣布全量正确或质量达标 |

不得把“未发生 OOM”“测试通过”“可以重载且 logits 有限”表述为模型质量
保证。最终还需要完整 tensor/index/config 校验、原推理容器 smoke，以及
可复现的固定输入输出或质量评估结果；runtime-smoke 通过本身也不能证明
质量无下降。当前完整计划未配置独立 evaluate 数据集，因此不得把其 report
称为精度基准通过。将来补充质量评估时记录数据集、seed、推理参数和参考
checkpoint，不能用先前 HIGGS 或 tiny 结果替代本次 GPTQ 产物的证据。

EP 默认将 global group 从 8 扩为 64，会改变 sequential propagation 的
更新边界；现有数值对照固定了相同 group 计划。不得承诺新的 group=64
产物与旧 group=8 或任意 oneshot 默认分组逐位一致。模型中含不可见副作用、
跨 owner alias 或未拆分 fused experts 时，也不能据现有测试泛化为已支持；
必须保持显式 preflight 拒绝或补充 tracing/ownership 适配与对照。

21:39 +08:00 性能窗口状态：首组 `[001/248]` 已完成 calibration 199.95 s、
modifier update 80.44 s，正在量化后 activation propagation，没有新 OOM。
这仍不足以估计普通组稳定吞吐。该组包含两个 Engram 投影、attention、
shared expert 与 60 个 routed experts，不能作为纯 expert 组的稳定样本。

**21:43 +08:00 完整八卡运行已启动。** 通过 API 主动取消性能窗口，释放
`res-20261115T132127Z-4f422fc0` 的 4.2 GPU-hours。窗口量化进程已退出，
watchdog 记录 `PROCESS_FINISHED`，没有等待 30 分钟上限，也没有发生新
算法失败；API cancel reason 和 `full-e2e-transition.json` 记录本次切换原因。
窗口没有完成首组 propagation，不能将 calibration/update 的分段计时相加
冒充一个完整 group 的耗时。未删除窗口工作目录、日志或部署快照。

确认服务 capacity=240、available=240 后，通过正常 preview/start API
启动完整任务，240 GPU-hours 预留成功。21:44:36 +08:00 检查 quantize job
为 RUNNING，实际脚本 PID 1986907；inference 和 report 等待量化完成。
该轮仍在初始化，没有本轮完整 group 耗时或全量正确性结论。

```text
run_id: 20261115T134000Z_dsv41_gptq_ep8_full
model_id: deepseek_v41_wNa8-15dc114c
attempt_id: attempt-20261115T133930Z-3e35e33e
launched_at: 2026-11-15T13:43:52.821108+00:00
quantize_job: job-20261115T134352Z-eac088fd
inference_job: job-20261115T134352Z-daae0cfd
report_job: job-20261115T134352Z-231e1402
reservation_id: res-20261115T134352Z-9fabedee
reserved_gpu_hours: 240
gpu_count: 8
plan_hash: 27a3a495cd66f97064e54e8ac387a1e2aabdf044863f9b940333d917f594898d
dashboard_root: /ssd2/model-quality/runs/20261115T134000Z_dsv41_gptq_ep8_full
actual_root: /ssd3/model-quality/runs/20261115T134000Z_dsv41_gptq_ep8_full
free_bytes_before_launch: 515472384000
expected_output_bytes: 493490084696
deployment_snapshot: codex-ep-deployment.tar
deployment_snapshot_sha256:
  c8e18a81f6a9ca4402dff996bb7a4d4477fb77400380edaaf8d95df934c14808
```

新运行的两个生产文件指纹与性能窗口相同。实际运行代码以快照和文件指纹
为准，manager 保留的旧 `git_sha=a619...` 不是本次 EP 实现的完整代码身份。
work/output 继续在同一实际文件系统，以免发布时再复制约 459.60 GiB；
上述容量仅是启动时快照，后续继续检查实际剩余空间。

**21:51 +08:00 Web 启动故障与修复设计。** 本地 health 请求 connection reset，
容器内相同端口 connection refused，Web 进程已退出，启动日志最后一个
异常为 `OSError: [Errno 98] Address already in use`。量化 PID 1986907
持续运行。本次故障属于 Web 服务生命周期，不是 GPTQ 失败。21:52 通过
正常 `up` 恢复，health 和 UI 均返回 HTTP 200。

启动脚本有两个确定的竞争窗口：向旧进程发送 TERM 后立即启动新进程，
没有等待旧进程释放监听端口；多个 up/restart 也没有共同的生命周期锁。
修复保持在 `scripts/model-quality-web-remote`：start/stop 共同使用
runtime 目录中的远端文件锁，锁内重新检查健康状态，停止仅匹配当前
host/port 的 Web 命令行，等待旧进程退出后才启动；超时明确失败，不再
叠加另一个监听者。新服务必须关闭继承的锁描述符，且锁内完成启动健康
确认，避免下一个 up 将尚在启动的服务误判为旧进程后终止。PID 文件只作
记录，不作为未经命令行核对的 kill 目标。验证覆盖延迟退出、并发启动和
过期 PID 文件，再执行真实 Web restart/up、UI/health/capacity 检查；保持
量化进程 PID、attempt 与 240 GPU-hours reservation 不变。

修复已部署。隔离临时端口的三项回归全部通过（3 passed，2.97 s），包括
旧监听者收到 TERM 后延迟 0.7 s 才退出、两个并发启动只创建一个监听者、
过期 PID 文件指向无关存活进程时 start/stop 均不终止该进程；同时验证
daemon 没有继续持有生命周期锁。`bash -n`、Ruff 和 diff whitespace
检查通过。真实 restart 后，再并发执行两次正常 `up`，两次均成功。
22:01:05 +08:00 检查本地 health/UI 均为 HTTP 200，仅有一个 Web
监听进程 PID 1988275；原量化 PID 1986907 已运行 17:10，仍为相同
attempt 的 RUNNING，capacity 和 reservation 均为 240 GPU-hours。

**22:30 +08:00 八卡吞吐与 ETA 复核。** 完整运行已经持续 46.74 分钟，
完成 3/248 组，第 4 组正在执行，没有新 OOM 或进程重启。第一组包含
Engram/attention/shared expert 与 60 个 routed experts，851.04 s；
随后两个纯 expert 组各有 64 个 owner，分别为 563.07 / 543.52 s，
中位数 553.30 s，约 9 分 13 秒。按 expert 数量折算，与此前每八个
owner 为一组、普通组中位 372.405 s 的单卡基线相比，吞吐约提高
`8 * 372.405 / 553.30 = 5.38` 倍。不同分组下这只是吞吐比较，不是
量化结果数值等价证明；两个普通组也不足以代表所有主干和 MTP 层。

两个普通组的平均阶段时间为 calibration 152.25 s、GPTQ update
91.21 s、activation propagation 280.08 s，其余约 29.77 s；分别约占
27.5%、16.5%、50.6%、5.4%。资源规划和权重 I/O 已不是主要阶段。
代码中仍有 expert 输入事件等待、逐 expert 的设备/主设备同步，以及
顺序执行的 GPTQ device batch。后续优先分解传播与同步开销，并验证
可并发的 GPTQ update，不能把移除同步当作无条件正确的优化。

22:24:01–22:24:16 的八次只读设备采样处于第 3 组 propagation：
各卡平均 utilization 为 0–0.625%，单次最大 1%；进程同期 CPU 使用约
18.23 个核心。该短窗口提示运行效率仍有问题，不能将它表述为整个
量化过程的平均利用率，也不能仅凭此确定每个等待点的根因。

实际 manifest 共 42 个含 attention 的组。以普通组中位数作为底数，
对余下 attention 组增加首组与普通组的差额，并扣除当前组已运行时间，
22:30 的点估计为剩余 41.00 小时、2026-11-17 15:31 +08:00 完成量化。
对外采用剩余 40–45 小时、11 月 17 日下午至晚间的粗估区间；不包含
后续 validate/runtime-smoke/report，且后层/DSpark 开销仍有不确定性。
该估计意味着完整任务很可能超过 240 GPU-hours 的原预留估计。

同时核实出启动配置遗漏：实际 run 的 `resources` 只有
`estimated_gpu_hours=240` 和 `gpu_count=8`，没有 `timeout_hours`。
远端 `ci/model_quality/executor.py:495` 因而采用默认 24 小时，
`_run_logged` 在 `process.wait(timeout=...)` 超时后终止整个量化进程组。
本轮量化始于 2026-11-15 21:43:55.797657 +08:00，硬超时约为
2026-11-16 21:43:56 +08:00，早于上述理论完成时间。此前只提高了准入
预算，漏掉了 stage timeout 对齐；不能把当前配置描述为能直接长跑到完成。
script-first 的 `_synthesize_model` 也没有透传请求中的 `timeout_hours`，
后续需要同时修复计划生成、显式 timeout 配置与测试。已经启动的
`process.wait` 持有既定 deadline，仅修改 YAML 不会热更新该 deadline，
不得伪称现有运行超时已解除。本次速度检查继续保留原进程采集基线。

原始证据保存在完整 run 根目录：
`ep-speed-device-sample-20261115T1423Z.json`、
`ep-speed-stage-summary-20261115T1423Z.json` 和
`ep-speed-latest-20261115.json`。

#### 16.2.7 激活传播优化：模型路由与通用执行路径

用户要求检查 `deepseekv41/model.py`。当前静态检查发现：eager MoE 已采用
稳定排序的分段路由，但 `_forward_traceable` 仍逐 expert 调用
`torch.where(indices == expert_id)`；实际 streaming 执行使用后者产生的 FX
图。因此 eager 的既有优化没有进入 streaming 路径。另一个确定的重复工作是
EP executor 仅缓存拼接后的 FX graph，每个 batch 仍重新构造依赖、alias、
副作用顺序和用户计数。边界 store 的 CPU snapshot、内容去重与 H2D/D2H
也必须单独计时，不能预先把全部传播耗时归因于模型 GEMM 或某一同步。

实施与验收顺序：

1. 在目标 P800 上构造与普通组相同的 64 experts / 8 devices、5120 hidden /
   2304 intermediate、8192 tokens、384-way / top-6 routing 的可复现实验。
   仅物化当前 expert group，分别记录 boundary get、FX/EP forward 和
   boundary commit；保留逐 expert where 的原始参考路由。微基准与当前
   大任务共存时必须注明资源竞争，不能将其直接当成完整 E2E 提速倍数。
2. 路由优化使用稳定排序后的 flat route rows、相应 pre-w2 weights 和
   Python 整数分段 offset，一次生成共享路由元数据。每个 expert 仍是独立
   FX `call_module`，slice 保留 row/slot 顺序，不把 expert 权重融合成大
   target，不更改原顺序的 FP32 index_add。trace/meta 路径必须可执行；
   grouped metadata 经 CPU boundary 存储后不能因 offset 被重新放到 GPU
   而恢复逐 expert 的 `.item()` 同步。优先将可复用部分放到通用 MoE helper，
   特化模型只负责调用适配，不给 streaming pipeline 添加 DSV4.1 分支。
3. EP 静态依赖、alias 和副作用顺序可按 graph/group 缓存，batch runtime
   只保留独立的 values、unmet 与 remaining-users。不得缓存 batch Tensor、
   Future 或输入 CUDA event，不能让前一批激活逃逸到下一批。移除或替换
   CUDA synchronize 需要单独的设备事件与 tensor 生命周期测试，不能与
   路由优化混在一起靠“结果看起来正常”验收。
4. 用 FP32/BF16、文本/图像 routing bias、空 expert、all-experts 和原生
   DSpark/MTP 验证输出、逐 Linear 输入、GPTQ Hessian/sample count、
   qparams、checkpoint 与 reload。覆盖普通 FX 和 HF meta tracer、
   多 group 边界及实际 2/4 卡。只有数值和生命周期验证通过，才采集目标
   设备新旧路径的同输入耗时，最后再安排完整长跑。

正在运行的旧进程用于保留基线。新文件不会热更新旧进程已导入的函数；
部署和完整重跑需要同时对齐 stage timeout，不能把改完源码描述为当前
任务已经获得提速，也不能继续忽略 §16.2.6 发现的 24 小时硬超时。

边界存储的进一步设计（依据首轮微基准，先记录再实施）：

- 2026-11-15 P800 首轮 microbenchmark 的旧路径每 batch 平均 get
  36–51 ms、forward 49–54 ms、commit 257–258 ms；commit 约占总时间
  74%。复核发现首版工具未像真实 adapter 一样对全部 slices 执行
  `trace_consumed_names`，保留了已经失活的内部输出；这一数字仅作为定位
  线索，不能用作性能验收或外推 E2E。修正版先计算全图 liveness，再选择
  中间的普通 64 expert group（默认 expert 64–127），使用前一组真实输出
  作为输入，比较同输入的新旧路由和完整/按需 boundary 两种读取模式。
  实验不含完整 attention/Engram，且与旧大任务共用设备，需注明竞争。
- 当前 store.get 会把整个存活 boundary 搬上设备，而 FX group 只读取其中
  一部分；executor 透传的未使用输入随后又被 snapshot 到 CPU。新的通用
  `get_for_execution` 按当前 group 的外部 FX 输入集合读取。In-memory
  store 对未参与执行的值保留私有 CPU snapshot，不向模型暴露这些对象。
  Disk store 保留完整读取的保守兼容实现。
- 如果未使用字段与活动字段共享同一 Tensor 对象（包括嵌套容器和内容
  dedup 已形成的 alias），它也必须加入活动集合，共用本批 move memo。
  这样对活动输入的原地修改仍能通过别名反映到下一 boundary。不同 view
  的原有独立 snapshot 语义不变，不推断未经验证的 storage alias。
- propagation commit 显式接收 consumed names；先 snapshot 执行结果，
  再把源 boundary 中未消耗、未被结果覆盖的私有 CPU carry 直接交接给
  目标 boundary。源/目标之间短暂共享只发生在 store 内部；读接口继续
  clone/move，不允许模型修改私有 snapshot。新结果 snapshot 或预算
  申请失败时，源 boundary 和既有目标仍完整存在。
  启用内容 dedup 时，私有 carry 同时作为只读候选参与新结果去重，保留
  新结果与旧 carry 内容相等时的既有共享语义，避免改变后续 alias 闭包。
- host admission 保守地预留完整目标 boundary 的字节，包括交接 carry；
  目标发布后再释放源 reservation。此阶段不扩大 host budget，不把
  “减少物理复制”当成放宽 admission 的理由，也不添加 Hessian offload。
  对无 host budget 的场景同样保持 source-before-destination 原子顺序。
- 验收必须覆盖活动输入与闲置 carry 的嵌套 alias、in-place mutation、
  多 batch、多 group、覆盖已有目标、失败 rollback、reservation 回收，
  并用同一微基准比较只改 routing、只改 boundary、合并优化三个路径。

本轮已实现的文件与职责：

| 文件 | 实现 |
| --- | --- |
| `modeling/moe/routing.py` | 通用稳定分段路由、Python offsets、meta 支持 |
| `modeling/deepseekv41/model.py` | trace 路径调用分段路由，每 expert 保持独立 FX leaf |
| `streaming/activations/store.py` | 按输入集合读取、传递性 alias 闭包、私有 CPU carry 交接、原子 snapshot 与预算生命周期 |
| `streaming/pipeline.py` | 根据各 slice 输出/输入计算 group 外部输入与 consumed 集合，校准和传播统一使用按需读取 |
| `streaming/parallel/expert.py` | 每 group 缓存 FX 静态依赖、alias 写入顺序、ready/unmet/user 模板，每 batch 复制计数 |
| `tools/benchmark_dsv41_streaming_propagation.py` | 保留旧 where 路由，计算全图 liveness，四路径同输入对照与分段 wall time |

边界优化是通用 storage/execution 能力，没有按模型类名分支。Disk store
通过默认完整读取与合并提交保持兼容；本轮性能收益主要针对 in-memory
CPU boundary。没有修改单 expert 的数学运算、权重更新顺序、pre-w2
route weights、FP32 汇合次序，也没有修改 EP CUDA synchronize 协议。
eager 路径既有的活跃 expert 跳过策略保持原样。GPTQ 更新与传播仍是
两个独立阶段，保持“更新后重新前向”的既有算法语义。

2026-11-15 目标设备实测（修正版工具）：

- 环境：8 × P800 OAM 96 GiB，同一量化容器，BF16；384 experts /
  top-6、hidden 5120、intermediate 2304、8192 tokens；测 expert
  64–127，前 64 expert 先生成输入 boundary；warmup 1、实测 5 batches。
- 每次对照的四条路径共享相同参数与输入；检查 accumulator 和传递 carry
  的 `rtol=0, atol=0` 一致性。计时包含 get、FX/EP forward、commit，
  不包含模型构造、prefix 执行、输入 boundary 初始化及额外正确性读取。
- 所有路径共用已缓存的 EP 静态 schedule，因此下面的旧路由/完整搬运
  对照没有单独计入 schedule 缓存相对旧实现的收益。

| 模拟闲置 carry | 路由 / boundary | get (ms) | forward (ms) | commit (ms) | 合计 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 128 MiB | 旧 where / 完整搬运 | 73.35 | 34.99 | 274.93 | 383.27 |
| 128 MiB | 旧 where / 按需读取 | 39.36 | 34.50 | 141.92 | 215.78 |
| 128 MiB | 分段路由 / 完整搬运 | 74.01 | 28.64 | 281.42 | 384.08 |
| 128 MiB | 分段路由 / 按需读取 | 44.73 | 28.60 | 142.07 | 215.40 |
| 1 GiB | 旧 where / 完整搬运 | 271.54 | 49.37 | 860.87 | 1181.78 |
| 1 GiB | 旧 where / 按需读取 | 41.07 | 58.67 | 151.53 | 251.27 |
| 1 GiB | 分段路由 / 完整搬运 | 259.48 | 136.53 | 882.37 | 1278.38 |
| 1 GiB | 分段路由 / 按需读取 | 43.00 | 38.45 | 147.37 | 228.81 |

最终候选的 128 MiB 同输入传播合计约快 **1.78×**，1 GiB 约快 **5.16×**。
前一轮候选曾测得 1.76× / 4.80×，其原始记录也保留。主要收益
来自避免闲置 carry 的 H2D/D2H 和重复 snapshot；路由优化收益较小。
1 GiB 测试的 forward 有明显抖动，分段路由/完整搬运出现离群耗时，
仍需在设备独占或完整任务中重新采样，不能把单轮差异归因为路由退化。
这些是 MoE 传播微基准结果，**不是整模型量化加速倍数**：旧大任务仍在
同机运行，attention、Engram、图像前处理、GPTQ update、checkpoint IO
均未包含在这张表中，不能据此承诺新的完成时间。

可复现命令（远端 repo，已设置正确的 Python 环境和 `PYTHONPATH`）：

```bash
python tools/benchmark_dsv41_streaming_propagation.py \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --iterations 5 --carry-mib 128 \
  --output /tmp/dsv41-propagation-final-128m.json
python tools/benchmark_dsv41_streaming_propagation.py \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --iterations 5 --carry-mib 1024 \
  --output /tmp/dsv41-propagation-final-1g.json
```

正确性验证记录：

- 本地最终联合集合 **261 passed / 13 skipped**，覆盖通用 route、边界存储、
  EP executor、subgraph pipeline、native DSV4.1 和 modifier 兼容。包含 FP32/BF16、
  图像路由、空 expert、all-experts、普通 FX 与 HF meta trace、
  统计量释放，以及 GPTQ/RTN/SmoothQuant/AWQ 与 oneshot 的既有对照。
- P800 定向集合 **36 passed**：包括真实 CUDA facade 的 2/4 卡 GPTQ、
  owner 装载卸载、CPU carry 生命周期及原生 DSpark/MTP。
- 随后加强原生 checkpoint 对照：单卡参考使用旧逐 expert where 和完整
  boundary 读取，四卡候选使用新路由和按需读取；比较每模块 Hessian
  （`rtol=1e-6, atol=1e-7`）、sample count（精确相等）、checkpoint 中
  浮点参数/qparams（同上容差）和整数权重（按字节精确相等）。
  包含 Engram、3 层 DSpark/MTP，重载 logits 有限且参数恢复 meta。
  加强后的单项对照在本地和 P800 均 **1 passed**；不能把它描述为真实
  500 GB checkpoint 的完整验收。
- 其中 tracing、HIGGS provider 与 modifier adapter 兼容集合为
  **44 passed / 7 skipped**，覆盖 QAD/AutoSmooth/AutoRound 的既有回归；
  GPU 依赖项目的跳过不代表完成相应大模型 HIGGS E2E。
- 最终补齐 carry 与新结果之间的内容去重后，P800 聚焦回归
  **10 passed**，再次覆盖原生新旧路径数值、alias、rollback 与预算交接。
- 静态检查通过。本轮候选已经同步到远端供新进程使用，未重启、取消或
  热修改旧量化进程。完整量化新速度和完成时间仍需新 attempt 测量，
  启动时必须同时解决上文所述 stage timeout 配置。

归档位置为当前 run 根目录的 `activation-propagation-20261115/`，
实际绝对路径：

```text
/ssd3/model-quality/runs/20261115T134000Z_dsv41_gptq_ep8_full/activation-propagation-20261115/
```

保留首轮有 liveness 缺陷的工具结果、修正后的两轮 128 MiB / 1 GiB
原始 JSON 和日志、P800 测试日志、旧实现 tar，以及最终源码/测试/文档
快照 `candidate.tar`；首轮缺陷不能在后续分析中被遗忘。
最终生产文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `modeling/moe/routing.py` | `21d2a9a1b32539ec797b30fc749b4f2e8049dca80732695ac2122f1f9266e409` |
| `modeling/deepseekv41/model.py` | `05c46d7ad3174aa80489fced541a75f2f03cb2565bef1b6a3400a6226985fc94` |
| `streaming/activations/store.py` | `597a49d8e20700ceb0b6af045fef66ab7ee68f89472731ad94c2278b5e8ee9be` |
| `streaming/parallel/expert.py` | `56d3dbd6f95cd27a523f48a6ed9ee5fdc57658b58ec3b15da2386888da34d1b6` |
| `streaming/pipeline.py` | `a525ca34ce6f65e13a32fc0ed5189a742aa0e658b004ab68421f5d286453ac31` |
| `tools/benchmark_dsv41_streaming_propagation.py` | `f1e348c65bd15d79dafd7c49197294575d2e60a2416b376f7ae2d8d0efef82e6` |

#### 16.2.8 DSV4.1 CED、CSA2 与 SWA Bounded Replay 核对

2026-11-16 读取了本地技术报告
`/Users/ang/models/DeepSeek-V4.1-Flash/DeepSeek_V41_Tech_Report.pdf`，
并逐页对照模型目录中的官方参考实现
`/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py`、当前 native
`model.py` 和 checkpoint 配置。报告中的关键位置是：第 1、4–6、7–12、
19–20、22 页；官方 Python 参考实现的对应入口是
`Transformer.forward`、`Attention.forward` 和 `forward_spec`。

报告给出的架构约束如下：

| 架构能力 | 报告描述 | 对量化传播的含义 |
| --- | --- | --- |
| CED | 20 层 causal encoder 产生 `H20`；decoder 的 global KV 从 `H20` 按层投影，prompt 的大部分 decoder global 计算可以跳过 | decoder 不能继续把完整 prompt 当作普通 40 层 dense prefill |
| CSA2 Full | 当前层生成 main KV、indexer K、Top-K 和 SWA KV | 需要保留一次 source state，并允许后续 target 直接消费 |
| CSA2 Reindex | 复用最近 Full 的 main KV/indexer K，只重新算 indexer Q/Top-K | indexer Q 的输入仍然是当前层 hidden，不能复用整个 attention 输出 |
| CSA2 Reuse | 复用 main KV、indexer K、最近 Top-K，只算当前层 Q、SWA KV 和 attention | 不能为每层复制 global KV 或重复 indexer |
| SWA Bounded Replay | decoder prefill 只 replay prompt 最后 `nwin=128` 个 token；encoder/decoder SWA miss 也用有界 replay | decoder 后半段的 Linear/MoE 校准输入可以从完整序列降到 replay window，但这是有意的近似执行语义 |
| CED + CSA2 | decoder Full Mode 的 global KV 从 `H20` 生成，后续 Reindex/Reuse 仍使用共享 state | 需要把“全序列 global state”和“最后 128 token 的局部 hidden”分开存活 |
| FP4 main KV | global main KV 训练后用 FP4；SWA KV 对精度更敏感，保留更高精度 | weight-only GPTQ 当前不等于 serving path；必要时需要独立 KV 精度仿真校准 |
| EPD/Engram | encoder、prefill、decode 可拆分；Engram lookup 可预取 | 属于运行时调度优化，不能把 Engram 表或 cache 当作普通 Linear boundary |

当前实现中，以下内容已经与报告的 CSA2 配置一致：

- `compress_ratios`、`kv_source_layer_ids=[2, 8, 14, 20]`、
  `index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36]`；
- `candidate_source_layer_id=20`、2048 个 block、每 block 8 个位置、
  `index_topk=512`；
- `Attention.forward` 中用 `shared` tuple 传递 compressed main KV、
  indexer K、Top-K 和 candidate mask；
- Full source、Reindex source 和 Reuse consumer 的计算职责；
- 每层仍单独计算 Q 和 SWA KV，不能把 shared global state 误认为完整层输出。

但当前量化路径有四个实质差异：

1. **没有 CED prefill 分相。** `Transformer.forward` 从 embedding 开始，
   对 40 个 backbone layers 都使用完整 calibration sequence；上半段
   decoder 的 `Attention` 也对完整 token 数生成 Q、SWA KV、MoE 和 Linear
   激活。`streaming/pipeline.py` 只按 target 顺序传播 boundary，没有
   encoder/decode replay transition。
2. **没有 SWA Bounded Replay。** 当前 `Attention.forward` 的非 cache 路径
   是完整序列 prefill；`DeepseekV41Cache` 只服务 append-only cached
   inference，不能表示“global KV 保留全序列、decoder hidden 只保留末尾
   128 token”的双工作集。
3. **官方参考 Python 也没有完整实现报告中的部署 replay。** 其
   `Transformer.forward(input_ids, start_pos=0)` 仍循环全部层和输入 chunk；
   `forward_spec` 在 `start_pos=0` 只为 DSpark seed SWA cache。报告中的
   CED/Bounded Replay 是更深的 inference-system execution policy，不能
   只因为当前 Python reference 能跑就认为量化已经复用了该优化。
4. **当前 weight PTQ 不仿真 FP4 main KV。** 这不会导致 GPTQ Hessian
   OOM，但会让 decoder attention 后续 Linear 的校准输入与部署精度路径
   不完全相同。它应作为独立的 KV-cache calibration/quality 任务，不能
   混入第一版 activation propagation 修复。

因此，当前 `model.py` 的问题不是 CSA2 shared tuple 完全失效，而是把报告
中的部署级 prefill policy 简化成了完整序列 forward。前一轮按需 boundary
优化已经减少了 CPU/GPU carry 搬运；下一项更大的收益来自 decoder replay
减少的 token 数，尤其是 20 层 decoder 的 routed expert、SWA projection、
QKV/FFN Linear 和 GPTQ activation hooks。

这次核对过的 capability 设计不进入当前实现。它可以作为未来推理 runtime
的研究草案，但不应被接入 `streaming/pipeline.py` 或量化 checkpoint
fingerprint：

- 官方 serving runtime 已经负责 CSA2 Full/Reindex/Reuse 和 append/decode
  KV cache；
- GPTQ/HIGGS 校准必须保留完整 sequence 激活统计；
- CED/SWA Bounded Replay 的近似 token slicing 会改变 decoder 后半段的
  observer/Hessian 输入；
- 因此当前 streaming PTQ 只优化 target 粒度、EP、权重生命周期和边界搬运，
  不实现 decoder prompt replay。

这里需要明确区分部署优化和量化校准：报告中的 CED/SWA Bounded Replay
允许近似 SWA state，而 GPTQ/HIGGS 的激活统计直接决定量化参数。经过与官方
reference 对照，决定**不把 bounded replay 接入当前 streaming PTQ 的实际
forward**，也不为了追求传播速度牺牲校准精度。

当前量化执行规则固定为：

1. 保留完整 sequence prefill，作为 GPTQ、HIGGS 和普通 PTQ 的唯一校准语义；
2. 继续复用官方已经实现的 CSA2 Full/Reindex/Reuse state，避免重复计算
   compressed KV、indexer K、Top-K 和 candidate state；
3. 不在 pipeline 中增加模型专用的 decoder replay、token slicing 或
   absolute-position workaround；
4. CED、SWA Bounded Replay 和 serving KV precision 只作为未来独立的推理
   runtime 优化研究，不进入当前量化 checkpoint identity。

因此不能用 CED 的部署收益估算当前 GPTQ 端到端时间；当前性能优化优先放在
target 粒度、EP 并行、HIGGS streaming provider、权重生命周期和激活搬运，
这些路径不改变完整 sequence 的激活统计。

#### 16.2.5 metadata 优化版重跑记录

完成基线采集后主动取消 `attempt-20261115T110755Z-9a853928`。该轮不是新
OOM；其 work 已保留为
`work-attempt-20261115T110755Z-9a853928-disjoint-baseline`。取消时 worker 已
结束 stage，但量化子进程 PID 1900475 变为 PPID 1 并继续占用资源。核对完整
argv 属于本 run 后，先 SIGTERM，等待 10 s 仍未退出再 SIGKILL；确认停止后
才归档 work、创建新 attempt。此处暴露的 worker 跨进程组取消清理问题需要
独立回归，本轮未修改队列实现。

```text
attempt_id: attempt-20261115T115518Z-8b92c0b3
created_at: 2026-11-15 19:55:18 +08:00
quantize_job: job-20261115T115518Z-8a9f118e
inference_job: job-20261115T115518Z-780ad8b3
report_job: job-20261115T115518Z-5ccae023
metadata_patch_sha256:
  8159ff0a0de6938d11d103e3fd7c2f38040c270a182ecc0ce7f498a74e2f144f
pipeline.py_sha256:
  b48e63ca40ff7f807e8a086c8daefb5f94add619fd5efb87c3e3c86e5dffe73f
```

其余三个生产文件的指纹与 §16.2.2 相同。新轮保留原数据、seed、recipe、
266 samples、max length 8192、固定 8 target group 与 synchronous save；
只替换 admission 的 metadata 读取路径。新轮首组与普通组资源/耗时、
checkpoint validate、runtime-smoke 与 report 均待完成，不能将上一轮基线
或单元测试结果代替本轮完整 E2E 结论。

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

### 17.1 旧 `streaming-ptq-ep` 分支复核与架构调整

2026-11-16 复核了本地历史分支 `streaming-ptq-ep`，其核心提交为
`02b6ca111`（`Add streaming PTQ expert parallel pipeline`）。该分支同时包含
`docs/developer/streaming-ptq-ep-design.md`、`streaming/ep.py`、本地 expert
bucket executor、rank-local staging/merge 和 EP 专用测试。它没有完成目标
P800 上的正式多 rank 端到端验收，但其中的执行拓扑比当前首版单进程
`ExpertParallelExecutor` 更适合作为八卡生产路径。

**结论。** 旧分支的总体方向更合理，后续 EP 优化以它的通信模型为准；当前
单进程 executor 保留为正确性基线、单进程回退和没有 distributed launcher
时的兼容路径。不能继续把“每个 expert 的输入复制到 owner、每个 expert 的
输出复制回主卡”作为八卡主路径，只在通用 FX fallback 或不满足批量条件时使用。

| 维度 | 当前单进程首版 | 旧 EP 分支 | 调整结论 |
|---|---|---|---|
| expert 执行 | 主卡切片后逐 expert D2D 到 owner，再逐 expert D2D 回主卡 | 每个 rank 只执行本地 experts，hidden/router 在 EP group 内复制 | 生产路径采用本地 expert 执行，消除 64 次级别的跨卡往返 |
| combine | 主线程按 FX mutation 顺序执行多个 `index_add_` | local partial output 一次 combine；分布式路径用 EP group reduce | 保留原始 FP32 累加顺序；优先 owner 批量 combine，distributed 路径再用一次 collective |
| 通信粒度 | expert 级输入/输出传输和 CUDA event | target/batch 级 output reduce；routed hidden 已复制时不需要 all-to-all | 禁止逐 expert host/device 同步，通信协议按 target/batch 定义 |
| hook/Hessian | 每个独立 FX expert leaf 直接调用，GPTQ hook 语义清晰 | `LinearExperts2D` 本地循环也保留单 expert module hook | 两者都保留；模型 adapter 负责将 DSV4.1 native MoE 映射到 local owner |
| 内存 | 当前 target 可按 expert owner 分片，主卡仍承担 dispatch scratch | replicated dense/shared + local routed weights；另有 expert bucket 降级 | 复用当前 admission，增加 replicated runtime、partial output 和 bucket 预算 |
| checkpoint | 单进程单 writer | rank-local staging，global barrier 后 rank 0 bounded merge | 多进程 EP 必须采用旧分支的 global commit 语义，不能让 rank 并写一个 shard |
| 模型边界 | 通用 FX executor，不识别模型名 | 外部 `StreamingMoEAdapter` 发现 expert container，并安装临时 proxy | 通用 EP 只依赖 adapter contract；模型本身不必增加 EP 方法 |

旧分支中以下内容不能直接复制：

1. 旧 `LinearExperts2D` 的 `one_hot` 路由会为 DSV4.1 的 384 experts 产生不必要
   的大路由张量；当前 `group_expert_routes` 的稳定排序、Python offsets 和
   `select_expert_routes` 应继续复用。
2. 旧模型适配针对 DeepSeek V4/旧 fused experts，不能直接覆盖新的 native、
   fused、MTP 或 multimodal MoE；这些差异必须通过各模型的 adapter contract
   表达，不能把 DSV4.1、CSA2 或 DSpark/MTP 逻辑放进通用 EP。
3. 旧分支虽然有分布式协议单元测试，但没有目标设备上的多 rank GPTQ/HIGGS
   完整运行证明；`all_reduce`、rank-local writer、异常退出和恢复必须在
   DSV4.1 tiny 与生产形状上重新验收。
4. 旧分支的 `dp_size>1` 仍明确未实现，不能把 topology dataclass 当作 DP
   statistics reduction 已完成；本轮先固定 EP-only，避免重复计入 GPTQ Hessian。

**调整后的实现顺序。**

1. 保留当前提交 `668afa138` 作为单进程异步 stream 回退点，并修复其测试中的
   staticmethod 装饰器问题。
2. 在通用 executor 中先实现 owner 内批量 dispatch：同一 owner 的兼容输入在
   主卡一次拼接、一次传入、owner 上按 expert 顺序调用，再一次回传并按原
   `index_add_` 依赖拆分；不兼容结构自动回退逐 expert。
3. 在现有 `streaming/moe/adapters.py` 的 `MoEModelAdapter` contract 上增加
   local-owner execution 能力，声明 expert container、replicated inputs、
   local expert ids、route metadata、partial output 和 combine event；streaming
   通过外部 adapter 为标准 `ModuleList`/`ModuleDict` 自动构造临时 proxy，并在
   进入/退出 target 时原子替换和恢复。模型本身不需要增加
   `configure_streaming_ep`。只有 fused/native 容器才由 adapter 提供
   `expert_accessor`、state-name parser 或 route conversion；DSV4.1 adapter
   只是第一个验收实现，不能成为核心接口的隐含前提。

标准容器的目标调用形态如下，proxy 生命周期由 streaming 管理，模型源码不变：

```python
with install_local_expert_proxy(
    model,
    "block.experts",
    local_expert_ids,
    process_group=ep_group,
):
    target_output = target_subgraph.forward(model, **batch)
```

proxy 保留 `block.experts.<id>` 的逻辑子模块名，因此现有 weight loader、GPTQ
hooks、Hessian key 和 writer 不需要认识 proxy。退出 context 时原始 container
和 child registration 恢复。标准 `ModuleList`/`ModuleDict` 使用通用 proxy；
不能表达 canonical route 或 fused expert layout 的模型才由外部 adapter 提供
route conversion/parent wrapper，不能要求模型类增加 EP 配置方法。
4. 将旧分支的 `EPTopology`、stable placement、rank-local output 和 global
   commit 语义以兼容层接入，先支持 `dp_size=1` 的多进程 EP-only。
5. 对每个 adapter 保持其完整 calibration 语义、独立 expert hook、原始 route
   weight 和规定的汇合顺序；DSV4.1 额外覆盖 CSA2/Engram/MTP，但这些只是
   adapter 验收项。只有在通用 tiny MoE、至少一个 fused/native MoE 和
   DSV4.1 单进程/多进程逐 tensor 对照通过后，才在目标 P800 重新测量八卡速度。

因此，旧分支不是直接替换当前代码的补丁，而是当前通用 EP 的正确性能方向：
**本地 expert 计算和粗粒度 combine 是主路径，逐 expert D2D executor 是兼容
回退路径，expert bucket 是容量模式，fused MoE kernel 仍不默认启用。核心只
依赖 `StreamingMoEAdapter`，不依赖任何具体模型类。**

**本轮实现检查点。** 当前分支已保留三个可回退提交：

- `668afa138`：owner CUDA stream、event dependency、卸载前统一同步；
- `d18efc39d`：同一 owner 的兼容 expert 输入按批次拼接，一次传入 owner，
  owner 内按原 expert 顺序调用，再按原输出边界拆回；单批输入上限默认
  128 MiB，超限或结构不兼容时逐 expert 回退；
- `cbb376da8`：外部 `LocalExpertProxy`，标准 expert container 通过 context
  manager 临时替换，保留 `experts.<id>` 名称和 hooks，退出时恢复原模块。

本地组合回归为 `70 passed, 5 skipped`，覆盖 EP executor、MoE adapter、
外部 proxy、subgraph pipeline、GPTQ/oneshot 对照和异常清理。该结果证明
当前改动没有改变已测 toy GPTQ 的 Hessian/sample count/qparams 语义；它不
证明 P800 上的跨卡通信收益。多进程 EP 仍需接入 topology、rank-local
materialization、global commit 和目标设备基准后，才能替换当前单进程路径。
