# Streaming PTQ：逐 Subgraph 闭环执行设计

> 文档定位：这是 Streaming PTQ 的目标架构和落地计划，不是提交历史。
> 详细的当前实现契约、资源公式、状态表和兼容边界见
> [Streaming PTQ 实现设计](streaming-ptq-implementation-design.md)。
>
> 本文明确区分“已经存在的可复用组件”和“尚待实现的新流水线”。用户用法、支持矩阵和资源实测应写入 `docs/guides/streaming_ptq.md`。

## 1. 结论

Streaming PTQ 不再把一次量化拆成三次完整模型遍历：

```text
旧方案：收集全模型统计 -> 重新遍历全模型量化 -> 组装 checkpoint
```

目标方案沿用 oneshot Sequential Pipeline 的执行节奏，对每个 subgraph 完成一个闭环：

```text
准备首个 boundary
       │
       ▼
加载当前 subgraph 权重
       │
       ▼
校准 forward 并收集统计
       │
       ▼
立即量化当前 subgraph
       │
       ▼
量化后 forward，生成下一 boundary
       │
       ▼
写出量化 tensor并提交当前 subgraph
       │
       └──────────────> 重复下一个 subgraph

所有 subgraph 完成
       │
       ▼
流式复制未量化 tensor -> 生成 index/config -> 原子发布
```

这个方案的关键不是把三个 Python 函数合成一个，而是恢复 oneshot 已有的因果关系：

```text
原始输入 -> 当前层校准 -> 当前层量化 -> 量化输出成为下一层输入
```

Streaming 只替换两项基础设施：

1. 完整模型权重常驻，替换为 checkpoint-backed 按需加载和释放；
2. `save_pretrained()`，替换为逐 tensor 写出和最终 checkpoint 发布。

## 2. 为什么修改设计

### 2.1 原三阶段方案的问题

原方案要求 Stage 1 为所有 target 保存 GPTQ Hessian 或 iMatrix 统计，Stage 2 再次加载权重完成量化。这带来四个问题：

- 统计量的生命周期被人为延长到全模型量级；
- 同一份原始权重至少经历校准和量化两次读取；
- GPTQ/iMatrix 的 modifier 行为被拆成独立 collector 和独立 quantizer，产生重复逻辑；
- Stage 1 使用原始前缀传播，默认只能对齐 `propagate_error=False`，不符合 GPTQ 常用的逐层误差传播。

三阶段设计解决了跨进程恢复，却把“阶段”错误地定义成完整模型遍历。恢复能力不要求这样做；恢复单位可以缩小为一个 subgraph transaction。

### 2.2 新方案为什么更省资源

当前 subgraph 完成量化后，其校准统计立即失效。因此统计量没有必要保存到所有层处理完成。目标峰值为：

```text
peak RAM/VRAM ~=
    当前 subgraph 的物化权重
  + 当前 subgraph 的算法统计量
  + 当前批次的执行临时 tensor
  + 当前/下一 boundary activations
  + 单个正在序列化的输出 tensor
```

峰值不应随以下值线性增长：

- 模型总权重大小；
- source checkpoint shard 总大小；
- 已处理 subgraph 的数量；
- 全模型 GPTQ Hessian 或 iMatrix 统计总量。

必须诚实保留两个算法下界：

- 如果一个 sequential target 本身放不进 device，新方案仍然无法执行；
- GPTQ 单个 Linear 的 Hessian 是 `in_features x in_features` 的 FP32 tensor，超宽层仍可能成为峰值。

## 3. 术语与处理单元

### 3.1 Sequential target

模型中适合逐层执行和换入换出的模块，例如 `model.layers.0`。target 名称由现有 `infer_sequential_targets()` 推导。

### 3.2 Subgraph

`trace_sequential_plan()` 从真实 forward 图划分出的有序执行片段。一个 target 属于一个 subgraph；subgraph 还可能包含执行该 target 所需的少量相邻操作或模块。

新流水线的校准、量化、传播和恢复单位是 target subgraph，不是 checkpoint 文件。

### 3.3 Boundary activation

相邻 subgraph 之间需要传递的命名 tensor 集合。它遵循 `Subgraph.input_names` 和 `Subgraph.consumed_names`，不等同于单一 `hidden_states`。这使 residual、position 信息和模型特有的 forward 参数仍由共享 tracing 表达，而不需要 `_qwen3_boundaries()` 一类模型专用函数。

### 3.4 Source checkpoint shard

原始 checkpoint 的 `.safetensors` 文件分片。它只是物理文件组织，不是内存加载或量化单位。一个 subgraph 的 tensor 可以分布于多个 source shards。

### 3.5 Output staging shard

最终发布前的量化输出容器。它是落盘组织和恢复的实现细节，不决定一次加载多少原始权重。

### 3.6 Subgraph transaction

一次不可分割的逻辑提交：

```text
当前 subgraph 的输出 tensor 已持久化
+ 下一 boundary 已持久化
+ transaction state 标记 completed
```

只有三者匹配，当前 subgraph 才可在恢复时跳过。

## 4. 目标架构

```text
streaming_oneshot() / CLI
          │
          ▼
PretrainedStreamingSetup
  - meta model
  - parsed recipe/modifiers
  - calibration dataloader
  - exact quantization schemes
          │
          ▼
SequentialExecutionPlan             CheckpointWeightSource
  - ordered subgraphs                         │
  - target names                              ▼
  - inputs/consumed names             WeightMaterializer
          │                              - BF16/FP16/FP32 cast
          │                              - FP8/packed-FP4 decode
          └──────────────┬─────────────────────┘
                         ▼
             StreamingSequentialPipeline
             ┌─────────────────────────────┐
             │ SubgraphWeightSession       │
             │ ModifierExecutionAdapter    │
             │ BoundaryActivationStore     │
             │ StreamingCheckpointWriter   │
             │ SubgraphTransactionStore    │
             └─────────────────────────────┘
                         │
                         ▼
              StreamingCheckpointFinalizer
              - copy untouched tensors
              - index/config/tokenizer files
              - validation and atomic publish
```

设计上只有 `StreamingSequentialPipeline` 负责主循环。其他模块分别管理一种资源或协议，不能各自再遍历完整模型。

## 5. 与 oneshot 的对应关系

新流水线应当复用 oneshot Sequential Pipeline 的执行语义，而不是复用完整的 `oneshot()` 入口生命周期。

| oneshot Sequential Pipeline | Streaming 对应实现 | 保持的语义 | 替换的机制 |
|---|---|---|---|
| `trace_sequential_plan()` | 直接共享 | subgraph 顺序、输入和消费关系 | 无 |
| `IntermediatesCache` | `BoundaryActivationStore` | batch 级命名 activation 传递 | 可落盘、可跨进程恢复 |
| Accelerate onload/offload | `SubgraphWeightSession` | 当前 subgraph 执行时权重可用 | checkpoint 按名物化，退出即释放 |
| calibration hooks | modifier adapter | 使用真实 module input 收集统计 | 只挂到当前 subgraph |
| `sequential_epoch_end` | 当前 subgraph 内调用或等价窄接口 | 校准结束后立即量化 | 结果转交 writer，不留在完整模型 |
| quantized propagation pass | 当前 subgraph 二次 forward | 下一层看到量化前缀误差 | 结果写入 boundary store |
| `save_pretrained()` | streaming writer + finalizer | 标准 compressed checkpoint | 不要求完整 state dict |

### 5.1 默认精度语义

目标默认行为等价于：

```python
oneshot(..., pipeline="sequential", propagate_error=True)
```

每个 subgraph 执行两次：

1. 原始当前权重 forward，触发统计 hooks；
2. 当前权重量化后关闭统计 hooks 再 forward，产生下一 boundary。

如后续需要 reference-only 模式，可以显式增加 `propagate_error=False`。它不是第一版主路径，也不能与默认模式共享含义含糊的恢复数据。

## 6. 主循环与资源生命周期

目标伪代码如下：

```python
plan = trace_sequential_plan(meta_model, sample_batch, ...)
boundaries.initialize(run_prefix(dataloader))
writer.initialize(source_checkpoint, quantization_config)
transactions.validate_run(manifest)

for target_index, target_subgraph in plan.target_subgraphs():
    if transactions.is_complete(target_index):
        boundaries.require(target_index + 1)
        writer.require_committed(target_index)
        continue

    with weight_session.load(target_subgraph) as loaded:
        modifier.attach(loaded.quantized_modules)

        for batch in boundaries.read(target_index):
            target_subgraph.forward(meta_model, **batch)

        modifier.quantize_current_subgraph(loaded.quantized_modules)

        with modifier.hooks_disabled():
            for batch in boundaries.read(target_index):
                output = target_subgraph.forward(meta_model, **batch)
                boundaries.write_pending(target_index + 1, output)

        writer.write_pending(target_index, loaded.output_state())
        modifier.detach(loaded.quantized_modules)

    writer.commit(target_index)
    boundaries.commit(target_index + 1)
    transactions.commit(target_index)
    boundaries.delete(target_index)

writer.copy_unmodified_tensors(source_checkpoint)
finalizer.publish(writer, output_dir)
```

伪代码表达的是生命周期，不规定提交函数必须按这个顺序暴露。实际实现必须满足第 12 节的崩溃一致性协议。

### 6.1 当前 subgraph 可以驻留的对象

- subgraph 自身参数和 buffer；
- traced graph 引用的外部依赖模块；
- 当前 target 下待量化 Linear 的统计量；
- 当前 batch 的输入、输出和临时 tensor；
- 正在写出的量化 tensor 和 qparams。

### 6.2 subgraph 结束必须释放的对象

- 物化后的原始和量化 weight；
- GPTQ Hessian/sample count；
- iMatrix sum/count；
- hooks、observer 及临时 quantization state；
- 已消费的旧 boundary；
- materializer 的解码依赖 tensor。

测试不能只检查 Python 属性被删除，还要在独立子进程中检查 RSS/VRAM 不随已完成层数持续增长。

## 7. 模块一：准备与共享执行计划

相关现有文件：

- `src/llmcompressor/streaming/pretrained.py`
- `src/llmcompressor/pipelines/sequential/plan.py`
- `src/llmcompressor/streaming/tracing.py`

### 实现目标

在不加载完整 checkpoint 权重的情况下，得到与 oneshot 相同的：

- meta model 结构；
- 有序 sequential targets/subgraphs；
- 每个 subgraph 的 `input_names`、`consumed_names`；
- recipe 精确匹配后的 module -> `QuantizationScheme`；
- 首个 subgraph 的 boundary batches。

### 实现技术

1. `AutoModelForCausalLM.from_config()` 在 meta device 建模；
2. 继续调用共享 `trace_sequential_plan()`，不复制 partition 逻辑；
3. prefix graph 只按需加载 embedding 等真实依赖；
4. `TracedBoundaryAdapter` 从“Stage 1 adapter”缩减成共享 execution-plan adapter；
5. recipe 在 meta model 上展开一次，后续所有模块使用精确 module name。

### 验收标准

- Qwen3 和 DeepSeek-V4 不需要模型专用 boundary 函数；
- 相同模型/输入下，streaming 与 oneshot 得到相同 target 顺序；
- 对每个 subgraph，输入名、输出名和 consumed name 与共享 plan 一致；
- tracing 和准备阶段不读取所有 checkpoint tensor。

## 8. 模块二：SubgraphWeightSession

相关现有文件：

- `streaming/loading/target.py`
- `streaming/checkpoint/weight_source.py`
- `streaming/materialization/*`

### 实现目标

进入 context 时，使当前 subgraph 可以正常 forward；退出 context 后，meta model 不再持有这些实体权重。

### 实现技术

`SubgraphWeightSession` 基于 traced `call_module/get_attr` 依赖求出本轮 tensor 集合，而不是把 source shard 中所有 tensor 一次加载。每个 logical tensor 经由：

```text
WeightSource 按名读取 raw tensor
  -> WeightMaterializer 加载必要依赖并解码
  -> 转成 target_dtype/device
  -> 临时绑定到 meta model 对应 parameter/buffer
```

退出时恢复 meta parameter/buffer，并清除 raw dependency。若量化会用新 parameter 替换 weight，session 必须显式记录并交给 writer，不能让退出清理误删尚未持久化的数据。

### 自定义源精度

所有源格式差异继续收敛在 `WeightMaterializer`：

- BF16/FP16/FP32：`CastWeightMaterializer`；
- DeepSeek-V4 FP8 + packed-FP4：`DeepSeekV4WeightMaterializer`；
- 用户自定义格式：实现 dependencies、logical shape 和 materialize。

量化器始终接收符合契约的 BF16 logical weight，不包含模型名或源 dtype 分支。

### 验收标准

- spy `load_tensors()`，断言每次只请求当前 logical tensor 及 materializer dependencies；
- session 退出后当前 target 的 parameter 回到 meta；
- 相同最大 tensor、不同 source shard 总大小时，峰值内存基本不变；
- DeepSeek materializer 的解码测试独立于 forward 和量化测试。

## 9. 模块三：ModifierExecutionAdapter

### 实现目标

复用现有 Quantization/GPTQ modifier 的初始化、hook、observer、量化数学和压缩格式逻辑，避免 streaming 维护第二套算法实现。

### 兼容性结论

逐 subgraph 闭环与现有 Sequential Pipeline 的 modifier 调用点一致，因此它不会像旧三阶段方案那样改变算法：校准 hooks、`sequential_epoch_end`、量化后传播和 modifier 顺序都可以保留。从架构上说，所有本来支持 oneshot sequential pipeline 的 modifier 都应当是 Streaming PTQ 的兼容目标，而不应为每种算法重新实现 collector 或量化公式。

但“架构不损失算法语义”不等于“当前代码无需适配就兼容仓库内所有 Modifier”。Modifier 的实际工作集不一定等于一个 target subgraph。Streaming 必须根据 modifier 声明的工作集加载相关权重，并让 writer 表达其结构/config 变化。

源码核对后的分类如下：

| 类型 | 当前例子 | 与闭环的关系 | 所需支持 |
|---|---|---|---|
| target-local 校准后变换 | GPTQ、Quantization、iMatrix、SparseGPT、Wanda、AutoRound | 与现有 `sequential_epoch_end` 天然一致 | 当前 subgraph 权重、hooks、量化后传播和 state 导出 |
| mapping-local 联合变换 | SmoothQuant、AWQ、AutoSmooth、LogarithmicEqualization | 算法仍可逐 subgraph 完成，但一次操作可能同时修改 norm/平衡层/多个 Linear | 工作集必须覆盖完整 mapping，不能只加载被 hook 的 Linear |
| 模型级预变换 | QuIP、SpinQuant | `calibration_start` 会处理 embedding、norm 或全模型 transform config | 增加可流式执行的 prepare transform pass，或把变换延迟到相关工作集 |
| 结构和 config 变更 | REAP | 每层决策可 sequential，但会删除 experts 并修改模型 config | writer 支持 tensor 删除/重命名，finalizer 支持 config update |
| 训练事件驱动 | Magnitude/Constant pruning 等 | 依赖 `start/update/end`、optimizer step 或训练 schedule，不属于 PTQ 校准闭环 | 只有定义 streaming training/checkpoint 协议后才能宣称支持 |

因此准确目标是：

1. 对所有 oneshot sequential-compatible modifier 保持算法语义；
2. 用通用的 modifier working-set 和 output-delta 协议承载差异；
3. 不为 AWQ、SmoothQuant、AutoRound 等另写 streaming 算法；
4. 对训练型或完整模型全局操作的 modifier，明确额外执行阶段，而不是宣称自动兼容。

### 实现技术

adapter 为 modifier 提供一个 subgraph-local lifecycle：

```text
prepare current modules
-> attach calibration hooks/observers
-> calibration forwards
-> quantize/compress current modules
-> disable hooks
-> quantized propagation forwards
-> export state_dict tensors
-> detach and cleanup
```

优先提取或复用 modifier 已有的窄接口，不能通过构造一个“假完整 oneshot session”掩盖依赖。确实依赖全局 `State` 的部分需要列清并最小化适配。

当前 `statistics/collectors.py` 和 `quantize.py:_quantize_module()` 可以作为数学等价测试的参考，但目标不是永久并存两套实现。迁移完成后：

- 不再跨 subgraph 持久化全模型 statistics；
- 不再由第二次模型遍历恢复 statistics 后量化；
- collector 只在 modifier 无法提供可复用接口时保留为薄适配层。

### GPTQ 特别要求

- Hessian 只属于当前 Linear/subgraph；
- 校准结束立即调用现有 `quantize_weight()` 路径；
- GPTQ 的 blocksize、dampening、actorder 等配置来自 modifier；
- 量化后 propagation 必须关闭统计 hooks；
- 导出的 weight、scale、zero point、g_idx 与普通 oneshot 格式一致。

### iMatrix + RTN 特别要求

- 当前 subgraph forward 累积 `_imatrix_sum/_imatrix_count`；
- weight observer 在量化前消费这些统计；
- `update_qparams()`、freeze 和 compress 复用现有 QuantizationModifier 路径；
- observer 和 accumulator 在写出后立即释放。

### 通用 modifier 协议

为了让兼容性来自公共机制而不是算法白名单，adapter 需要四个通用概念：

- **lifecycle**：保持 recipe 中 modifier 顺序，转发 initialize、calibration start、sequential epoch end、calibration end 和 finalize；
- **working set**：当前事件可能读取或修改的 module/tensor 集合，可以大于一个 target，但必须显式且可按需加载；
- **output delta**：事件产生的新增、修改、删除 tensor，以及 model config 更新；
- **propagation policy**：事件后是否需要用修改后的 subgraph 再 forward。默认沿用 Sequential Pipeline 的 `propagate_error` 行为。

第一轮实现可以先用 GPTQ 和 iMatrix + RTN 验证该协议，但接口不能以这两个算法的字段命名，也不能在入口通过永久白名单排除其他 sequential modifier。后续 modifier 的接入应主要是声明/推导 working set 和 output delta，而不是复制算法实现。

## 10. 模块四：BoundaryActivationStore

相关现有文件：`streaming/activations/store.py`。

### 实现目标

保存当前 subgraph 的输入以及量化后产生的下一 subgraph 输入，同时支持内存和磁盘 backend。正常执行时只需要相邻两个 boundary generation。

### 实现技术

- key 至少包含 run、boundary index、batch index 和 value name；
- 写下一 boundary 使用 pending namespace；
- 所有 batch 写完并校验后才能 commit；
- 当前 subgraph transaction 提交后删除旧 boundary；
- `consumed_names` 决定何时删除值，不能假设只有 hidden states；
- CPU/disk offload 不改变 dtype、shape 和嵌套结构。

### 验收标准

- residual、多输入和 kwargs 能跨 subgraph 正确传递；
- 校准 forward 的临时输出不会覆盖已提交 boundary；
- 进程在任一 batch 写入时退出，恢复后不会读取半个 boundary；
- 不同模型深度、相同 boundary shape 时，boundary 磁盘和内存峰值不随层数增长。

## 11. 模块五：StreamingCheckpointWriter

当前 `quantize.py` 已有逐 tensor 临时文件和 safetensors 组装能力，应提取成独立 writer，而不是让 writer 依赖“三阶段 Stage 2”的 statistics artifact。

### 实现目标

接收当前 subgraph 导出的量化 state tensors，在没有完整 state dict 的情况下构建最终兼容的 safetensors checkpoint。

### 实现技术

writer 维护：

- logical tensor name -> pending/committed tensor record；
- tensor dtype、shape、byte size 和 checksum；
- tensor 属于哪个 subgraph transaction；
- module format 和 tied-weight 决策；
- 最终 output shard 规划。

写入必须逐 tensor 执行：

```text
module.state_dict() 中取一个 tensor
-> 移到 CPU contiguous
-> 写临时 tensor record
-> 释放 CPU tensor
```

不能先将整个 subgraph state dict 的所有 tensor 复制到 CPU 再统一保存。对于单个 subgraph 仍很大的情况，这个约束决定峰值是否有界。

### 未量化 tensor

所有 target 完成后，writer 根据 source weight map 找出未被量化输出替换的 tensor，逐个读取和写出。处理规则包括：

- embedding、norm、lm_head、buffer 原样复制或由 materializer 输出 logical BF16；
- source-only 编码依赖不进入最终 checkpoint；
- tied weights 按兼容策略去重；
- 被量化 weight 由 compressed weight/qparams 替换，不能重复复制原 tensor。

### Output shard 含义

输出 shard 只负责文件组织。第一版可以沿用 source shard 名称以降低迁移成本，但 writer API 不能假设“当前 subgraph == 当前 source shard”。后续可以按目标大小重新分片，而不影响执行流水线。

### 验收标准

- writer 从未一次接收完整 source shard；
- 每个原始 logical tensor 在最终 index 中恰好出现一次，或作为 tied alias 被明确省略；
- safetensors offsets、dtype、shape 和 total size 正确；
- compressed checkpoint 可由 Transformers/compressed-tensors reload。

## 12. 模块六：SubgraphTransactionStore 与恢复

### 实现目标

进程在任意位置退出后，恢复执行得到的最终 checkpoint 与无中断执行一致。恢复不要求长期保存全模型统计。

### Manifest

run manifest 至少指纹化：

- source checkpoint 内容；
- 模型 config；
- normalized recipe 和算法参数；
- calibration dataset、样本数、长度和 seed；
- ordered targets 和 trace-plan schema；
- materializer 类型、版本和 target dtype；
- output format/tied-weight policy；
- streaming artifact schema version。

任一会改变量化结果的字段变化都必须拒绝复用旧 transaction。

### 原子提交协议

每个 subgraph 使用独立 pending 目录：

```text
transactions/<index>.pending/
    output tensor records
    next boundary records
    metadata.json
```

建议的提交顺序为：

1. 完整写出并 fsync 所有 output tensor records；
2. 完整写出并 fsync 下一 boundary；
3. 写 metadata，包含 tensor/boundary checksum；
4. 原子 rename pending -> committed；
5. 更新全局进度索引；
6. 删除旧 boundary。

`completed` 不能先于 tensor 和 boundary 发布。全局进度索引只是缓存；恢复时应能根据 committed transaction 重建。

### 恢复行为

- pending transaction：全部丢弃并重做当前 subgraph；
- committed transaction：验证输出记录和下一 boundary 后跳过；
- transaction 完整但旧 boundary 尚未删除：安全删除旧 boundary；
- state 声称完成但文件缺失/checksum 不符：报损坏，不静默继续；
- output 已 `FINALIZED` 且 manifest 匹配：直接返回。

### 为什么不保存统计

统计只用于产生当前 subgraph 的量化输出。transaction 未提交时，崩溃后重新校准当前 subgraph；transaction 已提交时，统计已无用途。保存 Hessian 只会增加磁盘、schema 和一致性成本。

## 13. 模块七：最终发布

相关现有文件：`streaming/finalize.py`。

### 实现目标

把 writer 的 committed records 发布成普通工具可加载的模型目录，并确保用户不会看到半成品。

### 实现技术

1. 确认所有 target transactions 完成；
2. 确认所有未量化 logical tensors 已复制；
3. 组装 `.safetensors` output shards；
4. 生成 `model.safetensors.index.json`；
5. 写入 compressed-tensors quantization config；
6. 复制 tokenizer、generation config 和必要模型文件；
7. 在临时 output 目录执行 reload/结构校验；
8. 原子 rename 到正式目录并写 `FINALIZED`。

### 验收标准

- index 的 weight map 指向实际存在的 shard；
- metadata total size 等于实际 tensor payload；
- config 中 target 和 format 与写出的 qparams 一致；
- 失败时正式 output 不包含伪完成模型；
- 重复 finalize 幂等。

## 14. 当前代码如何迁移

### 14.1 保留并增强

| 当前模块 | 处理方式 | 原因 |
|---|---|---|
| `pipelines/sequential/plan.py` | 保留 | execution plan 的唯一来源 |
| `streaming/tracing.py` | 保留并缩减职责 | 通用 prefix/target 图执行仍需要 |
| `checkpoint/*` | 保留 | 按名读取和 weight map 已符合目标 |
| `materialization/*` | 保留 | 源格式解码边界正确 |
| `loading/target.py` | 扩展为 subgraph session | 需要覆盖外部依赖和导出生命周期 |
| `activations/store.py` | 增加 pending/commit | 正常数据流和恢复都需要 |
| `finalize.py` | 保留并适配新 writer | 发布职责仍正确 |
| artifact fingerprint/validation | 保留核心逻辑 | 运行兼容性校验仍需要 |

### 14.2 合并或替换

| 当前模块 | 目标变化 |
|---|---|
| `collect.py` | 不再遍历完整模型；校准逻辑进入 pipeline 的当前 subgraph transaction |
| `statistics/collectors.py` | 优先由 modifier adapter 替代；仅保留必要薄封装和等价测试 |
| `quantize.py` | 拆成 subgraph quantization adapter、tensor writer 和通用 safetensors assembler |
| `oneshot.py` | 从三阶段编排改为 prepare -> pipeline -> finalize |
| `artifacts/store.py` | 从 target statistics store 改为 run manifest/transaction store |

### 14.3 兼容策略

迁移期间旧低层 `collect_calibration_statistics()` 和 `quantize_streaming()` 可以暂时保留，但新的 `streaming_oneshot()` 不应同时维护两条默认执行路径。新路径通过端到端和资源测试后再删除旧 API 或标记为 deprecated。普通 `entrypoints/oneshot.py` 不做行为修改。

## 15. 落地顺序

每一步都应产生可单独审查的功能和测试，提交信息不得使用 `PR1`、`PR2` 等编号字样。

### 15.1 当前进度（2026-11-14）

| 步骤 | 状态 | 已完成 | 尚未完成 |
|---|---|---|---|
| 一：transaction 与 writer | **已完成当前范围** | 通用 `StreamingCheckpointWriter`、pending/commit、逐 tensor payload、checksum、跨 transaction/shard 组装；subgraph transaction 同时保存下一 boundary；旧 quantize 路径已复用 writer；本地 checkpoint 可选子进程 RSS 回归已落地 | 扩展更多 checkpoint 格式时补充对应 transaction 测试 |
| 二：SubgraphWeightSession | **基本完成** | 从 traced `call_module/get_attr` 推导 working set；modifier `include/exclude`；跨 source shard；meta 恢复；临时 qparams/buffer 清理；共享 runtime buffer alias 独立物化；普通 tensor 属性恢复；tracing 已接入；独立进程资源回归可按本地 checkpoint 启用 | 更多 MoE 模型专项 working-set 测试 |
| 三：iMatrix + RTN 闭环 | **已完成并验证等价** | pretrained 入口已切换为逐 subgraph 校准、`sequential_epoch_end`、量化后传播、压缩和 transaction；两层 Qwen3 与 sequential oneshot 的压缩 checkpoint 逐 tensor 完全一致；Qwen/DeepSeek tiny E2E 通过 | 扩展量化格式和更大校准集的等价矩阵 |
| 四：GPTQ 闭环 | **已完成首个等价闭环** | 新闭环直接复用 GPTQ lifecycle hook、Hessian、`compress_modules()` 和量化后传播；两层 Qwen3 与 sequential oneshot 的压缩 checkpoint 逐 tensor 完全一致；每个 target 后 Hessian/sample count 均释放；端到端 reload 通过 | 扩展 actorder、不同量化格式和更大模型的等价矩阵 |
| 五：通用 Modifier 工作集 | **部分完成** | `include_modules` 可表达大于 target 的 working set；state tensor 可导出 modifier output delta；SmoothQuant、AWQ、SparseGPT、Wanda 已完成两层 Qwen3 streaming/sequential 对照及 reload logits 验证；AutoRound 已加入签名兼容和 bounded boundary reference 复用；AutoRound 明确禁止多 target group | modifier 通用声明/推导接口；AutoSmooth、AutoRound output adapter 及跨 subgraph mapping 等价测试 |
| 五点一：target grouping | **首个通用闭环完成** | 共享 tracer 的多 target subgraph 被转换为一个 streaming execution unit；统一 admission、权重 session、modifier lifecycle、传播、transaction 和 output shard；默认单 target；两层 Qwen3 grouped streaming/sequential 对照通过 | `sequential_targets_per_subgraph="auto"` 的 streaming admission；更复杂 MoE/特化模型的 group policy |
| 六：切换入口和清理 | **部分完成** | pretrained `streaming_oneshot()` 已使用新闭环；旧低层 boundary API 暂时保留兼容；guide 已更新逐 subgraph 资源模型与实测数据；相关 streaming/sequential 定向测试通过 | CLI/examples 全面对齐；旧全模型 statistics API deprecated/清理；子进程资源回归 |

当前代码仍处于迁移期：普通 pretrained `streaming_oneshot(model=..., dataset=...)` 已进入新的逐 subgraph 闭环；显式传递 `model_factory/checkpoint/calibration_batches/targets` 的低层 boundary API 仍走旧三阶段路径，以避免在未完成兼容迁移前破坏已有调用者。transaction 中的 boundary 已成为恢复事实源：未提交的外部 next-boundary 会在重试时清理，已提交 boundary 会覆盖外部缺失或残留数据，损坏的 committed transaction 会拒绝静默重算。Qwen3-0.6B BF16 本地实测完成：单样本、序列长度 8、CPU iMatrix+RTN 的进程 RSS 高水位为 847 MiB（其中导入框架后的基线约 435 MiB），低于 1.4 GiB 源权重体积；该测试同时暴露并修复了 writer 无法直接写 BF16 tensor bytes 的问题。`/Users/ang/models/DeepSeek-V4-Pro-Tiny-bf16` 也已完成三 subgraph 端到端 iMatrix+RTN、checkpoint reload 和有限 logits 推理；测试暴露并修复了 runtime buffer 共享 alias 只物化首个别名、以及 Indexer/Compressor 保留 meta tensor 属性的问题。两层 Qwen3 对照测试已覆盖 iMatrix+RTN、GPTQ、SmoothQuant、AWQ、SparseGPT 和 Wanda：前两者 checkpoint 逐 tensor 完全一致；SmoothQuant、SparseGPT、Wanda checkpoint 数值及 reload logits 接近；AWQ 的 grid search 对浮点扰动敏感，因此验收相同 schema 和接近的 reload logits，而不错误要求 packed tensor 完全一致。下一开发重点是把资源回归自动化，并定义跨 subgraph modifier working set 的正式协议。

当前验证基线：

```text
Streaming + Sequential tests: 142 passed, 1 skipped, 14 warnings
本次修改文件 Ruff: passed
git diff --check: passed
```

本次实现包含文档、AutoRound 兼容和 target grouping；与用户现有的无关工作区文件分开提交。

### 步骤一：定义 transaction 与 writer 接口

实现内容：

- 从 `quantize.py` 提取逐 tensor record 和 safetensors assembler；
- 增加 `StreamingCheckpointWriter`；
- 增加 subgraph transaction metadata 和 pending/commit；
- 暂不接入真实 modifier。

验证：

- tensor-by-tensor 写入峰值；
- transaction 崩溃点参数化测试；
- index/offset/checksum；
- 重复恢复幂等。

完成标准：用人工生成的两个 subgraph state dict 能在中断恢复后发布合法 checkpoint。

### 步骤二：实现 SubgraphWeightSession

实现内容：

- 从 plan/subgraph 推导 module 和 tensor dependencies；
- 物化、临时绑定、导出、恢复 meta 状态；
- 支持 materializer dependency。

验证：

- Qwen 风格 dense block；
- MoE block；
- tensor 跨多个 source shard；
- DeepSeek-V4 materializer；
- `load_tensors()` 调用集合和资源回归。

完成标准：连续执行两个 subgraph，任一时刻只有当前 subgraph 实体权重驻留。

### 步骤三：接入 iMatrix + RTN 闭环

实现内容：

- subgraph-local modifier initialization/hooks；
- 校准、立即 RTN、量化后 propagation；
- 输出交给 writer；
- 清理 observer/statistics。

验证：

- 与普通 oneshot 相同 boundary 输入时逐层 qparams/weight 对比；
- 下一层输入来自量化后而非原始前缀；
- 中断后从下一个 subgraph 恢复；
- Qwen3-0.6B end-to-end reload。

完成标准：新的默认 pipeline 能完成 iMatrix + RTN，不调用旧全模型 collect/quantize stages。

### 步骤四：接入 GPTQ 闭环

实现内容：

- 复用 GPTQ hook、Hessian 和 quantize/compress 路径；
- 支持配置参数与可选 iMatrix observer；
- 当前 subgraph 完成后释放 Hessian。

验证：

- Hessian 和样本计数与 oneshot 相同；
- compressed tensors 与 oneshot 数值/格式对比；
- propagation 误差语义；
- 宽度变化时的 Hessian 资源公式。

完成标准：GPTQ 端到端 reload 成功，内存不随已处理 target 数增长。

### 步骤五：验证通用 Modifier 工作集

实现内容：

- 增加 modifier working-set 和 output-delta 协议；
- 用 SmoothQuant/AWQ 验证跨模块 mapping；
- 用 AutoRound 验证 subgraph 内重复 forward；
- 用 SparseGPT/Wanda 验证非量化权重变换；
- 为模型级 transform 和结构变化提供 prepare/finalize 扩展点。

验证：

- 相同 recipe、输入和 seed 下，与普通 sequential oneshot 对比修改后的 tensors/config；
- mapping 跨 subgraph 边界时不会读取 meta weight；
- modifier 顺序及组合 recipe 的生命周期不变；
- 资源峰值由声明的最大 working set 决定。

完成标准：上述各类 modifier 至少有一个代表通过等价测试；新增 sequential modifier 无需修改主循环。

### 步骤六：切换入口并清理旧架构

实现内容：

- `streaming_oneshot()` 使用新 pipeline；
- CLI 和 examples 更新；
- 删除或 deprecated 全模型 statistics artifacts；
- guide 更新恢复语义和资源公式。

验证：

- streaming 全测试；
- sequential pipeline 回归测试；
- ruff；
- Qwen3 和完整可用 DeepSeek checkpoint 端到端测试。

完成标准：默认用户路径只有一套 streaming 执行架构，普通 oneshot 测试不变。

## 16. 测试矩阵与明确预期

| 功能 | 测试对象 | 必须观察到的结果 |
|---|---|---|
| shared tracing | Qwen、MoE synthetic model | target 顺序和 boundary names 与 oneshot plan 相同 |
| 按需加载 | weight-source spy | 不请求整个 source shard，不请求未来 target |
| BF16 转换 | FP16/FP32 source | 进入量化器的是 BF16 logical tensor |
| packed source | DeepSeek materializer fixture | dependency 解码正确，编码辅助 tensor 不写入 output |
| iMatrix | 固定输入小 Linear | sum/count、qparams、compressed weight 与基线一致 |
| GPTQ | 固定输入小 Linear | Hessian/count、qparams、误差补偿结果与基线一致 |
| quantized propagation | 两层模型 | 第二层统计来自第一层量化后的输出 |
| transaction recovery | 每个提交步骤注入退出 | 最终输出与无中断运行一致 |
| tensor writer | 大 shard/小最大 tensor | RSS 与最大 tensor 相关，不与 shard 总大小相关 |
| target lifecycle | 相同宽度、不同深度 | 峰值不随模型深度近似线性增长 |
| checkpoint compatibility | Transformers reload | config、index、tensor names 和 forward 均有效 |
| oneshot isolation | 原 oneshot test suite | 现有入口行为和结果不改变 |

### 16.1 Qwen3-0.6B 资源验收

约 1.4 GiB BF16 权重不应仅因 checkpoint 总大小产生 3--4 GiB 的 Stage 2 常驻峰值。新方案需要分别记录：

- baseline RSS；
- 当前 block 权重；
- boundary store；
- iMatrix 或 GPTQ statistics；
- writer 临时空间；
- Python/torch allocator 未归还内存。

验收不能只给总峰值；必须用 instrumented 生命周期解释每一项。如果 iMatrix + RTN 的增量峰值明显接近完整模型权重大小，应视为实现错误。GPTQ 则需单独扣除 Hessian 的算法成本。

### 16.2 DeepSeek-V4 验收

验收分两层：

1. materializer fixture 验证 FP8 + packed-FP4 到 BF16 logical tensor；
2. 完整 checkpoint 验证真实 subgraph forward、量化、写出和 reload。

不完整或 index 指向无效绝对路径的本地 checkpoint 只能用于局部读取诊断，不能作为端到端通过或失败的证据。

## 17. 支持边界

核心流水线首批验收支持：

- 本地 safetensors checkpoint；
- 单 CPU 或单 GPU；
- 可由共享 sequential tracer 处理的 CausalLM；
- dense `nn.Linear` target；
- iMatrix + RTN、GPTQ、GPTQ + iMatrix observer；
- materializer 支持的非 BF16 源权重；
- subgraph 粒度恢复。

架构兼容目标还包括现有 oneshot sequential modifier。首批未完成等价测试的 modifier 应标记为“尚未验证”，而不是由主架构永久排除：

- AWQ、SmoothQuant、AutoSmooth、AutoRound；
- SparseGPT、Wanda；
- QuIP、SpinQuant；
- REAP 等结构修改型 modifier。

下列能力不属于当前 Streaming PTQ 闭环，仍明确不支持：

- 依赖 optimizer/update schedule 的训练型 modifier；
- KV-cache quantization；
- 多 GPU/跨机器调度；
- PyTorch `.bin` checkpoint；
- 单个 target 本身无法放入 device；
- 没有 materializer 的未知 packed 格式。

## 18.1 自动调优能力：HIGGS

用户所说的“自动调优 Modifier”对应 upstream 提交 `ebc4e7559`
（`HIGGS: ILP-based mixed-precision quantization`）。HIGGS 的代码位于
`src/llmcompressor/entrypoints/higgs/`，公开入口是 `get_higgs_config`。
它不是普通的 `Modifier`，而是一个配置生成入口：

1. 用候选 quantization schemes 对目标权重做 MSE 评估；
2. 根据参数规模、层位置和模块类型估计 layer sensitivity；
3. 用 ILP 在平均 bitwidth 约束下选择每层 scheme；
4. 返回可以交给 `oneshot`、`GPTQModifier`、`QuantizationModifier` 或
   `model_free_ptq` 使用的 `QuantizationConfig`。

因此 HIGGS 与 streaming 的边界必须明确：

- HIGGS 本身不应被伪装成当前 subgraph 内的 modifier lifecycle；
- HIGGS 的 MSE 收集阶段如果直接按完整模型执行，会重新引入全模型权重驻留；
- streaming 的第一阶段只需要保证生成的 config 能被流式 GPTQ/RTN 消费；
- 后续如需让 HIGGS 本身也受 streaming 内存约束，应新增
  `StreamingHiggsConfigProvider`，按 target 复用 `WeightSource` 和
  `SubgraphWeightSession`，而不是增加 CPU Hessian/offload；
- HIGGS 的 `pulp`/ILP 求解属于全局配置决策，可以在所有 layer 的小型
  MSE 摘要收集完成后执行；不应保存完整权重、Hessian 或 activation。

HIGGS 的 streaming provider 需要复用上述有界权重生命周期，因此优先级
高于 MTP wrapper 和 QAD。它不应把 HIGGS 变成一个持有全模型权重的 modifier，
而应作为“候选 scheme 评估与配置生成”阶段，按 logical target 读取并释放权重。
QAD 仍是最低优先级。

### 18.1.1 HIGGS streaming provider 契约

新增 `StreamingHiggsConfigProvider`（具体模块位置可根据最终依赖方向调整），
公开职责包括：

1. 复用 HIGGS 现有的 scheme 解析、单层/融合层 MSE、MoE expert 拆分、ILP
   求解和 config group 生成；
2. 通过 `CheckpointWeightSource` 和 `WeightMaterializer` 按 logical target
   读取权重，而不是按完整 source shard 读取；
3. 每次只保留当前 target 或 fused group 的权重，完成所有 candidate scheme
   的 MSE 摘要后立即释放；
4. 在所有 target 的摘要（MSE、参数量、融合关系）收集完成后再执行一次 ILP，
   只在内存中保留小型摘要，不保留权重、Hessian 或 activation；
5. 返回可直接交给 streaming GPTQ、streaming RTN/iMatrix、普通 `oneshot`
   和 `model_free_ptq` 的 `QuantizationConfig`。

provider 必须遵守以下边界：

- fused group 的 candidate scheme 选择必须保持一致，不能将融合权重拆成彼此
  独立的 ILP 决策；
- 3-D fused expert tensor 必须按 expert slice 或其他 bounded logical group
  处理，不能为了调用已有 helper 而加载整个 checkpoint shard；
- DSV4.1、DeepSeek V4、Kimi K3 等 checkpoint 特化布局只能由
  `WeightMaterializer`/`WeightSource` 负责解码，provider 不判断模型名；
- `ignore`、候选 scheme 和 target 的匹配必须与 streaming quantizer 使用同一套
  module/tensor 语义；
- provider 的结果应包含稳定的 scheme/config fingerprint，便于 streaming
  artifacts 恢复和跨阶段校验。

第一版先覆盖普通 dense Linear、标准 fused group 和标准 MoE expert；每个
logical target 的峰值由该 target 的权重、融合 partner 和 MSE 临时 workspace
决定，不得随总 checkpoint shard 大小增长。

GPTQ 的自动 MoE target policy 采用两级边界：先按模块路径和实际量化后代识别
最小 expert owner，将同一 owner 下的多个 Linear 作为一个逻辑 target；再在
调用方未指定 group size 时，把最多八个连续 owner target 放进一个 subgraph。
这个策略只依赖模块树和 quantization scheme，不识别 DSV4.1、GLM 或其他模型名。
owner 不是 `ModuleList`/`ModuleDict`，因此不会把完整 expert collection 误当成
一个 target。每个 subgraph 进入统一 admission，若当前模型的 expert 宽度使八个
owner 仍超出显存预算，admission 会拒绝该 group，后续实现可继续缩小 group
size，而不需要启用 CPU Hessian offload。

## 18.2 upstream oneshot 特性适配清单

截至 2026-11-14，`main..upstream/main` 中与 streaming PTQ 直接相关的提交
需要按下表处理：

| upstream 提交/特性 | oneshot 行为 | streaming 处理 |
|---|---|---|
| `b3faa86f3`（upstream/main 当前最新）observer 数值统一 | MSE/iMatrix 共享 FP32/FP64 error numerics、scale rounding 和 patience 语义 | 直接复用共享 observer；streaming 只需补 checkpoint/qparams 等价回归，不能保留另一套误差数学 |
| `176df1cdd`（upstream/main 当前第二新）REAP DDP diagnostics | 增强 REAP router 的 DDP 诊断测试 | REAP 是结构修改型 modifier，当前 streaming 不支持；该测试提交不改变 streaming contract，也不应引入 REAP 特化分支 |
| `179140f73` AutoRound `fp_ref` 预填充 | `IntermediatesCache` 可提供 FP16 reference，跳过 AutoRound 的 `collect_reference` | 复用 boundary store 的当前 target reference；检测旧版 `auto-round` API，不传不支持的 keyword，并回退原生 reference 收集 |
| `f2adfbdaa` `sequential_targets_per_subgraph="auto"` | 按 node memory 自动把多个 target 放进一个 subgraph | streaming 必须实现真正的 target group execution unit，统一 admission、lifecycle、传播和 checkpoint boundary；不能只把 tracer 参数改成大于 1 |
| `8ea12bd63` sequential pipeline offloading | oneshot 可按 subgraph 管理 offload/prefetch | streaming 继续使用 checkpoint-backed session；只借鉴 working-set 和生命周期语义，不恢复淘汰的 CPU Hessian offload |
| `68a83df8e` GPTQ 显式 `actorder=None` | 保留配置中的显式语义 | streaming 的 GPTQ adapter 直接透传该配置，并在 target admission 中使用一致的 Hessian 估算 |
| `118c11d01` GPTQ DDP global scale broadcast | 多卡量化时同步 global scale | 当前 streaming 单卡边界保持不变；未来多卡必须以 wrapper 方式接入，不能让 DSV4.1 特化路径直接承担 MTP/DDP 语义 |
| `b3faa86f3` MSE/iMatrix 数值统一 | oneshot observer 与 iMatrix 使用一致误差数值 | streaming 只复用 observer/quantizer，不复制数值实现；在等价测试中锁定 dtype、detach 和 reduction 语义 |
| `ebc4e7559` HIGGS | 独立生成混合精度 config | 已实现 bounded provider；作为配置选择层输出普通 `QuantizationConfig`，组合接入 streaming GPTQ/RTN 仍需验证 |

其中，`MTP` 兼容不能通过给所有模型增加全局 special case 实现。后续应增加
一个 MTP wrapper/compatibility adapter：默认保持通用模型路径不变，只在检测到
特化模型的 MTP/extra-head ownership 时包裹 target plan、working set 和
finalizer config。DSV4.1 等特化模型必须能在没有 MTP 的场景下独立通过；
MTP 的新增输入、输出和 checkpoint tensor 不能污染基础模型的 GPTQ/Hessian
生命周期。

## 18. 防止设计再次复杂化的规则

| 新需求或差异 | 应进入的模块 | 禁止的捷径 |
|---|---|---|
| forward 数据流不同 | shared tracing/plan | `_model_name_boundaries()` |
| checkpoint key/dtype/layout 不同 | WeightSource/Materializer | collect/pipeline 中判断模型名 |
| 当前 subgraph 权重生命周期不同 | SubgraphWeightSession | 加载完整 source shard |
| 新算法统计/量化行为 | ModifierExecutionAdapter | 再建全模型 statistics stage |
| 输出 tensor/layout 不同 | writer/quantization format adapter | 完整 state dict 常驻后保存 |
| 崩溃恢复需求 | transaction protocol | 保存所有中间 activation/statistics |
| 输出 config 差异 | finalizer/config update hook | finalize 中散落模型名分支 |

最后用一句话判断实现是否偏离目标：

> 如果处理第 N 个 subgraph 时仍需要保留第 0 到 N-1 层的权重或校准统计，或者需要提前加载第 N+1 层权重，那么资源生命周期设计仍然有问题。

## 19. 当前实施顺序

GPTQ Hessian 生命周期、AutoRound 兼容、显式 target grouping、bounded HIGGS
provider、基础 MTP wrapper 和 QAD + GPTQ adapter 已经落地；本阶段按以下顺序
继续实现：

1. **HIGGS 组合接入**：验证 provider 生成的 config 交给 streaming GPTQ/RTN
   的路径，并把 provider fingerprint 纳入恢复身份。
2. **observer numerics parity**：适配 upstream `b3faa86f3` 的误差 dtype、
   reduction、scale rounding 和 `patience=0`，保留 bounded search。
3. **特化模型 MTP 完整矩阵**：覆盖 upstream MTP、多 MTP、native DSV4.1
   extra state、MTP + GPTQ 的 ownership 和 reload。
4. **QAD 完整验证**：补 numerical parity、grouped target、recovery 和资源曲线；
   QAD 仍然是最低优先级。
5. **自动 target grouping**：`sequential_targets_per_subgraph="auto"` 复用
   显式 grouping execution unit 和 admission 预算。
