# Streaming PTQ：多分支执行与局部候选比较

本文说明 E6/E7 的通用接口和验收范围。实现位于 `streaming/`，沿用已有
加载、modifier lifecycle、传播和事务主循环，不修改 GPTQ modifier。

## E6：执行计划、真实输入和 prefix 恢复

`streaming_oneshot(model=..., dataset=..., ...)` 在 trace 前将校准 batch
逐个保存到 `work_dir/inputs`，计算有长度分隔的内容摘要，之后重放同一份输入。
摘要覆盖所有 batch，包括图像、音频、mask、position 等张量的 dtype、shape
和字节内容。它补充调用方的 `dataset_fingerprint`，避免仅依赖数据集名称。
保存和摘要按 batch 进行，不在内存中拼接整个数据集。

`StreamingExecutionDescriptor` 将以下信息纳入 manifest 与事务 run identity：

- 实际 FX 图的操作、连接和执行次序；placeholder 插入次序不影响身份。
- modality、分支名称及所需阶段、共享参数别名、显式运行时依赖。
- 全部校准输入摘要、processor 配置摘要。

入口自动记录 tokenizer 的词表、special tokens、chat template，以及 checkpoint
中的 processor/tokenizer 配置文件内容摘要。外部 processor 或预处理器额外配置
可通过 `processor_identity={"processor": "...", "resize": ...}` 传入；
调用方应包含会影响预处理语义的选项和版本，不能只填一个固定标签。

启用 `checkpoint_progress=True` 时，prefix 输出单独提交为 `prefix-00000`
事务。prefix 提交后即使第一个 decoder 尚未完成，恢复也直接读取已提交的
boundary，无需重新执行视觉/音频 encoder。未提交的 prefix 整体重跑；
已提交且损坏的 payload 拒绝恢复。后续 decoder 提交后，prefix boundary
按已有 recovery window 回收，不长期保留全部视觉状态。

只量化 decoder 时，encoder、projector、norm 和共享 head 仍由原有静态权重
复制流程处理，采用 `StreamingDTypePolicy` 保持未量化参数的原始 dtype。
Transformers 的纯权重重命名同时应用于新生成的量化 scale、zero point、
packed weight 和 shape，确保模型可以重新加载。

### 通用扩展接口

标准、图可见的依赖由 FX 图自动推导。仅当调用隐藏在 autowrapped 函数中时，
模型可声明以下 metadata；它描述依赖，不实现量化算法：

```python
def streaming_execution_metadata(self):
    return {
        "modalities": ["text", "image"],
        "branches": {
            "decoder": ["prefix", "layers.0", "layers.1"],
            "mtp": ["layers.0", "mtp"],
        },
        "runtime_dependencies": [
            {"stage": "prefix", "modules": ["vision", "projector"]},
            {"stage": "mtp", "parameters": ["auxiliary_bias"]},
        ],
    }
```

阶段必须是 `prefix` 或已选择的具体 target。模块与直接参数在读取权重前合入
当前工作集计划，直接参数无需加载整个父模块。已有
`streaming_runtime_parameter_dependencies(stage)` 继续有效。
共享参数对象的别名自动发现，也可显式记录 `shared_tensors`。

此接口不允许将其他量化 target 的原始权重作为隐藏运行时依赖重新加载。
这种跨阶段依赖应变为显式传播的 boundary，或由未来的专用 adapter 管理。

### 已验证矩阵与边界

| 场景 | 覆盖 |
| --- | --- |
| 标准 Transformers LLaVA | 真实图像输入、decoder-only 量化、视觉分支原样导出、Transformers 重载与 logits 对照 |
| 通用 text/image/audio 图 | 变长图像与音频 token、跨两层保留的 live value、MTP target、共享 embedding/head |
| 外部工作集 | prefix 隐藏模块、直接参数、MTP 外部参数、norm/head |
| 故障恢复 | prefix 提交后中断、现有 target 提交前后中断、boundary 回收、输入/processor/图变化拒绝复用 |
| 既有 recipe | 保持原有 GPTQ、iMatrix/RTN、AWQ、SmoothQuant 和 E5 adapter 回归 |

这不是任意多模态模型的自动适配承诺。输入的 key、容器结构、张量 rank/dtype
和非张量分支选项必须符合本次 trace；允许图实际支持的变长张量。
不同输入组合或非张量分支改变会被拒绝，应单独建立计划。张量驱动的任意
Python 隐藏控制流无法仅靠 metadata 证明等价，仍受共享 tracer 的能力限制。
本次没有新增模型特化代码，也没有宣称特定 MTP 或音频 pretrained 架构已完成验收。

## E7：固定 boundary 的统计复用与候选

公开入口 `evaluate_streaming_candidates()` 对一个显式 target、同一组 Linear
模块比较 1–8 个 weight-only 候选。支持现有权重量化 helper 所支持的 GPTQ
以及 RTN/iMatrix 路径；不在候选间执行跨层旋转、平滑搜索或 activation 量化。

```python
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor.streaming import (
    CandidateStatisticsCache,
    StreamingCandidate,
    evaluate_streaming_candidates,
)

cache = CandidateStatisticsCache("/data/ptq-statistics-cache")
candidates = [
    StreamingCandidate(
        name=f"w{bits}",
        schemes={
            "layers.0.mlp.down_proj": QuantizationScheme(
                targets=["layers.0.mlp.down_proj"],
                weights=QuantizationArgs(num_bits=bits, strategy="channel"),
            ),
        },
        use_gptq=True,
        blocksize=128,
        dampening_frac=0.01,
    )
    for bits in (4, 8)
]
evaluation = evaluate_streaming_candidates(
    model_factory=model_factory,
    checkpoint="/data/model",
    target_name="layers.0",
    calibration_batches=boundary_batch_factory,
    candidates=candidates,
    work_dir="/data/candidate-work",
    dataset_fingerprint=dataset_digest,
    prefix_fingerprint=prefix_digest,
    statistics_cache=cache,
    target_dtype=torch.float32,
)
print(evaluation.best_candidate, evaluation.report_path)
```

`model_factory` 只构造模型，不能加载 checkpoint；入口在 meta device 上调用它。
`calibration_batches` 是**当前 target 的输入 boundary**，不是未经 prefix 的
token/image 数据。默认用 mapping、tuple 或单位置参数调用该 target。
自定义 `forward_target` 必须提供 `forward_identity`；额外执行语义、外部变换、
构造器闭包中的配置应通过 `statistics_identity` 明确纳入身份。

每次评估先保存固定输入和未量化 target 的参考输出。统计 miss 时，参考 forward
同时收集所需 raw Hessian/count 或 iMatrix sum/count；hit 时只重放参考 forward。
每个候选重新从 checkpoint 加载当前 target 的原始权重，独立量化与评分。
候选失败写入结果，后续候选继续从原始权重开始。

内容寻址 key 包含实际 boundary、物化后原始 target state、prefix、数据集、
原始计数语义、选定模块、算法、执行/变换配置、构造器配置、dtype、materializer
及实现版本。改变 bitwidth 或 dampening 可复用兼容统计；改变上游输入、权重、
prefix 或变换身份会 miss。原始统计按现有 collector 语义计数，不额外过滤 padding。

缓存默认上限为 **8 个条目、每条 512 MiB、合计 2 GiB**，可显式配置。
当前 target 的 raw 统计另受 `max_statistics_bytes` 限制，默认 512 MiB；
这是统计 payload 上限，不是完整进程 RSS 的严格上界。缓存采用锁、校验摘要、
pending 目录和原子提交；损坏的已提交对象报错。预算耗尽会报错，
可用 `invalidate_prefix(prefix_digest)` 显式清理，不隐式维护全模型 Hessian。
缓存目录应独立于会被清理的 PTQ work/output 目录。

低层 `collect_calibration_statistics()` 和 boundary-mode `streaming_oneshot()`
也可传入 `statistics_cache`、`candidate_config` 和 `prefix_fingerprint`。
其中 `candidate_config` 标识影响统计的执行/变换语义；自定义 forward/selector
开启缓存时必须提供它。pretrained 主入口不隐式开启候选缓存。

`candidates.json` 原子记录每个候选的配置摘要、局部输出 MSE、错误和当前最佳项。
这里的最佳项仅表示**当前校准 boundary 的输出误差最小**，没有计入模型体积、
运行速度或最终任务质量。入口不发布完整模型；选择配置后仍需独立执行完整
量化 pipeline，并用独立数据做质量评估。

## 验证记录

2026-11-06，在 node16 的 `llm-quant-base` 容器、代码目录
`/data/quant/llm-compressor` 执行：

```bash
python -m pytest -q tests/llmcompressor/streaming \
  --ignore=tests/llmcompressor/streaming/inference
```

结果为 **423 passed, 2 skipped**。新增端到端和故障注入测试采用 CPU/tiny 模型；
标准 LLaVA 验证涵盖真实图像张量输入、导出、重新加载及 logits 对照。
通用 text/image/audio + MTP 图覆盖混合 dtype、共享 head、跨层状态、
prefix/target 中断恢复与损坏拒绝。候选测试覆盖 GPTQ 和 iMatrix/RTN、
命中不重复收集、身份失效、候选次序独立、失败隔离和缓存预算/原子发布。
本次不包含大型多模态模型质量评测或完整 inference 测试目录。
