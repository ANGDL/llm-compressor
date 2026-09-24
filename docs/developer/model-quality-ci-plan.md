# LLM Compressor 模型量化与质量 CI 方案

## 1. 背景与目标

本方案用于在单机 8 卡、每卡 96 GiB 显存的环境中，为 LLM Compressor
建立一套可手动触发、可在夜间运行、可恢复、可审计的模型量化质量流水线。
流水线既要维护代码稳定性，也要持续验证量化产物的完整性、数值健康度和模型
精度，并具备通过 `bcecmd` 自动上传模型及报告的能力。

最终需要保证：给定任意一个 `run_id`，能够独立回答以下问题。

- 使用了哪一个代码提交、依赖环境和硬件环境。
- 使用了哪个源模型、模型 revision 和 checkpoint 摘要。
- 使用了哪份校准数据、数据顺序、样本数、token 数和数据摘要。
- 使用了什么量化 recipe、observer、target、ignore 列表和运行参数。
- 哪些模块完成了量化，哪些模块被忽略，是否存在漏量化或误量化。
- 量化产物是否完整，权重、scale 和 zero point 是否数值健康。
- 原模型和压缩模型在适用质量指标上的分数、相对退化和统计不确定性是多少。
- 精度测试是否执行；未执行时必须明确显示 `SKIPPED`。
- 产物上传到了哪个不可变 BCE 路径，远端是否校验成功。

## 2. 当前项目能力与缺口

项目已有以下可复用能力：

- `.github/workflows/quality-check.yaml` 提供基础代码质量检查。
- `.buildkite/` 已有 GPU 和 Transformers 测试流水线，可复用其 GPU 测试组织方式。
- `tests/e2e/` 提供量化、保存和 vLLM 加载验证。
- `tests/lmeval/` 已使用 `lm-evaluation-harness` 比较原模型与量化模型。
- 普通 oneshot 已支持部分算法的分布式校准和压缩。
- streaming PTQ 支持大模型逐 target 量化、持久化恢复和最终发布。
- `examples/imatrix/README.md` 已提供一个 WikiText-2 token-level perplexity
  实验，可作为因果语言模型通用评估能力的历史参考案例。

当前主要缺口如下：

- 缺少生产模型和 recipe 的声明式任务清单。
- 缺少统一的量化产物结构、数值检查和报告 schema。
- 缺少按模型类型选择评测套件的通用评估注册与适配层。
- 缺少可复现的因果语言模型 PPL 评测入口；这不是 iMatrix 专属能力。
- `tests/lmeval` 只支持“越高越好”的 metric，不能正确比较 PPL、WER、
  loss 等越低越好的指标。
- 缺少视觉语言、语音、embedding、reranker 和自定义业务模型的统一结果协议。
- 量化、评测和上传目前没有解耦，不能单独重跑。
- 缺少 BCE 不可变上传、远端校验和 latest/production 发布协议。
- 缺少同时考虑 GPU、CPU RAM、磁盘空间和 I/O 的单机调度。
- 现有 GPU 测试主要面向项目回归，不能代表完整生产模型量化质量。

## 3. 设计原则

1. 量化进程退出成功不等于量化模型质量合格。
2. 精度评测可以按需关闭，但结构、数值和最小推理验证不可关闭。
3. 每个模型独立执行、独立报告、独立上传；单模型失败不丢失其他结果。
4. 量化、评测、报告和上传是可分别重跑的阶段。
5. 所有质量比较必须基于同环境、同数据、同 tokenizer 和同评测协议。
6. 质量门禁使用未舍入的原始数值；舍入只用于报告展示。
7. 产物使用不可变 run 路径；只有完整校验后才能更新 latest 指针。
8. 基础设施错误可以有限重试，数值错误和精度回归不能“重跑到绿”。
9. 静态检查、CPU 测试、GPU API 兼容硬件测试和目标 runtime 验证应分别记录，
   不互相替代。
10. 硬件加速范围只包含兼容项目所依赖 GPU API 的平台，不设计或维护其他设备
    后端的专项 pipeline、配置、测试集或报告分支。
11. CI 不直接硬编码模型命令，而是从版本化清单生成运行矩阵。
12. 评估套件由模型类型和部署任务决定，不由 iMatrix、GPTQ、AWQ、RTN 等
    modifier 决定；不同压缩方法必须能复用同一套模型质量评估。
13. modifier 专项统计属于压缩过程诊断，不能代替端到端模型质量评估。
14. 校准集、阈值开发集和最终评测集分离，避免用评测结果反向调 recipe 后仍把
    同一数据称作无偏测试结果。
15. 质量验证分为三个正交维度：模型 profile 核心套件、压缩特性附加套件和
    modifier 专项诊断。三者不能相互替代。

## 4. 流水线总体结构

```text
手动触发 / 夜间调度
        |
        v
代码、模型、数据、硬件和上传预检
        |
        +---- 代码质量与单元测试
        |
        v
解析模型清单并生成任务矩阵
        |
        +---- 模型 A：量化 -> 验证 -> 报告 ----+
        +---- 模型 B：量化 -> 验证 -> 报告 ----+--> 可选精度评测
        +---- 模型 C：量化 -> 验证 -> 报告 ----+
                                                 |
                                                 v
                                      质量门禁与聚合报告
                                                 |
                                                 v
                                      BCE 上传与远端校验
                                                 |
                                                 v
                                  更新 latest/production 指针
```

建议将 Model Quality CI 作为独立 Buildkite pipeline，不与现有 PR 快速检查
强耦合。PR 流水线保持快速反馈；生产模型流水线按手动、nightly 和 weekly cadence
执行。

各模型任务应使用显式 Buildkite key 和依赖图，并让聚合报告在上游失败后仍然执行。
对共享的 8 卡机器使用全局 concurrency group 或节点级资源锁，确保两个 pipeline
build 不会同时超卖同一张卡、CPU RAM 或工作盘。任务还要处理取消信号，在退出前
写入状态、停止子进程并保留可恢复 work 目录。

## 5. 触发接口

手动触发至少提供以下参数：

```text
RUN_MODE=quantize|quantize_and_eval|eval_only|upload_only
MODEL_FILTER=all|model-a,model-b
EVAL_SUITE=smoke|standard|full
MAX_GPU_HOURS=<optional-budget>
PRIORITY_OVERRIDE=P0|P1|P2|P3|none
UPLOAD=true|false
PROMOTE_CHANNEL=none|latest|production
RESUME_RUN_ID=<optional-run-id>
```

推荐 cadence：

- 每个提交或 PR：代码质量、CPU 单测、轻量目标硬件测试。
- Nightly：选定模型的压缩、强制产物验证和该模型适用的 smoke/standard suite。
- Weekly：完整通用质量套件、业务任务和较大的模型矩阵。
- 手动：选择模型、评测级别、是否上传及是否晋升 production。
- 相关路径发生变化：根据模型清单中的影响路径选择受影响模型。

## 5.1 有限资源下的优先级原则

优先级需要分成两个独立问题：

1. **建设优先级**：有限人力先实现哪些 CI 能力。
2. **运行优先级**：有限夜间机时先运行哪些模型和评测。

两者不能混用。某项能力建设优先级较低，不代表对应模型永远不重要；某个模型运行
优先级很高，也不意味着为它单独开发一套不可复用的 CI。

统一采用四级优先级：

| 等级 | 含义 | 处理规则 |
|---|---|---|
| `P0` | 发布阻塞、线上回归或明确要求立即验证 | 第一顺位，允许占用当晚全部预算 |
| `P1` | 主力模型、核心 recipe、量化核心代码变更 | P0 后执行，nightly 重点保障 |
| `P2` | 次要模型、扩展评测、历史趋势建设 | 有剩余预算时执行或 weekly 执行 |
| `P3` | 实验模型、低收益优化、展示性能力 | 手动触发或资源充足时执行 |

默认不通过中断正在写 checkpoint 的任务来抢占资源。P0 到达时，调度器停止发放新的
低优先级任务，并在现有任务到达安全 checkpoint/阶段边界后释放资源。只有人工明确
要求时才强制终止可恢复任务。

## 5.2 每晚任务选择顺序

单机 8 卡不是要求每晚跑完全部矩阵。调度器接受 `MAX_GPU_HOURS`、夜间截止时间、
GPU 数、CPU RAM、磁盘和 I/O 预算，按下面顺序选择任务：

1. P0 发布/故障任务的 preflight、压缩、强制验证和 required suite。
2. 本次代码变更直接影响的 P1 模型和 recipe。
3. 已有压缩产物可复用的 P1 评测、报告或上传任务。
4. 超过最大新鲜度窗口、长期未成功验证的 P1 nightly 任务。
5. P2 weekly、扩展模型和 full suite。
6. P3 实验任务。

同一优先级内优先执行“质量风险高、结果新鲜度差、预计成本低”的任务。建议使用
可解释排序字段，不在第一版实现复杂机器学习调度器：

```text
effective_priority = declared_priority
                   + release_or_incident_boost
                   + affected_path_boost
                   + overdue_freshness_boost
                   + reusable_artifact_boost
                   - estimated_gpu_hour_penalty
                   - io_contention_penalty
```

排序结果、跳过原因和预算估算必须写入聚合报告。例如任务没有运行时，应显示
`DEFERRED_BUDGET`，而不是 `SKIPPED`；前者表示适用但本轮资源不足，后者表示用户或
cadence 主动关闭。

## 5.3 单模型内部的优先顺序

每个模型按由便宜到昂贵的门禁运行，前置硬失败后不浪费后续 GPU 时间：

1. 清单/schema、输入文件、checksum、磁盘和 capability preflight。
2. 压缩任务，或复用 fingerprint 完全匹配的已有产物。
3. 强制静态产物、覆盖率和数值验证。
4. 目标 runtime 实际加载与 profile-specific smoke。
5. 成本可控的 required core suite。
6. standard/full、业务扩展、性能和非必选 feature suite。
7. 报告汇总与 BCE 上传；失败报告也应保留。

优先复用已经验证的压缩模型和仍有效的原模型 baseline。不要为了重跑 PPL、GSM8K
或上传而重新量化同一 fingerprint 的模型。

## 6. 模型任务清单

建议新增 `ci/model_quality/models.yaml`，由动态 pipeline 解析。示例：

```yaml
schema_version: 1

models:
  - id: deepseek-v4-wna8
    enabled: true
    priority: P1
    business_tier: core
    max_result_age_hours: 24
    source:
      path: /models/DeepSeek-V4
      revision: local-checksum
    entrypoint: examples/streaming_oneshot/deepseek_v4_wNa8.py

    resources:
      strategy: single_device_streaming
      gpu_count: 1
      gpu_memory_gib: 96
      required_capabilities:
        - torch_accelerator
        - distributed_collectives
        - runtime:vllm
      host_memory_gib: 256
      io_weight: 3
      timeout_hours: 14
      estimated_gpu_hours: 14

    quantization:
      mode: w4a8
      calibration_dataset: /datasets/deepseek-calibration.jsonl
      calibration_samples: 256
      max_sequence_length: 4096
      batch_size: 1
      seed: 42
      checkpoint_progress: true

    validation:
      structural: true
      numerical: true
      smoke_prompts: ci/model_quality/prompts/general.jsonl

    evaluation:
      enabled_by_default: false
      profile: causal_lm
      baseline_model: /models/DeepSeek-V4
      suites:
        - name: language-modeling
          evaluator: lm_eval
          task: wikitext
          metrics:
            - name: word_perplexity
              direction: lower
              max_relative_increase: 0.05
        - name: reasoning
          evaluator: lm_eval
          task: gsm8k
          num_fewshot: 5
          metrics:
            - name: exact_match,flexible-extract
              direction: higher
              min_recovery: 0.95
              absolute_floor: 0.70
      feature_suites:
        - feature: kv_cache_quantization
          suite: long-context-kv-v1
          required_when_enabled: true

    upload:
      enabled: true
      remote_prefix: bos:/bucket/llm-compressor
```

清单中需要固定或计算：

- 源模型路径、revision、config 和 shard checksum。
- 量化入口、规范化 recipe、observer 和所有命令参数。
- 校准数据 revision、顺序、样本数、token 数及 checksum。
- tokenizer revision、chat template 和 BOS/EOS 行为。
- seed、dtype、target、ignore 列表及期望量化覆盖。
- 卡数、CPU RAM、I/O 权重、超时和可恢复策略。
- recipe 所需的 GPU API、dtype、collective、kernel 和 runtime capabilities。
- `priority`、业务层级、最大结果年龄和分阶段 GPU-hour 估算。
- 每个 metric 的方向、绝对门槛和相对门槛。
- 模型任务类型、evaluator adapter、suite 版本以及 suite 的 required/optional
  状态。
- 从 recipe 解析出的 `artifact_features` 及每个特性要求的 runtime/质量附加套件。
- BCE 目标前缀和晋升策略。

这些信息共同生成 run fingerprint。源模型、数据、recipe 或关键依赖改变后，
不得静默复用旧的恢复状态或原模型评测缓存。

实现时应拆分三种身份：artifact fingerprint 只覆盖压缩产物输入和量化 workflow；
evaluation fingerprint 覆盖 artifact fingerprint、evaluator、任务、数据、seed、limit
和 model args；report generator 版本单独记录。普通 checkout Git SHA 变化不应自动
使已有 artifact 失效，除非量化实现版本通过 workflow revision、镜像 digest 或入口
内容摘要明确进入 artifact identity。每次 Buildkite 执行还生成独立 attempt ID，阶段
状态和日志必须同时绑定 attempt 与 fingerprint，report 不读取其他 attempt 的旧状态。
量化 workflow 必须声明 `workflow.revision`，固定入口提交、镜像 digest 或等价的
实现版本；evaluation 与 runtime smoke 也必须声明各自的 `runtime_revision`。这样既
不让无关仓库提交使 artifact 失效，也不会在量化或评测实现变化时静默复用旧结果。
原始评测输出和完整报告都应先写到 attempt 专属目录；启动 evaluator 前删除该
attempt 下同名旧原始结果。只有 stage 状态、artifact/evaluation fingerprint 全部匹配
后，才原子更新共享报告视图和 `current-attempt.json`。publish 只能消费该指针指定的
报告，不能直接读取其他 attempt 的文件。
validate 成功后还必须生成 `artifact-manifest.json`，记录 finalized artifact fingerprint、
源模型 provenance、完整产物文件的大小/SHA256 及其聚合 checksum。`eval_only` 和
`upload_only` 必须重新计算并匹配该清单，产物内容发生变化时不得复用旧评测或发布。

命令参数替换只能识别 CI 自有占位符，例如 `{source_path}`、`{output_dir}`、
`{work_dir}` 和 `{reports_dir}`。不得对整个 argv 使用 Python `str.format()`；JSON、
正则表达式和普通花括号必须原样保留。量化、评测和上传命令共用同一个安全解析器。

## 7. 阶段一：Preflight

预检应在正式占用整晚计算资源前快速失败。检查内容包括：

### 7.1 代码与环境

- Git branch、commit、工作区状态。
- Python、PyTorch、Transformers、compressed-tensors、llm-compressor 版本。
- vLLM、lm-eval、驱动和 GPU runtime 版本。
- 评测环境必须是固定镜像或固定 lockfile，不临时安装不受控的最新版。

### 7.2 硬件

- 8 张卡全部可见，每卡显存符合预期。
- 卡的健康状态、温度和错误计数没有异常。
- 检查是否有其他进程占卡，并确认本任务获得清单声明的独占设备。
- 需要 DDP 的任务执行小规模 collective smoke test。
- 记录卡型号、设备 ID、驱动及拓扑。

本方案不按厂商名称编写多套业务逻辑，而是定义并预检所需 GPU API 能力，例如
PyTorch accelerator tensor/stream/event、分布式 collective、目标 dtype、目标 kernel
和目标推理 runtime。平台只有通过对应 recipe 的 capability check 才能调度该任务。
不在本方案中增加其他设备后端的专用安装源、环境变量、pytest 配置或兼容分支。

### 7.3 模型和数据

- 源模型 config、tokenizer、safetensors index 存在。
- index 引用的所有 shard 都存在且可读。
- 校准数据与评测数据可访问，样本数满足要求。
- 记录源模型、数据、recipe 和 tokenizer 摘要。

### 7.4 存储和上传

- 输出盘、work 盘、缓存盘空间满足任务预算。
- 检查 inode、写权限和临时目录，不只检查可用字节数。
- streaming 普通模式的 work/output 满足同文件系统要求。
- work 目录和 output 目录不能互相嵌套。
- 通过 CI secret 注入 BCE 凭证，并做不泄露密钥的连通性预检。

磁盘峰值不能只按参数量和位宽推算，应在首批模型中实测并记录：

```text
源模型 + 量化工作区峰值 + 最终模型 + 评测缓存 + 上传缓冲
```

## 8. 阶段二：代码稳定性 CI

代码稳定性分两档执行：

### 8.1 快速检查

- Ruff lint 和 format check。
- GPU API、kernel 和扩展相关 lint。
- CPU 单元测试和不依赖外部模型的 focused tests。
- 小模型或 tiny checkpoint 冒烟测试。

### 8.2 夜间 GPU API 兼容硬件检查

- GPU API 兼容硬件上的 quantization focused tests。
- streaming PTQ、保存、恢复和 finalize 测试。
- DDP observer 同步和多卡量化测试。
- vLLM 目标 runtime 加载测试。
- 与生产 recipe 相符的小模型 E2E。

测试报告必须区分：静态检查、CPU 测试、GPU API 兼容硬件测试、目标 runtime 测试。
外部模型下载失败、DNS 问题或依赖不匹配应标记为环境失败，不能直接归类为代码回归。

## 9. 阶段三：量化执行

### 9.1 Run ID 和目录

建议 run ID：

```text
20260923T220000Z_<git-sha>_<model-id>_<recipe-hash>
```

本地目录：

```text
runs/<run-id>/<model-id>/
├── input-manifest.json
├── artifact-manifest.json
├── logs/
│   ├── quantization.log
│   └── resource.jsonl
├── work/
├── model/
├── reports/
├── attempts/<attempt-id>/reports/
└── state/
    ├── QUANTIZING
    ├── QUANTIZED
    ├── VALIDATED
    ├── EVALUATED
    └── UPLOADED
```

状态标志必须在阶段真正成功后原子写入，不能根据目录存在或命令开始执行判断成功。

### 9.2 单机 8 卡使用策略

- 支持分布式并且验证有收益的普通 oneshot GPTQ、AWQ 或 AutoRound，可按
  2/4/8 卡运行。
- streaming PTQ 当前是逐 target 在单执行设备上运行，不应直接套用
  `torchrun --nproc_per_node=8` 当作专家并行。
- streaming 大模型优先采用“一卡一任务”或限制并发数的多模型调度。
- vLLM 精度评测可根据模型大小使用 tensor/pipeline parallel。
- 调度器同时预算 GPU 数、CPU RAM、输出空间和 `io_weight`。
- 每晚保留一部分预算给 P0/手动任务；未使用的保留预算可在截止时间前回填 P2。
- 历史实际耗时和峰值资源应回写为下一次调度估算，避免长期依赖人工猜测。

即使每个任务只需一张卡，也不应默认同时启动 8 个大模型 streaming 任务，
否则共享 NVMe、CPU 内存和数据盘可能成为瓶颈并导致失败。

### 9.3 恢复策略

- 长时间 streaming 任务建议启用 `checkpoint_progress=True`。
- recovery 模式具有额外磁盘与 I/O 成本，且不能与异步保存同时启用。
- 源 checkpoint、recipe、数据、target plan 或 materializer 改变时拒绝恢复。
- 普通模式已完成量化但发布失败时，优先使用 `finalize_only`，不重新量化。
- 每次恢复记录 attempt、前一次失败原因和复用的 transaction。
- 只有 agent 丢失、临时网络错误等基础设施错误允许有限自动重试。

## 10. 阶段四：强制产物验证

无论是否启用完整精度测试，以下验证都必须执行。

### 10.1 文件和结构验证

- 根据模型 profile 检查 `config.json`、tokenizer/processor、recipe、
  safetensors index 和必要的多模态辅助文件。
- index 中所有 shard 存在，无空 shard、临时文件或残留中间文件。
- index tensor 名称与 shard 中的实际 tensor 一致。
- 生成文件大小、数量和 SHA256。
- streaming 结果具有最终发布标志；不能把 work/publish 中间目录当作模型。

### 10.2 能力感知的压缩覆盖验证

逐模块和逐 tensor 统计：

- 总 Linear、MoE expert 和其他目标模块数量。
- 期望量化、实际量化、显式忽略和意外遗漏数量。
- 未匹配 target 和意外量化 target。
- bit width、group size、对称性、动态类型和 scale dtype。
- recipe 声明需要持久化的 weight、scale、zero point、transform 参数、KV qparams
  或稀疏元数据的 shape 和 dtype。
- 源模型大小、量化模型大小和实际压缩率。

显式 ignore 与本应量化却遗漏必须分开报告。验证器不能假设所有 scheme 都保存
同一组张量：动态 activation 量化可能没有静态 activation scale，某些浮点格式没有
zero point，KV cache qparams、transform 参数、稀疏 mask 和 mixed-precision 分组也
具有不同契约。应先把 recipe 解析为标准 `artifact_features`，再为每个 feature 调用
对应 validator；缺少适用 validator 是 `CONFIG_ERROR`，不能静默跳过。

### 10.3 通用数值验证

- 所有适用的浮点权重、scale、transform 参数和统计量是否包含 NaN 或 Inf。
- 按 scheme 的编码和语义检查 scale、zero point、稀疏元数据及其合法范围。
- 权重是否意外全零、常量化或出现异常饱和；合法的稀疏零值不应误报。
- 分层统计 min、max、mean、std、零值率和饱和率。
- 选定层计算 BF16 权重与反量化权重的误差。
- 对每个启用的 modifier 调用其诊断 adapter，输出独立、可选的专项统计。

modifier 专项统计不能写进通用质量 schema 的必选字段。例如 iMatrix adapter 可
记录无数据、全零、non-finite、OOM fallback，并定位输入、FP32 转换、平方、
reduction、累计和 DDP SUM 中的第一个失败阶段；GPTQ 可记录 Hessian/Cholesky
异常和 fallback；AWQ、SmoothQuant、AutoRound、SpinQuant、QuIP、SparseGPT、
Wanda 等分别记录自身的搜索、变换或稀疏度诊断。未启用对应 modifier 时该 adapter
状态为 `NOT_APPLICABLE`，不能报错，也不能影响通用评估套件的选择。

除静态 checkpoint 检查外，还应在目标 runtime 路径检查动态 activation、KV cache
和混合精度配置是否真正生效。只在 config 中看到配置，不足以证明运行时使用了
对应 kernel 或 dtype。

### 10.4 按模型类型执行最小推理验证

- 模型能被目标 runtime 完整加载。
- 文本生成模型对固定 prompt 做 greedy decoding，保存输入和输出 token ID。
- 视觉语言模型保存图像 checksum、processor 输入和输出 token ID。
- 语音模型保存音频 checksum、采样率、转写或生成结果。
- embedding/reranker 模型检查输出 shape、finite、归一化约束、排序和固定样本分数。
- 分类/回归模型检查 logits/数值输出 shape、finite 和固定样本结果。
- 输出必须符合对应模型任务契约，不能用“生成非空文本”作为所有模型的 smoke。
- 记录首 token 延迟、吞吐和峰值显存。初期性能作为报告项，不设硬门禁。

### 10.5 压缩特性驱动的附加验证

模型 profile 决定所有 recipe 共用的核心质量套件；`artifact_features` 决定只有某类
压缩产物才需要的附加验证。例如：

| Artifact feature | 必要附加验证 |
|---|---|
| 静态 activation quantization | 校准 scale 完整性及目标 runtime 静态量化路径 |
| 动态 activation quantization | 目标 kernel/dtype 生效证据及代表性 shape 推理 |
| KV cache quantization | 显式启用低精度 KV、长上下文质量和 KV 显存收益 |
| mixed precision | 每个层组实际 dtype/scheme 与 recipe 一致 |
| transform/rotation | 变换参数完整、加载后等价关系和目标 runtime 支持 |
| sparsity/pruning | 目标稀疏度、pattern 合法性及适用 runtime 路径 |
| multimodal component quantization | 各模态分支和 projector 的覆盖及端到端样本 |

这类附加验证由压缩“特性”而不是 modifier 名称选择。例如 GPTQ 与 RTN 都可能
产生 W4A16，核心模型质量套件相同；如果其中一个 recipe 还启用了 KV cache 量化，
则它额外运行长上下文 KV suite。

## 11. 阶段五：精度评估体系

建议将精度验证分成三个层次：

1. `smoke`：少量样本，快速检查评测链路和明显质量退化。
2. `standard`：模型类型对应的核心质量指标，作为 nightly/手动评测。
3. `full`：完整任务集，作为 weekly 或发布前评测。

精度评测整体可以关闭，但报告中必须生成 `evaluation.json`，并把适用但被关闭的
suite 状态写为 `SKIPPED`，同时记录关闭原因；不适用的 suite 写为
`NOT_APPLICABLE`。

### 11.1 通用评估架构

评估不应由 modifier 类型驱动。iMatrix、RTN、GPTQ、AWQ、SmoothQuant、
AutoRound 或任意组合只描述模型如何被压缩；评估套件由模型的输入模态、输出
契约和部署用途决定。相同模型的不同压缩 recipe 必须运行相同的核心 suite，才具备
可比较性。

建议采用三层抽象：

```text
model profile
    -> evaluation suite registry
        -> evaluator adapter
            -> normalized metric records
                -> generic comparator and policy engine
```

- `model profile`：`causal_lm`、`vlm`、`asr`、`embedding`、`reranker`、
  `classification`、`regression` 或项目扩展类型。
- `suite registry`：定义数据集、任务、版本、输入 adapter、metric、成本等级和
  required/optional 状态。
- `evaluator adapter`：封装 lm-eval、VLMEvalKit、ASR evaluator、自定义业务
  evaluator 等工具，并输出统一结果。
- `normalized metric record`：不论工具来源，统一包含 metric 名称、方向、原模型值、
  压缩模型值、有效样本数、不确定性和门禁结果。
- `policy engine`：只处理标准记录，不解析各工具的人类可读控制台表格。

建议通用 metric 记录至少包含：

```yaml
suite_id: wikitext2-ppl-v1
evaluator: lm_eval
evaluator_version: pinned-version
task: wikitext
metric: word_perplexity
direction: lower
aggregation: token_weighted
base_value: 6.24
compressed_value: 6.85
sample_count: 141
scored_units: 288768
stderr: null
confidence_interval: null
status: FAIL
```

### 11.2 评估工具选择

文本生成和因果语言模型优先使用 `lm-evaluation-harness` 作为评测编排工具，并
优先使用 vLLM 作为量化模型的加载后端。项目已有以下基础：

- `tests/lmeval/run_lmeval.py` 调用 `lm_eval.simple_evaluate()`。
- `tests/lmeval/test_lmeval.py` 已实现原模型与量化模型的成对评测。
- FP8 文档已有 `lm_eval --model vllm` 的使用示例。

某些压缩格式通过 Hugging Face evaluator 加载时可能不支持 `weight_scale` 等
量化参数，因此生产评测应以目标部署 runtime 为准。

评测环境应固定 `lm-eval`、vLLM、PyTorch、Transformers 版本和任务版本。当前
项目虚拟环境不应被假定已经安装 `lm_eval`；CI 应提供独立的固定评测环境。

`lm-evaluation-harness` 不是所有模型类型的唯一工具。建议按 profile 配置 adapter：

| 模型 profile | 推荐评估能力 | 典型指标 |
|---|---|---|
| `causal_lm` | lm-eval、固定 PPL evaluator、业务生成评测 | PPL、EM、F1、pass@k |
| `vlm` | 项目验证过的 VLM benchmark adapter | accuracy、ANLS、CIDEr 等 |
| `asr` | 固定音频预处理和转写 evaluator | WER、CER |
| `embedding` | 检索/语义相似度 evaluator | Recall@K、NDCG、Spearman |
| `reranker` | 排序 evaluator | MRR、NDCG、Recall@K |
| 分类/回归 | 对应任务 evaluator | accuracy、F1、AUROC、MAE、RMSE |
| 自定义业务 | 版本化内部 adapter | 业务定义指标 |

工具只有在项目中固定版本、输入协议和目标 runtime 加载路径后才能进入 production
门禁；尚未验证的工具先作为实验 suite，不能阻塞所有模型。

### 11.3 因果语言模型困惑度评估

PPL 是因果语言模型的一项通用质量指标，可用于 RTN、GPTQ、AWQ、iMatrix observer、
SmoothQuant、AutoRound 或其他适用 recipe 的压缩前后比较。它不属于 iMatrix，也
不应因为某个模型未使用 iMatrix 而被跳过。反过来，对 VLM、embedding、reranker、
ASR 等不适用 PPL 的模型，流水线也不能强制运行 PPL。

`examples/imatrix/README.md` 已记录以下历史结果：Llama-3.1-8B、W4A16、
group size 128、WikiText-2、141 个长度 2048 的 chunk，FP16 PPL 为 6.24，
RTN `imatrix_mse` 为 6.85，GPTQ + `imatrix_mse` 为 6.83。

当前缺少对应的可执行评测配置和完整协议，因此这些值只能作为历史参考，不能
直接作为 CI 基线。需要新增可复现的 WikiText-2 PPL suite，并固定：

- 数据集名称、config、split 和 revision。
- tokenizer 文件、revision 和文本预处理。
- chunk 长度、stride 和 chunk 数。
- BOS/EOS 处理。
- 首 token 和跨 chunk token 的 loss mask。
- 是否保留最后一个不完整 chunk。
- dtype、max model length、并行策略和 runtime 版本。
- 有效 token 总数。

PPL 必须根据全部有效 token 的总 NLL 计算：

```text
mean_nll = sum(all_scored_token_nll) / number_of_scored_tokens
ppl = exp(mean_nll)
```

不能先计算每个 chunk 的 PPL 再做算术平均。

### 11.4 按模型用途选择任务评估

PPL 能敏感地发现整体语言建模退化，但不能替代 instruction following、数学、
代码、工具调用和业务质量测试。建议模型按用途选择任务：

- 通用语言模型：WikiText-2 PPL。
- 数学/推理：GSM8K、MATH 等。
- 通用知识：MMLU 或项目认可的子集。
- 代码模型：HumanEval/MBPP 或内部代码集。
- 多模态模型：对应视觉语言 benchmark；PPL 仅在明确验证文本分支时作为附加项。
- 语音模型：WER/CER 以及需要时的生成质量。
- embedding/reranker：检索、相似度和排序指标。
- 分类/回归模型：任务定义的监督指标。
- 业务模型：固定且版本化的内部评测集。

每个 profile 至少配置一个必选核心 suite，production 策略不能要求所有 profile
都有 PPL。对因果语言模型，推荐同时包含“语言建模质量”“公开任务能力”和
“业务任务质量”；对其他 profile 使用对应的核心指标。

### 11.5 数据治理与统计可信度

- 校准数据、阈值调优数据和最终评测数据应逻辑隔离并分别版本化。
- 若根据某个 benchmark 反复调整 recipe，该 benchmark 结果属于开发集结果；发布
  前还需要未用于调参的 holdout 或业务回归集。
- small/limited eval 必须输出实际样本数，不能和 full eval 共用同一基线 key。
- evaluator 应保存逐样本结果或可审计摘要，以支持失败样本 diff。
- 随机生成、采样或 judge-based 评估要固定 seed、解码参数和 judge 版本，并运行
  足够重复次数。
- 有 stderr 或可计算置信区间的指标应一起报告；接近门槛时按预定义灰区策略
  复测，而不是根据单次点估计随意放行。
- 多 metric 或多 task 门禁要预先声明主指标、次指标和聚合策略，避免事后选择
  最有利的分数。
- 对不同 tokenizer 的模型，token-level PPL 通常不能直接横向比较；本方案的 PPL
  主要用于同一原模型与其压缩派生模型的成对比较。

## 12. 原模型相对分数与门禁公式

原模型和量化模型必须使用相同的 evaluator、数据、prompt/template、tokenizer、
并行策略和运行参数。所有报告同时提供原始分数、绝对变化和相对变化。

### 12.1 越高越好的 metric

适用于 accuracy、exact match、F1 等：

```text
absolute_drop = base_score - quantized_score
relative_drop = (base_score - quantized_score) / base_score
recovery = quantized_score / base_score
```

示例：

```text
BF16 GSM8K       = 0.750
W4A16 GSM8K      = 0.720
绝对下降         = 0.030，即 3.0 个百分点
相对下降         = 4.0%
recovery         = 96.0%
```

推荐门禁同时使用相对恢复率和绝对下限：

```text
recovery >= 0.95 AND quantized_score >= absolute_floor
```

当 base score 为零、接近零或 metric 允许负数时，比例可能无定义或没有解释意义。
这类指标必须使用绝对差、任务专用变换或显式 comparator，不能强行计算 recovery。

### 12.2 越低越好的 metric

适用于 perplexity、WER、CER、MAE、RMSE、loss、error 等非负指标：

```text
ratio = quantized_value / base_value
relative_increase = ratio - 1
recovery = base_value / quantized_value
delta_nll = ln(quantized_ppl) - ln(base_ppl)
```

使用项目 iMatrix 文档中的历史 PPL 数字作公式示例；公式本身与 modifier 无关：

```text
FP16 PPL            = 6.24
RTN imatrix PPL     = 6.85
PPL ratio           = 1.0978
PPL 相对增加        = 9.78%
PPL recovery        = 91.09%
delta NLL           = 0.0933
```

推荐 PPL 门禁：

```text
relative_ppl_increase <= configured_limit
```

初始 limit 不应对所有模型统一拍定。先对每个模型/recipe 重复运行 3 到 5 次，
建立均值、方差和硬件相关基线后，再设置 warning band 和 hard-fail band。

### 12.3 无天然方向或结构化 metric

有些结果不能归约成一个 higher/lower 标量，例如生成安全分类矩阵、工具调用 schema
通过率加参数准确率、延迟分位数集合或多目标 Pareto 结果。此类 suite 应注册任务
专用 comparator，输出标准的 `PASS/WARN/FAIL`、理由和原始子指标。禁止为了复用
单一公式而丢弃关键子指标。

### 12.4 比较器要求

现有 `tests/lmeval` 需要扩展：

- 支持 `direction: higher|lower`。
- lower-is-better metric 使用反向 recovery。
- 门禁前不对 recovery 做 `round()`。
- 处理 base score 为零或无效值。
- 同时支持相对门槛和绝对下限。
- 支持多个 task 和多个 metric。
- 支持不可使用比例的 metric 和任务专用 comparator。
- 支持 stderr/置信区间、灰区策略和预定义的重测规则。
- 保存完整 lm-eval 原始 JSON、stderr、task version 和样本数。
- 允许评测已有量化目录，不在评测后删除生产产物。

统一比较逻辑：

```python
if direction == "higher":
    recovery = quantized / base
    relative_degradation = (base - quantized) / base
else:
    recovery = base / quantized
    relative_degradation = (quantized - base) / base
```

## 13. 原模型基线管理

大模型原始 BF16 评测可能比量化评测更昂贵，可以缓存，但缓存 key 必须包含：

```text
model revision/checksum
+ tokenizer revision/checksum
+ evaluation dataset revision
+ lm-eval task/version/config
+ runtime and dependency fingerprint
+ dtype and parallel configuration
+ evaluator adapter and adapter version
+ decoding, prompt/template and preprocessing fingerprint
+ sample limit, seed and aggregation method
```

任一关键字段变化都必须重新计算基线。报告不能只引用一个历史数字，必须保存
基线来源、生成时间、运行 ID 和完整原始结果。

建议每次 production 发布至少执行一次同环境成对评测；nightly 可以复用仍有效的
原模型基线。

除原模型 baseline 外，还应保留“上一 production 压缩模型”的结果。原模型比较
衡量压缩损失，上一 production 比较检测压缩流程自身的版本回归；两者用途不同，
不得互相替代。

## 13.1 运行时一致性与成对执行

为了把差异归因于压缩而不是运行环境，base 和 compressed 评测应尽量在同一 agent、
同一 evaluator 镜像和相邻时间窗口执行，并保存实际生效的 runtime 参数。若源模型
只能由 Transformers 运行，而压缩模型只能由 vLLM 运行，则结果应标为
`CROSS_RUNTIME`；这类比较可以报告，但默认不能作为严格压缩 recovery 门禁，除非该
组合已通过独立的 runtime parity 校验。

评测开始前必须确认模型路径、revision 和 tokenizer/processor 没有被 evaluator
静默替换。评测结束后保存实际解析的 config、task version、模型加载后端和 kernel
选择证据。

### 13.2 阈值生命周期

阈值配置本身也需要版本化和审计，至少记录来源 run、样本规模、确定日期和审批者。
更新阈值必须作为独立变更审查，不能由当次待验收结果自动放宽。新模型或新 scheme
先进入 shadow 模式收集 3 到 5 次或足够样本的基线；稳定后再启用硬门禁。

对接近门槛的结果使用预定义灰区，例如置信区间跨过门槛时标记 `WARN` 并复测；
超过明确 hard-fail 界限时直接失败。复测次数和聚合方式必须预先固定，防止选择最好
的一次结果。

## 14. 报告设计

每个模型生成：

```text
reports/
├── summary.md
├── quantization.json
├── quantization.md
├── validation.json
├── evaluation.json
├── evaluation.md
├── modifier-diagnostics.json
├── environment.json
├── checksums.sha256
└── logs/
```

所有 JSON 包含 `schema_version`。`summary.md` 首屏至少展示：

| 项目 | 内容 |
|---|---|
| Run ID | 唯一运行标识 |
| 源 commit | 代码 SHA |
| 源模型 | 路径、revision、checksum |
| 量化方案 | W4A8、W8A8、FP8 等 |
| 校准数据 | revision、样本数、token 数 |
| 量化状态 | PASS/FAIL |
| 结构验证 | PASS/FAIL |
| 数值验证 | PASS/WARN/FAIL |
| 推理冒烟 | PASS/FAIL |
| 压缩特性验证 | 各 artifact feature 的 PASS/WARN/FAIL/N/A |
| 核心质量指标 | profile 对应的 BASE、COMPRESSED 和相对变化 |
| 附加任务指标 | 各 suite 的分数、方向和 recovery/差值 |
| 质量评估 | PASS/FAIL/SKIPPED/PARTIAL |
| 上传状态 | UPLOADED/NOT_UPLOADED |
| 耗时与资源 | 各阶段耗时、显存、RAM、磁盘 |
| 产物地址 | BCE 不可变路径 |

通用 evaluation JSON 中的 PPL metric 示例：

```json
{
  "schema_version": 1,
  "metric": "word_perplexity",
  "direction": "lower",
  "base": 6.24,
  "quantized": 6.85,
  "ratio": 1.097756,
  "relative_increase": 0.097756,
  "recovery": 0.910949,
  "delta_nll": 0.093264,
  "scored_tokens": 288768,
  "sequence_length": 2048,
  "chunks": 141,
  "dataset": "wikitext-2",
  "passed": false
}
```

流水线结束后还应生成所有模型的聚合报告：

| 模型 | Profile | 压缩 | 验证 | 核心质量 | 附加任务 | 上传 | 结论 |
|---|---|---|---|---|---|---|---|
| DeepSeek-V4 | causal_lm | PASS | PASS | PASS | SKIPPED | PASS | 可发布，评估不完整 |
| Qwen-VL | vlm | PASS | WARN | PASS | PASS | PASS | 可发布，有告警 |
| Embedding-X | embedding | FAIL | N/A | N/A | N/A | NO | 不可发布 |

## 15. 质量门禁

### 15.1 硬失败

- 量化进程非零退出。
- 输出未 finalized，或 index/shard 不完整。
- tensor 缺失或出现关键额外 tensor。
- 任一适用的权重、scale、transform 参数或运行时统计出现 NaN/Inf。
- 必须量化的模块被遗漏。
- 目标 runtime 无法加载。
- profile 对应的 smoke 输入失败。
- recipe 启用的 artifact feature 缺少 required validator/suite，或目标 runtime 中
  没有真正生效。
- profile 要求的核心 suite 缺失、执行失败或超过模型级门槛。
- 上传后远端文件或 checksum 校验失败。

### 15.2 告警

- 压缩率偏离预期但仍在容忍范围。
- scale 极值、零值率或饱和率相对历史显著漂移。
- 任一启用的 modifier 产生非致命 fallback 或异常诊断。
- 耗时、显存或吞吐明显退化。
- 完整精度测试未运行。
- 使用了未固定 revision 的模型、数据或依赖。
- base 与 compressed 使用不同 runtime，且尚无 parity 证明。
- 指标落在预定义统计灰区，等待按规则复测。

### 15.3 状态与依赖语义

阶段状态统一为 `PASS`、`WARN`、`FAIL`、`SKIPPED`、`NOT_APPLICABLE`、
`DEFERRED_BUDGET`、`CONFIG_ERROR` 和 `INFRA_ERROR`。聚合器必须保留原因码，不能把
以下状态合并成成功：

- `SKIPPED`：用户或 cadence 明确关闭了适用阶段。
- `NOT_APPLICABLE`：该模型/recipe 不适用该检查，例如 embedding 模型的 PPL。
- `DEFERRED_BUDGET`：任务适用且应运行，但本轮 GPU-hour、截止时间或其他资源不足。
- `CONFIG_ERROR`：required suite、validator 或必要输入未配置。
- `INFRA_ERROR`：硬件、网络、依赖或 agent 导致未获得有效质量结论。

如果压缩或强制产物验证失败，下游评估标为依赖失败而不是普通 `SKIPPED`。即使某个
模型失败，报告和其他模型任务仍应继续；最终 pipeline 结论由各模型的 required
阶段聚合。

生产发布门槛建议：

- `latest`：强制产物验证通过；允许任务精度为 `SKIPPED`，但状态必须显式。
- `production`：强制产物验证和 profile 配置的 required suites 全部通过，并经过
  人工批准或明确的自动晋升策略。

如果“精度测试可选”表示运行者可以手动关闭所有质量 suite，则该 run 最多只能进入
`latest` 或候选频道，不能进入 `production`。如果希望 production 仍支持轻量模式，
应在 profile 中定义一个成本可控但不可关闭的最小 required suite，而不是把全部评估
都设为 optional。

## 16. BCE 上传与发布协议

具体 `bcecmd` 命令参数需要在目标机器上根据实际版本的 `bcecmd version/help`
固化。建议封装单一 adapter，不在 pipeline 多处拼接上传命令。

远端目录：

```text
bos://bucket/llm-compressor/<model-id>/
├── runs/<run-id>/
│   ├── model/
│   ├── reports/
│   ├── checksums.sha256
│   └── SUCCESS.json
└── channels/
    ├── latest.json
    └── production.json
```

上传顺序：

1. 上传 model 和 reports 到新的不可变 `runs/<run-id>`。
2. 远端列举并校验文件数量、大小和关键 SHA256。
3. 上传 `SUCCESS.json`。
4. 最后更新 `latest.json`。
5. 完整质量门禁通过并获批准后更新 `production.json`。

失败策略：

- 量化验证失败：不上传模型，可上传失败报告。
- 验证通过、完整精度未启用：可上传 run 并更新 latest，标记 `SKIPPED`；默认
  不更新 production。
- 精度失败：保留诊断报告，可按策略保存 run，但不能更新 production。
- 网络错误允许断点续传和有限重试；checksum 不一致直接失败。
- BCE AK/SK 只能从 secret 注入，禁止出现在命令输出、日志和报告中。
- 上传脚本只接受显式且已验证的模型目录，拒绝空路径、`/` 和工作区根目录。
- 不把源模型、校准数据、逐样本业务评测内容或许可证受限文件默认打包上传；上传
  allowlist、脱敏规则和模型许可证检查必须在 publish 前执行。
- `latest.json` 和 `production.json` 应包含前一指针以支持原子回滚，并记录操作者、
  审批、时间和质量报告摘要。

## 17. 失败分类、重试和保留策略

### 17.1 可以自动重试

- Buildkite agent 丢失。
- 临时网络或对象存储超时。
- 远端模型仓库短暂不可用。
- 明确可恢复的基础设施错误。

### 17.2 不自动重试到成功

- NaN/Inf。
- profile 核心指标或 artifact-feature 附加质量指标下降。
- tensor 缺失、覆盖率异常。
- 模型加载失败。
- 远端 checksum 不一致。
- 同一输入和固定 seed 下结果异常漂移。

建议本地保留：成功任务 3 到 7 天；失败任务的 work 和报告保留更久，以便恢复和
诊断。远端 run 使用生命周期规则清理，但被 production 指针引用的 run 不得删除。

## 18. 推荐代码结构

```text
ci/model_quality/
├── README.md
├── pipeline.yml
├── models.yaml
├── schemas/
│   ├── models.schema.json
│   └── report.schema.json
├── prompts/
│   └── general.jsonl
├── scripts/
│   ├── preflight.py
│   ├── expand_matrix.py
│   ├── run_quantization.py
│   ├── validate_artifact.py
│   ├── run_smoke.py
│   ├── run_evaluation.py
│   ├── compare_metrics.py
│   ├── evaluation_registry.py
│   ├── modifier_diagnostics.py
│   ├── generate_report.py
│   ├── upload_bce.sh
│   └── aggregate_reports.py
└── baselines/
    └── <model-id>.yaml
```

职责边界：

- `run_quantization.py` 只负责量化进程、状态和资源记录。
- `validate_artifact.py` 只负责结构、覆盖率和数值验证。
- `run_evaluation.py` 可以独立评测任意已有模型目录。
- `evaluation_registry.py` 根据模型 profile 解析 suite 和 evaluator adapter。
- `compare_metrics.py` 负责 metric direction、相对退化和门禁。
- `modifier_diagnostics.py` 仅运行本次 recipe 适用的专项诊断。
- `generate_report.py` 消费结构化 JSON，不依赖不稳定的控制台文本。
- `upload_bce.sh` 只上传通过验证的显式目录。

## 19. 分阶段实施计划

在有限人力下，各阶段的建设顺序以“先防止坏模型发布，再提升自动化覆盖”为原则。
如果只能投入一轮开发资源，应完整完成第一阶段，不要同时铺开多个 evaluator。

### 第一阶段：最小闭环

目标：先证明从量化到上传的整个链路可靠。

- 新增模型清单 schema 和动态任务展开。
- 接入一个小模型和一个真实大模型。
- 实现 preflight、量化、结构验证、数值验证和推理 smoke。
- 实现 recipe 到 `artifact_features` 的解析和 feature validator 缺失时的硬失败。
- 生成统一 JSON/Markdown 报告。
- 实现 BCE 不可变 run 上传和远端校验。
- 精度可以 `SKIPPED`，但不能省略强制产物验证。
- 实现 P0/P1/P2/P3、GPU-hour 预算和 `DEFERRED_BUDGET`；第一版只需确定性排序。

验收：任务可手动运行，失败可定位，成功产物可从 BCE 按 run ID 完整还原。

第一阶段严格限制范围：只接入 `causal_lm`、一个轻量代表模型和一个主力生产模型；
只实现当前两者实际使用的 artifact feature validator；调度采用静态 P0/P1 队列和
GPU-hour 上限。以下内容明确不进入第一阶段：VLM/ASR/embedding/reranker adapter、
完整 benchmark 矩阵、judge 模型、跨 runtime 矩阵、自动阈值学习、复杂抢占调度和
可视化平台。

### 第二阶段：通用质量评估与相对精度

目标：建立与 modifier 解耦、按模型 profile 选择的可复现质量指标。

- 定义 model profile、suite registry、evaluator adapter 和标准 metric schema。
- 固定独立的文本评测 vLLM/lm-eval 环境。
- 为因果语言模型增加可复现的 WikiText-2 PPL suite。
- 为首批其他模型类型增加至少一个核心 suite adapter。
- 扩展比较器支持 higher/lower、无比例指标和任务专用 comparator。
- 修复比较前舍入、base=0、单 task 和产物删除问题。
- 实现原模型基线缓存和严格 cache key。
- 报告原值、绝对差、相对变化、不确定性和门禁理由。
- 对因果语言模型接入 PPL 和 GSM8K 等至少一个 higher-is-better 任务。
- 将校准、阈值调优和最终评测数据分离。

验收：同一个配置能够分别执行 baseline、quantized 和 compare；量化已有产物无需
重新量化即可重跑评测。

### 第三阶段：夜间调度与质量运营

目标：扩展到稳定的多模型 nightly/weekly 运行。

- 增加 GPU、RAM、磁盘和 I/O 联合调度。
- 建立每模型/recipe 的统计基线和质量阈值。
- 增加受代码路径影响的模型选择。
- 接入 standard/full 任务集和业务评测集。
- 增加历史趋势、通知、失败分类和保留策略。
- 实现 latest 和 production 晋升流程。

验收：连续多次夜间运行不发生资源争抢，报告可用于识别精度和性能趋势。

### 第四阶段：发布级扩展

- 增加跨 runtime 兼容性矩阵。
- 增加性能回归门禁。
- 增加多模态、代码、长上下文和工具调用评测。
- 增加 production rollback 指针及审计记录。

## 20. 首轮落地任务清单

建议按以下顺序实现：

1. `P0`：定义 `models.yaml`、输入 manifest、状态和 report JSON schema。
2. `P0`：实现 preflight、安全目录、产物完整性、覆盖率和数值验证。
3. `P0`：封装现有量化入口，支持稳定 run 目录、失败保留和安全恢复。
4. `P0`：实现按模型 profile 的目标 runtime 加载与 smoke。
5. `P0`：实现报告、`bcecmd` 不可变上传、远端校验和 production 防误发布。
6. `P1`：实现 P0/P1/P2/P3、GPU-hour 预算、确定性排序和延后原因报告。
7. `P1`：固定 vLLM/lm-eval 环境，实现 causal LM 的 PPL 和通用 comparator。
8. `P1`：加入 baseline cache、上一 production 对照和至少一个任务分数。
9. `P2`：实现完整 evaluation registry、其他实际模型 profile 的核心 suite。
10. `P2`：增加 nightly/weekly 新鲜度调度、趋势和性能门禁。
11. `P3`：扩展更多 benchmark、复杂 judge、多 runtime 矩阵和高级调度。

### 20.1 建议的首个可交付里程碑

如果当前只能安排一名工程师，先交付下面六项；其余需求进入 backlog：

1. 两个模型的 `models.yaml` 和手动 Buildkite 入口。
2. 输入 preflight、唯一 run ID、日志和状态文件。
3. 量化执行及失败后保留 work 目录。
4. shard/index、覆盖率、finite 和 vLLM 加载/最小生成。
5. 单模型 JSON/Markdown 报告和总报告。
6. 验证通过后使用 `bcecmd` 上传到不可变 run 路径并校验。

这个里程碑不等待完整精度体系即可阻止不完整、数值异常或无法加载的模型发布。
第二个里程碑再加入 causal LM PPL、原模型相对分数和一项任务精度。

### 20.2 建议的夜间预算模板

在尚无历史耗时数据时，第一版可采用保守静态预算：

| 预算部分 | 建议占比 | 用途 |
|---|---:|---|
| P0/手动保留 | 25% | 发布、故障定位和临时验证 |
| P1 主力模型 | 55% | 受影响模型量化、强制验证和 required suite |
| P2/补跑 | 20% | 过期任务、weekly 和扩展评测 |

如果到夜间窗口后半段仍没有 P0，保留预算可以回填 P1/P2。预算单位使用 GPU-hour，
但还必须满足整任务的 wall-clock 截止时间；预计无法在截止前到达安全阶段边界的任务
不应启动。运行两到四周后，再根据实际耗时、失败率和业务需求调整比例。

## 21. 完成标准

方案完成后应满足：

- 任意模型可以只量化、量化并评测、只评测已有产物或只上传已有产物。
- 中断的长任务可在输入 fingerprint 一致时安全恢复。
- 未运行完整精度测试不会被错误呈现为“模型质量已通过”。
- 评估选择与 modifier 解耦；同一模型的不同 recipe 使用相同核心 suite。
- recipe 的压缩特性会选择必要的附加验证，但 modifier 专项诊断不决定核心套件。
- 因果语言模型具备可复现的 PPL 和相对原模型报告，其他 profile 使用对应指标。
- higher/lower metric 都使用正确的相对退化公式和未舍入门禁。
- 不适合比例比较的 metric 使用显式 comparator，不产生误导性 recovery。
- 模型、数据、recipe、代码、环境、报告和 BCE 地址可以相互追溯。
- 不完整或校验失败的模型不会成为 latest/production。
- 单模型失败不阻塞其他模型生成报告和安全产物。
- 资源调度不会只看 GPU 数而忽略 RAM、磁盘和 I/O。

## 22. 实施前仍需现场确认的事项

以下项目需要在实际 8 x 96 GiB 机器上验证后写入最终配置：

- 卡的具体型号、runtime、互联拓扑和 DDP collective 性能。
- 最大模型的 GPU、CPU RAM、磁盘和 I/O 峰值。
- 每种 recipe 在目标硬件上的 DDP 支持和实际收益。
- vLLM 对各量化格式的目标 runtime 支持。
- `bcecmd` 的实际版本、上传/校验参数和凭证注入方式。
- WikiText-2 PPL 在固定 lm-eval 版本中的 task/metric 名称和切分行为。
- 实际模型清单包含哪些 profile，以及各 profile 采用的 evaluator 工具。
- 各模型核心指标、附加任务、性能 warning 和 hard-fail 阈值。

这些现场数据未验证前，应明确标记为待验收项，不能仅凭源码和小模型测试声称
生产流水线已经在目标硬件上通过。
