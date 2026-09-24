# Model Quality CI 中文部署与使用说明

本文对应当前 `ci/model_quality` 实现。长期设计见 [方案文档](model-quality-ci-plan.md)，不要把方案中的规划能力视为当前已经支持的功能。

## 1. 从哪里开始

建议按以下顺序上线：准备目标环境 → 配置一个小模型 → 生成计划 → Buildkite 执行量化和验证 → 增加精度评测 → 测试 BCE 上传 → 接入大模型和定时触发。

所有命令均在仓库根目录执行。仓库中的模型示例默认 `enabled: false`，上传默认关闭。默认生成空计划是正常现象，不代表模型测试通过。

## 2. 部署主机

当前生成的步骤使用 Buildkite queue `model-quality-gpu`。先将一台目标 GPU 主机的 agent 加入该 queue，确保各步骤都能访问同一代码版本、模型路径、数据路径和运行目录。仅有 queue 名称不会保证步骤落在同一物理主机；初期建议该 queue 只接入这台机器。

准备两个环境：

| 环境 | 用途 | 要求 |
|---|---|---|
| agent 的 `python3` | planner、stage、量化、checkpoint 检查 | 安装本项目及 PyTorch、PyYAML、safetensors 等依赖 |
| 独立 runtime Python | vLLM smoke、lm-eval | 安装兼容的 vLLM、lm-evaluation-harness，并能导入仓库中的 CI 模块 |

依赖版本应按目标 GPU 和本项目的安装说明固定，不在夜间任务中临时升级。runtime 环境使用 `python -m ci.model_quality...` 时，需要能从仓库根目录导入这些模块。

在 **Buildkite agent 环境或 hook** 中设置以下变量；只在个人终端 export 不会自动传给 agent：

```bash
export MODEL_QUALITY_RUNS_ROOT=/data/model-quality/runs
export VLLM_PYTHON_ENV=/opt/vllm/bin/python
```

预先创建运行目录并赋予 agent 读写权限。`MODEL_QUALITY_RUNS_ROOT` 在 Buildkite 中必须为绝对路径。`VLLM_PYTHON_ENV` 指向 Python 可执行文件，不是虚拟环境目录；它优先于 `runtime_smoke.python`，但不会替换 `evaluation.command` 中的 Python。

部署后检查：

```bash
python3 -c 'import torch, yaml, safetensors, llmcompressor; print(torch.accelerator.device_count())'
/opt/vllm/bin/python -c 'import vllm, lm_eval'
python3 -m pytest tests/ci/model_quality -q -o addopts=''
```

预期：主环境能导入依赖、看到所需 GPU；runtime 能导入对应工具；CPU 控制流测试通过。这些检查不等于真实模型量化和上传验收。

## 3. 配置第一个模型

复制并调整 `ci/model_quality/config/models.yaml` 中的 Qwen3 示例，或通过根清单的 `includes` 引入独立文件：

```yaml
schema_version: 1
includes:
  - models/my-model.yaml
models: []
```

include 路径相对于清单文件。DeepSeek 示例位于 `ci/model_quality/config/models/deepseek_v4_wna8.example.yaml`。

至少核对以下字段：

| 字段 | 配置内容 | 预期作用 |
|---|---|---|
| `id` / `enabled` | 唯一 ID；准备完成后改为 true | 决定模型是否进入计划 |
| `source.path` / `revision` | 本地 checkpoint 和版本标识 | 记录源模型身份 |
| `workflow.revision` | 量化入口 commit 或镜像 digest | 量化实现变化时使身份变化；必须由部署方维护 |
| `workflow.quantize` | argv 数组 | 启动量化入口 |
| `resources` | 卡数、GPU-hour、超时、磁盘等 | 预算选择及部分预检 |
| `validation` | required files、profile、streaming 等 | 定义产物检查要求 |
| `runtime_smoke` | Python、runtime revision、prompts、TP | 执行真实模型加载和最小生成 |
| `evaluation` | command、result_file、runtime_revision | 可选精度评测 |
| `upload` | 首次保持 `enabled: false` | 避免试运行直接发布 |

不要保留 `replace-with-*` 等版本占位符。Qwen3 CI 入口本身也使用 streaming；应为它设置 `validation.streaming: true`，并按实际产物要求配置 `FINALIZED`。

命令必须使用 argv 数组，每个元素是一个参数：

```yaml
workflow:
  revision: actual-entrypoint-commit
  quantize:
    - python3
    - ci/model_quality/entrypoints/qwen3_dense_w4a8.py
    - --model-id
    - "{source_path}"
    - --output-dir
    - "{output_dir}"
    - --work-dir
    - "{work_dir}"
```

不能写成一条带管道或重定向的 shell 字符串。多值参数逐项列出；布尔开关按入口的 `--flag` / `--no-flag` 语法填写。JSON 花括号无需转义。校准数据参数必须按入口实际支持的格式设置；示例可能会访问外部数据集，离线主机需要事先准备数据或缓存。

## 4. 先生成计划，不启动模型

```bash
python3 -m ci.model_quality.plan \
  --config ci/model_quality/config/models.yaml \
  --git-sha "$(git rev-parse HEAD)" \
  --model-filter qwen3-dense-w4a8-smoke \
  --run-mode quantize \
  --max-gpu-hours 8 \
  --plan-output /tmp/model-quality-plan.json \
  --pipeline-output /tmp/model-quality-pipeline.yml
```

预期得到两个文件：

- plan JSON：包含 `run_id`、`attempt_id`、selected/deferred、fingerprint 和预算。
- pipeline YAML：包含每个阶段的 Buildkite command 和依赖关系。

此命令只校验配置并生成计划，不执行硬件 preflight、量化、评测或上传模型。检查 selected 非空、模型 ID 正确、预算合理后再运行。

排序当前主要是 P0–P3、预计量化成本和模型 ID；预算不足的任务标为 `DEFERRED_BUDGET`。GPU-hour 是估算，不是强制运行时配额，也不是完整的 RAM/I/O 调度器。

## 5. 部署 Buildkite pipeline

在 Buildkite 创建独立 pipeline，连接本仓库。在初始步骤中上传仓库内 bootstrap：

```yaml
steps:
  - label: "Load model quality pipeline"
    agents:
      queue: model-quality-gpu
    command: "buildkite-agent pipeline upload .buildkite/model-quality/pipeline.yml"
```

bootstrap 会检查共享目录变量，运行 planner，保存 execution plan，并动态上传模型步骤。agent 的 PATH 必须让 `python3` 指向前面准备的主环境。

首次手动 build 设置：

```text
RUN_MODE=quantize
MODEL_FILTER=qwen3-dense-w4a8-smoke
MAX_GPU_HOURS=8
PRIORITY_OVERRIDE=none
```

预期：Buildkite 出现 `preflight → quantize → validate → runtime-smoke → report → summary`；配置启用上传时还会出现 publish。报告和聚合步骤允许上游失败后继续执行。

现有 concurrency group 对使用它的步骤串行限流；它不是整次 build 的资源锁。同一 run 的多个 build 仍可能在阶段之间交错，不要同时操作同一个 run。

`plan --upload` 的含义是上传 **Buildkite pipeline**，不是上传模型到 BCE。模型上传由 `upload.enabled` 和运行模式决定。

## 6. 四种模式与每一步效果

| 模式 | 执行顺序（最后均有聚合） | 适用场景 |
|---|---|---|
| `quantize` | preflight → quantize → validate → smoke → report → 可选 publish | 首次量化与基本健康检查；evaluation 显式 SKIPPED |
| `quantize_and_eval` | 在 smoke 后增加 evaluate | 量化并比较原模型与压缩模型精度 |
| `eval_only` | preflight → validate → smoke → evaluate → report → 可选 publish | 复用既有产物；匹配的 evaluation cache 可能直接命中 |
| `upload_only` | preflight → validate → smoke → report → publish | 重新验证已有产物并上传；仍需要 GPU 和源模型 |

| 阶段 | 实际操作 | 成功后预期效果 |
|---|---|---|
| plan | 配置校验、排序、预算选择 | execution plan 和动态 pipeline |
| preflight | 检查源文件、GPU 数、磁盘、runtime 等；比对身份 | 首次成功创建 input-manifest；失败不创建正式 manifest |
| quantize | 执行入口命令，保留 work 和日志 | model 目录存在；命令成功退出 |
| validate | 文件/index/tensor、finite、scale 等检查 | checksums、validation 结果；stage CLI 写 artifact-manifest |
| runtime-smoke | vLLM 加载、固定 prompts 生成 | token ID、生成文本、runtime 日志 |
| evaluate | 执行 evaluator 或使用匹配缓存 | attempt 内 evaluation JSON；门禁按原始未舍入值比较 |
| report | 汇总本 attempt 状态 | attempt 报告、共享报告视图、current-attempt 指针 |
| publish | 身份和内容检查、上传、verify、最后提交标记 | 成功后生成并上传 SUCCESS；返回远端 run 前缀 |
| aggregate | 读取 promoted attempt 和发布状态 | run 根目录 aggregate-report.json |

`quantize_and_eval` / `eval_only` 必须配置 evaluator；只有退出码 0 而没有新结果文件不能通过。`quantize` 的 PASS 只表示本模式必需阶段通过，不表示完整精度已通过。

## 7. 为已有 run 评测或上传

这两种模式必须显式传 `--run-id`。**当前默认 bootstrap 不读取 `RESUME_RUN_ID`，也不传 `--run-id`**；因此不能只在默认 build 中设置 RUN_MODE 就完成复用。

为复用任务增加一个自定义 Buildkite 初始步骤，在 agent 中运行以下命令，替换实际 run ID：

```bash
python3 -m ci.model_quality.plan \
  --config ci/model_quality/config/models.yaml \
  --run-mode eval_only \
  --run-id EXISTING_RUN_ID \
  --model-filter qwen3-dense-w4a8-smoke \
  --max-gpu-hours 8 \
  --persist-plan \
  --upload
```

只上传时改为 `--run-mode upload_only`，且模型必须 `upload.enabled: true`。显式 filter 允许选择 disabled 模型来复用历史产物。每次计划默认生成新的 attempt ID；保留同一 run 的 source、model、input/artifact manifest。

源内容、量化配置或产物 checksum 变化应使用新 run 或先解决不一致，不能靠删除 manifest 绕过检查。评测配置变化只应影响评测身份；修改量化实现时更新 workflow revision。

匹配缓存会返回 `EVALUATION_CACHE_HIT`，包括历史 FAIL。当前没有专用强制重评测开关；不要把创建新 attempt 当作一定会重新执行 evaluator。

## 8. 接入精度评测

可使用 `ci.model_quality.evaluators.lm_eval_pair`，其会在独立子进程中分别评估 base 和 compressed。参考 DeepSeek 示例中的 evaluation 配置，至少指定：

- runtime Python 和固定 runtime revision；
- base/compressed 路径；
- task、metric 名称、方向与门槛；
- fewshot、limit、seed、并行参数；
- `--output "{reports_dir}/evaluation-raw.json"` 与相同的 `result_file`。

higher 指标可用 `min_recovery`、`absolute_floor`；lower 指标可用 `max_relative_increase`。首次先运行小规模任务验证链路；小样本分数不能当作完整发布基线。task/metric 名称必须在固定 lm-eval 版本上确认。

## 9. 接入 BCE 上传

先保持上传关闭，在目标机器确认 `bcecmd` 版本、凭证注入和具体命令。不能把本文当作某个 bcecmd 版本的参数规范。

启用时需要：

```yaml
upload:
  enabled: true
  remote_prefix: bos:/YOUR_BUCKET/model-quality
  allowlist: [model, reports]
  # 以下三组必须填入已在目标主机验证过的 bcecmd argv 数组
  commands: []
  verify_commands: []
  success_commands: []
```

上述空数组是待填写模板，启用后保持为空会校验失败。

三组命令分别完成：

1. `commands`：把 `{output_dir}` 上传到 `{remote_run_prefix}/model`，把 `{reports_dir}` 上传到对应 reports 路径。
2. `verify_commands`：依据 `{reports_dir}/upload-manifest.json` 校验远端文件、大小和 SHA256；必须失败时返回非零。程序本身不会解析任意 bcecmd 输出判断校验是否充分。
3. `success_commands`：最后上传 `{run_dir}/commit-markers/SUCCESS.json` 至 `{remote_run_prefix}/SUCCESS.json`。

所有命令必须调用 bcecmd。数据上传本地路径必须为 allowlist 内绝对路径；不要传 `.`、`..`、`model` 这样的相对路径，也不要传源模型、work 或 run 根目录。当前 argv 校验支持固定的操作名及选项；带独立选项值的命令应在部署前确认可通过校验。

预期远端路径为 `<remote_prefix>/<model-id>/runs/<run-id>/`。SUCCESS 仅在 verify 成功后创建，上传重试会检查冻结清单。当前没有自动 latest/production 指针晋升功能，也没有自动远端不可覆盖保证；需用目标存储权限或经过验证的命令确保同一 run 不被不同内容覆盖。凭证只从 agent 环境注入。

## 10. 查报告和失败处理

```text
<runs-root>/<run-id>/
├── execution-plan.json
├── aggregate-report.json
└── <model-id>/
    ├── input-manifest.json
    ├── artifact-manifest.json
    ├── current-attempt.json
    ├── model/
    ├── work/
    ├── attempts/<attempt-id>/state/
    ├── attempts/<attempt-id>/reports/
    ├── logs/<attempt-id>/
    ├── reports/
    ├── evaluation-cache/
    └── commit-markers/SUCCESS.json
```

优先从 aggregate-report 找到失败模型，再读取 current-attempt 对应的 summary、state 和日志。quantization/evaluation 日志按 attempt 保存；runtime-smoke/publish 的部分日志仍在共享 logs 目录。报告保存在磁盘，当前 pipeline 不会自动将这些文件作为 Buildkite artifact 上传。

阶段失败通常表现为 `status: FAIL` 加 `reason_code`，并非所有原因码都是独立状态：

| 表现 | 下一步 |
|---|---|
| selected 为空 | 检查 enabled/filter；全 deferred 时检查预算 |
| PREFLIGHT_FAILED | 查看 failures；修路径、磁盘、GPU、runtime 或身份不一致 |
| CONFIG_ERROR | 修 manifest/argv/evaluator 配置后重新生成计划 |
| QUANTIZATION_FAILED / TIMEOUT | 查量化日志；保留 work；恢复是否可行取决于入口 checkpoint 支持 |
| VALIDATION_FAILED | 检查缺文件、tensor、NaN/Inf、scale；不要跳过门禁 |
| RUNTIME_SMOKE_FAILED | 检查量化格式、runtime 版本、TP 和 GPU 内存 |
| QUALITY_GATE_FAILED | 查看原始指标及门槛；不要无限重跑选最好结果 |
| DEPENDENCY_FAILED | 查最早失败阶段及 manifest/fingerprint |
| REMOTE_VERIFY_FAILED | 不发布成功标记；先检查远端缺失/损坏对象 |
| EVALUATION_CACHE_HIT | 使用了匹配缓存，未实际重跑 evaluator |

首次验收应至少完成：小模型 quantize、quantize_and_eval、eval_only、测试 bucket upload_only，以及一次故意失败后的报告检查。当前尚无完整 recipe-aware 覆盖率、多模态 adapter、联合资源调度和自动发布晋升；这些不应成为操作人员对当前工具的默认预期。
