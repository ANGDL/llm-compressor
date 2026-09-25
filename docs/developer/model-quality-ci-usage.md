# Model Quality CI 中文部署与使用说明

本文对应当前 `ci/model_quality` 实现。长期设计见 [方案文档](model-quality-ci-plan.md)，不要把方案里的规划能力当成已经支持的功能。

全文命令默认在**仓库根目录**执行，并且默认在 CI 控制面容器/agent 里跑。Buildkite 只是把下面这些命令按顺序串起来；没有 Buildkite 也能手动跑完整流程。

## 1. 最短路径

先跑通，再看细节。建议的上线顺序是：准备环境 → 配一个小模型 → 生成计划 → 跑 quantize 全链路 → 加精度评测 → 测 BCE 上传 → 再接大模型和定时触发。

**前置条件**

- 一台 GPU 主机，仓库已 checkout，`python3` 能 `import torch, yaml, safetensors, llmcompressor`。
- 源 checkpoint、校准数据、运行目录都在这台主机上，且路径对容器/agent 可见。
- `MODEL_QUALITY_RUNS_ROOT` 指向的目录存在且可写（见 §2.1）。

**第 1 步：生成计划，拿到 run 身份**

```bash
python3 -m ci.model_quality.plan \
  --config ci/model_quality/config/models.yaml \
  --git-sha "$(git rev-parse HEAD)" \
  --model-filter deepseek-v4-flash-0731-wna8 \
  --run-mode quantize \
  --max-gpu-hours 12 \
  --persist-plan \
  --plan-output /tmp/mq-plan.json \
  --pipeline-output /tmp/mq-pipeline.yml
```

`plan` 只做配置校验、排序和预算选择，**不碰 GPU、不跑模型、不上传**。所以 `selected` 为空是正常现象（模型没 `enabled`、filter 不匹配、或预算不够），不代表测试通过。

**第 2 步：取出身份参数**

后续每个阶段都要带上这几个值，用来保证审计记录和产物对得上：

```bash
P=/tmp/mq-plan.json
RUN_ID=$(python3 -c "import json;print(json.load(open('$P'))['run_id'])")
ATTEMPT=$(python3 -c "import json;print(json.load(open('$P'))['attempt_id'])")
FP=$(python3 -c "import json;print(json.load(open('$P'))['selected'][0]['fingerprint'])")
EFP=$(python3 -c "import json;print(json.load(open('$P'))['selected'][0]['evaluation_fingerprint'])")
SHA=$(python3 -c "import json;print(json.load(open('$P'))['git_sha'])")
```

**第 3 步：按顺序跑阶段**

每个阶段都是同一条命令换 `--stage`。先落一个模板脚本，前台后台都能用：

```bash
cat > /tmp/mq-stage.sh <<SCRIPT
#!/bin/bash
set -euo pipefail
cd "$PWD"
python3 -m ci.model_quality.stage --config ci/model_quality/config/models.yaml \
  --model deepseek-v4-flash-0731-wna8 \
  --run-id "$RUN_ID" --attempt-id "$ATTEMPT" \
  --run-mode quantize --git-sha "$SHA" \
  --fingerprint "$FP" --evaluation-fingerprint "$EFP" --stage "\$1"
SCRIPT
chmod +x /tmp/mq-stage.sh
```

然后：

```bash
bash /tmp/mq-stage.sh preflight
nohup setsid bash /tmp/mq-stage.sh quantize > /tmp/mq-quantize.log 2>&1 &   # 几十分钟到数小时
bash /tmp/mq-stage.sh validate
bash /tmp/mq-stage.sh runtime-smoke   # 需要先有可用的 runtime 服务，见 §4.4
bash /tmp/mq-stage.sh report
```

`quantize` 期间可以随时看进度，每层会打一条 `complete | time=`：

```bash
tail -f "$MODEL_QUALITY_RUNS_ROOT/$RUN_ID/deepseek-v4-flash-0731-wna8/logs/$ATTEMPT/quantization.log"
```

**完成标准**：`$MODEL_QUALITY_RUNS_ROOT/$RUN_ID/aggregate-report.json` 里该模型 `status: PASS`（聚合命令见 §3.2）。

## 2. 环境准备

### 2.1 环境变量

| 变量 | 作用 | 在哪里设置 | 默认值 |
|---|---|---|---|
| `MODEL_QUALITY_RUNS_ROOT` | 各阶段共享的 run 根目录，run/model/日志/报告都在这里 | Buildkite agent 环境或 hook；手动执行时 export 或写进 profile | `.model-quality/runs`（相对 CWD；Buildkite 中直接报错） |
| `VLLM_PYTHON_ENV` | vLLM runtime 的 Python 可执行文件 | 同上 | 无，缺省时用 `runtime_smoke.python` |
| `RUN_MODE` | planner 的运行模式 | Buildkite 手动 build 变量 | `quantize` |
| `MODEL_FILTER` | planner 的模型筛选 | 同上 | `all` |
| `MAX_GPU_HOURS` | planner 的 GPU-hour 预算 | 同上 | `80` |
| `PRIORITY_OVERRIDE` | 临时覆盖优先级 | 同上 | `none` |
| `BUILDKITE_COMMIT` | 记进 plan/报告的 git sha | Buildkite 自动注入 | 无 |

三个容易踩的点：

- 只在个人终端 export 不会传给 Buildkite agent，必须在 agent 环境或 hook 里设置。
- `MODEL_QUALITY_RUNS_ROOT` 在 Buildkite 中必须是绝对路径。
- `VLLM_PYTHON_ENV` 指向的是**可执行文件**，不是虚拟环境目录。它优先于 `runtime_smoke.python`，但**不会**替换 `evaluation.command` 里写死的 Python。

### 2.2 两个 Python 环境

| 环境 | 用途 | 要求 |
|---|---|---|
| agent/主环境 `python3` | planner、stage、量化、checkpoint 检查 | 安装本项目及 PyTorch、PyYAML、safetensors 等依赖 |
| 独立 runtime Python | vLLM smoke、lm-eval | 安装兼容的 vLLM、lm-evaluation-harness，并能从仓库根导入 CI 模块 |

依赖版本按目标 GPU 和本项目安装说明固定，不要在夜间任务里临时升级。

### 2.3 部署后自检

```bash
python3 -c 'import torch, yaml, safetensors, llmcompressor; print(torch.accelerator.device_count())'
/opt/vllm/bin/python -c 'import vllm, lm_eval'
python3 -m pytest tests/ci/model_quality -q -o addopts=''
```

预期：主环境能导入依赖并看到所需 GPU；runtime 能导入对应工具；CPU 控制流测试通过。**这些检查不等于真实量化和上传验收。**

## 3. 触发 pipeline

### 3.1 Buildkite（正式路径）

1. 让目标 GPU 主机的 agent 加入 queue `model-quality-gpu`。queue 名本身不保证各步骤落在同一台物理机，初期建议该 queue 只接这一台，确保代码版本、模型路径、数据路径、运行目录一致。
2. 在 Buildkite 建独立 pipeline 连接本仓库，初始步骤上传仓库内 bootstrap：

```yaml
steps:
  - label: "Load model quality pipeline"
    agents:
      queue: model-quality-gpu
    command: "buildkite-agent pipeline upload .buildkite/model-quality/pipeline.yml"
```

bootstrap 会检查共享目录变量、跑 planner、保存 execution plan，再动态上传模型步骤。agent 的 PATH 必须让 `python3` 指向 §2.2 的主环境。

3. 手动 build 时设置 build 变量：

```text
RUN_MODE=quantize
MODEL_FILTER=deepseek-v4-flash-0731-wna8
MAX_GPU_HOURS=12
PRIORITY_OVERRIDE=none
```

预期 Buildkite 出现 `preflight → quantize → validate → runtime-smoke → report → summary`；配置启用上传时还会多一个 publish。report 和 summary 允许上游失败后继续执行，用来留下报告。

两个注意点：

- 现有 concurrency group 只对使用它的步骤串行限流，**不是**整次 build 的资源锁。同一 run 的多个 build 仍可能在阶段之间交错，不要同时操作同一个 run。
- `plan --upload` 的含义是上传 **Buildkite pipeline**，不是把模型传到 BCE。模型上传由 `upload.enabled` 和运行模式决定。

### 3.2 手动分阶段执行（没有 Buildkite 时）

排障和首次验收建议直接手动跑，等价但更可控。命令与 §1 相同，补充几点：

- 每个阶段的机器可读结果在 `<runs-root>/<run-id>/<model-id>/state/<stage>.json`，人读报告在 `reports/`。
- 最后一步聚合要额外传 `--models-json`：

```bash
python3 -m ci.model_quality.stage --config ci/model_quality/config/models.yaml \
  --run-id "$RUN_ID" --attempt-id "$ATTEMPT" --stage aggregate \
  --run-mode quantize --git-sha "$SHA" \
  --models-json '["deepseek-v4-flash-0731-wna8"]'
```

- `runtime-smoke` 之前必须先有可用的 runtime 服务（见 §4.4）。

### 3.3 复用已有 run：eval_only / upload_only

这两种模式**必须显式传 `--run-id`**。当前默认 bootstrap 既不读 `RESUME_RUN_ID` 也不传 `--run-id`，所以不能只改 `RUN_MODE` 就完成复用，需要加一个自定义初始步骤：

```bash
python3 -m ci.model_quality.plan \
  --config ci/model_quality/config/models.yaml \
  --run-mode eval_only \
  --run-id EXISTING_RUN_ID \
  --model-filter deepseek-v4-flash-0731-wna8 \
  --max-gpu-hours 12 \
  --persist-plan
```

只上传时改成 `--run-mode upload_only`，且模型必须 `upload.enabled: true`。显式 filter 允许选中 `enabled: false` 的模型来复用历史产物。每次计划默认生成新的 attempt ID，同一 run 的 source/model/input/artifact manifest 保持不变。

## 4. 配置模型

### 4.1 清单与 include

复制 `ci/model_quality/config/models.yaml` 里的 Qwen3 示例，或通过根清单的 `includes` 引入独立文件（路径相对于清单文件）：

```yaml
schema_version: 1
includes:
  - models/my-model.yaml
models: []
```

仓库里现成的参考：

- `config/models/deepseek_v4_flash_0731_wna8.yaml`：实际在跑的 DeepSeek-V4-Flash WNA8 定义（含 xSGL runtime smoke）。
- `config/models/deepseek_v4_wna8.example.yaml`：量化入口所有参数的写法示例，默认 disabled。

### 4.2 字段

| 字段 | 配置内容 | 预期作用 |
|---|---|---|
| `id` / `enabled` | 唯一 ID；准备完成后改为 true | 决定模型是否进入计划 |
| `priority` | P0–P3 | 排序 |
| `source.path` / `revision` | 本地 checkpoint 和版本标识 | 记录源模型身份 |
| `workflow.revision` | 量化入口 commit 或镜像 digest | 量化实现变化时使身份变化；必须由部署方维护 |
| `workflow.quantize` | argv 数组 | 启动量化入口 |
| `resources` | 卡数、GPU-hour、超时、磁盘等 | 预算选择及部分预检 |
| `validation` | required files、profile、streaming、量化后缀等 | 定义产物检查要求（见 §4.5） |
| `runtime_smoke` | Python 或 command、runtime revision、prompts、TP | 执行真实模型加载和最小生成（见 §4.4） |
| `evaluation` | command、result_file、runtime_revision | 可选精度评测（见 §7） |
| `upload` | 新模型先 `enabled: false`；DeepSeek 参考模型已开启 | 控制是否发布到 BCE（见 §8.1 已验证示例） |

- `resources` 常用键：`gpu_count`（量化所需，必填）、`estimated_gpu_hours`（必填，用于预算）、`runtime_gpu_count`（smoke 用卡数）、`evaluation_gpu_count`、`estimated_eval_gpu_hours`、`timeout_hours`、`minimum_free_disk_gib`、`host_memory_gib`、`io_weight`、`required_capabilities`。
- `workflow` 可选键：`env`（追加到量化进程的环境变量）、`cwd`、`forbidden_argument_pairs`、`exactly_one_argument_groups`（后两个用来拦掉入口不接受的参数组合）。
- 不要保留 `replace-with-*` 之类的版本占位符。
- Qwen3 CI 入口本身也使用 streaming，应为它设置 `validation.streaming: true` 并按实际产物要求配置 `FINALIZED`。

### 4.3 命令写法

命令必须是 argv 数组，一个元素一个参数：

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

- 不能写成一条带管道或重定向的 shell 字符串。
- 多值参数（`nargs="+"`）逐项列出，放在 flag 之后、下一个选项之前。
- 布尔开关按入口的 `--flag` / `--no-flag` 语法填写；省略即用脚本默认值。
- JSON 花括号无需转义：`{"name":"exact_match"}`、`a{2,4}` 都按原样保留，只替换 CI 自己拥有的占位符。
- 校准数据参数必须按入口实际支持的格式设置；示例可能访问外部数据集，离线主机需要事先准备数据或缓存。

各命令可用的占位符：

| 命令 | 可用占位符 |
|---|---|
| `workflow.quantize` | `{source_path}`、`{run_dir}`、`{output_dir}`、`{work_dir}`、`{logs_dir}`、`{reports_dir}` |
| `runtime_smoke.command` | 同上 |
| `evaluation.command` | 同上（其中 `{reports_dir}` 指向 attempt 内报告目录） |
| `upload.commands` / `verify_commands` / `success_commands` | `{output_dir}`、`{reports_dir}`、`{remote_run_prefix}`、`{model_id}`、`{run_id}` |

### 4.4 runtime_smoke 的两种运行时

**方式一：vLLM（默认）** 用 `runtime_smoke.python`（或 `VLLM_PYTHON_ENV`）执行 `ci.model_quality.vllm_smoke`，由 vLLM 加载产物并生成固定 prompts。预检会检查该 Python 能否 `import vllm`。

**方式二：自定义 command** 目标主机不用 vLLM 提供产物时，配置 `runtime_smoke.command`（argv 数组，占位符规则同 §4.3），替换内置 vLLM 路径：

```yaml
runtime_smoke:
  enabled: true
  runtime_revision: sglang-0.5.10+44a64804b
  prompts:
    - The capital of France is
  command:
    - python3
    - -m
    - ci.model_quality.xsgl_smoke
    - --model
    - "{output_dir}"
    - --base-url
    - http://127.0.0.1:30000
    - --served-model-name
    - deepseek-v4-flash
    - --prompts-json
    - '["The capital of France is"]'
    - --wait-ready-seconds
    - "1800"
    - --output
    - "{reports_dir}/runtime-smoke-output.json"
  result_file: "{reports_dir}/runtime-smoke-output.json"
```

契约：

- 配了 `command` 后，预检只校验 argv 第一个元素是可执行文件，不再要求 `import vllm`。
- 命令必须自己写出 `result_file`，内容形如 `{"status": "PASS", "outputs": [...]}`。退出码为 0 但没有结果文件、结果里 `status` 不是 `PASS`、或 `outputs` 为空，都判定 `RUNTIME_SMOKE_FAILED`。
- `runtime_revision` 仍然必填，用于标识运行时身份。

**xSGL 适配器** `ci.model_quality.xsgl_smoke` 查询一个**已经在运行**的 SGLang OpenAI 兼容服务：

- `--base-url` 指定服务地址；`--wait-ready-seconds` 用于轮询 `/health` 直到就绪。
- `--served-model-name` 要与 `sglang serve --served-model-name` 一致。
- 当前固定的 SGLang 版本不返回输出 token id，因此适配器会用 checkpoint tokenizer 重新编码生成文本，并在记录里用 `token_ids_source: local_checkpoint_tokenizer` 标明来源；tokenizer 不可用时该字段为 null，`completion_tokens` 仍来自服务端 usage。

服务生命周期由部署方管理——CI 控制面容器通常没有 docker 访问权限。仓库提供 `ci/model_quality/xsgl_launcher.sh`，在**宿主机**上从服务容器拉起服务并等待 `/health`：

```bash
bash ci/model_quality/xsgl_launcher.sh <model_dir> [port] [container] [ready_timeout_seconds]

# 例：把刚量化出来的产物喂给 xSGL 容器（在宿主机上执行）
bash ci/model_quality/xsgl_launcher.sh \
  /ssd2/model-quality/runs/$RUN_ID/deepseek-v4-flash-0731-wna8/model \
  30000 zhuang_xsgl_0923 2400
```

两个容器共享 host network 时，`http://127.0.0.1:30000` 在控制面容器里可直接访问。

### 4.5 validation 检查项

`validation` 决定 validate 阶段查什么：

| 键 | 含义 |
|---|---|
| `profile` | 目前只实现 `causal_lm` |
| `required_files` | 必须存在的产物文件，如 `config.json`、`model.safetensors.index.json`、`recipe.yaml`、`FINALIZED` |
| `streaming` | 为 true 时缺失 `FINALIZED` 直接判 FAIL；否则只记 warning |
| `require_quantization_config` | 产物 config 里必须有非空 `quantization_config`；若写了 `quantization_status`，必须是 `compressed` |
| `min_quantization_auxiliary_tensors` | 量化辅助张量（scale/zero_point/packed）的最小数量 |
| `quantization_auxiliary_suffixes` | 辅助张量的命名后缀，默认 `.weight_scale`、`.weight_zero_point`、`.weight_packed`、`.weight_compressed` |
| `quantization_scale_suffixes` | 参与"非正 scale"门禁的后缀，默认 `.weight_scale`、`.weight_zero_point` |

validate 还会检查 index 与实际张量是否一一对应、有无 NaN/Inf、scale 是否非正。

**原始格式要显式配后缀。** 例如 DeepSeek 原始 checkpoint 的量化 scale 叫 `...scale` 而不是 `.weight_scale`；不配的话辅助张量数会是 0、validate 直接 FAIL，而且"非正 scale"门禁会静默空转：

```yaml
validation:
  min_quantization_auxiliary_tensors: 1
  quantization_auxiliary_suffixes:
    - .weight_scale
    - .weight_zero_point
    - .weight_packed
    - .weight_compressed
    - .scale
  quantization_scale_suffixes:
    - .weight_scale
    - .weight_zero_point
    - .scale
```

## 5. 先生成计划，不启动模型

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

`--persist-plan` 会把 execution plan 同时写进 `<runs-root>/<run-id>/execution-plan.json`，让 aggregate report 保留预算和 deferred 信息；用默认 bootstrap 时必须加。

此命令只校验配置并生成计划，不执行硬件 preflight、量化、评测或上传模型。检查 selected 非空、模型 ID 正确、预算合理后再运行。

排序当前主要是 P0–P3、预计量化成本和模型 ID；预算不足的任务标为 `DEFERRED_BUDGET`。GPU-hour 是估算，不是强制运行时配额，也不是完整的 RAM/I/O 调度器。

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
| runtime-smoke | vLLM 或配置的 command 加载/访问产物并生成固定 prompts | token ID、生成文本、runtime 日志 |
| evaluate | 执行 evaluator 或使用匹配缓存 | attempt 内 evaluation JSON；门禁按原始未舍入值比较 |
| report | 汇总本 attempt 状态 | attempt 报告、共享报告视图、current-attempt 指针 |
| publish | 身份和内容检查、上传、verify、最后提交标记 | 成功后生成并上传 SUCCESS；返回远端 run 前缀 |
| aggregate | 读取 promoted attempt 和发布状态 | run 根目录 aggregate-report.json |

`quantize_and_eval` / `eval_only` 必须配置 evaluator；只有退出码 0 而没有新结果文件不能通过。`quantize` 的 PASS 只表示本模式必需阶段通过，不表示完整精度已通过。

## 7. 接入精度评测

可使用 `ci.model_quality.evaluators.lm_eval_pair`，其会在独立子进程中分别评估 base 和 compressed。参考 DeepSeek 示例中的 evaluation 配置，至少指定：

- runtime Python 和固定 runtime revision；
- base/compressed 路径；
- task、metric 名称、方向与门槛；
- fewshot、limit、seed、并行参数；
- `--output "{reports_dir}/evaluation-raw.json"` 与相同的 `result_file`。

higher 指标可用 `min_recovery`、`absolute_floor`；lower 指标可用 `max_relative_increase`。首次先运行小规模任务验证链路；小样本分数不能当作完整发布基线。task/metric 名称必须在固定 lm-eval 版本上确认。

## 8. 接入 BCE 上传

先保持上传关闭，在目标机器确认 `bcecmd` 版本、凭证注入和具体命令。不能把本文当作某个 bcecmd 版本的参数规范。

启用时需要三组非空命令：

```yaml
upload:
  enabled: true
  remote_prefix: bos:/YOUR_BUCKET/model-quality
  allowlist: [model, reports]
  commands: []          # 上传产物
  verify_commands: []   # 校验远端确实存在
  success_commands: []  # 最后写 SUCCESS 标记
```

上述空数组是待填写模板，启用后保持为空会校验失败。

当前 DeepSeek V4 WNA8 配置已启用上传。生成的 Buildkite pipeline 在 `quantize`、`quantize_and_eval`、`eval_only` 模式下都会加入 publish 步骤；`quantize` 并不表示禁止上传。仅量化或评测的实验应复制一份配置，将对应模型的 `upload.enabled` 设为 `false`，并在 plan 和所有 stage 中使用同一份配置。手动逐阶段执行时，上传需要显式执行 publish。

该模型的 `workflow.revision` 已修正为完整提交 `c5921a139d9f076d07e26bfedec3fd88bf75c2ec`。这会改变 request fingerprint，并连带改变 evaluation fingerprint；后续量化应重新生成计划并使用新的 run。已发布的 `20260924T235322Z_c5921a139d9f` 使用当时冻结的身份，当前配置不能复现它；不要改写旧 run 的 plan、manifest 或 SUCCESS 标记来匹配新配置。

三组命令分别完成：

1. `commands`：把 `{output_dir}` 上传到 `{remote_run_prefix}/model`，把 `{reports_dir}` 上传到 `{remote_run_prefix}/reports`。
2. `verify_commands`：确认远端对象真的存在；缺失时必须返回非零。
3. `success_commands`：最后上传 `{run_dir}/commit-markers/SUCCESS.json` 至 `{remote_run_prefix}/SUCCESS.json`。

所有命令必须调用 bcecmd。数据上传本地路径必须为 allowlist 内绝对路径；不要传 `.`、`..`、`model` 这样的相对路径，也不要传源模型、work 或 run 根目录。当前 argv 校验支持固定的操作名及选项；带独立选项值的命令应在部署前确认可通过校验。

### 8.1 一个已在真机验证的示例

下面这段在 node16 `llm-quant-base` 上完整跑通（bcecmd v0.5.1，远端根 `bos:/klx-public/llm-demo/quant`），可以直接抄：

```yaml
upload:
  enabled: true
  remote_prefix: bos:/klx-public/llm-demo/quant
  allowlist: [model, reports]
  commands:
    # `cp -r <dir> <dst>` 复制的是 <dir> 的内容（不会多套一层目录），
    # 所以 object key 与 allowlist 一一对应。
    - [bcecmd, bos, cp, -r, -y, --quiet, --disable-bar, "{output_dir}", "{remote_run_prefix}/model"]
    - [bcecmd, bos, cp, -r, -y, --quiet, --disable-bar, "{reports_dir}", "{remote_run_prefix}/reports"]
  verify_commands:
    - [bcecmd, bos, cp, -y, --quiet, --disable-bar, "{remote_run_prefix}/model/config.json", /tmp/mqci-verify-model-config.json]
    - [bcecmd, bos, cp, -y, --quiet, --disable-bar, "{remote_run_prefix}/reports/summary.json", /tmp/mqci-verify-reports-summary.json]
  success_commands:
    - [bcecmd, bos, cp, -y, --quiet, --disable-bar, "{run_dir}/commit-markers/SUCCESS.json", "{remote_run_prefix}/SUCCESS.json"]
```

动手前先用 `bcecmd bos cp --help` 确认本机版本的参数名（`-r/--recursive`、`-y/--yes`、`--quiet`、`--disable-bar`、`--concurrency`、`--restart`、`--storage-class` 等），不要照抄别的版本。

### 8.2 三个容易踩的坑

- `bcecmd bos ls` 对**不存在的路径也返回 0** 且不打印内容，不能当作存在性校验。`bcecmd bos cp` 下载不存在的对象会返回非零、且不落任何文件，所以 verify 用「下载一个小的代表性对象」实现。
- 单对象 `cp` 的目标是**完整 object key**，不是目录；对目录才用 `-r`。
- `bcecmd bosapi get-object-meta --bucket-name B --object-name O` 也能在缺失时返回非零，但要求把 bucket 和 object 拆开，而配置里只有 `{remote_run_prefix}` 整体，无法拆分，只能硬编码 bucket/run-id，不适合复用。

程序不会解析 bcecmd 输出判断校验是否充分，也不会逐对象比对 SHA256（149G 级别不现实）。`reports/upload-manifest.json` 冻结了每个文件的 `path`、`size_bytes` 和 `sha256`，需要人工复核时以它为准；上传命令的退出码加上对象大小核对是自动化层面的保证。

### 8.3 跑 publish

```bash
bash stage.sh publish
```

`publish` 不需要重跑上游阶段：它自己会重新校验 `current-attempt.json`、`reports/summary.json`、`artifact-manifest.json` 的身份指纹和产物内容指纹，任何一项对不上都会以 `DEPENDENCY_FAILED` 失败。`--run-mode upload_only` 要求 `upload.enabled: true`，否则直接报 `CONFIG_ERROR`。

预期远端路径为 `<remote_prefix>/<model-id>/runs/<run-id>/`，包含 `model/`、`reports/` 和 `SUCCESS.json`。`reports/upload-manifest.json` 与 `SUCCESS.json` 不计入冻结清单，但会随目录一起上传。SUCCESS 仅在 verify 成功后创建，上传重试会检查冻结清单是否与本次一致。当前没有自动 latest/production 指针晋升功能，也没有自动远端不可覆盖保证；需用目标存储权限或经过验证的命令确保同一 run 不被不同内容覆盖。凭证只从 agent 环境注入。

## 9. 查报告与失败处理

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

优先从 `aggregate-report.json` 找到失败模型，再读 `current-attempt.json` 指向的 attempt 的 `summary.json`、`state/` 和日志。quantization/evaluation 日志按 attempt 保存在 `logs/<attempt-id>/`；runtime-smoke/publish 的部分日志仍在共享 `logs/` 目录。报告保存在磁盘，当前 pipeline 不会自动把这些文件作为 Buildkite artifact 上传。

常用排查命令：

```bash
R="$MODEL_QUALITY_RUNS_ROOT/$RUN_ID/<model-id>"

python3 -c "import json;d=json.load(open('$R/reports/summary.json'));print(d['status'], d['stages'])"
cat "$R/attempts/$ATTEMPT/reports/summary.md"
tail -50 "$R/logs/$ATTEMPT/quantization.log"
tail -50 "$R/logs/runtime-smoke.log"
python3 -m json.tool "$R/state/validate.json" | head -40
```

阶段失败通常表现为 `status: FAIL` 加 `reason_code`，并非所有原因码都是独立状态：

| 表现 | 下一步 |
|---|---|
| selected 为空 | 检查 enabled/filter；全 deferred 时检查预算 |
| PREFLIGHT_FAILED | 查看 failures；修路径、磁盘、GPU、runtime 或身份不一致 |
| CONFIG_ERROR | 修 manifest/argv/evaluator 配置后重新生成计划 |
| QUANTIZATION_FAILED / TIMEOUT | 查量化日志；保留 work；恢复是否可行取决于入口 checkpoint 支持 |
| VALIDATION_FAILED | 检查缺文件、tensor、NaN/Inf、scale；不要跳过门禁 |
| RUNTIME_SMOKE_FAILED | 检查量化格式、runtime 版本、TP、GPU 内存，以及 runtime 服务是否真的起来了 |
| QUALITY_GATE_FAILED | 查看原始指标及门槛；不要无限重跑选最好结果 |
| DEPENDENCY_FAILED | 查最早失败阶段及 manifest/fingerprint |
| REMOTE_VERIFY_FAILED | 不发布成功标记；先检查远端缺失/损坏对象 |
| EVALUATION_CACHE_HIT | 使用了匹配缓存，未实际重跑 evaluator |

身份不一致类错误（`stale ... fingerprint`、`source checkpoint changed`）说明源内容、量化实现或产物 checksum 变了。正确处理是换新 run 或先解决不一致，**不能靠删除 manifest 绕过检查**。评测配置变化只影响评测身份；修改量化实现时更新 `workflow.revision`。

匹配缓存会返回 `EVALUATION_CACHE_HIT`，包括历史 FAIL。当前没有专用强制重评测开关；不要把创建新 attempt 当作一定会重新执行 evaluator。

## 10. 首次验收清单

按顺序做完这些，才算把这个工具在当前主机上验收通过：

- [ ] §2.3 自检通过（主环境依赖 + GPU 可见 + CPU 控制流测试）
- [ ] 小模型 `quantize` 全链路 PASS
- [ ] `quantize_and_eval` 跑通，门禁按预期判定
- [ ] `eval_only` 复用既有产物跑通（显式 `--run-id`）
- [ ] 测试 bucket 上的 `publish` 跑通并 verify 成功（DeepSeek 模型已在 `bos:/klx-public/llm-demo/quant` 验证）
- [ ] 故意制造一次失败，确认报告和 `reason_code` 可定位

当前尚无完整 recipe-aware 覆盖率、多模态 adapter、联合资源调度和自动发布晋升；这些不应成为操作人员对当前工具的默认预期。
