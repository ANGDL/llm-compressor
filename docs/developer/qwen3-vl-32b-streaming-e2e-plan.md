# Qwen3-VL-32B Streaming PTQ 端到端测试计划

目标模型：

```text
/ssd4/models/Qwen3-VL-32B-Instruct
```

执行环境：

```text
node16 / llm-quant-base
代码：/data/quant/llm-compressor
GPU：8 × 96 GiB
```

测试脚本：

```text
scripts/e2e_qwen3_vl32b_streaming.py
scripts/run_qwen3_vl32b_e2e_node16.sh
```

脚本使用确定性合成图像，避免端到端测试依赖外部数据集。图像尺寸在
`224×224`、`336×224`、`448×256`、`256×384` 之间变化，校准 batch 固定为
1，processor 输出保留 `input_ids`、`attention_mask`、`mm_token_type_ids`、
`pixel_values` 和 `image_grid_thw`。因此测试覆盖真实多模态 prefix、变长视觉
token、processor identity 和 decoder-only 量化。

## 执行顺序

### 0. Preflight

确认 checkpoint 是标准 `qwen3_vl`、14 个 safetensors shard 可见、目标 GPU
可用，并记录 checkpoint 大小、GPU 和模型架构。不加载 32B 权重。

```bash
STAGE=preflight \
scripts/run_qwen3_vl32b_e2e_node16.sh
```

### 1. Processor 与 calibration snapshot

加载 `AutoProcessor`，构造 8 个确定性图像样本，执行 chat template 和视觉
预处理，记录字段 shape、图像尺寸及校准身份摘要。

```bash
STAGE=prepare \
scripts/run_qwen3_vl32b_e2e_node16.sh
```

验收：所有样本都包含图像输入；`image_grid_thw` 与 `pixel_values` 存在；
不同分辨率能够被 processor 接受；报告中有非空 64 位 calibration fingerprint。

### 2. Streaming RTN/W8A16 主路径

使用标准 `QuantizationModifier`，只量化文本 decoder 的 Linear；视觉塔、
embedding、norm 和 `lm_head` 保留原始权重。启用 `checkpoint_progress=True`，
验证 prefix boundary 事务、decoder 子图事务、输出组装和最终配置。

```bash
STAGE=quantize \
scripts/run_qwen3_vl32b_e2e_node16.sh
```

默认输出：

```text
/data/quant/e2e/qwen3-vl-32b-streaming/output-rtn-w8a16
/data/quant/e2e/qwen3-vl-32b-streaming/work-rtn-w8a16
/data/quant/e2e/qwen3-vl-32b-streaming/report-rtn-w8a16.json
```

验收：

- `FINALIZED` 存在；
- 输出可被 `Qwen3VLForConditionalGeneration.from_pretrained(...,
  local_files_only=True)` 重载；
- 视觉塔、projector、embedding、norm、`lm_head` 的原始 dtype 和权重完整；
- 量化模块只位于文本 decoder；
- report 记录总耗时、输出字节数和 GPU/RSS 峰值。

### 3. 重载与图像生成 smoke

先加载原始模型，在同一张确定性合成图像上记录最后位置 logits 的 top-10
和 greedy continuation；释放原始模型后加载量化模型，重复同一输入。

```bash
STAGE=validate \
scripts/run_qwen3_vl32b_e2e_node16.sh
```

验收分三层：

1. 量化模型成功重载，配置和 processor 输入契约有效；
2. logits 全部有限，首选 token 和原始模型 top-10 有记录；
3. 生成结果有非空文本，并记录 token overlap。量化模型不要求逐 token
   完全一致，最终质量需要独立数据集评测。

### 4. GPTQ 第二轮

RTN 路径通过后再运行 GPTQ，避免首轮同时引入统计算法和多模态 trace 问题。
GPTQ 使用同一组固定输入，输出目录和 report 自动切换到 `gptq-w8a16`。

```bash
ALGORITHM=gptq \
scripts/run_qwen3_vl32b_e2e_node16.sh
```

GPTQ 额外验收 Hessian 统计完成、目标层不出现空样本，并比较 RTN/GPTQ 的
报告、峰值显存、耗时和生成结果。GPTQ 仍沿用现有 modifier，不修改其实现。

## 失败处理与资源约束

- `/ssd4` 只读模型源，输出和 work 放到 `/data/quant/e2e`，避免 95% 磁盘
  使用率下继续产生临时 shard。
- 首轮只使用 8 个样本和 512 token；通过后再扩大到 32/128 样本。
- 发生 OOM 时保持同一 recipe，先把 `--samples` 降到 4，再检查单层
  admission；不要改成整模型常驻。
- 发生 processor 或 trace 错误时保留 report 和 work，先查看实际输入字段；
  不要删除 checkpoint-progress 事务目录。
- 如果量化阶段中断，使用相同 `ROOT_PATH` 和参数重跑，检查已提交的
  `prefix-00000` 与 `subgraph-*` 是否被恢复，而不是重新建立全新 work。

## 结果记录

每个算法独立记录：

```text
report-<algorithm>-w8a16.json
calibration-<algorithm>.json
output-<algorithm>-w8a16/
```

最终报告至少包含模型源 fingerprint、calibration fingerprint、recipe、
量化模块范围、完成标记、输出大小、耗时、GPU/RSS 峰值、重载结果、logits
top-10、生成文本和 token overlap。质量结论应基于独立图像问答集合，不能只用
校准样本的生成文本判断。

本轮重点验收 E6 的多模态执行、prefix 恢复、输入/processor identity 和
decoder-only 导出；E7 的候选分支在固定 boundary API 上单独验收，不把
32B 全模型候选搜索混入首轮主路径。

## Node16 实测结果

2026-11-06 在 `node16 / llm-quant-base` 使用
`/ssd4/models/Qwen3-VL-32B-Instruct` 完成了 RTN/W8A16 主路径：

- preflight 识别为标准 `qwen3_vl`，14 个 safetensors shard，源 checkpoint
  约 66.7 GB；
- 8 个确定性合成图像样本全部通过 processor，字段包含
  `input_ids`、`attention_mask`、`mm_token_type_ids`、`pixel_values` 和
  `image_grid_thw`；
- 64 个文本 decoder 层、448 个 Linear 模块完成量化，最终输出 14 个 shard，
  约 36.0 GB，`FINALIZED` 存在；
- 量化 checkpoint 可重载，448 个量化模块全部位于
  `model.language_model.layers.*`，视觉塔、embedding、norm 和 `lm_head`
  未被量化；
- 原模型与量化模型在固定图像上的 16-token greedy continuation 完全一致，
  首 token top-10 一致，token overlap 为 `1.0`；
- RTN 量化阶段耗时约 278 秒，量化单层峰值显存约 2.87 GiB，重载生成阶段
  峰值约 63.6 GiB；
- 报告位于
  `/data/quant/e2e/qwen3-vl-32b-streaming/report-rtn-w8a16.json`。

GPTQ/W8A16 也已启动并连续完成到第 22/64 层，首层 Hessian 工作集峰值约
29.3 GiB，未发生 OOM。由于完整 GPTQ 轮次预计需要较长时间，本次在第 23 层
开始处停止；`work-gptq-w8a16` 中的 checkpoint-progress 事务和 boundary
文件保留，`output-gptq-w8a16` 尚未发布。恢复时使用相同参数运行
`STAGE=quantize`，不要带 `OVERWRITE=1`。
