# DeepSeek-V4.1-Flash 流式量化

采用与 native V4 相同的接入方式：独立模型、MoE 校准 wrapper、
按需权重解码器及流式示例。
实现位于 `src/llmcompressor/modeling/deepseekv41`，复用 V4 的基础参考算子。
原始类型的 AutoConfig fallback 仅在上游未注册 `deepseek_v41` 时生效。

## 当前工作流

```bash
python examples/streaming_oneshot/deepseek_v41_wNa8.py \
  --model-id /path/to/DeepSeek-V4.1-Flash \
  --dataset-id lmms-lab/flickr30k \
  --text-dataset-id HuggingFaceH4/ultrachat_200k \
  --text-calibration-samples 128 \
  --output-dir /path/to/DeepSeek-V4.1-Flash-WNa8 \
  --quant-mode wna8 \
  --num-calibration-samples 128 \
  --max-sequence-length 2048 \
  --batch-size 1
```

示例默认使用稀疏路由；需要让每个专家看到全部 token 时，再添加
`--moe-calibrate-all-experts`。

模型目录须包含索引引用的全部真实分片。主校准源按多模态数据集处理，默认每条
记录需要 `image` 和可选的 `caption` 字段；示例会将图像编码成 DeepSeek V4.1
官方格式的 ViT patches、image token span 和 `image_token_types`，并通过原生
模型的 vision/aligner 路径参与校准。可选的 `--text-dataset-id` 与
`--text-calibration-samples` 会把文本对话追加到每个图像样本后面。没有
`encoding/encoding.py` 时回退到本地 chat template。若模型目录包含
`inference/image_processor.py`，示例直接复用其中的 `load_image` 和
`image_token_types`；tiny fixture 缺少该目录时才使用兼容回退。文本数据仍接受
`messages`、`text` 或 `input_ids`；`--reasoning-effort` 范围为 1..100，默认 75。
`--device cpu` 或 `--device cuda:0` 可显式选择校准设备。
多模态校准当前要求 `--batch-size 1`，每条图像样本只支持一张图片。

量化范围：

| 模式 | routed experts | attention、shared experts、DSpark main_proj |
|---|---|---|
| `wna8` | INT4 权重 / 动态 INT8 激活 | INT8 权重 / 动态 INT8 激活 |
| `w8a8` | INT8 权重 / 动态 INT8 激活 | INT8 权重 / 动态 INT8 激活 |
| `w4a8` | INT4 权重 / 动态 INT8 激活 | INT4 权重 / 动态 INT8 激活 |

`--quantize-engram-wkv` 控制是否额外量化 Engram 的 `wkv` 投影，默认关闭。
开启后，三种主量化模式都为它使用独立的 `engram_w8a8` 组：
INT8 权重、动态 token INT8 激活及 FP32 per-channel weight scale，
与参考 INT4 权重的 Engram 投影格式一致。可用
`--no-quantize-engram-wkv` 显式关闭；关闭时保留执行精度，默认 BF16。
两种设置均保留 FP8 `engram.embed` 及其 UE8M0 scale，不量化
Engram 的 `q_weight`、`k_weight`。

与 V4 示例一致，`--use-float32-scale-dtype` 将所有 PTQ 量化组的 weight
scale 指定为 FP32，适用于三种主量化模式。该选项默认关闭，也可用
`--no-use-float32-scale-dtype` 显式关闭；关闭时普通量化组沿用默认 scale
精度，BF16 执行时通常为 BF16。已开启的 `engram_w8a8` 组仍固定使用 FP32
scale，与参考权重格式保持一致。该参数只控制 PTQ weight scale，不改变
校准执行精度、动态激活量化设置或 FP8 查表的 UE8M0 scale。

具体 target/ignore 以示例 `_schemes` 为准。这里的 INT4 是源 E2M1 FP4 解码后
再次量化；不能将二者的存储字节或数值格式互换。

Vision/aligner、router、norm/mHC、compressor、indexer 控制投影、
grouped `wo_a`、embedding/head、Markov/confidence 头不做额外 PTQ。
普通源 FP8/FP4 权重由 materializer 解码为执行精度；这些未选模块不保证维持
原始存储 dtype。只有 Engram 巨表及其行级 scale 逐字节保留，读取时先按行 gather
再转浮点。若源分片只含待保留表，会优先创建硬链接；否则使用有界缓冲复制。
最终配置分别记录目标 `quantization_config`、原始 `source_quantization_config`
以及 `engram_storage`。

V4.1 示例的 `--save-raw-checkpoint-format` 默认开启：tensor 使用
`layers.*`、`mtp.*`、`head.weight` 和 `.scale` 名称，`config.json` 中的
量化 targets/ignore 由 V4.1 materializer 同步转换到该命名空间。
这项配置转换只用于 V4.1。`--no-save-raw-checkpoint-format`
保留 native 模型的 `model.*`、`model.lm_head` 和 `.weight_scale`。
项目 streaming 推理可通过同一个 materializer 回载这两种格式，并兼容旧产物中
使用 `model.*` targets 的配置。

配置转换只处理选择器，不会改写 index。tensor 保存直接继承 V4 的
`output_tensor_name()` → `_logical_to_raw()`，没有新增名称缓存或保存路径。
最终 `model.safetensors.index.json` 从已写入 shard 的真实 header 生成。集成测试
直接比较源 index 的非 scale 名称、输出 index 和全部输出 shard header，覆盖
主干、MTP、vision/aligner、Engram 和 `head.weight`。

raw tensor 名称和量化 targets 必须一致。否则推理引擎可能将已量化的 DSpark
专家判为浮点模块，按完整输入宽度加载 packed INT4 权重。INT4 两个值合为一个
INT8 字节后，存储形状最后一维减半属于预期行为，不能据此把存储形状改回
浮点权重形状。该导出配置检查不代替目标推理引擎的实际启动验证。

## DSpark 专用校准

普通参考生成路径不会自动执行 DSpark。native forward 显式执行 3 层草稿网络，
从目标主干层执行前的 residual 均值构建上下文，并采用各样本的绝对 RoPE 位置。
窗口、padding 和 Single-Pass pre-mix 均作为输入/输出明确传递。

每条样本预留一段末尾真实 token 作为 teacher tokens。草稿输入为一个真实 seed
和配置中的 noise tokens；Markov/confidence 头使用对应 teacher token。
草稿 MoE 按 128 experts/top-3 独立构造，校准时始终覆盖所有草稿专家。
这条路径用于收集 PTQ 统计，不实现生产 speculative decoding 的接受/拒绝逻辑。

## 验证与限制

常规测试：

```bash
python -m pytest tests/llmcompressor/modeling/test_deepseek_v41.py -q
```

可选的独立参考对齐测试：

```bash
DEEPSEEK_V41_REFERENCE_DIR=/path/to/DeepSeek-V4.1-Flash/inference \
python -m pytest tests/llmcompressor/modeling/test_deepseek_v41_reference.py -q
```

参考对齐使用相同的随机缩小 BF16 权重，以 CPU 等价算子替换 GPU kernel，
对比主干、DSpark 各层、草稿 logits 和 confidence；包括短序列、ratio=2 的
不完整分组及 DSpark 窗口回绕。Engram/vision 不包含在这项参考数值比较中，
另有 hash、按行解码/保留、视觉形状及 image routing 单元检查。
常规测试还检查 FX 分段执行与直接执行一致、三种量化模式的 INT4/INT8 实际导出，
以及有/无 checkpoint-progress 的流式路径。

真实权重/GPU kernel、长上下文下的内存和速度、生产 KV-cache/replay、
DSpark 接受率及目标推理引擎加载需要在目标环境中分别验证。模型类同时支持
`encode_image`、预构建的 `inputs_embeds/image_mask` 和校准入口使用的
`image_patches/image_token_types`。导出保留原始键名和 Engram FP8 表；目标
引擎仍需支持该架构、目标 INT4/INT8 布局及混合格式。AutoConfig 能解析输出
不等于已验证推理可加载。

## 完整层分布的 tiny 模型

创建工具默认生成 32,202,678 参数的 fixture，保留 40 层主干、32 层 ViT、
3 层 DSpark、两处 Engram、全部 CSA2 来源/重索引/复用分布及 129,280 词表。
缩小 hidden size、专家数、Engram 桶和运行时序列长度，以适合 CPU 测试。

创建新 fixture：

```bash
python tools/create_tiny_dsv41.py \
  --source /path/to/DeepSeek-V4.1-Flash \
  --output /path/to/new-tiny-fixture
```

执行完整校验及三种量化：

```bash
python tools/validate_tiny_dsv41.py \
  .cache/tiny-models/DeepSeek-V4.1-Flash-Tiny
```

目录包含 `native-finetuned/`、分片的 `source-fp8-fp4/`、
`wna8/`、`w8a8/`、`w4a8/`、校准数据、README、`creation.json`、
`validation.json` 和各量化日志。已有 `FINALIZED` 的量化输出会复用并重新校验。
raw/量化产物通过工具中的 CPU reference loader 解码，native checkpoint 可用
`AutoModelForImageTextToText` 加载，需传入对应 tokenizer 以重建 Engram 哈希。

工具使用 toy 文本进行微调与困惑度检查，用于验证模型可学习和量化流程可运行。
视觉和 DSpark 不参与微调；图像路径使用合成 patches 和预构建 embeddings。
这些检查不替代真实模型、GPU kernel、生产引擎或量化质量验证。
