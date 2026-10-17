# Streaming PTQ 推理能力：项目评估与实施方案

日期：2026-10-17。状态：通用入口、按需反量化及 V4.1 增量适配已实现；
GPU/原尺寸验收和旧版 native V4 cache 适配待完成。

实现入口为 `llmcompressor.streaming.load_streaming_model()`，使用方式与当前
能力边界见 [推理指南](../guides/streaming_inference.md)。
Llama/Qwen3 复用原生 Cache，V4.1 使用其模型专有的 CPU request cache。
本地回归已覆盖真实 streaming PTQ 输出接入推理、逐 token decode、
工作集卸载、失败清理、量化格式及独立原始实现对照。

下文保留设计阶段的基线分析、探针和后备方案。第 2、3 节中
“当前模型无 cache”等表述描述的是实现前的检查结果，不代表现版本状态。
完整 tiny 的流式/常驻执行一致；量化舍入敏感的 cache/recompute 数值诊断
另见推理指南，不能将加载一致性直接视为所有数值检查通过。

已根据后续 offload 与量化执行检查修订为“通用接口，优先复用原生推理；
无匹配量化内核时，按当前工作集在加载阶段反量化为 BF16”。
检查证据、API 边界和探针结果见
[offload 复用评估](streaming-inference-offload-reuse.md)。
默认复用 Transformers forward/generate/cache 和 Accelerate 权重 hooks；
本文中的独立执行计划及 state store 仅作为具体缺口出现后的后备设计。

基线：`feat/deepseek-v41-native-quantization`，提交 `29af84e1b`。
设计阶段仅分析项目、执行临时 CPU 探针并编写方案。
工作区原先暂存的三个 V4 文件不属于本方案的实现前提。

## 1. 结论与目标

可行，建议在 `llmcompressor.streaming` 内增加薄推理入口，优先复用
Transformers 原生 forward/generate/cache 和 Accelerate CPU/disk offload。
公共入口不绑定 DeepSeek：标准 HF 模型使用 AutoConfig/AutoModel 与原生
cache，特殊模型通过已有注册机制或小范围 adapter 扩展。
新增工作主要是缺失的模型增量适配、特殊 checkpoint 桥接和数值对照。
现有 subgraph 权重 session 保留为特殊图/严格资源需求的后备能力。

目标是提供一个资源受控、结果可检查的参考推理器：

- 正常执行 prefill，然后每次仅输入新增 token 进行 decode。
- 计算设备同时只安装当前 subgraph 的权重。
- 模型适配和量化产物都能通过相同的推理流程验证。
- 能定位首个产生异常的 subgraph、token、cache owner 或量化模块。
- 支持原始浮点、原始 DeepSeek FP8/FP4 和本项目 INT8/INT4 PTQ 输出。
- 量化执行能力按 module 的 scheme、格式、设备和 backend 判断；
  没有匹配内核时，只在加载当前权重时反量化到 BF16，默认保留 A8 Q/DQ。

推理速度受权重读取和传输影响。性能优化按验证需要逐步加入，首先保证
执行语义、checkpoint 解码和内存所有权可以独立检查。

这里的“一个 subgraph”是可执行片段，不强制等于完整 decoder block。
大 MoE 优先利用现有 module/Expert 级 offload，按需评估更复杂的分区。

## 2. 项目现状与证据

### 2.1 可以复用的设施

| 现有设施 | 已有能力 | 推理中的用途 |
| --- | --- | --- |
| `streaming/checkpoint/weight_source.py` | 按 tensor 读取 safetensors | 所有权重输入 |
| `streaming/materialization/` | 命名转换、源权重解码、dtype policy | CPU 准备执行权重 |
| `streaming/loading/target.py` | meta 模型、加载并恢复 module 状态 | 当前工作集安装与卸载 |
| `streaming/loading/session.py` | subgraph working set、prepare/move/install | 单计算设备权重生命周期 |
| `streaming/loading/prefetch.py` | 至多一个 CPU 预取任务 | 准备下一工作集 |
| `streaming/loading/host_memory.py` | 主存 reservation | 权重、cache 和边界预算 |
| `pipelines/sequential/helpers.py` | `Subgraph`、图分区、变量最后使用点 | 可复用的图表达和部分分区工具 |
| `tools/validate_tiny_dsv41.py` | tiny 原始/PTQ 解码及激活 Q/DQ 参照 | 推理读取与数值验收基准 |

`SubgraphWeightSession.working_set()` 能识别 `call_module` 和一部分
`get_attr` 依赖。autowrap 隐藏的调用仍需显式声明或发现；目前 prefix
已有运行时依赖发现机制。推理计划需要覆盖所有阶段，包括最终输出。

### 2.2 不能直接作为推理执行器的部分

1. `trace_subgraphs()` 使用 `calibration_forward_context()`。
   该上下文关闭 cache、替换 LM head，并强制校准所用 attention 行为。
2. `populate_concrete_args()` 对未显式提供的 `use_cache` 固定取 `False`。
   这类 trace 不能通过运行时传 `True` 自动变为有状态推理。
3. `trace_streaming_boundaries()` 投影出校准 target 所需的边界，
   不承诺保留最终 logits 和生成输出。
4. `BoundaryActivationStore` 明确拒绝非空 KV cache 字段和不透明对象。
   它适合独立校准 batch，不适合直接承载 request 的可变 cache。
5. PTQ pipeline 的顺序是“当前层处理所有校准 batch，再下一层”。
   自回归推理的顺序是“当前 token/chunk 通过所有层，再处理下一 token”。
6. PTQ 有 modifier 初始化、校准、传播、压缩和写出生命周期。
   推理执行器只需要读取和 forward，应独立编排。

因此推理入口优先调用原生模型，不复用校准 trace。
`streaming_oneshot()` 的量化行为和入口继续保持原有职责。

### 2.3 当前模型适配的具体限制

**V4.1**

- `Attention.forward()` 从 `freqs_cis[:sequence_length]` 取位置，
  没有增量位置偏移或历史 window KV。
- `Compressor.forward()` 只处理当前完整序列，未保存未完成 group。
- `Indexer.forward()` 的可见长度按当前输入长度生成。
- Engram hash 只读取当前输入，没有跨调用 token history。
- 顶层 forward 通过 `**kwargs` 接收未声明参数，但没有使用传入 cache，
  返回的 `CausalLMOutputWithPast` 也没有有效 `past_key_values`。
- `calibrate_dspark=True` 会执行 teacher-forced draft 分支。
  普通自回归推理需要明确选择 backbone 模式。

**V4**

- 内部 attention/compressor 已含 `start_pos` 分支和运行时 cache，
  但顶层 CausalLM forward 未向内部转发增量位置或 request cache。
- `_reinitialize_non_persistent_buffers()` 会重置 KV 和 compressor 状态。
  当前 session 安装工作集时会调用模型初始化钩子，不能将 request cache
  的生存期绑定在这类初始化钩子上。

继承 `GenerationMixin` 本身不足以证明任一模型已经实现正确的 cache decode。
现有 V4.1 tiny 验证中的生成使用 `use_cache=False`，不覆盖这一能力。

## 3. 本轮诊断结果

环境版本：PyTorch `2.12.1`，Transformers `5.19.0`，
compressed-tensors `0.17.2.a20260806`。当前环境无可用 CUDA/MPS。
以下是临时诊断程序的结果，不代表已经交付推理 API。

### 3.1 现有权重 session 可以用于正常 prefill/decode

构造两层 tiny Llama：hidden=16，head=2，词表=32。
使用同一 checkpoint，显式划分 embedding、单 decoder、norm/head 阶段，
调用现有 `SubgraphWeightSession.loaded_modules()`：

| 检查项 | 结果 |
| --- | --- |
| 五 token prefill 对照完整模型 logits 最大绝对误差 | `0.0` |
| 随后一个 token decode 对照完整模型最大绝对误差 | `0.0` |
| 两层 cache 的累计长度 | `[6, 6]` |
| 每次 decoder 执行只安装一个 decoder 的权重 | 是 |
| 完成执行后所有参数恢复 meta | 是 |

这个探针验证 CPU 上的执行顺序与权重生命周期。GPU 的 cache 搬运、
stream 同步和峰值显存仍需单独验证。

### 3.2 校准 trace 不能直接提供完整推理结果

同一个 tiny Llama 经现有 tracing 得到三段图。prefix 的 embedding 调用
被 autowrap 隐藏，需要现有 prefix 依赖发现才会加载。
末段图虽然含 decoder、norm 和 LM head 调用，最终输出却是 `return {}`。

这说明即使遍历 `adapter.plan.subgraphs`，也不能假定能获取 logits。
推理计划必须显式保留结果，并声明 prefix/suffix 的实际权重依赖。

### 3.3 当前 V4.1 不具备增量 cache 合同

使用已有五层 tiny V4.1 配置（开启 Engram，关闭 DSpark）：

- `use_cache=True` prefill 后没有返回 cache。
- 带上该结果并传 `position_ids=[[5]]`，单 token 调用仍无 cache。
- 单 token 输出与完整六 token 上下文最后位置的 logits 最大差为
  `0.1873918623`。

该数值用于暴露缺失合同，不是精度门槛或真实模型质量指标。

### 3.4 通用 W8A8/W4A8 BF16 回退探针

使用原生 tiny Llama、Qwen3，各 2 层、hidden=16、intermediate=32、
head=2、词表=32，seed=13。通过 compressed-tensors API 构造标准量化
checkpoint，以 `CompressedTensorsConfig(dequantize=True)`、
`dtype=torch.bfloat16` 在 CPU 加载。每组执行五 token cached prefill、
一次单 token decode，以及从三 token prompt 生成两个 token：

| 模型 | 权重/激活 scheme | 存储格式 | 执行权重 dtype | prefill/decode/generate |
| --- | --- | --- | --- | --- |
| Llama | INT W8A8 | int-quantized | BF16 | 通过 |
| Llama | INT W4A8 | int-quantized，未打包 | BF16 | 通过 |
| Llama | INT W4A8 | pack-quantized，int32 打包 | BF16 | 通过 |
| Qwen3 | INT W8A8 | int-quantized | BF16 | 通过 |
| Qwen3 | INT W4A8 | int-quantized，未打包 | BF16 | 通过 |
| Qwen3 | INT W4A8 | pack-quantized，int32 打包 | BF16 | 通过 |

六组各有 14 个量化模块，状态均为 `DECOMPRESSED`。每组记录到 56 次
input Q/DQ、0 次 weight Q/DQ，prefill/decode 输出有限，两层 decode
cache 长度为 `[6, 6]`，生成 shape 为 `[1, 5]`。
加载完成后没有整模型 `ct_decompress_hook`。
这是功能及激活量化语义探针，不是量化质量或 GPU kernel 验收。

另对已有 V4.1 tiny 的 W8A8、W4A8、WNa8 config 单独测试
`dequantize=False, use_optimized_inference=True`：三者均为纯 INT scheme，
`has_fp8_modules=False`，`post_init()` 均关闭优化推理。
因此不能将“可加载 W8A8/W4A8”解释为“当前 backend 原生执行整数 GEMM”。

上述六组标准 loader 探针会物化整个 tiny CPU 模型，用于确认 BF16 与 Q/DQ
合同；大模型实现须改为 lazy Mapping 逐当前权重解码。
此前 V4.1 offload prefill 探针使用 FP32 参考值，其零误差不构成 BF16 或
整数 kernel 的证据。BF16 量化 lazy-load 与 offload 的组合仍需独立验收。

## 4. 内存可行性与分区粒度

对本地原尺寸 V4.1 config 构建 meta 模型，并将 Engram 表设为外部 lookup，
得到以下参数计数。表中按每参数两字节统一估算，未计 FP32 控制参数增量、
activation、cache、计算 workspace 或 CUDA allocator。

| 工作集 | BF16 权重估算 |
| --- | ---: |
| 最大完整 backbone block，`model.layers.14` | 25.933 GiB |
| 一层全部 384 个 routed experts | 25.313 GiB |
| 一个 routed expert 的三个矩阵 | 67.5 MiB |
| `model.layers.14.attn` | 0.256 GiB |
| `model.layers.14.engram` 投影/门控，不含外部表 | 0.293 GiB |
| embedding 或 LM head | 1.233 GiB |
| 第一个完整 DSpark block | 8.889 GiB |

结论：

- 完整 block 模式适合 tiny 或显存足够的设备，24 GiB 卡无法仅靠卸载其他层
  容纳原尺寸 BF16 V4.1 block。
- 首批真实大模型支持必须有足够小的工作集；优先使用 Accelerate 的
  Linear/Expert 粒度 hook，不要求先实现独立 expert group 调度器。
- 即使 checkpoint 是 INT4，参考推理若先反量化到 BF16，
  计算卡上仍要按 BF16 工作集估算，不能按 checkpoint 字节数计算峰值。
- Engram 表继续 mmap 并按需 gather rows，不物化为整个计算设备参数。

建议将预算检查落实为两部分：

```text
设备峰值 ≈ 当前 subgraph 执行权重
         + 当前 subgraph 所需 cache 的设备副本
         + 本阶段 live inputs/outputs/residual
         + 激活 Q/DQ、临时 FP32 转换及算子 workspace
         + 明确的安全余量

主存峰值 ≈ request 的已提交 cache
         + 当前有效边界
         + 当前准备权重 + 至多一个下一工作集
         + 解码 workspace + 诊断输出缓冲
```

全模型 KV 可以随 token 数增长，必须有独立上限；不能把它描述成与
sequence length 无关的恒定内存。共享 cache 按 storage owner 计量一次。

## 5. 建议结构

```text
薄入口 / compare
       |
Transformers 原生 generate / forward / Cache
       |
Accelerate cpu_offload / 模块 hooks
       |
有界 lazy weights Mapping
       |
现有 WeightSource + Materializer + 必要的 PTQ codec
```

### 5.1 通用能力边界

以下是职责边界，不要求每项新增一套 manager 或框架：

| 职责 | 选择依据 | 复用与扩展方式 |
| --- | --- | --- |
| 模型构建与增量数学 | config/architecture 的已有注册 | 标准 HF 原生模型；特殊模型只补自身 forward/cache 合同 |
| checkpoint key/格式解释 | header、量化 scheme、逻辑 shape、命名映射 | 现有 source/materializer，加经过验证的格式 codec |
| 量化执行选择 | 每 module 的 scheme、布局、设备及实际 backend 能力 | 可验证的原生内核，或加载时 BF16 回退 |
| 权重驻留与卸载 | module 调用依赖及预算 | Accelerate hooks、lazy Mapping |
| 数值与资源验证 | 输入、checkpoint、执行路径及语义 tap | 共享 compare/report，不按模型名复制工具 |

公共 inference 代码不写 `if model_type == "deepseek_v41"` 分支。
源 FP4 命名映射、Engram 外部表、CSA2 等规则留在对应 adapter/codec；
Llama/Qwen3 不需要实现这些接口才能使用相同入口。
新增模型时，优先验证其原生 cache、module 调用方式和实际格式，
而不是重新实现 generation、mask、RoPE 或 weight mover。

执行能力声明分别记录 cached prefill、单 token decode、chunked prefill、
量化 scheme/format 与 offload 兼容性；“模型类可导入”不等于全部支持。
WNa8 按模块解析实际 W8A8/W4A8 等 scheme，不能用一个全局位宽推断执行。

建议首批新增：

- `streaming/inference/api.py`：原生推理的薄入口与配置。
- `streaming/inference/weights.py`：有界 lazy weights 桥接及格式选择。
- `streaming/inference/compare.py`：逐模块/语义 tap、逐 token 诊断。
- 在既有模型适配文件中补充 cache-aware forward 和生成输入准备。
- `examples/streaming_inference/` 与 `tests/llmcompressor/streaming/inference/`。

上述薄入口、权重桥接、比较工具和 V4.1 模型 cache 已按该分工实现。
未新增通用 engine、executor、planner 或标准 HF KV state store。

## 6. 原生 forward 优先与后备推理计划

标准 HF 模型直接复用完整 forward、原生 cache 和 `.generate()`，
不逐模型重新实现 mask/RoPE/decoder loop。V4/V4.1 只补其模型特有的
增量合同与 checkpoint 解释，不通过新执行器绕过缺失的模型能力。

如果 module hooks 无法表达真实 subgraph，才使用现有 session 并引入
后备 `InferencePlan`，与校准 `SequentialExecutionPlan` 分开，描述：

1. 阶段顺序、阶段类型及稳定名称。
2. 输入、输出和每个值的最后消费者。
3. checkpoint-backed modules，以及不经 `call_module` 暴露的依赖。
4. 读取和写入的 cache owner、访问方式及别名关系。
5. prefill/decode 的能力及允许的输入长度。
6. 明确的 terminal result：logits、hidden-state taps 或 scoring output。
7. 当前分区的设备预算和 CPU 预取预算。

首批模型为 Llama、Qwen3 dense 和 native V4/V4.1。
支持声明需有对应数值测试；相似名称或 forward signature 不能证明兼容。

后续可增加推理专用 FX 生成器，但它必须：

- 使用 inference context，保留 LM head。
- 明确表示 cache 副作用和 terminal outputs。
- 区分 prefill/decode 分支，不能复用将 cache 固定为 None 的校准图。
- 对 hidden dependency、未覆盖调用和未知输出提前报错。

这一设计避免为了 trace 一个有状态模型而将整个 cache 对象复制进每条边界。

## 7. 状态所有权与后备执行生命周期

标准模型的 request 状态直接使用 Transformers Cache，
权重安装/卸载直接使用 Accelerate。下面的 owner/lease/state store
只用于 DeepSeek 特有状态或已经验证官方策略无法满足的严格资源需求。
不能先为标准 HF 模型复制一套通用 KV 存储系统。

区分四类数据：

| 类型 | 所有者 | 生存期 | 默认位置 |
| --- | --- | --- | --- |
| checkpoint 权重 | 权重 session | 当前 subgraph | CPU 准备，当前阶段上设备 |
| RoPE/hash 常量 | adapter/runtime constants | engine | 有界 CPU 常量，按需切片上设备 |
| request cache | `InferenceState` | prefill 到请求结束 | CPU，当前依赖临时上设备 |
| 当前 chunk/token 的 boundary | execution frame | 本轮最后消费者结束 | CPU，执行当前片段时搬运 |

现有 `BoundaryActivationStore` 的拒绝 cache 规则保留。
如果确需 `RuntimeStateStore`，使用显式 owner/handle/lease，
不依赖递归 pickle 任意 `Cache` 对象。普通 tensor 树复制及 alias
计数可提取为共享工具。

`InferenceState` 至少包含：

- checkpoint/adapter/plan 身份及 compute/quantization 配置指纹。
- 已提交 token 数、序列 attention mask、逻辑位置和物理 cache slot 元数据。
- 各 cache owner 的有效长度、容量、dtype、CPU storage。
- Engram token history 及边界/图像 DEAD 标记。
- active/failed/closed 状态。

单阶段顺序：

1. 准备当前权重；可预取下一阶段 CPU 权重，最多一个。
2. 上一个阶段权重已经卸载后，才把当前权重安装到设备。
3. checkout 当前 cache 依赖与 boundary。
4. 执行当前 subgraph，保存新增 cache 和必要输出。
5. 将状态写回 CPU，释放 lease 和设备临时引用。
6. 恢复当前权重至 meta，再进入下一个阶段。

默认不在 GPU 上预取下一层权重。KV 优先使用官方 offloaded cache，
允许其有界的下一层预取并计入预算。只有预算实测不满足时，才增加严格
cache 搬运策略。后备自定义搬运须用 event 保证 copy/compute 完成后
才释放或复用 storage，不能用频繁 `empty_cache()` 代替所有权管理。

每次完整 token/chunk 遍历成功后才推进 request 的 committed length。
初版阶段失败即标记 request failed，要求 reset/重新 prefill；
不允许带着部分层已经更新的 cache 重试同一个 token。
后续若加入回滚，应为追加状态和 ring overwrite 分别实现 journal。

安装权重不能触发全模型 cache reset。常量重建与 request cache 初始化
必须拆开，尤其需要调整 V4 的全局重初始化钩子使用方式。

## 8. 正常 prefill 和 decode

### 8.1 Prefill

输入 prompt，执行 embedding → 每个 decoder subgraph → norm/head。
每层计算 prompt 对应的 KV，并提交到 request state。
生成模式只对最后一个有效位置计算 LM head；scoring 模式分块输出 logits
或直接累计 loss，避免分配完整 `[B, T, vocab]`。

首个 release 支持 batch=1 的完整 prompt prefill。
随后加入 chunked prefill 和 padding/batch 支持，每项有独立能力标记。
不能让输入 chunk 静默走模型只适用于 `start_pos > 0, length=1` 的 decode 分支。

### 8.2 Decode

prefill logits 采样得到第一个新增 token。接下来：

```text
输入刚采样的 1 个 token
  -> 恢复各阶段所需历史 cache
  -> 使用正确的绝对位置/有效 mask
  -> 每阶段仅计算这个新增 token
  -> 更新 cache
  -> 最后位置 logits
  -> 采样下一个 token / EOS / 长度终止
```

在 prompt 长度 P 的 prefill 后，cache 长度为 P；第一次单 token decode
后为 P+1。一次生成最后采样的 token 不一定已经喂回 cache，
返回值要同时报告 sequences 长度和已处理长度，避免 off-by-one。

`decode()` 的默认合同是输入 `[1, 1]` 新 token。传完整 prefix 或错误长度
应被拒绝。多 token suffix 是未来显式能力，不伪装成单 token decode。

### 8.3 Chunked prefill

每个 chunk 完整通过所有层后再进入下一 chunk，保持正确因果顺序。
跨 chunk 要保存 compressor partial group、Engram history、window ring
和有效位置。首个 chunk 的结果与后续 chunk 的 attention mask 分别验证。

长 prompt 可用 chunk 减少 boundary/attention workspace，但会增加权重遍历。
API 允许用户选择 chunk size，并记录这一开销。

## 9. Hugging Face cache 适配

以本地安装的 Transformers 5.19.0 为首个兼容基线。
实际 Llama/Qwen3 forward 使用 `past_key_values: Cache`、`position_ids`、
`use_cache`。当前 `Cache.get_mask_sizes()` 接收 query length 和 layer index，
不照搬旧版以 `cache_position` 为核心的合同。

默认直接使用官方 Cache 和 offloaded cache，不新增通用 facade/搬运层。
模型特殊 cache 或严格预算确有缺口时，扩展须满足：

- 所有层的长度、mask 元数据在 CPU 可查询。
- 仅激活当前阶段读写的 cache tensor。
- 将官方 cache container 对象作为 session handle 传递，不放入 boundary snapshot。
- 在本轮开始时固定 committed position，不能因为第 0 层先 append 就让
  后续层看到不同的本轮起始位置。
- full attention、sliding attention 分别验证，不默认适配 hybrid/recurrent cache。

官方 cache offloading 会提前搬运下一层 cache。该行为不等于同时执行
两个 subgraph；默认接受并纳入预算。严格单 cache 驻留是按需扩展能力，
不作为推翻官方复用方案的前提。

外部接口参考：

- [HF cache explanation](https://huggingface.co/docs/transformers/main/en/cache_explanation)
- [HF cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [Accelerate big model inference](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference)

在线文档会变化；运行和测试以记录的本地版本及实现源码为准。

## 10. DeepSeek V4.1 增量适配

普通 backbone decode 需要新增以下状态合同：

| 状态 | owner 与语义 |
| --- | --- |
| SWA window KV | 每个 attention layer 独立 ring，保留写指针和有效长度 |
| compressed KV | 只由 KV source `[2,8,14,20]` 持有，后续 consumer 引用 |
| index K | KV source 生成，Reindex consumer 读取同一 owner |
| compressor partial state | 保存 ratio=2 未完成 group 的 KV/score 投影 |
| 当前 query 的 top-k/candidates | 跨层边界值，按 source 到最后 consumer 的生存期传递 |
| Engram history | 最大 n-gram 为 4，当前块之外至少保留最近 3 个 compressed ids/DEAD |
| mHC `pre_mix` | 当前 token/chunk 的跨层边界，不当成跨 token 历史 cache |

source/consumer 映射由配置生成，不依赖进程全局变量，也不为每个 reuse
层复制一整份 compressed KV。每个新 token 从其 source 层刷新当前 query
索引，不能沿用上一个 token 的 top-k。

关键数学要求：

1. query 和 window KV 使用绝对位置。
2. compressed group 在完整形成后才可见。
3. 该 group 的 RoPE 位置是 group 首 token 的位置。
4. index K 必须从 RoPE 前 latent 生成；随后压缩 KV 才做自己的 RoPE/QDQ。
5. 保留 window FP8、compressed KV 的 group16/E4M3 和 index 的
   group32/E8M0 参考量化语义。
6. Engram 的 pad/image DEAD 边界跨 prefill/decode 不变。
7. mask、有效长度与 ring slot 分开表示，不能仅靠 tensor shape 推断。

实现上给原有完整序列路径增加显式 cache-aware 合同；
prefill/decode 共用投影、mHC、gate、expert 等数学原语。
旧完整序列路径保留为校准及 recompute oracle，避免同时改掉对照基准。

本地官方参考的 compressor、SWA、index cache 和 `NgramHashState`
已有 prefill/decode 语义，可逐项建立 tiny 数值对照。

## 11. 大 MoE 的更小 subgraph

先验证 Accelerate 默认 module 粒度和 `preload_module_classes=["Expert"]`
等最小分组。原始 MoE forward 仍负责 routing、expert 顺序和合并；
未被调用的 expert 自然不会因 leaf hook 被加载。

只有存在 `.weight` 直接访问、非 module 依赖或最小分组仍超预算时，
才使用下列后备显式分解：

```text
Engram（若存在）/ mHC pre / attention / attention post
                        |
                FFN norm / gate / dispatch
                        |
            expert group 0 -> group 1 -> ...
                        |
                 shared expert / reduce / mHC post
```

每段明确列出权重和 live tensor。expert 阶段之前卸载 gate、attention
等上一阶段权重，只保留必需的 residual、路由结果与累加器。
`SubgraphWeightSession.plan_modules()` 可用于每个阶段的显式工作集。

必须满足：

- routing 计算一次，依据当前 token/chunk 决定实际激活 experts。
- 每次只加载一个预算内 group；没有 routed rows 的 experts 可以跳过。
- prefill 的一个 chunk 可能命中全部 experts，容量仍只按 group 驻留。
- 按原实现的 expert 顺序累加，记录浮点 reduction 顺序引起的差异。
- V4.1 的 routing weight 在 `w2` 之前应用；有激活 Q/DQ 时不能任意移到输出。
- 使用真实推理路由，`calibrate_all_experts` 不进入推理执行。

offload 策略根据预计峰值选 block、Expert 或 leaf module 粒度。
后备 `partition="auto"` 才选择独立 MoE 分解。
无法在最小已支持阶段内满足预算时，在权重加载前给出具体阶段和预计峰值。
LM head 若也超预算，先明确拒绝；后续可增加 vocabulary 分块。

## 12. Checkpoint 读取与量化执行

### 12.1 必须先确定格式

由 `CheckpointExecutionSpec` 联合读取 config、量化 scheme、索引和
safetensors header，选择 codec：

| 输入 | 解码合同 |
| --- | --- |
| 普通 BF16/FP16/FP32 | 浮点读取与 dtype policy |
| 原始 DeepSeek FP8 | 正确 block shape 与源 scale dtype |
| 原始 DeepSeek FP4 | E2M1 codebook + 源 group scale |
| PTQ INT8 | 整数值、scale、可选 zero-point 与 scheme |
| PTQ INT4 | 签名 nibble/已有 packed format，随后按 scheme 反量化 |

当前 DeepSeek materializer 认为 INT8/UINT8 scaled weight 是源 FP4，
且逻辑 shape 为存储末维乘二。PTQ INT8 不能进入这个分支。
raw key 命名与 codec 选择要组合处理，`.scale` 名称本身也不能决定类型。

对于本项目 `pack_to_int8` 输出，结合 INT4 scheme、模型逻辑 shape 和
物理 shape 确认是否为 nibble packing；不能只看 int8 dtype。
所有映射必须覆盖 target、ignore、辅助参数和共享权重 alias，缺少 scale、
shape 不一致、未知 format 或冲突 scheme 要在执行前失败。

### 12.2 当前安装栈的量化推理能力

检查 Transformers `5.19.0` 的 `CompressedTensorsConfig.post_init()`、
compressed-tensors quantizer 和 CT `ModelCompressor`，得到：

| 输入 scheme/格式 | 当前 CT backend 的执行路径 | 本项目默认策略 |
| --- | --- | --- |
| INT W8A8 / int-quantized | 没有该优化整数 kernel 路径；解压后浮点 forward + A8 Q/DQ | 当前权重在 CPU 加载时反量化成 BF16 |
| INT W4A8 / int-quantized 或 pack-quantized | 同上；packing 只改变存储解释 | 经匹配 codec 解码当前权重为 BF16 |
| 项目自有 signed nibble INT4 / int-quantized | 标准 CT loader 不能仅凭 format 推断项目布局 | 项目 codec 解包，再按 scheme 反量化为 BF16 |
| 混合 WNa8 | 逐模块包含 W8A8/W4A8，不能整体推断 | 各模块分别选择执行与回退 |
| FP8 W8A8 | 有条件的 opt-in 优化路径，需 scheme、设备与模块都匹配 | 只有通过设备及 offload 验收后才保留原生 FP8 执行，否则 BF16 |

这里的 W8A8/W4A8 指 INT 权重与 INT 激活，不与 FP8 W8A8 混用。
结论只覆盖已检查的 compressed-tensors backend，不代表 Transformers
所有其他量化 backend 都不支持 INT8/INT4。
其他 backend 的存在也不能证明它能直接读取本项目 packing 或满足
当前模块加载、计算后卸载的要求。

`CompressedTensorsConfig(dequantize=False)` 在本版本不是逐 module
反量化：CT 在模型根节点注册 `ct_decompress_hook`，首次 forward 调用
`decompress_model(model)` 遍历所有量化模块。
直接叠加这个 hook 会破坏流式工作集约束，也可能对 meta 参数解压失败。
`dequantize=True` 的标准 loader 则会在加载结束前解压整模型。
两种整模型路径都不能作为大模型 lazy-load 的实现。

### 12.3 通用执行选择与加载时 BF16 回退

每个量化模块先检查 scheme、物理布局、逻辑 shape、module forward、
计算设备，以及所选 backend 的真实可用性和 offload 生命周期。
只有这些条件均匹配且已有验证，才允许使用原生量化执行。
初版已确认的 INT W8A8/W4A8 走 BF16 回退；不为此引入新的量化 kernel。
后续接入其他 backend 不需要改变模型 forward 或 generation 公共接口。

回退发生在 lazy weights Mapping 响应当前 module 的权重读取时：

```text
读取当前 tensor 的 packed 数据、scale/zero-point、shape
    -> CPU 按匹配 codec 解包、反量化
    -> 形成当前权重的 BF16 执行值
    -> Accelerate 将其安装到计算设备
    -> 按原生模型 forward 执行
    -> Accelerate 卸载为 meta，释放有界 CPU 准备值
```

复用 CT 的逐 tensor/逐 module 解码原语及既有 materializer，
不调用整模型 `decompress_model()`，也不为模型注册全局解压 hook。
不预先生成全模型 BF16 `state_dict` 或另存完整 BF16 checkpoint。
当前准备值和至多一个下一工作集受主存预算限制；GPU 默认不预取下一组权重。
预算包括 packed 输入、scale、可能的 FP32 解码 workspace 和 BF16 输出，
设备权重预算按 BF16 字节数计算。

BF16 是回退计算权重的默认 dtype；scale、norm、mHC 等控制参数按模型和
scheme 的 dtype policy 保留必要 FP32，索引/shape/hash 保留整数。
不能对整个 state dict 一律 `.to(torch.bfloat16)`。
设备或模型不能正确执行 BF16 时明确报告限制；改变计算 dtype 必须显式配置。
比较 resident 与 streaming 时使用相同的 dtype policy，避免把 dtype
变化当作 streaming 或量化格式错误。

没有匹配 kernel、但具有正确 codec 时允许回退；缺少 scale、未知 packing、
不支持的 group/asymmetric 规则等格式问题必须报错，不能猜测解码。
首批实现覆盖本项目 symmetric、channel INT4/INT8、dynamic token INT8
activation；其他 scheme 按格式逐项增加验收。

### 12.4 激活量化语义与两种参考模式

权重反量化为 BF16 不会自动取消激活量化。默认模式
`quantized_reference` 保留 scheme 声明的 input/output activation Q/DQ：

```text
输入激活 -> A8 量化/反量化 -> 原生浮点算子（BF16 权重）
         -> 可选 output activation Q/DQ
```

这是 W8A8/W4A8 的数值参考执行，不是原生 INT8/INT4 GEMM；
也不能据此宣称具有量化 kernel 的加速或设备存储优势。
原始模型自身的 FP8/FP4 参考行为仍按 source/PTQ 合同处理，
不因为外部 A8 wrapper 被重复叠加或遗漏。

显式诊断模式 `dequantized_weights` 绕过 checkpoint 声明的激活 Q/DQ，
只检查序列化及权重量化误差。报告必须标为权重 BF16 回退对照，
不能当作完整 W8A8/W4A8 数值验证结果。

优先复用 compressed-tensors 的 Q/DQ 原语；已反量化权重不再执行 weight
Q/DQ。本地 `DECOMPRESSED` 状态会跳过权重 Q/DQ，仍执行激活 Q/DQ，
第 3.4 节六组探针已确认这一行为。
不能直接假设安装其 forward wrapper 就数值等价：本地 wrapper 调用
`self.__class__.forward()`，可能覆盖模型实例绑定的 reference forward。
桥接需保留实际 callable、累加与 dtype 行为；这条规则也适用于其他模型。

每个模块的 inspect/report 至少记录 source weight/activation 位宽与类型、
storage format/packing、实际 backend、执行 weight dtype、activation mode、
fallback reason；支持按 scheme/backend 汇总，避免海量重复日志。
原生量化 kernel 与参考执行的数值差异通过独立同输入对照验收。

## 13. DSpark 与图像验证

普通生成默认执行 backbone；使用显式执行模式避免误入 teacher-forced
校准分支。用户不启用 draft 时，DSpark 权重不应被顺带加载。

为验证完整适配和量化产物，增加独立 auxiliary 评估：

- DSpark：主干 target layer 的输入 hidden、draft attention/cache、
  三 stage、Markov head 和 confidence head 分别比较。
- 首批辅助接口使用固定 teacher-forced 输入，验证权重和数学；
  不能据此声称 speculative accept/reject 已正确。
- speculative generation 后续需单独实现 draft prefill、候选验证、
  接受长度及 cache rollback/commit。
- 图像：prefill 前逐 ViT block 执行，aligner 后合并 image embeddings，
  再进入文本计划。图像 span 的 DEAD history 延续到 decode。
- processor/patch grid/span 需独立验收；不能在 prefix 隐藏调用中一次
  安装整个 vision tower。

## 14. 验证框架

### 14.1 建立互补的对照关系

| 对照 | 用途 |
| --- | --- |
| 官方/上游完整模型 vs native 完整模型，同权重 | 检查模型数学适配 |
| native resident vs native streaming，同 checkpoint | 检查调度、卸载与读取 |
| full-prefix recompute vs prefill+decode，同 checkpoint | 检查增量状态与位置 |
| 内存量化参考 vs 写出再读回的量化产物 | 检查 packing、scale 和序列化 |
| 原始权重 vs PTQ，固定 token 序列 | 量化数值变化和 PPL/loss |
| 参考执行 vs 目标引擎 | 检查生产 kernel/部署，单独报告 |

只有流式和 resident 一致，不能排除两边共用了同一个错误 native 实现。
同样，生成文字看起来合理或 tiny toy PPL 较低都不能代替数值对齐。

### 14.2 比较方式

先用固定 prompt 和固定 continuation 做 teacher forcing。
候选/reference 看到相同 token，逐 token 比较，不让采样分叉混入层误差。
然后再运行独立 greedy generation smoke。

报告至少包括：

- checkpoint/config/tokenizer/adapter/plan 指纹与软件版本。
- execution mode、dtype、quantization scheme 与 attention backend。
- 源权重/激活位宽、packing、实际执行 backend、BF16 回退原因和激活 Q/DQ 模式。
- 每阶段 max abs、RMS abs、相对 L2、finite 检查及 shape。
- logit top-1/top-k agreement、概率 KL 和基准 top-1 margin。
- cache owner、有效长度、写入位置、共享 alias 与首次异常。
- 源码语义位置的 hidden taps，而非仅绑定不稳定 FX node 名称。

精度规则区分：

- 整数 packing、tensor 名、shape、hash ids、cache 长度：严格相等。
- 同 CPU FP32 路径：根据实际相同算子设严格 allclose，
  tiny 探针目前可以到零误差。
- BF16/不同设备/backend：按模型和算子设定实测门槛并记录误差分布。
- 原始 vs PTQ：作为质量评估，不要求逐 tensor 或 logits 相等。

低 margin 时 tiny 浮点差异可能翻转 top-1；应同时看 logit error 和
margin，不用单一 top-1 指标断言实现错误。

### 14.3 诊断输出内存

默认只保存统计和 top-k。需要完整张量时按选定阶段/token 逐个落盘。
不同时保留两个完整模型、所有层 activation 或整个 prompt 的全词表 logits。
大模型可分别运行 reference/candidate，使用语义 tap 文件对齐。
也可按同输入比较局部阶段，另行报告自由传播的累计误差。

## 15. API 草案

默认暴露配置后的原生模型，直接调用原生 API。小模型且 RAM 足够时可用：

```python
import torch
from accelerate import cpu_offload
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/standard-supported-checkpoint",
    device_map="cpu",
)
cpu_offload(model, execution_device=torch.device("cuda:0"))
result = model.generate(
    input_ids=input_ids.to("cuda:0"),
    use_cache=True,
    cache_implementation="offloaded",
    max_new_tokens=16,
)
```

上例要求模型本身已支持 cache、checkpoint 被标准 loader 支持，
且 CPU 放得下加载权重。大模型及特殊格式使用 meta 构建加 lazy
weights Mapping，避免先物化完整 CPU 模型。V4.1 cached decode 仍待适配。

薄入口只负责格式、offload 配置与验证选项。
生成参数复用 Transformers，score/compare 使用原生 forward 与诊断 hooks，
不另建通用 generation loop 或强制要求新的 request API。

CLI 建议采用 `generate`、`score`、`compare`、`inspect-plan` 子命令。
`inspect-plan` 不加载整模型，输出每段权重、cache 依赖、工作集估算和
预算结果；真实 header 缺失时明确标注哪些量只是 config 推导。

对不支持的模型、cache 类型、输入形状和 checkpoint 格式在执行前失败。
手动 meta+hook 路径显式设置 execution device 与输入位置，
验证 `.generate()` 的 device 提示及输入准备，不把输入移到 meta。

## 16. 分阶段实施与验收

| 阶段 | 交付 | 验收条件 |
| --- | --- | --- |
| M1 原生复用 | 薄入口、Accelerate hooks、lazy weights，tiny Llama/Qwen3 原生 generate/cache | resident、offload、recompute 三方对齐；卸载后权重为 meta |
| M2 量化桥接 | 通用逐 module 能力选择、加载时 BF16 回退、codec/QDQ、raw key adapter，score/compare | Llama/Qwen3 与项目三种 PTQ 产物读回对齐；A8 Q/DQ 恰好一次；无整模型解压；未知格式拒绝 |
| M3 DeepSeek 主干 | V4/V4.1 增量 state，cache reset 拆分，原始 FP8/FP4 输入 | 官方/native、resident/streaming、recompute/cache 逐项对齐 |
| M4 容量与长序列 | module/Expert 粒度、官方 KV offload、设备预算、chunked prefill | 小设备预算下执行成功；跨窗口/压缩组一致；GPU 峰值受控 |
| M5 完整拓扑诊断 | 现有完整 tiny，vision 逐块，DSpark auxiliary，CLI/report | 40 text/32 vision/3 draft 分别覆盖，原始与三种 PTQ 产物均有报告 |

M1/M2 可以用同一个最小执行架构贯通；M3/M4 必须完成后，
才宣称支持原尺寸 V4.1 的低显存流式推理。
M5 的普通生成和 auxiliary 结果分开验收。

建议按阶段提交，不在第一步大规模抽取 PTQ pipeline。
共享底层 API 有改动时，原有 PTQ、prefetch、host budget、tied-weight、
materialization 和 checkpoint 回归应继续通过。

### 必须覆盖的边界测试

- prompt 长度 1、window 前后、ratio=2 的奇偶长度、candidate block 边界。
- 多次 decode 至 window ring wrap；新请求/reset 不混入上次状态。
- chunk 与 unchunked prefill，chunk 在压缩 group 中间结束。
- Engram 2/3/4-gram 跨调用，图像 DEAD/pad 截断。
- CSA2 Full/Reindex/Reuse、不同 source 切换和 cache alias 去重。
- PTQ INT4 packed/unpacked、INT8、scale 缺失、名字冲突和错误 shape。
- 无原生量化 kernel 时按当前 module 载入 BF16；A8 Q/DQ 保留，weight Q/DQ 为零；
  初始化不读全量权重，首次 forward 也不触发全模型解压。
- 混合 WNa8 的逐模块 scheme 选择、FP32 控制参数和整数索引 dtype；
  显式关闭 A8 Q/DQ 时报告与默认参考模式有区别。
- V4.1 pre-w2 routing weight 与 dynamic activation Q/DQ。
- 阶段失败、加载失败、预取失败及预算拒绝后的清理。
- decode 阶段的 embedding/decoder 输入长度确实为 1，
  而非悄悄重新计算整个 prefix。
- 所有阶段结束后无未申明设备参数；GPU live allocation 不随已处理层数增长。

CPU 测试验证数学与所有权；GPU 测试验证真实搬运、同步和峰值。
GPU allocator reserved bytes 可以缓存，验收应同时看 live allocated、
峰值及仍被引用的 tensor，不能要求每层 reserved bytes 都归零。

## 17. 风险、限制与完成定义

主要风险集中在模型 cache 状态、直接 `.weight` 访问导致 hook 未触发、
nonpersistent buffer 驻留，以及 raw FP4/PTQ INT4 的解释冲突。
优先用官方 cache、最小安全 module 分组及现有格式桥接处理。
自定义 FX 计划/MoE 调度只在已有组件出现具体缺口后引入。

每个 decode token 都要重新遍历所需权重。完整 block 计划会搬运该 block
全部 experts；expert group 计划可跳过未激活 experts，但长 prefill 仍可能
命中全部。CPU 预取最多隐藏部分准备耗时，不能消除 I/O 与传输成本。

本地原尺寸 DeepSeek 模型目录当前只有配置、索引和参考代码，没有实际
权重分片，因此本轮只做 meta 计数和 tiny 探针。真实权重/GPU 验收需要在
有相应 checkpoint 和计算卡的环境执行，不能由现有 CPU 结果替代。

完成本需求的判断标准：

1. 公共入口确实走 prefill → cache → 单 token decode。
2. 每个阶段的权重、状态和临时内存都满足声明的预算及所有权。
3. 原始模型和量化产物均可读取并生成可解释的对照报告。
4. 报告区分模型适配、权重格式、量化数值和目标 kernel 验证范围。
5. tiny 全拓扑验证完成后，再用真实 checkpoint/GPU 确认容量与数值。
