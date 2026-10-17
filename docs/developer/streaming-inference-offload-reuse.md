# Streaming 推理：优先复用 Transformers / Accelerate 的评估

日期：2026-10-17。状态：评估结论已落地为通用推理入口，
CPU 回归与完整 tiny 对照已执行；GPU/原尺寸验收待完成。

使用方式、测试范围和数值限制见
[Streaming reference inference](../guides/streaming_inference.md)。
下文的“本轮探针”“仍需实现”保留评估时的历史记录。当前已有 lazy
weights Mapping、CPU 按需 BF16 解码、A8 Q/DQ、失败清理及 V4.1 cache，
旧版 native V4 的增量适配仍待完成。

实现中还处理了 Accelerate `1.14.0` 递归 preload 对 descendant
nonpersistent buffer 使用裸名字的问题：先安装官方非递归权重 hooks，
再把目标 preload group 的 hooks 组合到父模块调用；参数安装和卸载仍由
官方 hooks 完成。派生常量按实际 owner 搬运，V4.1 RoPE 仅传当前位置切片。

本评估修订 `streaming-inference-plan.md` 的实现优先级。默认使用原生
forward / generate、官方 cache 和 Accelerate 权重 offload；
不先构造新的通用执行器、推理图规划器和 KV 存储系统。

## 1. 结论

Transformers 具备 CPU/disk offload 推理入口，权重搬运主要通过
Accelerate 实现。标准模型可以继续执行原生 prefill、cache decode
和 `.generate()`，无需项目重新编排 decoder。

需要分清两个独立开关：

| 功能 | 主要实现 | 解决的问题 |
| --- | --- | --- |
| 权重 offload | Accelerate `cpu_offload` / `dispatch_model` | 模型权重放不进计算卡 |
| KV offload | Transformers `DynamicCache(offloading=True)` 或 `cache_implementation="offloaded"` | 历史 KV 放不进计算卡 |

KV offload 不会替代权重 offload，权重 offload 也不会自动实现模型缺失的
cache 数学。V4/V4.1 的增量适配仍需完成。

设计面向模型和量化格式的通用组合：原生模型提供数学与 cache，
checkpoint codec 解释存储，执行选择按每个 module 的 scheme/布局/设备判断。
本地 INT W8A8/W4A8 没有匹配的 CT 优化 kernel 路径，默认在读取当前工作集时
反量化为 BF16，并保留激活 A8 Q/DQ；不在 CPU 或 GPU 预先解压整模型。

## 2. 本地版本与源码

验证环境：

- Transformers `5.19.0`。
- Accelerate `1.14.0`。
- compressed-tensors `0.17.2.a20260806`。
- PyTorch `2.12.1`，没有可用 CUDA/MPS。

主要检查位置：

- `accelerate/big_modeling.py`：`cpu_offload`、`cpu_offload_with_hook`、
  `disk_offload`、`dispatch_model`、`load_checkpoint_and_dispatch`。
- `accelerate/hooks.py`：`AlignDevicesHook`、`attach_align_device_hook`。
- `accelerate/utils/offload.py`：`PrefixedDataset`、`OffloadedWeightsLoader`。
- `transformers/cache_utils.py`：`Cache`、`DynamicCache` 及每层 offload/prefetch。
- `transformers/integrations/accelerate.py`：`accelerate_dispatch`。
- `transformers/quantizers/quantizer_compressed_tensors.py`。
- `compressed_tensors/offload/`：项目依赖已有的另一套 offload 实现。

## 3. 已有 API 的复用价值

### 3.1 `cpu_offload`

将权重 backing state 放在 CPU，实际参数置为 meta。
forward 前由 hook 从 `weights_map` 取需要的权重并移到执行设备，
forward 后恢复 meta。

默认按拥有直接参数/buffer 的 module 递归加 hook，而非自动加载整个
decoder block。使用 `preload_module_classes` 可以在指定 module 入口加载
该 module 及全部子模块，并停止向其子模块递归安装 offload hook。

因此可以选择：

- dense tiny：预加载单个 decoder，验证“一次一个 block”。
- 大 MoE：保持细粒度 hook，或只预加载单个 Expert，避免整层 experts 驻留。
- 使用 `.weight` 直接运算而不调用子模块 forward 的实现：
  对最小安全父模块预加载，或局部改为显式 module 调用。

父模块的直接控制参数可能在子模块调用期间仍然驻留；
这属于当前调用依赖，峰值要按完整调用栈的 live weights 统计。
不能将“按 module hook”误写成任何时刻只有一个 tensor。

### 3.2 Lazy `weights_map`

本地实现接受 `Mapping[str, Tensor]` 作为 hook 的 backing weights。
传给 `cpu_offload(state_dict=...)` 后，递归 hook 用 `PrefixedDataset`
代理访问，每次只请求当前 module 的 tensor。

这允许项目用一个小桥接层连接：

```text
Accelerate hook 请求逻辑 tensor
    -> 现有 CheckpointWeightSource
    -> 现有 WeightMaterializer / 必要的 PTQ codec
    -> 当前 tensor 的 CPU 执行值
    -> Accelerate 安装、执行、卸载
```

不需要先把整个反量化模型载入 CPU，再调用 `cpu_offload()`。
Mapping 必须实现不读取 tensor 的 key iteration / membership / length，
并设置有界 CPU 缓存；不能在初始化中调用 `dict(mapping)` 或遍历 values。

### 3.3 `device_map`

`device_map="auto"` 优先使用可用计算设备，然后 CPU/disk；
它可能保留许多 GPU 权重，不满足严格单工作集约束。

`device_map={"": "cpu"}` 的普通加载会在 CPU 执行，不代表自动在 GPU 逐层计算。
需要通过 `cpu_offload(execution_device=...)`，或配置明确执行设备的
dispatch hooks，才能实现“CPU 存储、GPU 计算”。

`load_checkpoint_and_dispatch()` 适用于可由标准 loader 解释的 checkpoint。
它不能代替本项目 raw DeepSeek / 自有 INT4 packing 的格式转换。

### 3.4 `cpu_offload_with_hook`

该 API 在 forward 后保持模型驻留，直到显式 `hook.offload()` 或下一模块
的 hook 触发卸载。它适合重复调用同一阶段，但不作为本需求默认方案：
对完整模型调用可能导致一次安装整个模型。

### 3.5 Disk offload

标准 API 已支持 mmap/磁盘 backing。优先使用已有格式可直接支持的路径。
本项目原始 FP8/FP4 若仍需 CPU 解码，则通过 lazy Mapping 复用原始
safetensors 和 materializer，避免预先落一份巨大 BF16 checkpoint。

### 3.6 KV cache offload

标准 Transformers 模型直接使用：

```python
model.generate(
    input_ids=input_ids,
    use_cache=True,
    cache_implementation="offloaded",
    max_new_tokens=16,
)
```

也可向 forward 传 `DynamicCache(config=model.config, offloading=True)`。
该机制通常同时涉及当前 KV 和下一层预取 KV；默认接受这一有界预取，
并计入显存预算。“只执行一个 subgraph”不要求另一个 cache 副本绝对不存在。

若实际预算要求严格限制 cache 驻留，才评估少量搬运策略扩展。
本轮没有 GPU，未运行 offloaded KV 的 CUDA stream 路径。

## 4. CPU 探针

这些是临时诊断，未提交新的推理 API 或修改第三方库。

### 4.1 原生 Llama/Qwen3 生成

两层 tiny 模型：hidden=16、intermediate=32、head=2、词表=32，
prompt 为五 token，seed=13。

流程：保存 checkpoint → meta 构建 → lazy Mapping →
`cpu_offload(execution_device="cpu", preload_module_classes=[DecoderClass])` →
直接调用模型原生 forward 和 generate。

| 检查 | Llama | Qwen3 |
| --- | ---: | ---: |
| 挂载 hook 时的 tensor 读取次数 | 0 | 0 |
| prefill 对 resident 最大绝对误差 | 0.0 | 0.0 |
| 连续三步单 token decode 最大绝对误差 | 0.0 | 0.0 |
| 两层 cache 长度 | [8, 8] | [8, 8] |
| 四 token greedy generate 与 resident 相同 | 是 | 是 |
| decoder 内观察到仅当前 decoder 的参数已物化 | 是 | 是 |
| 调用结束所有参数恢复 meta | 是 | 是 |

`.generate()` 在这种手动 meta+hook 设置下产生输入 device 与
`model.device == meta` 的提示，但探针正常执行。
正式封装需正确传递执行设备，不能按提示将输入移到 meta。

### 4.2 完整 V4.1 tiny 与量化产物

复用已有 40 text / 32 vision / 3 DSpark 的 tiny 模型。
本探针执行 backbone 文本 prefill，显式关闭校准 draft 分支，
不声称覆盖本次未执行的 vision/DSpark 路径。

权重通过现有源 materializer 或现有验证工具的 INT4/INT8 解码规则按需读取，
PTQ 激活 Q/DQ 复用已有 reference hook。
Accelerate 按默认 module 粒度搬运，保留 V4.1 实例绑定的 reference forward。

| checkpoint | prefill 最大绝对误差 | hook 初始化读取权重 | 结束恢复 meta |
| --- | ---: | ---: | --- |
| source FP8/FP4 | 0.0，逐元素相同 | 0 | 是 |
| WNa8 | 0.0，逐元素相同 | 0 | 是 |
| W8A8 | 0.0，逐元素相同 | 0 | 是 |
| W4A8 | 0.0，逐元素相同 | 0 | 是 |

三种 PTQ 模式各挂接 1306 个 Linear 的 scheme/input QDQ。
这里验证的是现有参考解码与 offload 的组合，未证明标准
`AutoModel.from_pretrained()` 可以不经桥接读取这些原始命名和 packing。
V4.1 当前没有增量 cache，本轮未对它执行有效 cached decode。
该组使用 FP32 参考权重；不能将零误差解释为 BF16 或原生整数 kernel 验收。

### 4.3 标准 HF 模型的 BF16 + A8 Q/DQ

为确认通用性，使用 Llama、Qwen3 各 2 层、hidden=16、intermediate=32、
head=2、词表=32、seed=13 构造标准 CT checkpoint。
以 `CompressedTensorsConfig(dequantize=True)` 和 `dtype=torch.bfloat16`
加载，每组执行 cached prefill、单 token decode 和原生 greedy generate：

| 模型 | scheme | format | 执行权重 | prefill/decode/generate |
| --- | --- | --- | --- | --- |
| Llama | INT W8A8 | int-quantized | BF16 | 通过 |
| Llama | INT W4A8 | int-quantized，未打包 | BF16 | 通过 |
| Llama | INT W4A8 | pack-quantized，int32 打包 | BF16 | 通过 |
| Qwen3 | INT W8A8 | int-quantized | BF16 | 通过 |
| Qwen3 | INT W4A8 | int-quantized，未打包 | BF16 | 通过 |
| Qwen3 | INT W4A8 | pack-quantized，int32 打包 | BF16 | 通过 |

六组每组 14 个量化模块，状态为 `DECOMPRESSED`；
记录到 56 次 input Q/DQ、0 次 weight Q/DQ。
五 token prefill 后再 decode 一个 token，两层 cache 长度为 `[6, 6]`，
输出均有限；从三 token prompt 生成两 token，结果 shape 为 `[1, 5]`。
加载后没有整模型解压 hook。

这些标准 loader 测试使用完整 tiny CPU 模型，确认 BF16 与 A8 Q/DQ 功能，
不作为大模型有界加载的实现或精度门槛。
第 4.1 节验证了浮点 lazy offload，第 4.2 节验证了项目 PTQ 解码与 offload；
BF16 量化 lazy-load、offload 和 GPU 路径的组合仍需单独验收。

### 4.4 排除整数 scheme 的原生优化路径

对已有 V4.1 tiny 的三个真实 PTQ config 单独设置
`dequantize=False, use_optimized_inference=True` 并运行 `post_init()`：

| checkpoint | scheme | has_fp8_modules | 最终 use_optimized_inference |
| --- | --- | --- | --- |
| w8a8 | INT W8A8 | False | False |
| w4a8 | INT W4A8 | False | False |
| wna8 | 混合 INT W8A8/W4A8 | False | False |

该检查不依赖 `dequantize=True` 强制关闭优化的分支，因此能独立确认：
当前 installed CT backend 没有这些整数 scheme 的优化执行路径。
不把结论外推到 Transformers 的所有其他量化 backend。

## 5. 仍需要项目实现的内容

### 必须实现

1. **V4/V4.1 模型增量数学。** 绝对位置、SWA ring、压缩 partial group、
   CSA2 source/reuse、Engram history，以及正确的 cache 返回和生成输入准备。
2. **特殊 checkpoint 桥接。** raw key、FP8/FP4 与 PTQ INT4/INT8 codec、
   scale、逻辑 shape 和 dtype；复用现有 source/materializer/CT 原语。
   桥接按格式组织，公共入口不依赖 DeepSeek 模型名。
3. **参考量化 forward。** 保留模型原有 callable 与累加行为，确保 input/output
   Q/DQ 正确且权重不被二次量化。没有可用原生 kernel 时，在读取当前 module
   权重阶段反量化成 BF16，不能延迟到整模型首次 forward。
4. **数值验证及资源审计。** 逐模块/语义 tap 比较、recompute 对 cached decode、
   单工作集峰值检查和错误格式提前拒绝。

### 有证据显示缺口后再实现

- 严格单 cache 驻留策略。
- module hook 无法覆盖的非 module subgraph 调度。
- 超过已支持最小 module 工作集的细分。
- 面向长序列的特殊 cache 容量控制及诊断持久化。

### 默认不实现

- 新的通用 autoregressive generation loop。
- 为标准 HF 模型复制 mask、RoPE、decoder 调度和 KV 存储。
- 新的通用 FX inference planner。
- 新的权重 CPU/GPU 搬运、meta 安装、tied-weight 或 disk offload 框架。
- 同一模型同时叠加 Accelerate 和 CT 两套 offload owner。

## 6. 模型 buffer 与第三方库的边界

Accelerate 的 `offload_buffers=True` 不等于自动管理运行时 KV。
本地实现会把 nonpersistent buffers 放到执行设备，而不是把其修改后的值
当作持久 checkpoint 权重自动存回 CPU。

因此 V4 内部的 nonpersistent KV/ring/state 仍需模型 cache 适配。
V4.1 的 RoPE/hash 常量也要检查是否被所有层提前搬上设备。
这属于模型运行时状态问题，不需要据此重写权重 offload。

本地 compressed-tensors 也提供 CPU/disk offload、参数访问时 onload、
`load_offloaded_model()` 和 `from_accelerate()`。它适合现有 CT 工作流，
但原始格式与 runtime cache 的限制同样存在。
首个参考路径选 Accelerate，避免同时维护两套 backend；
只有具体格式/已有调用链需要时，才单独验证 CT 路径。

### 6.1 CT 解压 hook 不能直接复用为流式解码

本地 `CompressedTensorsConfig` 的普通加载语义：

- `dequantize=True`：标准 loader 加载后反量化整模型。
- `dequantize=False`：CT 注册根模型 `ct_decompress_hook`，
  首次 forward 的 `decompress_model()` 遍历所有量化模块。
- `use_optimized_inference=True`：只有当前支持的 FP8 scheme 和设备才会
  保留量化执行；纯 INT W8A8/W4A8 会关闭该选项。

因此“延迟到首次 forward 解压”不等于“逐 module 解压”，不能直接用这条
整模型 hook 满足单工作集约束。
应只复用 CT 逐 tensor/module 解码原语，由 lazy Mapping 在权重加载阶段
返回当前 BF16 tensor；不注册根解压 hook，也不生成完整 BF16 state dict。
项目自有 int8 容器的 signed nibble INT4 与 CT int32 `weight_packed` 格式
分别解析，不能只看 `format="int-quantized"`。

反量化后的权重执行 BF16 浮点算子；scheme 中的 A8 由 input/output Q/DQ
保留，属于数值参考执行。显式禁用 A8 的诊断结果必须单独标记，
不能当成完整 W8A8/W4A8 验证。
scale/控制参数所需 FP32 和索引整数 dtype 保留，不统一强转。
设备预算按 BF16 权重加 activation/cache/workspace 计算；
CPU 预算还需包含 packed 输入与解码临时值。

### 6.2 通用接口的扩展方式

模型适配只负责原生 forward/cache/特殊外部状态；checkpoint codec 负责
key、packing、scale 和逻辑 shape；执行选择按 module 的 scheme、设备及
backend 能力完成。公共入口及 lazy weights 不写 DeepSeek 专用分支。
标准 Llama/Qwen3 直接使用现有 HF 类；新模型优先验证同一套能力合同。

默认策略是“经过验证且兼容 offload 的原生量化执行，否则加载时 BF16”。
WNa8 必须逐 module 判定。缺少 kernel 可回退；缺少正确 codec、scale 或
shape 信息则直接失败，不能以 BF16 回退掩盖格式错误。
report 记录源位宽/类型、format/packing、实际 backend、权重 dtype、
activation mode 与 fallback reason。

## 7. 修订后的开发顺序

1. 把已成功的原生 HF offload 探针整理为最小入口和回归测试。
2. 实现有界 lazy weights 桥接和逐 module 能力选择；
   INT W8A8/W4A8 在加载阶段 BF16 回退并保留 A8 Q/DQ，
   使用原生 `.generate()` / forward / cache。
3. 在模型文件中补齐 V4/V4.1 cached prefill/decode 和生成合同。
4. 对源 FP8/FP4 与三种 PTQ 产物做相同 token 序列的数值验证。
5. 在 GPU 上验证单 module/block/Expert 的驻留、官方 KV offload、
   nonpersistent buffer 和实际峰值。
6. 根据实测缺口决定是否启用已有 `SubgraphWeightSession` 的特殊后备调度。

阶段验收仍沿用原方案中的三方对照、完整 tiny、DSpark/vision 辅助验证、
失败清理和真实权重/GPU 验收。默认实现范围已收缩为薄入口、权重桥接、
模型 cache 和比较工具。

## 8. 官方参考

- [Accelerate 1.14.0 big modeling API](https://huggingface.co/docs/accelerate/v1.14.0/en/package_reference/big_modeling)
- [Accelerate 1.14.0 big model inference](https://huggingface.co/docs/accelerate/v1.14.0/en/concept_guides/big_model_inference)
- [Transformers cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [Transformers compressed-tensors：优化推理与加载时反量化](https://huggingface.co/docs/transformers/quantization/compressed_tensors)

最终兼容性以本地已记录版本和上述 CPU/GPU 验收结果为准。
