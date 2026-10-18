# DeepSeek V4.1 适配与 streaming 推理修改记录

记录日期：2026-10-18，时区 Asia/Shanghai。
归档分支：`feat/deepseek-v41-native-quantization`。

本记录覆盖本会话中已完成的模型适配、量化、tiny 模型、streaming 推理、
精度检查和 GPU kernel 对比工作。文件清单按 Git 提交核对，验证状态同时参考
本地报告和用户返回的 GPU 报告。

## 1. 统计范围与提交

主清单对应分支 `feat/deepseek-v41-native-quantization` 上的以下 4 次提交，
基线为 `77f874751`，统计终点为 `00e29c74a`。

| 编号 | 提交 | 日期 | 内容 |
| --- | --- | --- | --- |
| T1 | `29af84e1ba0274fbcecea52fe32752e18eb577e6` | 2026-10-17 | V4.1 原生模型、streaming 量化、完整拓扑 tiny 创建及验证 |
| T2 | `9c9c91554170ac7b54a22a1203c6d38e60e7b270` | 2026-10-18 | 通用 streaming 推理、V4.1 cache、精度诊断及 GPU 算子验证 |
| T3 | `872d2bdc8347d0ed5eed9460eeb7c17036445f9b` | 2026-10-18 | 对齐官方 sparse attention 计算顺序和模型 FP32 参数边界 |
| T4 | `00e29c74a4c2f15b4ad4b4146b27b9eb48eb3585` | 2026-10-18 | 独立 kernel 级直接对照工具 |

上述范围合计 **48 个文件：新增 38 个、修改 10 个、删除 0 个**，
净差异为新增 10,826 行、删除 78 行。同一文件跨多个提交修改只计一次。
下表“新增/修改”均相对基线，提交列列出该文件涉及的提交。

| 类别 | 文件数 |
| --- | ---: |
| 模型实现与注册 | 10 |
| streaming 量化及通用推理 | 12 |
| 示例 | 3 |
| 创建、诊断及对照工具 | 7 |
| 测试 | 7 |
| 文档与分析报告 | 9 |
| 合计 | 48 |

本记录与 V4.1 适配代码一同归档在
`feat/deepseek-v41-native-quantization` 分支。清单根据指定提交范围核对，
记录文件自身作为后续文档提交，不计入前述 48 个历史文件。

归档本记录前，本地特性分支 HEAD 为 `1874bbd5f`，本地远端跟踪引用
`origin/feat/deepseek-v41-native-quantization` 指向 `00e29c74a`。
后续两次提交涉及的另外 4 个文件见第 3 节，单独列出以保留工作边界。
此处保留的是归档前的本地引用快照，后续文档提交不改变历史统计范围。

## 2. 逐文件主清单

### 2.1 模型实现与注册：10 个文件

| 文件 | 类型 | 提交 | 修改用途 |
| --- | --- | --- | --- |
| `src/llmcompressor/modeling/__init__.py` | 修改 | T1 | 导入 V4.1 MoE 校准模块，触发注册 |
| `src/llmcompressor/modeling/deepseek_v41.py` | 新增 | T1 | V4.1 MoE 校准适配，保留模型模态路由及专家权重应用方式 |
| `src/llmcompressor/modeling/deepseekv4/kernels.py` | 修改 | T1、T3 | 官方 kernel backend 接口及 sink FP32 ABI；共享 sparse attention 改为每 64 个 KV 的在线 softmax、块内 BF16 exp 权重、FP32 累加和重缩放 |
| `src/llmcompressor/modeling/deepseekv41/__init__.py` | 新增 | T1、T2 | 注册原生配置、文本和图文模型工厂；在上游缺少 V4.1 注册时提供兼容注册；导出 cache |
| `src/llmcompressor/modeling/deepseekv41/config.py` | 新增 | T1 | 原生配置与 Transformers 配置映射，涵盖文本、vision、CSA2、Engram 和 DSpark |
| `src/llmcompressor/modeling/deepseekv41/model.py` | 新增 | T1、T2、T3 | 文本、MoE、mHC、CSA2 共享状态、DSpark 和模型包装；接入 prefill/decode cache；显式保留官方要求的 FP32 compressor、head、confidence 参数 |
| `src/llmcompressor/modeling/deepseekv41/engram.py` | 新增 | T1 | Engram 哈希、查表、行分组量化权重读取及历史 token 处理 |
| `src/llmcompressor/modeling/deepseekv41/quantization.py` | 新增 | T1、T3 | FP8/FP4 软件 Q/DQ 与 Linear 工厂；支持显式参数 dtype |
| `src/llmcompressor/modeling/deepseekv41/vision.py` | 新增 | T1 | vision 编码及其与 V4.1 图文模型的连接 |
| `src/llmcompressor/modeling/deepseekv41/cache.py` | 新增 | T2 | 请求独有的窗口 KV、压缩 KV、index K、部分压缩状态和 Engram 历史；在 attention 调用之间保存在 CPU |

模型实现按 DeepSeek 官方语义特化。通用化要求针对推理 pipeline，
不要求将 V4.1 的计算顺序改成其他模型的通用近似。

### 2.2 streaming 量化及通用推理：12 个文件

| 文件 | 类型 | 提交 | 修改用途 |
| --- | --- | --- | --- |
| `src/llmcompressor/streaming/__init__.py` | 修改 | T1、T2 | 导出 V4.1 materializer，以及 `StreamingInference`、`load_streaming_model`、`compare_cached_to_recompute` |
| `src/llmcompressor/streaming/activations/store.py` | 修改 | T1 | 递归快照和设备迁移共享 memo，保持跨结构引用的张量复用关系，支持 CSA2 共享状态 |
| `src/llmcompressor/streaming/finalize.py` | 修改 | T1 | 支持 E8M0 元数据；把原样保留的 Engram shard 纳入最终 checkpoint 索引和状态校验 |
| `src/llmcompressor/streaming/materialization/__init__.py` | 修改 | T1 | 导出 V4.1 权重源和 materializer |
| `src/llmcompressor/streaming/materialization/base.py` | 修改 | T1 | 增加 `save_preserved_tensors` 扩展接口，使模型适配器能够保留外部量化状态 |
| `src/llmcompressor/streaming/materialization/deepseek_v41.py` | 新增 | T1 | V4.1 block32 FP8/FP4 源权重物化、Engram payload 原样保留、输出配置与 DSpark/MTP 配置处理 |
| `src/llmcompressor/streaming/materialization/transformers.py` | 修改 | T1 | 在 Transformers 转换包装层转发原样张量保留接口 |
| `src/llmcompressor/streaming/inference/__init__.py` | 新增 | T2 | 通用 streaming 推理模块公共接口 |
| `src/llmcompressor/streaming/inference/api.py` | 新增 | T2 | 复用原生模型 forward/generate 和 Accelerate CPU offload hooks；按模块/预加载组装载与释放权重，检查设备权重预算及调用生命周期 |
| `src/llmcompressor/streaming/inference/weights.py` | 新增 | T2 | checkpoint 支撑的惰性权重 Mapping；按请求读取、解码当前张量，描述执行 dtype、量化格式、回退方式和准备内存预算 |
| `src/llmcompressor/streaming/inference/quantization.py` | 新增 | T2 | 解析量化方案、安装 activation Q/DQ；支持量化权重解码后以浮点算子的参考推理路径 |
| `src/llmcompressor/streaming/inference/compare.py` | 新增 | T2 | 比较缓存 decode 与完整前缀重算 logits，输出误差及一致性结果 |

当前推理路径遵循模型声明的 dtype：普通权重可解码为 BF16，
官方要求保留 FP32 的参数按 FP32 加载。W8A8/W4A8 的参考执行采用权重解码
和 activation Q/DQ；不能将其表述为已经实现项目原生 INT8/INT4 GEMM。
Mapping 不持有完整解码后的 state dict。设备预算检查针对权重工作集，
不等同于所有运行时显存的硬上限。

### 2.3 示例：3 个文件

| 文件 | 类型 | 提交 | 修改用途 |
| --- | --- | --- | --- |
| `examples/streaming_oneshot/deepseek_v41_wNa8.py` | 新增 | T1 | V4.1 streaming 量化入口，覆盖 WNa8、W8A8、W4A8 和 DSpark 校准选项 |
| `examples/streaming_oneshot/README_deepseek_v41.md` | 新增 | T1 | V4.1 量化示例用法、checkpoint 格式及适配边界 |
| `examples/streaming_inference/generate.py` | 新增 | T2 | 加载 streaming checkpoint，执行原生生成并报告权重加载与执行信息 |

### 2.4 创建、诊断及对照工具：7 个文件

| 文件 | 类型 | 提交 | 修改用途 |
| --- | --- | --- | --- |
| `tools/create_tiny_dsv41.py` | 新增 | T1 | 保留 40 层文本、32 层 vision、3 级 DSpark、CSA2 与 Engram 拓扑，缩小通道、专家和表规模，创建并微调 tiny fixture |
| `tools/validate_tiny_dsv41.py` | 新增 | T1、T2 | 检查 tiny checkpoint、源格式、量化导出、文本/vision/DSpark 和回载路径 |
| `tools/validate_tiny_dsv41_inference.py` | 新增 | T2 | 检查 source、WNa8、W8A8、W4A8 的 streaming/resident prefill、decode、生成与 cache/recompute 一致性 |
| `tools/diagnose_tiny_dsv41_precision.py` | 新增 | T2 | 追踪完整前缀与增量执行的中间差异，定位 mHC 投影、系数及残差合并的精度分歧 |
| `tools/audit_tiny_dsv41_precision.py` | 新增 | T2 | 审核官方模型参数和中间计算的精度边界，辅助 FP32/BF16 诊断 |
| `tools/validate_dsv41_gpu_kernels.py` | 新增 | T2、T3 | H20/FP8 GPU 上的官方与项目算子对照；包含 HC、sparse、FP8、mHC 形状诊断及 FP4 软件语义，记录环境、源文件 hash 和异常 |
| `tools/compare_dsv41_kernel_level.py` | 新增 | T4 | 绕过模型、checkpoint、Transformers 和 pipeline，直接对照官方 kernel 公共函数、项目实现与计算顺序参考 |

两个 GPU 工具均不要求执行原生 FP4 kernel。kernel 级工具当前对照的是官方
`hc_split_sinkhorn`、`sparse_attn`、`act_quant`、`fp8_gemm` 的输入输出，
尚未导出 TileLang kernel 内部逐块中间张量。

### 2.5 测试：7 个文件

| 文件 | 类型 | 提交 | 修改用途 |
| --- | --- | --- | --- |
| `tests/llmcompressor/modeling/test_deepseek_v41.py` | 新增 | T1、T2、T3 | 配置/工厂、模型、CSA2、Engram、DSpark、cache、加载和 FP32 参数精度边界测试 |
| `tests/llmcompressor/modeling/test_deepseek_v41_reference.py` | 新增 | T1、T2 | 与官方 Python 模型源码比较；CPU 环境下 GPU primitive 由参考实现替代 |
| `tests/llmcompressor/modeling/test_deepseek_v4_kernels.py` | 修改 | T1、T3 | 官方 backend ABI、64-KV 在线计算顺序、全屏蔽输出及有限梯度检查 |
| `tests/llmcompressor/streaming/inference/test_inference.py` | 新增 | T2 | 惰性读取、offload、量化回退、预算、生成、异常释放与缓存对照工具测试 |
| `tests/tools/test_create_tiny_dsv41.py` | 新增 | T1 | tiny 配置、完整拓扑和源权重格式检查 |
| `tests/tools/test_validate_dsv41_gpu_kernels.py` | 新增 | T2、T3 | GPU 工具参考计算、形状矩阵、生产形状和 CPU 自检测试 |
| `tests/tools/test_compare_dsv41_kernel_level.py` | 新增 | T4 | kernel 级工具导入、形状矩阵和 CPU 自检测试 |

### 2.6 文档与分析报告：9 个文件

| 文件 | 类型 | 提交 | 修改用途 |
| --- | --- | --- | --- |
| `docs/reports/deepseek_v41_vs_v4_flash_quantization_2026-10-17.md` | 新增 | T1 | V4/V4.1 结构差异、权重格式、Transformers 支持与量化适配分析 |
| `docs/developer/streaming-inference-offload-reuse.md` | 新增 | T2 | Transformers/Accelerate CPU offload 复用评估 |
| `docs/developer/streaming-inference-plan.md` | 新增 | T2、T3 | 通用推理 pipeline 方案、接口、预算、模型责任与验收边界 |
| `docs/guides/streaming_inference.md` | 新增 | T2、T3 | 推理 API、示例、量化回退、dtype 与限制 |
| `docs/guides/dsv41_gpu_kernel_alignment.md` | 新增 | T2、T3、T4 | GPU 对照运行命令、依赖、检查项、数值指标和报告解释 |
| `docs/guides/streaming_ptq.md` | 修改 | T2 | 增加 streaming 推理文档入口 |
| `docs/reports/dsv41_streaming_precision_2026-10-17.md` | 新增 | T2 | tiny streaming 精度结果与 mHC BF16 中点问题 |
| `docs/reports/dsv41_official_precision_audit_2026-10-17.md` | 新增 | T2 | 官方代码精度审查，记录 FP32/BF16 计算边界 |
| `docs/reports/dsv41_official_alignment_2026-10-18.md` | 新增 | T3 | GPU 报告分析、在线 softmax 修正、FP32 参数边界和更新后的验证状态 |

## 3. 特性分支上另有的后续提交

以下两次提交出现在上述 4 次适配提交之后。它们不计入第 2 节的 48 个文件，
在此单独登记，避免把其他后续工作归入 V4.1 kernel 对齐改动。

| 提交 | 文件 | 用途 |
| --- | --- | --- |
| `b62887113e60f337313be40d04612a4e74d50932` | `examples/streaming_oneshot/deepseek_v4_wNa8.py` | V4 原生模型加载入口修正 |
| `b62887113e60f337313be40d04612a4e74d50932` | `src/llmcompressor/modeling/deepseekv4/config.py` | V4 配置与加载兼容修正 |
| `b62887113e60f337313be40d04612a4e74d50932` | `tests/examples/test_deepseek_v4_wna8.py` | 对应 V4 示例与加载测试 |
| `1874bbd5f33945a2ad0f8ac38ea1a54ead82de8f` | `usage.md` | P800 WNa8/W8A8 量化运行命令与用法说明 |

其中 V4 的 3 个文件曾作为已有暂存修改排除在 T1～T4 之外，目前已由
`b62887113` 单独提交。核对时 `main` 的 `146720b5f` 在这 3 个文件上的内容
与该提交一致。

若统计整个特性分支 `77f874751..1874bbd5f`，则是 **52 个文件：
新增 39 个、修改 13 个、删除 0 个**。本记录文件自身不计入这些历史统计。

## 4. 本地模型产物与外部报告

以下是验证输入或输出，不属于上述 Git 源码修改清单。

| 路径 | 内容 |
| --- | --- |
| `.cache/tiny-models/DeepSeek-V4.1-Flash-Tiny/` | 完整拓扑 tiny 模型、source FP8/FP4 及 WNa8/W8A8/W4A8 验证产物 |
| `/tmp/dsv41-streaming-official-order-2026-10-18.json` | 在线计算顺序修正后的完整 tiny 推理验证报告 |
| `/Users/ang/Downloads/dsv41-gpu-alignment.json` | 第一轮 GPU 对照：17 passed、0 failed、0 error、1 diagnostic，项目实现未参与 |
| `/Users/ang/Downloads/dsv41-gpu-project-full.json` | 第二轮 GPU 对照：51 passed、0 failed、5 error、2 diagnostic，项目实现已参与；在 T3 修正前运行 |
| `/Users/ang/Downloads/dsv41-kernel-level.json` | 最新 kernel 级对照：8 passed、2 failed、0 error；项目源码 hash 与 T4 快照匹配 |

模型权重和 GPU 报告不会随上述源码提交自动分发。`/tmp` 中的报告属于临时产物。

## 5. 验证结果与未完成事项

以下保留已有验证结果；本次整理记录没有重新运行模型或 GPU 测试。

### 5.1 本地回归与完整 tiny

T3 阶段相关回归记录为 `330 passed, 4 skipped`。T4 阶段新旧 GPU 工具的
相关测试记录为 `17 passed`，CPU 自检通过。官方 Python 源码对照使用 CPU
参考 primitive，不等于执行官方 CUDA kernel。

完整 tiny 使用 40 层文本、32 层 vision、3 级 DSpark，复用已有 checkpoint，
并非每次精度修正后重新量化。四种权重模式的 streaming/resident prefill、
decode、生成、vision 与 DSpark 对照均一致；cache/recompute 结果另行验收：

| 权重模式 | cache/recompute 最大 logit 绝对差 | 状态 |
| --- | ---: | --- |
| source FP8/FP4 | 约 `9.536743e-7` | 通过已有验收 |
| WNa8 | 约 `1.907349e-6` | 通过已有验收 |
| W8A8 | 约 `0.0138549805` | 未通过 |
| W4A8 | 约 `0.0305302143` | 未通过 |

W8A8/W4A8 的缓存一致性尚未解决。已有诊断中 mHC FP32 合并值约 `1.91e-6`
的差异可以跨过 BF16 中点，形成较大的 BF16 残差差异。不能用
streaming/resident 一致代替 cache/recompute 一致，也没有通过统一提高模型
dtype 或放宽验收阈值宣布修复。

### 5.2 最新 GPU kernel 对照

环境为 H20、PyTorch `2.13.0+cu130`、CUDA `13.0`、TileLang `0.1.13`；
TF32 和 BF16 reduced-precision reduction 在脚本中关闭。原生 FP4 未执行。

| 算子 | 结果 | 解释 |
| --- | --- | --- |
| HC split/Sinkhorn | 4/4 通过 | 官方/项目最大绝对差约 `2.3842e-7`，通过严格 FP32 阈值，部分结果非逐元素相等 |
| FP8 activation | 2/2 通过 | 编码、scale、Q/DQ 相关检查精确相等 |
| FP8 GEMM | 2/2 通过 | 官方与分块 scale 参考精确相等；项目尚无独立原生 FP8 GEMM |
| sparse，H=4、D=512、K=65 | 失败 | 官方/项目有 6/28,672 个不同元素，最大绝对差 `0.00390625`，最大 BF16 步数差 1 |
| sparse，H=64、D=512、K=640 | 失败 | 官方/项目有 933/458,752 个不同元素，最大绝对差 `0.001953125`，最大 BF16 步数差 171 |

两组 sparse 的项目实现与计算顺序参考均精确相等；官方和项目各自的完整
query 序列/单独最后一个 query 对照、全屏蔽零输出也精确相等。
这些测试固定 KV 与索引，不覆盖完整模型的 cache 构建和更新。

64-head 用例中，最大 BF16 步数差 171 的位置和值没有被报告保存。
不能把该用例的全部差异归结为只差一个 BF16 舍入。尚需逐块比较 QK、
running max、exp/BF16 权重、分母、value 累加器和最终输出转换，定位首次分歧。
当前报告只能确认存在官方/项目数值差异，尚不能确定具体底层运算是根因。

### 5.3 当前能力边界

- 通用 pipeline 复用原生模型推理与 Accelerate offload；模型专用计算和 cache
  由模型实现负责。
- V4.1 cache 当前支持无 padding、batch=1、追加式 prefill/decode；
  beam 扩展和非零回滚等场景尚未支持。
- 旧 V4 原生模型的 cache 集成仍待完成。
- FP8 activation 对齐不等于 W8A8/W4A8 checkpoint 权重及完整 logits 已对齐。
- 当前 kernel 级工具未覆盖完整 mHC 投影和残差合并，也未覆盖真实模型权重。
- 保留的模型差异包括为缓存一致性使用稳定 top-k tie-break，以及继承的
  text RMSNorm FP32 参数存储；不能宣称整个模型与官方逐位一致。
- 生产规模完整权重推理和全部模型路径尚未通过本记录中的 tiny/算子测试完成验收。

## 6. 清单复核命令

在仓库根目录执行：

```bash
# 本会话 4 次适配提交涉及的文件和净差异。
git diff --name-status 77f874751 00e29c74a
git diff --stat 77f874751 00e29c74a

# 适配提交历史。
git log --oneline --reverse 77f874751..00e29c74a

# 包含后续 2 次提交的固定快照。
git diff --name-status 77f874751 1874bbd5f
```

本记录新增路径为
`docs/reports/dsv41_adaptation_change_log_2026-10-18.md`。
