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

## 7. 后续：sparse attention 逐块诊断

本轮继续在 `feat/deepseek-v41-native-quantization` 分支开展，
不改变第 1～3 节的历史提交统计。修改文件如下：

| 文件 | 类型 | 修改内容 |
| --- | --- | --- |
| `tools/dsv41_sparse_trace.py` | 新增 | 官方源码副本的中间值写出、逐块 PyTorch trace、首次差异和选中输出点轨迹 |
| `tools/compare_dsv41_kernel_level.py` | 修改 | 增加 `--trace-sparse`、原始/插桩输出一致性校验、原始失败结果保留和增量报告、可选张量与 CUDA 源码保存 |
| `tools/validate_dsv41_gpu_kernels.py` | 修改 | 比较指标增加最大 BF16 步数差对应的坐标和值，最大绝对差也增加原始坐标 |
| `tests/tools/test_dsv41_sparse_trace.py` | 新增 | 原始 AST 计算语句保持、mask、跨块 trace、首次差异定位和 head padding 检查 |
| `tests/tools/fixtures/dsv41_sparse_kernel.py.txt` | 新增 | 官方 sparse factory 源码节选，供无需 TileLang/CUDA 的 AST 测试使用 |
| `tests/tools/test_compare_dsv41_kernel_level.py` | 修改 | 诊断异常保留失败结果、无效插桩不得归因、可选真实 GPU 插桩验证 |
| `docs/guides/dsv41_gpu_kernel_alignment.md` | 修改 | 逐块 GPU 命令、输出字段、归因校验、张量保存及 checkpoint 单独验证说明 |
| `docs/reports/dsv41_adaptation_change_log_2026-10-18.md` | 修改 | 补充本节后续修改记录 |

本地工具测试为 `36 passed, 1 skipped`，跳过项为真实 CUDA 插桩验证；
CPU 自检通过。本地没有 CUDA，尚未确认插桩 TileLang kernel 的 GPU
编译、最终输出等价性或首次官方/项目分歧。

本轮另行运行已有 W8A8/W4A8 tiny checkpoint，报告为
`/tmp/dsv41-cache-separate-2026-10-18.json`。streaming/resident 的
prefill/decode 仍精确相等；缓存/重算最大 logit 差仍分别为
`0.01385498046875` 和 `0.030530214309692383`，按原有验收条件失败，
验证程序退出 1。该结果未被算子对照或诊断检查通过所替代。

快速 GPU 报告 `/Users/ang/Downloads/dsv41-sparse-trace-quick.json` 已确认：
插桩最终输出与原始官方 kernel 精确相等，trace 输出与项目及独立参考精确
相等，诊断归因有效；首次差异出现在 block 0 的 QK 阶段，最大绝对差约
`5.340576171875e-05`，但该次报告的 PyTorch trace 仍按公开 H=4 形状执行，
而官方 wrapper 实际将 H=4 补齐为 H=16。后续修正将 reference 也按 H=16
执行后裁剪，避免把 head padding 与 GEMM 归约顺序混在一起；需要在 GPU 上
重新运行后才能确认 QK 差异是否仍存在。
v2 报告 `/Users/ang/Downloads/dsv41-sparse-trace-quick-v2.json` 进一步确认：
head padding 修正后，`trace_reference_vs_project.exact` 仍为 true，官方插桩
输出校验仍为 true，首次差异仍位于 block 0 的 QK；最大绝对差约
`6.4849853515625e-05`。两个 block 的 `bf16_weights` 都精确相等，
所以当前主要假设是官方 `T.gemm` 与项目 PyTorch FP32 einsum 的 QK
归约路径差异，尚未证明是 softmax、BF16 权重转换或 value GEMM 的问题。
后续工具将比较 BF16 输入/FP32 累加的 PyTorch bmm 变体。

本轮新增 QK backend 诊断：

| 文件 | 修改内容 |
| --- | --- |
| `tools/dsv41_sparse_trace.py` | 增加 `fp32_einsum`、`bf16_bmm_out_fp32`、`bf16_bmm_default_cast` 三种 QK 路径的逐 block 对照 |
| `tools/compare_dsv41_kernel_level.py` | 在有效 sparse trace 中写入 `qk_backend_comparison` |
| `tests/tools/test_dsv41_sparse_trace.py` | 增加 QK backend 参考一致性测试 |
| `docs/guides/dsv41_gpu_kernel_alignment.md` | 记录 QK backend 字段和解释 |

本地工具测试更新为 `38 passed, 1 skipped`，CPU 自检通过。GPU 仍需重新运行
以获得三种 QK backend 与官方 `T.gemm` 的相似度排序。

### 7.1 v3 QK backend 结论与实现修正

GPU 报告 `/Users/ang/Downloads/dsv41-sparse-trace-quick-v3.json` 的
`h4_d512_k65` 结果确认：

- `fp32_einsum`：block 0 最大绝对差
  `0.000064849853515625`，与官方 QK snapshot 不精确；
- `bf16_bmm_out_fp32`：两个 block 均逐元素精确一致；
- `bf16_bmm_default_cast`：最大绝对差
  `0.12450790405273438`，不是官方路径；
- 官方与项目最终 BF16 输出仍有 6 个元素、最大 1 个 BF16 step 的差异，
  但 `first_bf16_weight_divergence` 仍为 `null`，说明差异首先出现在
  QK GEMM，而不是 exp 的 BF16 权重转换。

据此，本轮将以下实现统一为官方的 QK dtype contract：

| 文件 | 修改 |
| --- | --- |
| `src/llmcompressor/modeling/deepseekv4/kernels.py` | CUDA + BF16 输入时使用 BF16 输入、FP32 输出的 `torch.bmm`；保留 BF16 gathered KV，value 累加继续使用 FP32 |
| `tools/validate_dsv41_gpu_kernels.py` | 独立 sparse reference 使用同一 QK contract，CPU 回退 FP32 einsum |
| `tools/dsv41_sparse_trace.py` | GPU trace 默认选择 `bf16_bmm_out_fp32`，CPU 选择 `fp32_einsum` |
| `tools/compare_dsv41_kernel_level.py` | 报告增加 `reference_qk_backend` 字段 |
| `docs/guides/dsv41_gpu_kernel_alignment.md` | 记录 v3 结论和 v4 复测边界 |

本地回归为 `43 passed, 1 skipped`。本地没有 CUDA，因此尚未把 sparse
官方/项目比较标记为通过；需要在 H20 上重新运行 v4 命令，继续检查 QK
修正后是否还存在 running max、exp/BF16 权重、denominator、value
accumulator 或最终 BF16 cast 阶段的差异。W8A8/W4A8 cache/recompute
问题仍独立保留，未由本次 QK 修正代替。

### 7.2 v4 sparse trace 结果与 value GEMM 修正

快速报告 `/Users/ang/Downloads/dsv41-sparse-trace-quick-v4.json` 在 H20、
PyTorch `2.13.0+cu130`、CUDA `13.0`、TileLang `0.1.13` 上通过。该结果
不是放宽阈值：`official_vs_project`、`project_vs_order_reference`、
`instrumented_vs_official` 和最终 BF16 输出均为精确一致。

逐块阶段的结果为：

| 阶段 | v4 结果 |
| --- | --- |
| QK | 精确一致 |
| scaled scores / running max / rescale | 精确一致 |
| exp / BF16 weights | 精确一致 |
| block sum | 首次逐元素差异；4 个元素，最大绝对差 `1.9073486328125e-6`，严格阈值内 |
| denominator | 继承 block sum 的同量级差异 |
| value accumulator | 同量级 FP32 差异，最终 BF16 输出未受影响 |
| normalized | 最大绝对差 `2.384185791015625e-7`，严格阈值内 |
| final BF16 output | 精确一致 |

`block_sum` 的差异来自官方 `T.reduce_sum` 与 PyTorch `torch.sum` 的归约
树/顺序，不是 QK、mHC 控制系数或 BF16 权重转换差异。官方源码中的 value
路径是 `T.copy(acc_s, acc_s_cast)` 后执行 BF16 `acc_s_cast × kv_shared`
到 FP32 `acc_o` 的 `T.gemm`；此前项目/reference 把两侧先转成 FP32 后做
einsum，虽然 v4 最终输出已精确一致，但中间 accumulator 不完全一致。
本轮已修正为 CUDA 上 BF16 输入、FP32 输出的 value `torch.bmm`，CPU 回退
FP32 einsum。

本轮修改文件：

| 文件 | 修改 |
| --- | --- |
| `src/llmcompressor/modeling/deepseekv4/kernels.py` | 增加 `_value_gemm_fp32`，对齐官方 value GEMM dtype contract |
| `tools/validate_dsv41_gpu_kernels.py` | 独立 reference 对齐 BF16 value GEMM |
| `tools/dsv41_sparse_trace.py` | trace reference 对齐 BF16 value GEMM |

本地回归仍为 `43 passed, 1 skipped`，CPU self-test 通过。下一轮 GPU 复测
应重点确认 `accumulator`/`rescaled_accumulator` 是否变为精确一致；即使
这些中间值仍有归约顺序差异，只要严格阈值和最终 BF16 输出保持通过，也必须
在报告中保留该差异，不能将其抹掉。W8A8/W4A8 cache/recompute 仍是独立问题。

### 7.3 v5 sparse trace 结论

报告 `/Users/ang/Downloads/dsv41-sparse-trace-quick-v5.json` 在 H20、
PyTorch `2.13.0+cu130`、CUDA `13.0`、TileLang `0.1.13` 上通过。所有基础
检查、插桩输出 gate、项目/reference 精确比较、整段/尾 token 一致性和全屏蔽
query 检查均通过；最终 BF16 output 为逐元素精确一致。

中间阶段的最终状态：

| 阶段 | v5 结果 | 结论 |
| --- | --- | --- |
| QK 到 BF16 weights | 全部精确一致 | QK 和 BF16 权重 contract 已对齐 |
| block 0 accumulator | 精确一致 | value GEMM 修正生效 |
| block 1 rescaled accumulator | 精确一致 | block 间在线重缩放一致 |
| block 1 accumulator | 127 个元素不同，最大 `4.76837158203125e-7` | 仅剩 value GEMM 的 FP32 归约顺序差异 |
| block sum | 4 个元素不同，最大 `1.9073486328125e-6` | 首次逐元素差异，来自 `T.reduce_sum` 与 PyTorch reduction tree |
| denominator / normalized | 严格阈值内 | 继承 block sum 的微小误差 |
| final BF16 output | 精确一致 | 对模型可见输出无差异 |

因此当前 sparse 适配已经达到“官方 dtype contract + 最终 BF16 输出精确”
的状态，但还没有达到“所有 FP32 中间张量逐位一致”。这两个中间 reduction
差异必须继续保留，不能把 `status=passed` 解读成内部 bitwise 对齐完成。
下一步应运行不带 `--quick` 的完整 sparse 形状矩阵，重点覆盖
`h64_d512_k640`；该用例可以判断 reduction 差异是否随 head 数和 block 数
扩大。W8A8/W4A8 cache/recompute 仍独立验收。

### 7.4 v6 完整 sparse 矩阵结果与 reduction 诊断

报告 `/Users/ang/Downloads/dsv41-sparse-trace-full-v6.json` 的环境仍为
H20、PyTorch `2.13.0+cu130`、CUDA `13.0`、TileLang `0.1.13`。完整矩阵
包含两个 case：

| case | 结果 | 关键指标 |
| --- | --- | --- |
| `h4_d512_k65` | 通过 | 最终 BF16 输出精确一致；仅保留 block 0 `block_sum` 的 4 个 FP32 微差 |
| `h64_d512_k640` | 失败 | 56 个 BF16 输出不同，最大绝对差 `0.0009765625`，最大 BF16 步数差 10 |

H64 用例的归因 gate 全部通过，且：

- `project_vs_order_reference.exact = true`；
- QK、scaled scores、running max、rescale、exp、BF16 weights 均逐元素一致；
- 首次逐元素差异仍是 block 0 的 `block_sum`，最大绝对差
  `1.9073486328125e-6`；
- 首次严格分歧出现在 block 2 的 `accumulator`，最大绝对差
  `4.76837158203125e-6`；
- 后续 block 的 `rescaled_accumulator` 和 `accumulator` 差异逐步放大，
  到 block 9 的 accumulator 最大绝对差约 `2.47955322265625e-5`；
- 最终 normalized 仍在严格阈值内，但 56 个值在 BF16 cast 时跨过舍入点。

最大绝对输出差坐标为 `[0, 0, 16, 55]`：

```text
official:  -0.16796875
reference: -0.1689453125
```

该点的 FP32 normalized 值为：

```text
official:  -0.16845703125
reference: -0.16845709085464478
```

两者只差约 `5.96e-8`，但位于两个相邻 BF16 值
`-0.16796875` 和 `-0.1689453125` 的中点附近，因此最终 cast 结果不同。
最大 BF16 步数差坐标为 `[0, 6, 28, 31]`，两侧最终输出分别为
`1.909211277961731e-7` 和 `2.0023435354232788e-7`，相差 10 个 BF16
表示步数。这个结果证明中间 FP32 reduction 差异已经足以影响最终 BF16
舍入，不能只看 `normalized` 的相对误差。

这表明 H64 失败不是 QK 或 BF16 权重转换问题，而是 block-sum/value
GEMM 的 FP32 reduction 顺序在多 block 在线累加中积累，并在最终 BF16 cast
时触发 midpoint/邻近舍入差异。本轮只增加诊断候选，不改生产 sparse kernel：

| 诊断字段 | 候选 |
| --- | --- |
| `block_sum_backend_comparison` | `torch_sum`、`pairwise_sum`、`sequential_sum` |
| `value_gemm_backend_comparison` | FP32 einsum、BF16 bmm FP32 输出、BF16 默认输出、相邻二叉树 BF16×BF16→FP32 |

这些候选只用于定位官方 `T.reduce_sum/T.gemm` 的归约树，不会自动替换生产
路径，也不会放宽现有失败条件。W8A8/W4A8 cache/recompute 继续单独验收。
