# DeepSeek-V4.1-Flash 与 DeepSeek-V4-Flash：量化适配分析

分析日期：2026-10-17。结论针对本地两个目录的实际配置、权重索引与参考实现，以及当前工作区的 llm-compressor；不假定其他同名模型版本具有完全相同的布局。

V4.1 需要独立的模型适配层。可以复用 V4 的量化算法、普通 FP4 解码与部分 streaming 基础设施，但不能只修改 V4 的维度或替换 `kernel.py`。主要新增工作是 CSA2 跨层状态、Single-Pass mHC、Engram 按行加载及原格式保存，以及新的配置和 DSpark 图。

后续实施状态：已按本文分析加入 native V4.1 配置/模型、MoE wrapper、materializer
和 WNa8/W8A8/W4A8 示例，并显式适配 DSpark。第 9 节保留的是实施前缺口记录。
目前完成缩小模型的流式量化和 CPU 参考数值对齐；真实权重/GPU 及目标推理引擎
验证仍待完成。用法、范围和验证入口见
[V4.1 流式量化说明](../../examples/streaming_oneshot/README_deepseek_v41.md)。

## 1. 分析范围与证据强度

本地输入：

- [V4 config.json](/Users/ang/models/DeepSeek-V4-Flash/config.json)、[权重索引](/Users/ang/models/DeepSeek-V4-Flash/model.safetensors.index.json)、[参考模型](/Users/ang/models/DeepSeek-V4-Flash/inference/model.py)。
- [V4.1 config.json](/Users/ang/models/DeepSeek-V4.1-Flash/config.json)、[权重索引](/Users/ang/models/DeepSeek-V4.1-Flash/model.safetensors.index.json)、[参考模型](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py)、[技术报告](/Users/ang/models/DeepSeek-V4.1-Flash/DeepSeek_V41_Tech_Report.pdf)。
- 当前仓库的 native V4、0731 DSpark、streaming materializer 和 WNa8 示例。保留了工作区已有改动。

两个目录均没有索引引用的 `.safetensors` 分片。索引只记录张量名、分片名和总字节数，不包含张量 shape/dtype。因此：

- 张量名称、数量、层归属、分片归属和配置差异已直接核对。
- 下文 shape 由配置和参考模型在 PyTorch `meta` 设备上推导，未分配实际模型权重。
- 原始存储 dtype 根据参考模型和 `convert.py` 判断，尚未读取真实 safetensors header 验证。
- 已进行小张量解码检查；没有进行真实模型推理、量化精度或 GPU kernel 验证。

`meta` 构建使用原尺寸配置，仅将运行时缓存长度设为 256、batch 设为 1；禁用 kernel 执行，并跳过不含可训练权重的 tokenizer/hash 状态初始化。模型与索引的名称差集均可解释：参考运行时提前将每层 `wo_a` 解码为 BF16，因而少了其 `.scale`；MTP 运行时含共享 embedding/head 的别名，原始索引省略这些别名。

## 2. 总体结构对比

| 项目 | V4-Flash | V4.1-Flash | 适配影响 |
|---|---|---|---|
| `architectures` | `DeepseekV4ForCausalLM` | `DeepseekV41ForCausalLM` | 新模型注册 |
| `model_type` | `deepseek_v4` | `deepseek_v41` | 新配置分发 |
| 配置布局 | 文本参数平铺 | `text_config` + `vision_config`，源量化信息在顶层 | 不能直接用旧 config 类 |
| 输入 | 文本 | 文本和图像 | processor、image mask、校准数据有变化 |
| 主干层数 | 43 | 40，20 层 encoder + 20 层 decoder | CED 执行语义 |
| hidden size | 4096 | 5120 | +25% |
| MoE intermediate | 2048 | 2304 | +12.5% |
| routed experts / 层 | 256 | 384 | +50% |
| activated routed experts / token | 6 | 6 | 主干 top-k 不变 |
| shared experts | 1 | 1 | SwiGLU 基础结构可复用 |
| Q LoRA rank | 1024 | 1280 | Q 投影变化 |
| attention heads / head dim | 64 / 512 | 64 / 512 | 主注意力头布局不变 |
| O groups / rank | 8 / 1024 | 8 / 1024 | `wo_a` shape 不变 |
| index heads / dim / top-k | 64 / 128 / 512 | 32 / 128 / 512 | 索引器 Q 和 weights_proj 变化 |
| SWA window | 128 | 128 | 窗口大小不变，缓存精度路径变化 |
| 全局注意力 | CSA ratio=4 与 HCA ratio=128 交替 | CSA2 ratio=2 / ratio=1，跨层共享 | 需要新的 Attention/Compressor/Indexer |
| 前置 hash MoE | 前 3 层有 `tid2eid` | 无 `tid2eid`，均为 score routing | 不能继承 `num_hash_layers=3` 默认值 |
| mHC | 本子层生成并消费 pre-mix | 消费上个子层产生的 pre-mix | 新的 block 边界状态 |
| RMSNorm epsilon | `1e-6` | `1e-20` | 必须保留准确 epsilon 与 FP32 统计 |
| MTP | 1 层传统 MTP | 3 层 DSpark，draft block=5 | 新校准图 |
| Engram | 无 | 第 1、14 层，2/3/4-gram lookup | 巨型 FP8 记忆表 |
| vision | 无 | 32 层 ViT + 两层 aligner | 可分阶段量化 |
| 普通 FP8 权重块 | 128×128 | 32×32 | materializer 必须改分块规则 |
| routed FP4 权重组 | 沿 K 每 32 元素一个 E8M0 scale | 相同 | FP4 解码主体可复用 |

词表大小均为 129280，最大位置长度均为 1048576。mHC expansion=4、Sinkhorn iterations=20、SwiGLU clamp=10、主干路由缩放=1.5 也保持一致。配置证据见两份 `config.json`。

技术报告 §2.1 明确区分 **552B backbone** 和 **196B Engram**。552B 不能当作包含记忆表、DSpark 的全部 checkpoint 参数量。按参考模型推导，V4 文本主干约 284.33B；V4.1 文本主干（不含 Engram）约 551.57B，另有约 0.485B vision/aligner、0.315B Engram 投影/门控、196.614B Engram 表和 14.225B DSpark 参数。这些是结构计数，不是实际权重文件测量值。

## 3. 权重索引与关键 shape

| 项目 | V4 | V4.1 |
|---|---:|---:|
| 索引张量数量，含 scale | 69,187 | 96,085 |
| 索引分片数量 | 46 | 48 |
| 本地实际存在的索引分片 | 0 | 0 |
| `metadata.total_size` | 159,609,485,896 bytes | 510,286,023,000 bytes |
| 十进制 GB / 二进制 GiB | 159.61 GB / 148.65 GiB | 510.29 GB / 475.24 GiB |
| 主干 routed expert 实例数 | 43×256=11,008 | 40×384=15,360 |
| 主干 routed expert 三个 weight 数量 | 33,024 | 46,080 |
| MTP routed expert 实例数 | 1×256=256 | 3×128=384 |

V4.1 索引体积约为 V4 的 3.20 倍；其中新增记忆表占很大比例，不能只按主干参数增长估算存储。

以下矩阵采用 `[out_features, in_features]`，为逻辑形状：

| 模块 | V4 | V4.1 |
|---|---|---|
| `embed.weight` / `head.weight` | `[129280, 4096]` | `[129280, 5120]` |
| `attn.wq_a` | `[1024, 4096]` | `[1280, 5120]` |
| `attn.wq_b` | `[32768, 1024]` | `[32768, 1280]` |
| `attn.wkv` | `[512, 4096]` | `[512, 5120]` |
| `attn.wo_a` | `[8192, 4096]` | `[8192, 4096]` |
| `attn.wo_b` | `[4096, 8192]` | `[5120, 8192]` |
| `attn.indexer.wq_b` | `[8192, 1024]` | `[4096, 1280]` |
| `attn.indexer.weights_proj` | `[64, 4096]` | `[32, 5120]` |
| `attn.indexer.wk` | 无，走独立 compressor | `[128, 512]`，仅 Full 层 |
| `ffn.gate.weight` | `[256, 4096]` | `[384, 5120]`，DSpark 为 `[128,5120]` |
| `ffn.experts.i.w1/w3` | `[2048, 4096]` | `[2304, 5120]` |
| `ffn.experts.i.w2` | `[4096, 2048]` | `[5120, 2304]` |
| `hc_attn_fn` / `hc_ffn_fn` | `[24, 16384]` | `[24, 20480]` |
| `engram.wkv` | 无 | `[25600, 6144]` |
| `mtp.0.main_proj` | 无，传统 MTP 为 e_proj/h_proj | `[5120, 15360]` |
| `mtp.2.confidence_head.proj` | 无 | `[1, 5376]` |
| `aligner.w1` / `aligner.w2` | 无 | `[5120, 9216]` / `[5120, 5120]` |

FP4 专家的原始 checkpoint 预期用 INT8 字节装两个 E2M1 值；转换后的运行时用 `float4_e2m1fn_x2`。V4.1 `w1/w3` 的 packed shape 为 `[2304,2560]`，scale 为 `[2304,160]`；`w2` 的 packed shape 为 `[5120,1152]`，scale 为 `[5120,72]`。不能把“字节以 INT8 保存”解释成普通 INT8 数值。证据：[V4.1 convert.py](/Users/ang/models/DeepSeek-V4.1-Flash/inference/convert.py:17)、[Linear](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py:210)。

`wo_a` 在两版源索引里都有 scale，但参考 `convert.py` 将其解码为 BF16 后再运行 grouped einsum。它是按组的输出投影，不能直接等同于一个普通 dense Linear 的乘法。第一阶段建议继续保留解码后的 BF16 精度，若要量化需有独立的 grouped 运算适配和验证。

## 4. CSA2 / CED 对 streaming 校准的影响

V4 主干第 0、1 层纯 SWA；第 2..42 层交替使用 ratio=4 与 ratio=128。索引确认有 41 个主 KV compressor、21 个 indexer；这些 indexer 还有各自独立的 compressor。

V4.1 的分配如下，层号均为从 0 开始：

| 层范围 | 压缩率 | Full 层 | Reindex 层 | 其他层 |
|---|---:|---|---|---|
| 0..1 | 0 | 无 | 无 | 纯 SWA |
| 2..7 | 2 | 2 | 无 | 3..7 Reuse |
| 8..13 | 2 | 8 | 无 | 9..13 Reuse |
| 14..19 | 2 | 14 | 无 | 15..19 Reuse |
| 20..23 | 1 | 20 | 无 | 21..23 Reuse |
| 24..27 | 1 | 无 | 24 | 25..27 Reuse |
| 28..31 | 1 | 无 | 28 | 29..31 Reuse |
| 32..35 | 1 | 无 | 32 | 33..35 Reuse |
| 36..39 | 1 | 无 | 36 | 37..39 Reuse |

索引确认只有 4 个主 KV compressor（2、8、14、20）、8 个 indexer（2、8、14、20、24、28、32、36），其中只有 4 个 indexer 具有 `wk/k_norm`。第 20 层 ratio=1，没有 `compressor.wgate`；只有 2、8、14 层保留该权重。V4 的 compressor APE 和 overlap 路径均已移除；V4.1 indexer K 从主 KV 的 RoPE 前 latent 投影，不再单独从 hidden states 压缩，也不沿用 V4 indexer 的 Hadamard 路径。

三种模式均计算本层的 Q 和 SWA KV。Full 产生共享 main KV、indexer K 和 Top-K；Reindex 复用 KV/K，但重新计算索引；Reuse 复用 KV 和最近一次索引。第 20 层还选出至多 2048 个 block、每 block 8 个位置的候选池，即最多 16384 个候选位置，供后续 Reindex 使用。

参考实现把共享状态放在模块外全局 `shared_attn` 中。为了让 FX/streaming 能跨子图正确保存、重放这些依赖，适配时应显式传递或管理：

- 当前 `hidden_states` 和 Single-Pass `pre_mix`；
- main KV、indexer K，以及它们各自的来源和有效长度；
- 最近的 Top-K、候选池和 SWA/全局 KV 拼接 offset；
- Engram 的哈希索引/文本 mask；
- 启用 DSpark 时的 target hidden states。

来源：[Attention / CSA2](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py:614)、[SharedAttentionRuntime](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py:1169)、技术报告 §2.2–2.3。

CED 的 decoder 全局 KV 来自最终 encoder 表示。在当前配置里只有 decoder 第 20 层创建这份全局 KV，后续 decoder 共享它。**ratio=1 表示不做 token 池化，仍然执行 Top-K 稀疏注意力。**

技术报告的 prefill=8B / decode=16B 激活量依赖 CED 与 SWA Bounded Replay：多数 prompt token 只跑 encoder，decoder 的局部 KV 通过末尾 128 token 的 replay 近似构建。发布的简明参考代码仍遍历全部 40 层，并未实现这些生产优化。量化第一阶段可以用完整 40 层 teacher-forced forward 收集全部主干统计；不应直接套生产 prefill 的跳层路径，否则 decoder 专家可能没有充分校准数据。若最终部署启用 replay，还应单独测其误差和量化误差的叠加。

参考层级索引也先计算完整 score 再 mask candidates；其语义可用于对齐，但不能据此声称该 Python 实现已经获得报告所述的候选域计算复杂度。

## 5. mHC 与 router 变化

V4 当前子层输入为 `pre_current × residual`；V4.1 为 `pre_previous × residual`。V4.1 attention 用传入的 `pre_mix`，FFN 用该 attention 产生的 `attn_pre`，再返回 `ffn_pre` 给下层。初始 pre-mix 为选择第一个 residual 分支的 one-hot；末尾直接用最后的 pre-mix collapse 后接 norm/head。

因此，V4.1 索引不再含 V4 的主干或 MTP `hc_head_fn/base/scale`。同名 `hc_attn_*` / `hc_ffn_*` 参数并不意味着 block 计算可直接复用。应保留这些控制参数的 FP32、RMS 统计的 FP32，以及 `norm_eps=1e-20` 与 `hc_eps=1e-6` 的区别。来源：[V4.1 Block](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py:909)、[最终 collapse](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py:1268)。

V4 前 3 层通过 token→expert 表决定专家；V4.1 改为 score-based routing，并为图像 token 增加 `ffn.gate.bias_vl`。索引中主干 40 层及 3 个 DSpark 层都存有该 bias。校准 wrapper 必须接受 `image_mask`，保留 bias 只影响选择、不参与路由权重计算的语义。现有 `CalibrationDeepseekV4MoE.forward(hidden_states, input_ids)` 不能原样替换 V4.1 MoE。

V4.1 专家参考路径在 `w2` 之前对 SwiGLU 激活乘 routing weight。对齐原生量化执行时，应保持这个位置，不能只依据无限精度线性代数把该乘法移动到量化线性层之后。来源：[Gate / Expert](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py:794)。

## 6. 必须区分的量化格式

| 对象 | V4.1 格式 / scale 布局 | 当前适配建议 |
|---|---|---|
| 普通 FP8 Linear 权重 | E4M3，32×32 block，UE8M0 scale | 复用普通 FP8 解码，分块改为 `(32,32)` |
| routed FP4 expert 权重 | E2M1，沿 K 每 32 值一个 E8M0 scale | 复用现有 packed FP4 解码 |
| Engram embedding 表 | E4M3，**每行**沿通道每 32 值一个 E8M0 scale | 独立按行读取、解码与原格式保存 |
| 全局 main KV cache | E2M1，每 16 通道一个 E4M3 scale，RoPE 后量化 | 独立 cache/QAT 模拟 |
| indexer Q/K | E2M1，每 32 通道一个 E8M0 scale | 保留参考量化路径 |
| SWA KV | FP8，每 32 通道一组，包含 RoPE 维度 | 保留参考量化路径 |
| `wo_a` 前的输出激活 | FP8 Q/DQ，每 32 通道一组 | baseline 对齐时也要覆盖 |

全局 KV 的格式参考 NVFP4，但没有第二级全局 scale，不能直接等同于任意标准 NVFP4 preset。它也不同于 checkpoint routed FP4 权重的 MXFP4 分组和 scale 类型。V4 参考实现对主 KV/SWA KV 只量化 non-RoPE 部分、每 64 通道一组；V4.1 主 KV 和 SWA 都覆盖 RoPE 部分。来源：[V4.1 _window_kv / _compress_kv](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py:701)、[FP4 kernel](/Users/ang/models/DeepSeek-V4.1-Flash/inference/kernel.py:128)、技术报告 §2.4.4。

普通 V4.1 `wq_a` 的 scale 推导为 `[40,160]`，V4 为 `[8,32]`；Engram 则是 `[table_rows,8]`。同为 `.scale` 后缀，不能使用一个统一的 block shape 解释。

小张量检查直接提取并执行当前仓库 materializer 的解码方法，结果：

- 普通 FP8 `(32,32)` 解码与显式展开 scale 的参考计算一致，最大绝对误差 0。
- 旧默认 `(128,128)` 对相同 V4.1 格式数据报 scale shape 不匹配。
- 即使普通 block 改为 `(32,32)`，对 Engram 行级 scale 仍报 shape 不匹配。
- FP4 group=32 解码与显式 nibble 查表结果一致，最大绝对误差 0。

这些检查验证格式处理判断，不证明真实权重、特殊 E8M0 值或 GPU kernel 数值正确性。

## 7. Engram 是加载与导出的首要工程问题

两张表推导如下：

| 所在层 | weight shape | scale shape | 仅表参数 | 按参考 FP8+E8M0 估算的存储 |
|---|---|---|---:|---:|
| 1 | `[384006168,256]` | `[384006168,8]` | 98,305,579,008 | 101,377,628,352 bytes |
| 14 | `[384016682,256]` | `[384016682,8]` | 98,308,270,592 | 101,380,404,048 bytes |
| 合计 | — | — | 196,613,849,600 | 202.76 GB / 188.83 GiB |

索引把第 1 层表的 weight/scale 放在 `model-00047-of-00048.safetensors`，第 14 层放在 `model-00048-of-00048.safetensors`。可把这些表与主干量化数据分别管理，但改写输出时仍需维护正确索引。

若把两表完整转 BF16，仅权重就约 393.23 GB / 366.22 GiB，且解码还有 FP32 中间量；通用 `get_tensor → weight.float() → dequant → BF16` 路径会产生极大的内存压力。

建议保留其源 FP8+scale 格式。哈希只依赖 token，按校准 batch 所需 hash IDs 读取对应行并解码，必要时用 CPU mmap/行缓存/预取。原表不应作为普通 Linear 的 IMatrix/GPTQ 对象，也不应仅通过 `ignore` 后交给通用“未量化参数导出”流程；后者依然可能全表解码并丢失源 scale。

每层有 `(4-1)×8=24` 个哈希列，lookup 后展开成 6144 维，再通过 `[25600,6144]` 的 `engram.wkv` 生成 key/value。该投影是可单独讨论量化的普通 FP8 Linear，和巨大 lookup 表需要不同策略。第一阶段建议将表原样保留，也先保留这个投影的源精度，待 baseline 对齐后再评估 W8。

Engram 的压缩 token map、素数桶大小、随机种子、2/3/4-gram 顺序，以及图像 token 断开 n-gram 的规则，都影响访问哪一行。它不只是 embedding 形状问题。来源：[Engram embedding / forward](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py:296)、[hash 实现](/Users/ang/models/DeepSeek-V4.1-Flash/inference/engram.py:18)。

## 8. 视觉、DSpark 与校准样本

视觉分支为 32 层、hidden=1024、16 heads、intermediate=2816、patch=14 的 ViT。3×3 下采样将通道变为 9216，再经 `9216→5120→5120` aligner 接入文本 embedding。图像跨度含 image start/end/newline 参数；图像 token 走专门的 router bias，且不贡献 Engram lookup。来源：[vision.py](/Users/ang/models/DeepSeek-V4.1-Flash/inference/vision.py)、[图像合并](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py:1231)。

可以先量化文本主干并原样保留视觉/aligner，但 Engram 在纯文本推理中仍是必需路径。若随后要量化视觉模块或保证视觉任务质量，需要图片输入以及混合 image/text 的语言层统计；纯文本统计不能覆盖这部分分布。

V4.1 DSpark 的关键配置为 3 个 block、每块 128 routed experts / top-3、SWA=128、draft 长度=5、target layers=`[37,38,39]`、Markov rank=256。第一个 block 有 `main_proj/main_norm`，最后一个有 Markov 与 confidence head，主 embedding/head 与主干共享。

当前仓库已有的 0731 DSpark 能提供代码参考，但有明确差异：

- 当前 `TraceFriendlyBlock` 取目标层**输出**，V4.1 参考实现取目标层执行前的 residual stream 均值。
- 当前沿用 V4 mHC 和末尾独立 hc_head；V4.1 使用 Single-Pass pre-mix collapse。
- 当前 Markov 参数是 `markov_w1/markov_w2`；V4.1 索引是 `markov_head.embed/head`。
- 当前 DSpark MoE 没有按 V4.1 的 128 experts / top-3 配置建模，不能复用主干 384 / top-6。
- 发布的普通生成循环不调用 `forward_spec`，因此正常主干校准不会自动覆盖 DSpark。需显式构建 draft/teacher-forcing 校准路径，或保留其源格式。

证据：[V4.1 DSpark](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py:1103)、[target hidden 收集](/Users/ang/models/DeepSeek-V4.1-Flash/inference/model.py:1263)、[当前仓库 DSpark](/Users/ang/Projects/llm-compressor/src/llmcompressor/modeling/deepseekv4/model_dspark.py:205)。

两份 `tokenizer_config.json` 相同，但 `tokenizer.json` 并不相同。程序对比发现基础 tokenizer 配置未变，变化集中在 `added_tokens`，包括 ID 128799 改为 `<｜System｜>`，129264 改为 `<｜deepseek_image｜>`。DSpark 配置的 noise ID 也为 128799，适配时应按配置保留。V4.1 的 DSML tag、数值 reasoning effort、中途 system message 格式均有变化，不能沿用示例中 V4 的 `encode_messages`。本地 V4.1 另有 `chat_template.jinja`，应与其 Python encoding 用测试样例核对，而不是仅依据 README 关于模板是否发布的描述。

来源：[V4.1 encoding 说明](/Users/ang/models/DeepSeek-V4.1-Flash/encoding/README.md:8)、[本地模板](/Users/ang/models/DeepSeek-V4.1-Flash/chat_template.jinja)。

## 9. 当前 llm-compressor 的具体适配缺口

| 位置 | 已有能力 / 实测 | V4.1 所需工作 |
|---|---|---|
| `modeling/deepseekv4/config.py` | V4 平铺配置和 0731 DSpark 字段 | 新建 V4.1 text/vision 配置，正确处理顶层源量化参数 |
| `modeling/deepseekv4/model.py` | V4 compressor/indexer/mHC | 新建 CSA2、Engram、Single-Pass block；复用基础算子 |
| `modeling/deepseek_v4.py` | V4 MoE 校准 wrapper | 支持 image mask 和不同层专家数的 V4.1 wrapper |
| `streaming/materialization/deepseek_v4.py` | 普通 FP8、packed FP4、raw key 映射 | 普通块 32；Engram 行级路径；源压缩表透传及 scale 保留 |
| streaming trace / loading / output | 子图、张量级加载与剩余张量导出 | 显式共享状态；巨表按行读取；防止静态导出解码整表 |
| `model_dspark.py` | 0731 DSpark | 对齐新输入位置、pre-mix、Markov 命名和 top-3 |
| `deepseek_v4_wNa8.py` | V4 W8A8/W4A8/WNa8、V4 encoding | 独立 V4.1 示例，先明确量化 target，再处理新 ignore 项 |
| finalize/config metadata | 默认配置主要按顶层更新 | 保留 text/vision 结构，定义源 Engram 与新量化层混合格式契约 |

直接实测旧 `ModelConfig.from_pretrained(V4.1目录)` 不会正确提取 `text_config`，得到旧默认值：

```text
hidden_size=7168, num_hidden_layers=61, moe_intermediate_size=3072,
num_hash_layers=3, q_lora_rank=1536, dspark_block_size=0
```

这比显式报“不支持”更容易误导。当前本地环境 Transformers 为 5.19.0，`AutoConfig.from_pretrained(..., local_files_only=True)` 对 `deepseek_v41` 报不识别模型类型；本地目录也没有 HF `configuration/modeling` remote-code 实现。新 adapter 需注册自己的配置/模型，不能假定配置文件写的 `transformers_version` 就保证可直接加载。

当前示例实例化 materializer 时没有从配置传入 FP8 block size，仍采用 `(128,128)` 默认值。即使修为 `(32,32)`，也只解决普通 FP8 Linear，不解决 Engram。当前原始 key 主要仍是 `embed/layers/head/mtp`，因而基础 key 规范化可以复用；新增 `vision/aligner/image_*`、Engram 和 Markov head 的映射需要独立核对。

`ignore` 建议必须结合模块用途判断：router、norm、mHC、attn sink、compressor、indexer weights_proj/wk、Engram 门控、confidence head、head/embedding 等先保持源精度。源 FP8 部分如何保留 scale 必须有明确机制，不能只列 ignore 正则。现有 WNa8 targets 还不会主动覆盖 `engram.wkv`、vision/aligner，因此应检查实际选中的 module 清单，而非仅观察任务成功退出。

## 10. 推荐实施顺序与验收条件

1. **P0：配置和结构契约。** 独立 V4.1 config/model 注册；从 nested config 精确构图，按索引核验主干、视觉、Engram 和 DSpark 全部键。明确源 FP4/FP8 与目标 INT4/INT8 的格式边界，保留已有 V4 路径。
2. **P0：不再量化的参考基线。** 先实现纯文本、完整 40 层 forward，保留 Engram lookup、CSA2 共享与候选限制、Single-Pass mHC 和原生 cache Q/DQ。与发布参考在相同权重和相同精度规则下对齐，避免把架构误差计入 PTQ 误差。
3. **P0：Engram 加载和保存。** 表保持 FP8+行级 scale；按所需行读取；导出不做全表 BF16 转换。确认保留索引和源/目标混合精度元数据能被目标推理引擎加载。
4. **P1：主干 W8A8。** 先覆盖普通 attention 投影和专家；报告 router/Top-K 变化、逐层输出误差、MoE 覆盖率及端到端指标。W8 可作为量化链路基线，但将原生 FP4 专家升为 INT8 不会恢复原始 BF16 信息，且可能增加 checkpoint 体积。
5. **P1：主干 WNa8。** routed experts 用 INT4，其余选定 Linear 用 INT8；对比的是已发布 FP4/FP8 源模型。原生 E2M1 FP4 转 INT4 是再次量化，不能因位数相同就假定无损。确认 2304 中间维度在目标 kernel 的分组和对齐限制内可用。
6. **P2：视觉与 DSpark。** 视觉使用多模态样本；DSpark 显式运行 draft 校准，并单独统计专家覆盖、接受率和 confidence 偏差。保留源格式的阶段也必须验证正确加载。
7. **P2：长上下文和生产优化。** 验证候选限制真正生效的长样本、prefill/decode 一致性、prefix cache/reset、bounded replay 和量化误差叠加，以及输出 checkpoint 在指定推理引擎的可加载性。

建议的关键正确性检查：

- 普通 FP8 block32、FP4 group32、Engram 行级 scale 三条解码路径分别对齐。
- 全局 KV 的 FP4 group16/E4M3 与 indexer 的 FP4 group32/E8M0 分别对齐。
- 完整 forward 与 streaming 子图输出一致，共享 KV/Top-K/pre-mix 无丢失或跨样本污染。
- ratio=2 的奇偶 token 边界、短于窗口的序列、长于候选池的序列均符合参考语义。
- 每条新样本独立重置 hash history；图像跨度正确屏蔽 Engram 并选择 VL router bias；padding 不进入真实 token 哈希。
- 如不运行 DSpark，确保所有相关权重保持可用，或明确从配置与输出中成套裁除，不能只是不收集统计却把未覆盖模块量化。
- 验证实际产物的 key、shape、dtype、scale 布局和最终推理，而不只验证量化流程本身。

当前可以确定适配工作量和格式风险；真实 safetensors header、解码后的数值、具体目标引擎的 kernel 支持以及量化质量，仍需在拥有实际分片的环境继续验证。
