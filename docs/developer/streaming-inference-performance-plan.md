# Streaming Inference 性能优化计划

日期：2026-11-09。

状态：P1（INT8 CPU 解码 fast path）已验证；有界单层预取已作为 opt-in
实验能力实现，但当前 Qwen3-VL-32B 端到端测试未取得收益，仍需重新设计
调度；完整 CPU/GPU 权重缓存已明确排除，batch 调度仍未实现。本文只规划
`StreamingInference` 的运行时优化，
不修改 `modifiers/gptq/*`，也不改变 GPTQ 的统计或量化语义。

## 1. 基线与问题

当前 `StreamingInference` 使用 Accelerate 将模型参数保持在 meta device，
每次模块调用按需执行：

```text
safetensors 读取
  -> CPU packed weight 解包
  -> CPU 反量化为 BF16/FP16
  -> 搬运到执行设备
  -> 模块 forward
  -> 卸载参数
```

2026-11-06 在 node16、`llm-quant-base`、Qwen3-VL-32B W8A16 上的基线为：

| 场景 | 结果 |
| --- | ---: |
| 多模态单次 forward | 约 83 s |
| `generate(max_new_tokens=1)` | 约 66 s |
| tensor reads | 1954 |
| materializations | 1058 |
| 单次权重工作集 | 约 1.45 GiB |
| 单个最大 CPU 解码准备量 | 约 3.54 GiB |

生成阶段的关键问题是每次 decode 调用都可能重新读取和反量化 decoder
权重。优化必须同时报告首次 prefill、单 token decode 和多 token decode，
不能只用单次 forward 的结果推断生成吞吐。

## 2. 设计约束

1. 优化集中在 `streaming/inference` 和通用 checkpoint source/materializer
   接口，不为 Qwen3-VL 增加模型名称分支。
2. 不修改 `modifiers/gptq/*`，不复用或改变 GPTQ Hessian 生命周期。
3. 数值路径保持与 `compressed_tensors` 一致；fast path 只替换等价的
   packed INT8 解包，反量化仍使用公开的 `dequantize` API。
4. fast path 必须有明确条件，条件不满足时回退到当前通用 compressor。
5. 默认继续保持无缓存和有界设备驻留。预取只能保留一个下一层的 CPU
   准备结果，不能悄悄改变设备或主机内存上界。BF16/FP16 权重常驻 GPU
   不属于本系统的优化方向，因为它会破坏流式执行的设备内存边界。
6. 每个优化阶段都记录硬件、checkpoint、输入、dtype、量化格式、峰值内存和
   数值对照结果。

## 3. 分阶段方案

### P1：INT8 packed weight CPU fast path（本轮）

对 `pack-quantized`、对称 INT8、little-endian CPU、沿最后一维 packed 的
权重使用 byte-view 解包：

```text
int32 packed
  -> uint8 byte view
  -> subtract 128
  -> int8
  -> compressed_tensors.dequantize
```

该路径避免通用解包器的逐元素 bit index、shift、mask 和临时 int32
workspace。它覆盖标准 Transformers Linear 的 2D 权重，也覆盖任意前导维的
packed tensor；4-bit、非对称量化、其他 packed dimension 和非 little-endian
环境继续使用通用实现。

验收条件：

- 与 `unpack_from_int32` 在随机 2D/3D/非 4 对齐形状上逐元素相等；
- 与现有 compressor dequant 输出逐元素相等；
- 现有 streaming inference 全部回归测试通过；
- report 标明 fast path 使用次数、通用回退次数、source read 时间和 decode
  时间；
- 实际模型复测至少包含 forward、1/8/16 token generation。

本地 `torch 2.12.1 + compressed-tensors 0.20.2a` 的单 tensor 基准：
典型 `4096×14336` INT8 group 权重的完整 CPU 解码约提升 3.6 倍；这是
microbenchmark 结果，不能直接当作端到端提升，具体结果随 CPU 线程数和内存
带宽变化。

### P2：分阶段耗时与安全预取

P1 复测后，如果 CPU decode 占单次 forward 的显著比例，再增加下一模块的
有界预取：

```text
当前模块 GPU forward
  || 下一模块 safetensors read + CPU decode
```

预取必须复用现有 host memory budget，最多保留一个 pending module，
失败时释放 reservation 并回退到同步加载。不能在没有预算的情况下后台持有
整层或整模型权重。该阶段先做单线程 CPU prefetch，再评估异步 H2D。
当前实现的 node16 结果是 `358.24s/10 tokens`，无预取 fast path 基线为
`323.69s/10 tokens`；预取命中率高，但 CPU 解码线程竞争使总耗时增加约
`10.7%`。因此预取当前不作为默认推荐配置。

2026-11-10 对 Qwen3-VL-32B W8A16 的单次多模态 forward 复测（`--mode forward`，
8 GiB device/preparation/prefetch 预算）如下：

| 配置 | forward 总耗时 | decode 合计 | prefetch prepare | prefetch wait | 命中 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 关闭 prefetch | 18.35 s | 5.13 s | - | - | - |
| 线程 prefetch | 27.97 s | 11.70 s | 13.52 s | 12.98 s | 969 |
| 进程 prefetch（16 线程） | 33.52 s | 0.40 s（仅父进程） | 20.16 s | 20.89 s | 1005 |

后台线程确实提交并执行了加载（89 个 group），但主线程等待时间接近 worker
总准备时间，且主线程 decode 被 GIL/CPU 竞争拖慢约 2.3 倍。因此当前实现属于
“有异步 prefetch，但没有形成有效的加载重叠”，仍不应作为默认配置。

把 decode 放进独立进程后，GIL 竞争确实消失：`7.27 GiB` 权重的解码在 16
PyTorch 线程下为 `1.71 s`，104 线程下为 `1.87 s`；两个并发线程为 `1.43 s`，
而两个进程可达到接近线性的 `0.87 s/半`。但端到端进程 prefetch 仍需要
`33.52 s`，因为每层解码结果要写入共享内存，消费者仍等待 `20.89 s`。该 W8A16
推理是内存带宽受限，而不是单纯 GIL 受限；额外的共享内存写入抵消了 GIL 收益。
下一步应优先减少解码字节数或使用 fused/native 解码，而不是继续增加并发。

量化 pipeline 的 target 级预取与 inference prefetch 是两条独立路径，不在本文
范围内；其调度与实测结果见 `docs/guides/streaming_ptq.md`。

### P3：有界单层预取

为 generation 和长序列 forward 提供有界的单层预取：

| 模式 | 语义 | 适用场景 |
| --- | --- | --- |
| `none` | 当前行为，每次调用后卸载 | 最小内存、严格基线 |
| `prefetch` | 只保留当前层和下一层的有界 CPU 准备结果 | CPU 解码可与 GPU forward 重叠 |

预取缓存按模块边界管理，最多保存一个下一模块的完整 CPU 准备结果；
当前模块完成后立即释放旧结果。预取、当前模块解码 workspace 和输入状态
共同计入 `HostMemoryBudget`，因此内存上界只随单层工作集增长，与模型总层数
和总权重大小无关。模型单层也超过 host budget 时，preflight 必须拒绝，
不能退化为整模型缓存或交换到磁盘。

设备常驻权重和完整 CPU BF16 cache 都明确排除在本计划之外。

### P4：批量与服务侧调度

StreamingInference 已可接受 batch tensor，但还没有请求级 continuous batching。
后续可在同一次模块执行中合并多个请求，摊薄每层的读取和解码成本。该阶段
属于服务调度能力，必须单独报告 latency、throughput、padding 和 KV cache
增长，不能与 P1/P2 的单请求 latency 混报。

## 4. 观测指标

`StreamingInference.report()` 逐步增加以下累计指标：

```text
source_read_seconds
decode_seconds
materializations
fast_path_materializations
generic_decode_materializations
prefetch.hits
```

设备搬运和 GPU forward 时间由端到端脚本单独记录。当前 report 的计时器按
session 累积；端到端脚本应在一次 `forward` 或 `generate` 前后做差，避免把
setup 时间混入 decode 时间。后续若需要服务侧观测，再增加显式的 last-call
快照，而不是改变累计字段的含义。

## 5. 验证矩阵

| 层级 | 覆盖 |
| --- | --- |
| 单元 | INT8 packed 解包、非对齐尾部、3D 前导维、little-endian 条件 |
| 回归 | Llama/Qwen3、channel/group、INT4 通用回退、tied weights |
| 数值 | streaming logits 与现有 reference compressor 对照 |
| 性能 | 典型 MLP/attention projection 矩阵的 read/decode benchmark |
| 端到端 | Qwen3-VL-32B 多模态 prefill、1/8/16 token decode |
| 内存 | no-cache 与 bounded-prefetch 分别记录 RSS、设备峰值和 host budget |

性能优化完成的判定需要同时满足：

1. 结果保持在既定数值容差内；
2. no-cache 模式的设备/主机内存上界不回退；
3. fast path 只在已验证格式启用；
4. 端到端收益和 microbenchmark 收益分别记录；
5. 通用 Transformers MoE 仍走统一路径，特化模型适配另行规划。

## 6. 当前进度

- [x] 记录 node16 Qwen3-VL-32B StreamingInference 基线。
- [x] 完成 INT8 byte-view 与通用解包的本地对照基准。
- [x] 实现 P1 fast path 和通用回退。
- [x] 增加 P1 数值与计时回归。
- [ ] 在 node16 复测 P1 的端到端收益。
- [x] 依据 decode 占比确定 P2 为有界单层预取。
- [x] 排除完整 CPU BF16 cache 和 device weight cache。
- [x] 实现有界单层 prefetch（opt-in，当前端到端未加速）。
- [ ] 评估 batch/continuous batching。
