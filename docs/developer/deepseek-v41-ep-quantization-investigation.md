# DeepSeek-V4.1 EP Quantization Investigation Log

本文件按时间倒序记录 DeepSeek-V4.1 WNa8 streaming EP 量化的问题、证据、
修复和已验证的坑。新增记录放在最上方，不删除旧记录。

## 维护规则

- 每条记录包含：现象、证据、结论、避免重复踩坑。
- 路径优先记录实际实验目录、commit、wheel 和日志时间。
- 没有 trace 或日志支撑的结论必须标为“待验证”，不能写成根因。
- 每次改变 kernel、batch、并发或派发方式后，记录可对比的首个 routed
  MoE window 数据。

## 2026-11-23 17:20 CST - 批量 D2H 仍会 OOM：改为有界分块回填

**现象**

`filtered_boundary` 实验在首个 routed region 的 combine 阶段再次 OOM：

```text
/ssd3/model-quality/e2e/20261123_dsv41_wna8_imatrix_gptq_ep8_filtered_boundary_20261123T165000Z
combine batch 110: requested 1,783,888,460 bytes,
available 1,583,915,428 bytes after safety reserve on cuda:0
```

该实验的 update 为 `36.89–41.75s`，propagation 为 `43.28–54.48s`；
与 EP4 的 update `36.77s`、propagation `49.19s` 相比，EP8 尚未出现
两倍收益。combine 未完成，不能作为有效对比。

**结论**

- 只过滤下游 output 不能解决 OOM。批量 D2H 把所有 batch 的 GPU 输出
  留到 combine 全部结束才回传，累计占用超过 admission 预算。
- `prepare_batch` 按 `2 * size` 预留 device budget，但 pinned host
  buffer 属于 host memory，device 侧实际只有一块 `size` 的 staging。
  该预留口径在可用显存接近 1.6 GiB 时会把本可成功的 batch 拒掉。
- propagation 的 dispatch 计时合计仅约 `11.7s`，而 wall 为 `54.3s`，
  仍有约 `42s` 未归因。仅凭现有字段不能判定是 GIL 下的逐 batch 切片、
  host 预留、output reserve 还是 capture 后处理，必须先补计时再定性。

**修复与验证**

- [combine 回填](../../src/llmcompressor/streaming/parallel/region.py#L1380)
  按待提交和回传中的实际 backing storage 字节数限制队列。GPU 字节
  阈值取 `512 MiB` 和初始可用预算的 `1/8` 中较低者，每块还受
  `8` 个 batch 限制。单个输出超过阈值时立即回传并等待完成。
- `prepare_batch` 按对齐后的 staging 尺寸分别申请 device 和 host
  预算。双缓冲准入失败时保留下一批输入，排空回填队列后按单缓冲重试。
  `prefetch_fallbacks` 记录预取回退次数。
- [D2H 依赖](../../src/llmcompressor/streaming/activations/transfer.py#L357)
  在 copy stream 上等待 combine producer。Pinned staging 纳入
  `HostMemoryBudget`，该预算不足时逐 batch 同步回填。
- combine 按整个下游的输入活跃性过滤输出，保留跨越中间 group 的
  activation。
- propagation 增加 `expert_s / loop_s / submit_s / reserve_s /
  slice_s / output_reserve_s / split_s`，用于把 wall 时间归因到
  逐 batch 循环和 capture 后处理。

本地 CPU 回归已覆盖双缓冲准入失败后的单缓冲重试、跨组 activation
和 staging 释放。P800 EP8 上的 batch 110 和首个 routed region
峰值仍待实测。

**避免重复踩坑**

- 不要把“批量 D2H”实现成“全部 batch 的 GPU 输出一直驻留”。批量必须
  有字节或 batch 数上限，否则会重现 `batch 110` 的 OOM。
- device budget 不应把 pinned host buffer 计入 device 预留；host 侧
  容量应由 `HostMemoryBudget` 单独管理。
- 跨 stream 的 `non_blocking` 拷贝必须显式 event 排序，不能依赖默认
  stream 语义。

## 2026-11-23 16:20 CST - xSGL / SGLang EP 技术迁移评估

**现象**

旧版 EP8 首个 routed region 的 combine 由主线程串行搬运主导。按 15:31
记录的切点，`cudaMemcpy` API 区间并集为 `197.060s`，其中与 expert
输出搬运标记 `move_cuda:0` 相交 `16.961s`，其余时间仍需归因。当前
工作区已把 combine 的 H2D 按 batch 合并，并用独立 stream 提交下一
batch。Dispatch 仍在各 owner 上构造、拼接 expert 输入，再用普通
`.to()` 搬运。 [Dispatch](../../src/llmcompressor/streaming/parallel/region.py#L873) ·
[Combine](../../src/llmcompressor/streaming/parallel/region.py#L1297)

**源码依据**

本次对照 `../xSGL` 的 `97fe8bc1a4`、`../sglang` 的 `bf94411d8e`，
以及 `llm-compressor` 的 `1447f61c9` 和当前工作区改动。

![SGLang 的 EP 派发、卡内重排、跨卡回填和 xSGL 重叠时序](figures/sglang-ep-dispatch-combine.svg)

- SGLang 的 standard + DeepGEMM 路径通过 `ep_scatter` 把行排入按 expert
  分组的连续 buffer，并保留 `src2dst`。`post_reorder_deepgemm` 复用映射，
  把结果归并回原 token 行。
  [输入布局](../../../sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py#L1180) ·
  [输出回填](../../../sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py#L1272)
- 图中的 DeepEP Normal + DeepGEMM 路径使用 `output_index` 供本卡
  `ep_gather` 还原接收行，并使用 DeepEP `handle` 做跨卡回传。两种映射
  的作用域分别是 expert 输出行和源 rank 的 token 行。
  [Normal 本卡重排](../../../sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py#L1390) ·
  [Normal 本卡归并](../../../sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py#L1493) ·
  [Normal 跨卡归并](../../../sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L720)
- xSGL 的 normal DeepEP 路径在 `dispatch_a`、`combine_a` 提交通信，在
  `dispatch_b`、`combine_b` 为消费 stream 设置 event 依赖。两批交错时，
  等待前的间隙可以执行另一批的独立计算。Dispatch 的 notify 仍需返回
  host 侧 expert 行数。 [Dispatch 提交与等待](../../../xSGL/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L598) ·
  [Combine 提交与等待](../../../xSGL/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L740)
- xSGL 的 KLX fused down GEMM + combine 依赖低延迟 DeepEP layout、
  BKCL 上下文及特定量化输入。
  [Fused 路径条件](../../../xSGL/python/sglang/srt/layers/moe/ep_moe/layer.py#L327) ·
  [BKCL 与 layout 参数](../../../xSGL/python/sglang/klx/klx_ops.py#L905)

**结论与优先级**

本次移植无需引入 DeepEP。单进程持有 CPU boundary，可以按 owner 直接
打包并 H2D 到目标卡。Propagation 输出经 CPU staging 汇聚到主卡，按本地
记录的 batch、token 行和 expert offsets 归并。用现有张量拷贝和 stream /
event 实现连续布局、批量回填、双缓冲和延迟等待。
 [单进程执行入口](../../src/llmcompressor/streaming/pipeline.py#L1176) ·
 [CPU 派发及回传](../../src/llmcompressor/streaming/parallel/region.py#L873)

![同一组 Top-2 路由下的 SGLang EP 与 Streaming PTQ 移植方案](figures/streaming-ptq-ep-port-comparison.svg)

1. **先复用路由布局，批量回填。** 从 `ExpertRouteCache` 的分组结果构造
   token 行索引、expert offsets 和逆映射，校准和 propagation 复用这些
   元数据。按 owner、window 和 token chunk 直接填连续 staging buffer，
   减少逐 expert gather、拼接和小拷贝。对可识别的纯归并子图批量做 FP32
   累加，其余沿用 FX combine。 [现有路由缓存](../../src/llmcompressor/streaming/parallel/route_cache.py#L37)
2. **按 owner 双缓冲，推迟等待。** 在 resident window 内让 CPU 打包和
   `chunk N+1` 的 H2D 与 `chunk N` 的 expert forward 重叠。传输和计算
   分别使用 stream，消费者在用到数据时等 event，buffer 在上一消费者
   完成后复用。Pinned host buffer 和 device buffer 都计入预算，device
   预算还需计入已预取的权重。 [窗口调度](../../src/llmcompressor/streaming/parallel/region.py#L1251)
3. **按去重率评估 owner 局部归并。** 对纯 `index_add` 归并，可先在 owner
   上按原 token 行做 FP32 累加，再回传紧凑的 `(row_ids, partial_sum)`。
   先测每个 owner/window 的路由行数和去重后行数。若原输出为 BF16，
   忽略索引开销时，去重后行数低于原路由行数的一半才会减少传输字节。
   [原有归并语义](../../src/llmcompressor/modeling/deepseekv41/model.py#L565)

**避免重复踩坑**

- 保留逐 expert 的 Linear 校准调用、token 与 batch 的对应关系，以及
  校准、GPTQ update、propagation 的顺序。DeepSeek-V4.1 在 `w2` 前应用
  路由权重。把乘权移动到输出归并会改变 `w2` 的校准输入及舍入行为。
  [Expert 前向](../../src/llmcompressor/modeling/deepseekv41/model.py#L486)
- 旧 trace 的 `0.291s` 是 `aten` copy op 与已记录 kernel 的交集。若用
  `cudaMemcpy` API 与 kernel 的交集，combine 为 `0.000s`。设备侧 DMA
  overlap 待验证。把 boundary H2D、owner D2H、FX combine 和 boundary
  commit 分开计时，有助于定位下一步的优化对象。
- 在 P800 上以相同 recipe、校准输入和 token budget 对比改动前后的首个
  routed region，记录拷贝次数和字节数、打包和提交耗时、设备侧的实际
  传输与重叠、各 owner 和主线程的阶段耗时、host/device 峰值。比较
  Hessian、样本计数、量化权重、qparams 和下一层 boundary。迁移收益
  待验证。 [当前阶段计时](../../src/llmcompressor/streaming/parallel/region.py#L1381) ·
  [现有数值对照](../../tests/llmcompressor/streaming/test_expert_region.py#L422)

## 2026-11-23 15:31 CST - overlap 分析：combine H2D 实际是同步串行

**问题**

理论上 H2D 应设计为非阻塞，并与计算或其他 rank 的工作重叠。trace 需要
确认当前实现是否真的 overlap。

**overlap 结果**

区间并集和交集按阶段计算：

| 阶段 | wall | copy op 并集 | cudaMemcpy 并集 | kernel 并集 | copy×kernel |
|---|---:|---:|---:|---:|---:|
| prepare | 44.40s | 10.947s | 0.978s | 0.006s | 0 |
| calibration | 75.00s | 0 | 7.304s | 12.881s | 1.287s |
| update | 45.20s | 0 | 7.568s | 31.696s | 2.106s |
| propagation | 55.20s | 0 | 17.449s | 2.170s | 0.677s |
| window_complete | 17.70s | 4.852s | 6.463s | 0.326s | 0 |
| combine | 224.06s | 203.329s | 197.060s | 0.291s | 0.291s |

combine 的 copy op 全部发生在主线程：

```text
main_copy_ops_union_s=203.329
copy_ops_x_kernel_s=0.291
```

**结论**

- combine 阶段几乎是纯同步拷贝，copy 与 XPU kernel 基本没有重叠。
- propagation 的 `cudaMemcpy` 17.45s 也只与 kernel 重叠 0.68s。
- update 有一定重叠：kernel 31.7s，memcpy 7.57s，重叠 2.11s。
- 当前 routed region 的 `_move()` 使用同步 `.to(device)`，没有
  `non_blocking=True`；CPU 端也没有 pinned staging。
- expert 输出先 D2H 到普通 CPU 内存，再在 combine 阶段逐个 H2D 回
  primary，形成主线程串行关键路径。

**避免重复踩坑**

- “设计上应该非阻塞”不能替代 trace；当前代码实际是同步串行。
- 只加 `non_blocking=True` 不够，源 CPU tensor 需要 pinned memory，且需要
  正确 stream/event 依赖。
- 应把 expert 输出按 batch 聚合成连续 staging，批量 H2D，而不是逐 tensor
  搬运。

**产物**

原始 trace 保留在：

```text
/ssd3/model-quality/e2e/20261123_dsv41_wna8_imatrix_gptq_ep8_profile_20261123T150000Z/profile/first_routed.trace.json
```

聚合结论：

```text
profile/summary.txt
profile/summary_phases.txt
profile/summary_overlap.txt
```

原始 trace 约 6 GiB，不纳入交付；只用于后续复核。

## 2026-11-23 15:24 CST - 完整 trace 结论：combine/输出搬运是首要瓶颈

**采集范围**

`torch.profiler` 连续覆盖首个 routed MoE region：

```text
/ssd3/model-quality/e2e/20261123_dsv41_wna8_imatrix_gptq_ep8_profile_20261123T150000Z
```

覆盖 calibration、update、propagation、window complete 和最终 combine；
不覆盖准备阶段和后续 target。原始 trace 约 `6 GiB`、`10,937,672` 个事件，
已只保留聚合摘要，不交付原始 trace。

**全局聚合**

| 类别 | 时间 |
|---|---:|
| `cpu_op` | 674.843s |
| `cuda_runtime` | 280.593s |
| 其中 `cudaMemcpy` | 252.052s |
| XPU `kernel` | 80.508s |

CPU op 的主要项：

| op | 时间 | 次数 |
|---|---:|---:|
| `aten::to` | 216.329s | 535,700 |
| `aten::_to_copy` | 215.862s | 532,253 |
| `aten::copy_` | 215.399s | 533,402 |

**按阶段拆分**

| 阶段 | 主要成本 |
|---|---|
| prepare | `cudaMemcpy` 5.8s，CPU op 47.1s |
| calibration | kernel 20.1s，`cudaMemcpy` 8.1s |
| update | kernel 57.5s，`cudaMemcpy` 13.1s |
| propagation | `cudaMemcpy` 21.5s，kernel 2.2s |
| combine | CPU copy op 约 606s，`cudaMemcpy` 197.1s，kernel 0.29s |

combine 阶段：

- `aten::to`: 203.329s，511,504 次
- `aten::_to_copy`: 202.886s，511,504 次
- `aten::copy_`: 200.280s，511,504 次
- `cudaMemcpy`: 197.060s，295,530 次
- 全部 XPU kernel 仅 0.291s

**结论**

首个 routed region 的首要瓶颈不是 native factorization，而是 expert 输出
先落到 CPU，最终 combine 时再按 expert/batch/tensor 逐个搬回 primary
device。该阶段被大量小 `aten::to/_to_copy/copy_` 和 `cudaMemcpy` 主导，
GPU kernel 几乎不占时间。

update 仍有明显的 8 路 native factorization 争用，但它是第二优先级；
propagation 也是 memcpy-bound，而不是计算-bound。

这是通用 EP 架构问题，可以通过批量/连续 staging、pinned memory、合并
H2D copy、避免逐 tensor roundtrip 或 owner-side partial combine 解决，
不应针对 DeepSeek-V4.1 写特例。

**NUMA 证据**

- 进程 CPU 亲和性为 `0-207`，未绑定 NUMA。
- `numastat` 采样：heap 约 `144 GiB` 在 NUMA0、`27 GiB` 在 NUMA1；
  private 约 `178 GiB` vs `1.5 GiB`。
- XPU0-3 属 NUMA0，XPU4-7 属 NUMA1，因此当前 host 输出 staging 对
  XPU4-7 很可能跨 NUMA。

**避免重复踩坑**

- 不要先入为主把 EP8 慢归因于 factorization；必须按阶段看 memcpy。
- 不要按单个 tensor 做 CPU->primary 搬运；这会放大为数十万次小拷贝。
- 不要交付 GB 级原始 trace，保留聚合摘要和可复现筛选脚本即可。

## 2026-11-23 15:01 CST - 优化目标和边界

**目标**

EP8 并行量化性能达到 EP4 的 `2x`。这是 EP 架构优化目标，不是针对
DeepSeek-V4.1 的模型特调。

**允许的优化层**

- owner/window 并发调度和负载分配规则
- CPU boundary 读取、owner 线程亲和性和 NUMA-local 派发
- native factorization 的批量执行和多 owner 并发扩展性
- H2D/stream/事件同步
- 通用准备阶段复用：target/FX 计划缓存、boundary 缓存、reason 统计

**禁止的优化方式**

- 针对 dsv4.1 的层号、expert 数、shape 或固定 batch 常量
- 只对某个模型有效的“经验 cap”或特例分支
- 用降低量化质量、改变 recipe 或跳过工作量的方式制造加速

**验收方式**

同一模型、同一 calibration 数据量、同一 recipe，对比 EP4/EP8 的：

- 首个 routed MoE window 的 CPU prepare、update、propagation
- 准备阶段各阶段耗时
- CPU/XPU trace 中的 host launch、kernel、memcpy 和等待时间

## 2026-11-23 14:59 CST - 准备阶段 cProfile 会阻塞实验，暂时跳过

**现象**

对完整准备阶段启用 `cProfile` 后，`Grouped quantized MoE modules` 从约
111 秒膨胀到 7 分 7 秒，FX trace 也长时间无法完成。

**证据**

- 正常准备：`20261123_dsv41_wna8_imatrix_gptq_ep8_native_factorize_20261123T140500Z`
- cProfile 准备：`20261123_dsv41_wna8_imatrix_gptq_ep8_profile_20261123T143000Z`
- `target grouping`: 正常约 `111s`，cProfile 下约 `427s`。

**结论**

准备阶段有约 1.5 万个 target 和大量 Python 调用，`cProfile` 的逐调用
插桩不适合直接覆盖完整准备阶段。当前先采集首个 routed MoE window 的
CPU/XPU trace；准备阶段后续改用低开销采样或分段 profile。

**避免重复踩坑**

- 不要在完整 target 构建/FX trace 上直接启用 `cProfile`。
- 准备阶段 profile 必须限制在短时间窗或单个子阶段。

## 2026-11-23 14:57 CST - 轻量 trace 采集启动

**执行**

已重新启动不带准备阶段 `cProfile` 的 trace 采集：

```text
/ssd3/model-quality/e2e/20261123_dsv41_wna8_imatrix_gptq_ep8_profile_20261123T150000Z
```

环境：

```bash
XPU_ENABLE_PROFILER_TRACING=1
XPU_CUPTI_ENABLE_DEVICE=0,1,2,3,4,5,6,7
STREAMING_EP_PROFILE_PREPARATION=0
```

profiler 只在首个 non-dense routed MoE region 启停，导出
`profile/first_routed.trace.json` 后进程自动退出。

**避免重复踩坑**

- 准备阶段 profiling 默认关闭，避免影响正式实验进度。
- trace 只覆盖首个 routed window，避免完整量化 trace 过大和额外开销。

## 2026-11-23 14:50 CST - 确认 expert 输入是 CPU 派发，不是 GPU0 转发

**代码路径**

- `InMemoryBoundaryActivationStore` 强制 boundary 存储在 CPU。
- `read_for_routing()` 返回 CPU tensor。
- `region.forward_expert()` 在 owner 线程中先组装 CPU inputs，再调用
  `_move(combined, owner_device)` 做 H2D。
- 当前 routed region 不经过 non-region
  `ExpertParallelExecutor._copy_inputs()` 的 primary GPU -> owner GPU 路径。

**部署和 NUMA**

- 进程 `Cpus_allowed_list=0-207`，`Mems_allowed_list=0-1`。
- NUMA0 对应 XPU0-3，NUMA1 对应 XPU4-7；两组内部是 XPULINK，跨组是
  `SYS/NODE`。
- 进程没有 CPU/NUMA 亲和性绑定，`numastat` 显示 heap/private 内存分布
  在两个节点，因此 host 侧仍有跨 NUMA 首触内存访问的可能。

**结论**

专家输入的数据路径是 CPU -> 各 owner XPU 的直接 H2D，不是 GPU0 ->
GPU4-7 的跨 NUMA/XPULINK 转发。但不能据此认为 host 内存全部 NUMA-local：
线程和设备没有绑定到各自 NUMA 节点。

**避免重复踩坑**

- 区分“GPU0 跨设备转发”和“CPU 内存跨 NUMA 访问”，这是两个不同问题。
- 后续若要优化 H2D，应检查 owner 线程的 CPU 落点和 boundary 内存
  first-touch 节点。

## 2026-11-23 14:26 CST - 8 路 native factorize 并发扩展性不足

**基准**

固定 `batch=3, width=5120`，每个线程在独立 XPU 上调用
`factorize_hessian`：

| 并发数 | wall | 单次平均 |
|---|---:|---:|
| 4 | 0.393s | 0.342s |
| 8 | 1.006s | 0.897s |

8 路单次成本约是 4 路的 `2.62x`。

单设备 batch 基准中，batch `3/6/12/24/48` 基本线性，约
`57ms/module`。`torch.set_num_threads(1)` 没有改善 8 路扩展性。

**结论**

单个设备上的 batch 组装不是主要问题。8 路 owner update 同时进入 native
factorization 时存在明显争用，导致 EP8 每 expert 成本显著高于 EP4。

**避免重复踩坑**

- 内存上能放下 96 个模块，不代表 8 路并发下性能最优。
- 不能只用 wall-clock 猜瓶颈，需要结合 CPU/XPU trace 定位争用发生在
  host launch、native kernel 还是设备执行。

## 2026-11-23 14:21 CST - native kernel 修复生效，但仍未达到 EP8 2x EP4

**修复安装**

旧环境安装的是 2026-11-12 的
`/data/quant/llmcompressor-kernels/dist` wheel。强制安装：

```bash
/root/miniconda/envs/python312_torch212_cuda/bin/python -m pip install \
  --force-reinstall --no-deps \
  /data/quant/llmcompressor-kernels-native/dist/\
llmcompressor_kernels-0.1.0-cp312-cp312-linux_x86_64.whl
```

安装后：

```text
ops = gptq_block_update, imatrix_search, bmm_fp32,
      gptq_factorize_hessian
setup | gptq_factorize_hessian=available
Using precompiled native operator 'gptq_factorize_hessian'
```

**首个 routed MoE window，取最慢 rank**

| 版本 | update | propagation |
|---|---:|---:|
| EP4，native，per-expert | 36.77s | 49.19s |
| EP8，native，per-expert | 71.41s | 65.27s |
| EP8，native，动态 batch | 58.41s | 70.09s |

当前 EP8 update 比旧 EP8 快约 `18%`，但仍比 EP4 慢约 `1.59x`。
按“EP8 快 2x”目标，update 应约 `18.4s`，实际 `58.41s`。

**避免重复踩坑**

- 新旧 wheel 都是 `0.1.0`，普通 `pip install` 会认为已安装，必须
  `--force-reinstall`。
- 算子存在且被选中，不等于多卡扩展性达标。

## 2026-11-23 13:23 CST - 动态 batch 消除 OOM，但性能仍受 factorization 限制

**结果**

动态 batch 版本首个 routed window：

- 预测 cap：`94` 到 `96`
- batch budget：约 `61.6-61.8 GiB`
- OOM：`0`
- update：`56.50-58.41s`
- propagation：`61.58-70.09s`

固定 cap4 对比：

- update：`291.24-319.65s`
- propagation：`50.59-59.64s`

**结论**

内存容量预测修复有效，解决了大 batch OOM 和小 batch 过慢的容量问题；
但 native factorization 激活后，剩余瓶颈转向多 owner 并发执行效率。

**避免重复踩坑**

- 不要把“无 OOM”当作“性能达标”。
- 不要把 96 模块大 batch 当作固定目标；它只是当前显存预测给出的上限。

## 2026-11-23 13:06 CST - 删除固定 batch cap，改为设备预算预测

**代码**

- `src/llmcompressor/modifiers/gptq/helpers.py`
- `src/llmcompressor/streaming/gptq_adapter.py`

`auto` 在 owner 权重和 Hessian 就绪后，按模块 shape、Hessian 峰值和设备
可复用显存计算一个 batch 上限。显式整数配置仍保持原语义。

**避免重复踩坑**

- 不要重新引入 `cap4`/`cap32` 这类跨模型固定经验值。
- admission 只保证单模块能放下；实际 batch 必须在 owner window 加载后计算。

## 2026-11-23 12:33 CST - 固定 cap4 可用但吞吐过低

**结果**

EP8 固定 cap4：

- update：`291.24-319.65s`
- 每 4 模块 batch 的 GPTQ 时间约 `8s`，即约 `2s/module`
- 无 OOM

**避免重复踩坑**

- cap4 是安全性实验值，不是通用方案。
- 仅缩小 batch 会掩盖 `auto` 的容量预测错误，并牺牲所有模型吞吐。

## 2026-11-23 11:30 CST - 旧 auto 在 128 模块 batch OOM

**证据**

`20261123_dsv41_wna8_imatrix_gptq_ep4_batch_update_5234f9eab`

```text
Quantizing 128 module(s)
torch.OutOfMemoryError: Tried to allocate 25.59 GiB
GPU 0 free: 18.66 GiB
```

**结论**

`auto` 在未扣除完整 batch 峰值和当前窗口占用时，会把同 shape 模块合批
到 128，导致 activation ordering 的 Hessian 临时空间 OOM。

**避免重复踩坑**

- 只按权重数量或固定 cap 评估 batch 都不够，必须包含 Hessian、激活排序
  临时张量和当前设备实际可用字节。

## 2026-11-23 11:25 CST - 历史 EP4/EP8 日志中的 native 状态曾不一致

**现象**

- 10:04、11:25 的 EP4 日志显示
  `gptq_factorize_hessian=available`。
- 12:17 之后的 EP8 日志显示
  `gptq_factorize_hessian=missing`。

**结论**

修复版 wheel 曾在环境中可用，后来环境被旧 wheel 覆盖。根因不是
`llm-compressor` 桥接代码回退，而是量化环境安装包状态回退。

**避免重复踩坑**

- 每次正式量化前检查 setup 日志中的
  `gptq_factorize_hessian=available`。
- 修复 kernel 版本时同时记录 wheel 路径、构建时间和 `--force-reinstall`。

## 2026-11-23 11:00 CST - 准备阶段基线耗时

首个 routed target 前约 `11m24s`：

| 阶段 | 时间 |
|---|---:|
| recipe/modifier 初始化 | ~51s |
| MoE 替换 | ~124s |
| target 分组 | ~111s |
| automatic target grouping | ~53s |
| FX trace | ~103s |
| 初始 266 boundary 采集 | 227.44s |
| modifier 初始化 | ~14s |

**避免重复踩坑**

- 准备阶段占端到端时间很大，不能只优化 GPTQ update。
- 初始 boundary 采集、FX trace 和 target 分组需要分别 profile，避免再次
  用全量 `cProfile` 阻塞实验。

## 2026-11-23 10:00 CST - EP4 与 EP8 旧版本对比

**首个 routed window**

| 阶段 | EP4 | EP8 |
|---|---:|---:|
| CPU prepare | 20.95s | 29.27s |
| update | 36.77s | 71.41s |
| propagation | 49.19s | 65.27s |

**结论**

旧版本 EP8 不但没有达到 2x 加速，反而比 EP4 慢。每 rank expert 数减半，
但每 expert update 从约 `0.38s` 增至约 `1.49s`，并发扩展性是主要问题。

**避免重复踩坑**

- 不能只用总 expert 数评估负载，必须按并发 owner 数和单 expert 成本看。
- 先确认 kernel provider 状态，再做 EP4/EP8 对比，避免版本不一致。

## 参考资料

- P800 profiler 文档：
  `/Users/ang/Projects/POC/chuling/profile_p800.md`
- 当前提交：
  - `1447f61c9 fix(streaming): predict GPTQ owner update batches`
  - `5234f9eab fix(streaming): prepare routed MoE weights once per layer`
