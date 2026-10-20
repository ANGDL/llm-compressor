# DeepSeek V4.1 量化速度分析与优化记录

记录日期：2026-10-20，Asia/Shanghai。
工作分支：`feat/deepseek-v41-native-quantization`。

## 已确认的耗时来源

使用已经完成的真实量化日志：

`/ssd2/model-quality/runs/20261019T093114Z_a619a1629881/deepseek_v41_wNa8-4c4ba690/logs/attempt-20261019T093114Z-8ba4826d/quantization.log`

全流程耗时约 `26992.4 s`，其中 43 个 sequential target 的累计时间为：

| 阶段 | 累计时间 | 占总耗时 |
| --- | ---: | ---: |
| calibration forward | `12871.1 s` | 约 47.7% |
| activation propagation forward | `11991.3 s` | 约 44.4% |
| prepare weights | `1528.4 s` | 约 5.7% |
| modifier update | `949.0 s` | 约 3.5% |
| load weights | `343.1 s` | 约 1.3% |
| queue checkpoint | `294.8 s` | 约 1.1% |

校准和传播两轮前向合计约 6.91 小时，是量化速度的主要瓶颈。实际量化
任务已完成，当前测试使用其他空闲卡，没有中断原任务。

## MoE 热点

### V4.1 实际校准路由

`examples/streaming_oneshot/deepseek_v41_wNa8.py` 的
`--moe-calibrate-all-experts` 使用 `store_true`，因此默认值是
`False`，主干 MoE 按真实稀疏路由校准。带有 draft/MTP 语义的 MoE 会通过
`requires_all_experts_for_calibration` 单独要求 all-experts，这是模型代码的
覆盖规则，不改变入口的默认值。

因此，性能优化的主要路径是稀疏路由：只遍历实际激活的专家。all-experts
路径仍保留完整统计覆盖，并复用每次前向的权重缓冲，减少临时张量分配。

官方 V4.1 MoE 同样按专家循环，并且每个专家使用 `torch.where` 获取 token
索引。当前实现之前还存在两项额外开销：

1. 量化适配层的 Linear 在输入和权重 dtype 相同的 BF16/FP32 情况下仍强制
   转 FP32；
2. 每个专家单独执行 `torch.where(indices == expert_id)`，在设备后端上会产生
   大量小索引操作和同步。

新增 [MoE benchmark](/Users/ang/Projects/llm-compressor/tools/benchmark_dsv41_moe.py)
测量四种路径：

- `legacy_fp32`：所有 Linear 强制 FP32；
- `native_where`：原始 dtype + 官方逐专家 where；
- `native_counted_where`：原始 dtype + 计数后跳过空专家；
- `current`：原始 dtype + 一次稳定排序分组。

### P800 合成权重

模型尺寸为 hidden `5120`、intermediate `2304`、384 experts、top-6，
输入为 BF16 synthetic tokens，使用空闲 P800 卡的 `cuda:7`。

| tokens | legacy FP32 | native where | current | current 相对 legacy |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 224.7 ms | 183.8 ms | 87.2 ms | 约 2.58x |
| 8192 | 231.6 ms | 195.7 ms | 102.5 ms | 约 2.26x |

带 iMatrix observer hooks 后：

| tokens | legacy FP32 | native where | current | current 相对 legacy |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 339.8 ms | 299.3 ms | 184.7 ms | 约 1.84x |
| 8192 | 351.1 ms | 306.2 ms | 195.0 ms | 约 1.80x |

Profiler 统计中，512 tokens 时每次前向的 `aten::nonzero` 从 384 次降为 0 次；
`aten::where` 也从 768 次降为 0 次。`aten::bincount` 仅保留 1 次。

### P800 真实源权重

使用 `/ssd4/models/DeepSeek-V4.1-Flash-1017` 第 3 层 MoE，权重按原始 FP8/FP4
checkpoint materializer 解码为 BF16，输入为 synthetic BF16 tokens，没有执行
source FP8/FP4 QDQ：

| tokens | 激活专家数 | legacy FP32 | native where | current | current 相对 legacy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 299 | 253.8 ms | 225.5 ms | 137.4 ms | 约 1.85x |
| 8192 | 337 | 297.3 ms | 264.2 ms | 164.6 ms | 约 1.81x |

带 iMatrix hooks 的真实源权重测试中，current 路径输出、每个 Linear 输入、
iMatrix sum/count 都与 native where 路径完全一致。

### 2026-10-20 GPU 复测

在 node16 的 `llm-quant-base` 容器中，使用 `cuda:6`、`CT_ENFORCE_EAGER=1`
和同一份真实源权重重新运行候选实现。结果如下，`current` 是稳定排序分组
实现，`native_where` 是官方逐专家 `torch.where` 对照：

| 路径 | tokens | native_where | current | 加速 |
| --- | ---: | ---: | ---: | ---: |
| 稀疏路由 | 512 | 229.8 ms | 145.8 ms | 1.58x |
| 稀疏路由 | 8192 | 273.6 ms | 168.1 ms | 1.63x |
| all-experts | 512 | 304.8 ms | 208.2 ms | 1.46x |

三组测试均满足 `finite=true`，并且 `current` 与 `native_where` 的输出
逐元素相同，`imatrix_exact=true`。all-experts 结果用于覆盖 draft/MTP
层可能触发的强制校准分支；V4.1 示例入口本身仍默认使用稀疏路由。

## 已实现的 MoE 优化

`DeepseekV41MoE.forward()` 现在：

1. 用一次 `argsort(stable=True)` 将 `(token, topk slot)` 路由结果分组；
2. 用 `bincount` 得到每个专家的分段范围；
3. 每个专家仍只调用一次原来的 Expert/Linear 结构；
4. 普通路由通过 `index_select` 取得专家输入和 routing weight；
5. `calibrate_all_experts=True` 仍让所有专家看到完整 token 集，未路由 token
   使用单位权重；
6. 保持专家处理顺序、token/slot 稳定顺序和 `index_add_` 累加顺序。

已通过 CPU 和 GPU 对照：

- 普通路由与原 `torch.where` 实现输出逐元素完全相同；
- all-experts 校准输出逐元素完全相同；
- 每个专家 Linear 的输入逐元素完全相同；
- iMatrix sum/count 逐元素完全相同；
- 结果均为 finite。

## 已实现的 streaming pipeline 优化

纯 RTN/iMatrix 量化时，`calibration` forward 和 `modifier update` 之间只会
计算观察统计和 qparams；浮点权重在之后的 `compress weights` 阶段才被替换。
因此，传播到下一 target 的 boundary 可以直接复用 calibration forward 的输出，
省去一次完整 subgraph forward。

为避免误用于会改变权重或激活的算法，新增显式 modifier 能力声明：

```python
calibration_outputs_reusable = True
```

当前只有以下两个具体 modifier opt in：

- `QuantizationModifier`
- `IMatrixGatherer`

只有 recipe 中所有具体 modifier 都显式 opt in 时才启用复用。GPTQ、AWQ、
SmoothQuant、SparseGPT、Wanda 等流程仍执行原来的 propagation forward。

复用路径在 calibration 阶段直接提交下一 boundary；最终 target 没有 successor，
不会创建无用 boundary。`checkpoint_progress=True` 和普通内存 boundary 两条
路径都覆盖。

GPU 测试结果（空闲 P800 第 6 张卡、BF16 测试夹具、`CT_ENFORCE_EAGER=1`）：

```json
{
  "checkpoint_progress": false,
  "reused_forwards": 4,
  "recomputed_forwards": 6,
  "checkpoint_tensors_exact": true
}
```

`checkpoint_progress=true` 结果相同。这里的 4 和 6 对应两层、两批校准数据：
复用路径每层每批执行一次，旧路径第一层还会额外执行一次传播。

## 验证

本地定向回归：

- streaming、V4.1 modeling、MoE context、工具和 V4 示例相关测试：修复测试
  夹具后通过；
- GPU reference grouped MoE 对照：`2 passed`；
- Ruff 和 `git diff --check` 通过。

新增测试和 benchmark：

- `tests/tools/test_benchmark_dsv41_moe.py`
- `tests/llmcompressor/modeling/test_deepseek_v41.py`
- `tests/llmcompressor/modeling/test_deepseek_v41_reference.py`
- `tests/llmcompressor/streaming/test_subgraph_pipeline.py`

当前优化尚未重新运行完整 256 图像 + 256 文本、8192 序列的生产量化。预计
主要收益来自两部分：每个校准/传播 forward 的 MoE 路由与 Linear 执行加速，
以及纯 RTN/iMatrix pipeline 删除约一半的 propagation forward。完整收益仍需
在新的输出目录中重跑小样本和完整任务分别确认。

没有修改 `src/llmcompressor/observers/imatrix.py`，没有改写昨天的 checkpoint，
也没有中断已有量化进程。
