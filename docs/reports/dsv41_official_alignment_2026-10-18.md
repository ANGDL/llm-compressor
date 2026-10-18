# V4.1 官方计算顺序对齐与推理 pipeline 边界

日期：2026-10-18。基线：`9c9c91554`。

通用化范围是推理 pipeline：checkpoint 读取、权重 offload、工作集预算和
prefill/decode 验证入口。attention、mHC、DSpark、模型专有 cache 和精度
边界由模型实现负责，按官方源码特化。不能为了共享模型代码而替换低精度
计算顺序，也不能用放宽验收阈值代替对齐。

## GPU 报告中的证据

输入报告为用户提供的 `dsv41-gpu-project-full.json`，运行环境为 H20、
PyTorch 2.13.0+cu130、TileLang 0.1.13，项目比较已启用。官方及项目源码
SHA256 均与本地基线一致。报告为 51 passed、5 error、2 diagnostic。

- mHC/Sinkhorn 的 10 个 case 通过严格 FP32 比较，项目最大绝对误差
  `2.384185791015625e-7`。
- FP8 activation 的 70 项比较（含 10 项项目 Q/DQ）全部精确相等。
- FP8 GEMM 的 8 个 case 通过 BF16 预算，6 个精确相等。
- sparse 的 5 个 error 均来自官方 d=32 kernel 的 8-warp GEMM 编译布局
  限制。它们没有产生可比较的输出，不能算作数值对齐通过。
- sparse 的 18 个已执行 case 通过宽 BF16 预算，但其中跨 64-KV 的项目
  路径存在额外偏差。例如 h16/d512/k129：
  官方与独立在线参考最大差为 `0.001953125`，官方与项目最大差为
  `0.0078125`，项目有 7389/114688 个不同元素。

官方 V4、V4.1 kernel 都使用每 64 个 KV 的在线 softmax、块内 BF16
权重转换及 FP32 累加重缩放。项目此前使用全局 softmax 后再转换 BF16，
跨块时不保持相同的舍入顺序。

## 本轮修改

共享 sparse reference 改为官方的 64-KV 在线顺序，末块补齐为 64，
每次仅 gather 当前 block。V4.1 全屏蔽 query 保持零输出；校准路径额外避免
`0 / inf` 对反向传播产生 NaN。原始官方 GPU kernel 不作改写。

V4.1 显式保留 ratio > 1 compressor wkv/wgate、LM/Markov head 和
DSpark confidence 投影的 FP32 参数，ratio=1 compressor wkv 按官方保留
BF16。测试用 `1 + 2**-16` 的 FP32 checkpoint 值检查加载过程，避免先降为
BF16 再转回 FP32。通用推理 pipeline 直接遵循模型声明，无需添加模型名称分支。

GPU 对齐脚本补充：

- 项目与独立在线参考之间的严格 FP32 阈值比较。
- 项目全屏蔽输出、整段与独立 query 的精确相等检查。
- 生产形状 64 heads、d=512、K=128/512/640。
- 默认保留 K=1，但使用可编译的 d>=64；`--include-small-head-dim` 显式
  复查 d=32，编译失败仍记录 error。
- 单行 FP8 输入使用非零数据，避免单行 case 仅测试零组。

## 本地验证与限制

CPU 上固定输入的 K=32/64/65/128/129，以及 64 heads、d=512、K=640，
项目输出与独立在线参考逐元素相同。手算的跨块 rescale case 可以区分
全局 BF16 权重舍入与官方在线顺序。全屏蔽和首块全屏蔽 case 的梯度有限。
这些结果不能替代更新后代码的官方 GPU 对照。

相关 streaming、模型、官方源码对照、tiny fixture 和 GPU 脚本回归为
330 passed、4 skipped。新增的自动模型工厂 FP32 加载分支另行验证。
Ruff、格式及相关代码的 CUDA API lint 通过。

复用现有完整 tiny checkpoint，未重新量化或修改权重。报告另存为
`/tmp/dsv41-streaming-official-order-2026-10-18.json`，原有基线报告保留：

| 权重 | Streaming / resident prefill、3 步 decode、vision、DSpark | BF16 cache/recompute 最大 logit 差 |
| --- | --- | ---: |
| source FP8/FP4 | 精确一致 | `9.536743e-7` |
| WNa8 | 精确一致 | `1.907349e-6` |
| W8A8 | 精确一致 | `0.0138549805`，严格验收失败 |
| W4A8 | 精确一致 | `0.0305302143`，严格验收失败 |

完整 tiny 校验脚本因此按设计退出 1，没有将 stream/resident 对齐误报为
cache/recompute 通过。W8A8/W4A8 问题继续保留，未提高默认精度或修改阈值。

额外 V4 model-free 入口测试未能收集：仓库缺少其导入的
`examples.model_free_ptq.dsv4_w4a8`，错误发生在测试收集阶段。该文件不属于
本轮修改范围。
