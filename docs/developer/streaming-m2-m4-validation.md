# M2/M4 单参与者实现与验证

测试日期：2026-11-06。环境为 `node16` 的 `llm-quant-base` 容器，
代码路径 `/data/quant/llm-compressor`，Python 3.12。
设备测试使用容器提供的 `torch_xmlir` CUDA 兼容接口 `cuda:0`。
本次不修改 `modifiers/gptq/*`，不实施 E1。

原始指标的结构化摘录保存在
[streaming-m2-m4-20261106.json](measurements/streaming-m2-m4-20261106.json)。

## M2：独立进程恢复曲线

每个深度分别使用一个进程准备事务链，再启动全新进程构建 `ResumePlan`
和消费 frontier。测量进程不执行 fixture 准备，避免其 RSS 高水位污染恢复结果。
固定每个 boundary 为 4 个 `[1, 1024, 64]` FP32 batch，保留最近两个 boundary。

| 子图数 | 扫描事务数 | 元数据字节 | boundary 校验字节 | 增量峰值 RSS | frontier 活跃 batch 峰值 |
| --- | --- | --- | --- | --- | --- |
| 8 | 8 | 39,184 | 2,110,280 | 3.06 MiB | 1 |
| 32 | 32 | 156,781 | 2,110,280 | 3.36 MiB | 1 |
| 128 | 128 | 627,238 | 2,110,280 | 4.82 MiB | 1 |

保留 boundary 文件总量在三个 case 中均为 2,110,280 bytes。
历史输出 payload 每次恢复仍校验一次，其读字节量随事务数线性增长；
元数据也允许线性增长。这些数据验证了固定宽度下 frontier 的有界恢复，
不表示整个恢复过程的元数据内存与深度无关。
RSS 为 `ru_maxrss` 高水位差值，包含 Python/allocator 波动。

复现命令：

```bash
python tools/measure_streaming_resource_curve.py \
  --depth 8 --depth 32 --depth 128 \
  --work-root /tmp/streaming-recovery-new-run > recovery-curve.json
```

`work-root/depthN` 必须不存在，避免覆盖已有事务链。
工具也支持 `--case LABEL=CHECKPOINT_DIRECTORY`，以独立进程测量完整 PTQ，
并通过 `--algorithm`、`--device` 选择执行方式。失败 case 记录退出码及诊断，
不混入成功曲线。

## M4：资源 admission

每个 target 在读取权重前计算 materialized bytes，与 modifier 的 held
statistics、scratch 及 observer 估算联合 admission。held 按模块求和，
同一 modifier 的逐模块 scratch 取最大值，各 modifier 之间保守求和。
静态声明 `memory_estimate=(held, scratch)` 表示该 modifier 的整个工作集；
Identity 探测值只用于 contract 描述，不能替代实际 target 的动态估算。

输入 token 总数与最大 batch token 数分别从当前 boundary 推导，并保留配置值
作为下限。AWQ 的输入缓存使用总 token 数；iMatrix 的输入临时张量使用最大
batch token 数，避免将整个校准集误算成一个 batch。

- host：CPU 权重解码、boundary、writer snapshot 共用已有预算；新增 CPU
  modifier 工作集和 AWQ 显式 CPU offload 输入缓存记账。
- device：独立查询可用设备内存，保留 10% 安全余量。没有设备容量查询能力时，
  accelerator admission 明确拒绝；CPU 执行由 host budget 负责。
- disk：在读取 target 前检查 work filesystem 的临时输出余量，保留 5%。
  这是当前 target 的容量检查，不是文件系统配额或全任务空间预留；后续写入仍
  可能因并发占用或最终组装空间不足而失败。
- prefetch：先检查下一 target 的资源能力和设备容量，再在 host 上为当前
  writer 保留 completion headroom。预算不足时延后预取，正常继续当前 target。
- 估算失败：calibration modifier 返回未知工作集，或 observer 没有资源适配时，
  在读取该 target 权重前抛出明确错误。

对 RTN、iMatrix、GPTQ、AWQ、SmoothQuant 各运行一个独立进程；
输入为随机种子 7 的 tiny Qwen3，两层、hidden size 32、intermediate size 64，
source/target 均为 BF16，1 个长度 8 的 batch，方案 W8A8。
每个进程均完成两个 target 并发布 `FINALIZED`。

| 算法 | 每 target 联合设备估算（bytes） | 两个 target 的最大 allocator peak（bytes） |
| --- | --- | --- |
| RTN | 100,512 | 43,008 |
| iMatrix + RTN | 105,688 | 74,752 |
| GPTQ | 293,024 | 231,936 |
| AWQ + RTN | 111,776 | 63,488 |
| SmoothQuant + RTN | 104,864 | 49,152 |

估算列为 target weights + modifier/observer held + scratch；
实测列为设备接口报告的每 target 绝对 allocated peak，包含当时仍驻留的设备
状态。独立进程 RSS 另行记录，不能与设备 working set 直接比较。

复现单个算法：

```bash
python tools/measure_streaming_resource_curve.py \
  --case qwen=/path/to/bf16-tiny-qwen3 \
  --algorithm gptq --device cuda:0 \
  --work-root /tmp/resource-work-new \
  --output-root /tmp/resource-output-new > gptq-resource.json
```

## 验证范围与后续接口

专项回归覆盖 admission 失败不读取 target、动态估算与静态声明的区别、
reservation 释放、不同设备隔离、预取退让、AWQ CPU cache、
8/32/128 子图恢复扫描，以及五种算法与 sequential oneshot 的结果对照。
包含 manifest、artifact、pretrained 和 MoE expert group 的最终回归结果为
`158 passed, 1 skipped`；跳过项需要另行配置大型本地 checkpoint 的完整 PTQ RSS
测试，不影响本次独立恢复曲线和五算法实测。所有本次修改 Python 文件的 Ruff
检查通过。

本次交付单参与者预算。contract 预留 parameter placement、statistics sync
scope、collective owner 和通信字节字段，多 rank 执行在 preflight 明确拒绝，
后续由 EP-B 接入具体 group 和每 rank 预算。
特化模型、未知 observer 和跨子图变换的额外资源能力需要独立适配，不复制主循环。

这里的 admission 不是任意模型所有算子、backend workspace 和缓存 kwargs 的
绝对峰值证明。AWQ 现有估算仍不包含任意额外 forward kwargs，GPTQ auto batching
仍由原有代码根据运行时容量决定批量；本次保留这些算法实现。
tiny 实测不能替代更大模型上的资源曲线或质量验证。未提供可靠估算的自定义
modifier/observer 继续拒绝，而不是自动换算法。
