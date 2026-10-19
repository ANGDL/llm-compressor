# V4.1 tiny：W8A8 / WNa8 logit 误差复查

日期：2026-10-17。范围：已有完整拓扑 tiny checkpoint，CPU 参考执行。
未重新量化、训练或修改 checkpoint，也未修改默认模型计算路径。

> **重点未解决问题：W8A8 cache/recompute 数值分叉**
>
> 当前 W8A8 的整体量化 logit MSE 低于 WNa8，但在特定 prefill 长度下，
> cache/recompute 仍会失败。已定位到第 21 层 FFN mHC residual post merge：
> 两条执行形状产生约 `1.91e-6` 的 FP32 微差，随后跨过 BF16 中点，
> 使 `13.4375` 与 `13.5` 两个中间值分叉，首次 decode 最大 logit 差为
> `0.01385498046875`。该失败必须继续保留，不能通过放宽阈值或用较低的
> WNa8 路径一致性替代验收。该问题与 vision 校准、streaming/resident
> 一致性和 sparse attention 对齐分别验收。

## 结论与比较口径

本次相同源模型对照中，W8A8 的整体 logit 误差小于 WNa8，符合精度预期。
上一轮展示的是 **同一 checkpoint 的缓存解码与整段重算差异**，不是
**PTQ checkpoint 与源模型之间的量化误差**。此前没有同时列出后者，
容易把缓存路径一致性当成量化质量指标。

这两种差异必须分别报告：

| 检查 | 两端 | 用途 |
| --- | --- | --- |
| 量化后相对源模型的误差 | PTQ 与 source，同一完整序列、同一执行 dtype | 比较量化方案 |
| cache/recompute 一致性 | 同一 PTQ checkpoint，增量与整段重算 | 检查增量适配与数值敏感性 |
| streaming/resident 一致性 | 同一 checkpoint，同一输入和 cache 路径 | 检查流式权重加载与执行 |

不能要求不同 checkpoint 的第二项严格随权重位宽单调变化。
也不应以单个 logit 的最大绝对差作为整体量化精度的唯一排序依据。
本报告仍保留 cache/recompute 的失败，不以量化质量符合预期代替它。

## 相同源模型对照

参考为量化输入 `source-fp8-fp4`，保留其原有激活模拟。
各模式使用相同 BF16 执行权重与显式 FP32 控制参数策略，
PTQ 保留 A8 Q/DQ。使用 8 段已有 toy 文本，每段最多 64 token，
合计 344 token；均为无 padding、无 cache 的完整序列 forward。
MSE/RMSE 按全部有效位置和整个词表聚合。

| 模式 | Logit MSE | Logit RMSE | 源模型 top-1 一致率 |
| --- | ---: | ---: | ---: |
| W8A8 | 0.0018227344 | 0.04269349 | 99.7093% |
| WNa8 | 0.0034085986 | 0.05838320 | 99.7093% |
| W4A8 | 0.0046083839 | 0.06788508 | 99.1279% |

W8A8 的聚合 MSE 比 WNa8 低约 46.5%，8 段文本各自的 MSE 也均更低。
这里的参考本身采用源 FP8/FP4 权重和模拟激活，
不能把结果外推为原始大模型质量或推理内核精度。

权重核查发现：

- WNa8 为 310 个 W8 模块与 996 个 routed-expert W4 模块。
- W8A8 为 1306 个 W8 模块。
- 两者共同的 310 个 W8 模块，解码并应用执行 dtype 后逐点相同。
- 所有非量化参数及参数 dtype 相同。
- 两者的 1306 个量化模块使用相同的激活 scheme。
- WNa8/W8A8 的差别集中于 996 个 routed-expert 权重的 W4/W8 表示。

同一段文本改用不同 prefill 长度后，缓存一致性的排序还会反转。
下表为每组 prefill 加连续三次 decode 的最大 logit 绝对差：

| Prefill token 数 | WNa8 cache/recompute | W8A8 cache/recompute |
| --- | ---: | ---: |
| 3 | 0.00790167 | 0.01678562 |
| 5 | 0.00000191 | 0.01385498 |
| 7 | 0.01929736 | 0.00000143 |

原先只报告 prefill=5，不能据此排序方案精度。

## W8A8 的 cache/recompute 偏差

复现输入为 “The capital of France is Paris. Paris is the largest city in France.”。
取前 5 token prefill，再连续输入 3 token decode；重算端每次输入完整前缀。
流式与常驻加载此前已确认相同，本次使用常驻模型定位数值来源。

首次 decode 的最大 logit 差为 `0.01385498046875`。首次 BF16 状态差异出现在
`model.layers.21` 的 FFN mHC 残差合并，层号从 0 开始：

| 项目 | Cached decode | 完整前缀重算 |
| --- | ---: | ---: |
| 舍入前 FP32 值 | 13.468748092651367 | 13.468750000000000 |
| BF16 值 | 13.4375 | 13.5 |

这个合并的 FFN 输出与输入 residual 逐点相同；mHC 控制系数含微小差别。
两个 FP32 值相差约 `1.91e-6`，其中重算值恰好位于两个 BF16 数值的中点，
经过 BF16 舍入后产生 `0.0625` 的状态差。后续层将该差异传到最终 logits。

对第 0 层 mHC 投影冻结完全相同的输入和权重，只比较批量 `F.linear`
与其中最后一个 token 的单独 `F.linear`，已能观测到 `2.3841858e-7` 的差别。
这说明控制投影的计算形状本身就能引入浮点扰动。

进一步只在诊断进程中，将所有 mHC 投影改为固定逐行的 `F.linear`：

- checkpoint、A8 Q/DQ、cache、attention 与其他模型算子不变。
- 原始 BF16 执行策略不变。
- 三次 decode 与完整前缀重算的最大 logit 差均降到 `9.5367432e-7`。
- 诊断结束后恢复所有原生方法，未修改模型实现或 checkpoint。

这组受控实验定位了本次 W8A8 的触发链：
投影计算形状 → mHC 系数的浮点扰动 → BF16 中点舍入 → 后续状态差异。
WNa8 的 routed-expert 权重不同，内部值也不同；原测试没有触发同样的
中点舍入，并不说明它比 W8A8 的量化精度更高。

逐行 mHC 仅用于定位，未加入默认推理路径；它会影响性能和计算顺序。
W4A8 的全部数值敏感性也不能仅由这一个 W8A8 实验宣称已经解决。

后续[官方精度合同核查](dsv41_official_precision_audit_2026-10-17.md)确认：
mHC 的中间计算已经是 FP32，官方同样在 post 合并后返回 BF16。
直接替换官方原函数仍复现相同偏差。还发现了 compressor/head 的 FP32
参数声明差异，但因现有 tiny 源值就是 BF16，补齐声明不会改变本次结果。

## 复现

```shell
.venv/bin/python tools/diagnose_tiny_dsv41_precision.py \
    .cache/tiny-models/DeepSeek-V4.1-Flash-Tiny
```

输出 `precision-diagnosis.json`，分开记录：

- `quality_vs_source`：PTQ 相对同一 source 的误差。
- `cache_consistency`：同一 checkpoint 在多个 prompt 长度上的缓存一致性。
- `checkpoint_audit`：位宽、公共权重与 dtype 核查。
- `w8a8_rounding_diagnosis`：首次 mHC 舍入差及受控重放。

该诊断工具在 CPU 加载常驻 tiny 模型，有 100M 参数上限，
不用于原尺寸模型，也不替代流式资源测试。
