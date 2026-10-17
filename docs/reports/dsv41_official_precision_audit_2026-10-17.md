# V4.1 官方网络精度合同核查

日期：2026-10-17。核查对象：
`/Users/ang/models/DeepSeek-V4.1-Flash/inference/`。
本轮仅诊断，未修改默认网络计算、权重加载策略或 checkpoint。

## 缓存一致性的含义

固定相同的 token 序列，比较每一步最后位置的 logits：

- Cached：先 prefill，再复用历史状态，仅计算新增 token。
- Recompute：每次输入完整前缀，重新计算所有 token。

V4.1 历史状态除了 window/compressed KV，还包括 index K、
未完成压缩组的投影、Engram token 历史。比较的不是随机采样的两个生成结果。
该检查应作为增量适配的验收项；不能只用 top-1 相同或“浮点舍入”
解释掉严格比较失败。

## 官方代码的精度合同

`generate.py:118` 将默认 dtype 设为 BF16，但网络对敏感路径显式指定 FP32。
不能由全局 BF16 推断所有权重、计算和状态都是 BF16。

| 阶段 | 官方精度 | 本项目当前实现 |
| --- | --- | --- |
| mHC `hc_*_fn/base/scale` 参数 | FP32，`model.py:942` | 显式 FP32，一致 |
| `hc_mixes` 输入、RMS、投影 | FP32，`model.py:954` | 显式 `.float()`，一致 |
| Sinkhorn 的 `mixes/pre/post/comb` | kernel 参数与内部归约均 FP32，`kernel.py:414` | CPU `hc_split` 保持 FP32；TileLang 本轮未执行 |
| `hc_pre` 乘加 | FP32，随后转回输入 dtype，`model.py:961` | 一致 |
| `hc_post` 乘加 | FP32 系数参与，算术结果 FP32 | 一致 |
| `hc_post` 返回值/跨层残差流 | `y.type_as(x)`，通常 BF16，`model.py:968` | `.to(x.dtype)`，一致 |
| 压缩比 > 1 的 `compressor.wkv/wgate` | 参数、投影、partial state 为 FP32，`model.py:446` | 投影/partial state 为 FP32；参数构建未显式 FP32 |
| LM/Markov 输出 head | 参数 FP32、logits FP32，`model.py:1008` | 计算 FP32；参数构建跟随默认 dtype |
| DSpark confidence 投影 | 参数与计算 FP32，`model.py:1096` | 计算 FP32；参数构建跟随默认 dtype |

官方 `hc_post` 没有将跨层 residual 永久保留 FP32。将 residual stream
全面改为 FP32，属于改变官方计算合同，而非补回被遗漏的官方 cast。

还观察到文本 RMSNorm 参数的存储差异：官方跟随 BF16 默认 dtype，
本项目复用的 RMSNorm 显式 FP32；二者归一化算术都在 FP32，输出回到输入
dtype。视觉 RMSNorm 在官方也显式 FP32。

## 直接执行官方函数的受控实验

诊断脚本从本地官方 `model.py` 的 AST 中提取原封不动的
`hc_mixes/hc_pre/hc_post` 函数，绑定到已有 tiny W8A8 模型实例。
不改源文件、不改权重、不改 attention/cache，不修改全局模型类。
官方 GPU Sinkhorn 由同一个 CPU 参考原语替代。
测试前 5 token prefill，再逐个输入 3 token。

| 实验 | decode 1 最大 logit 差 | decode 2 | decode 3 |
| --- | ---: | ---: | ---: |
| 现实现 | 0.0138549805 | 0.0031156540 | 0.0048778653 |
| 直接替换官方 mHC 原函数 | 0.0138549805 | 0.0031156540 | 0.0048778653 |
| 仅补齐官方 FP32 参数例外 | 0.0138549805 | 0.0031156540 | 0.0048778653 |
| 仅将 mHC RMS/投影改为 FP64，系数仍 FP32 | 0.0000009537 | 0.0000009537 | 0.0000009537 |

官方函数替换前后的完整比较记录完全相同，表明本次触发点不是
`hc_pre/hc_post` 的适配中遗漏 FP32。
最后一行是超出官方精度的诊断，不是默认推理修复。
之前的固定逐行 FP32 mHC 投影实验也获得相同量级的改善。

已定位的首个 BF16 residual 差仍为第 21 层 FFN mHC 合并：

| 值 | Cached | Recompute |
| --- | ---: | ---: |
| FP32 合并值 | 13.468748092651367 | 13.468750000000000 |
| 官方规定的 BF16 返回值 | 13.4375 | 13.5 |

这里的中间计算已经是 FP32；FP32 投影归约在不同计算形状下产生的微差，
经过官方规定的 BF16 输出边界后被放大。不能据此声称 FP32 控制系数
误用了 BF16，也不能据此声称严格缓存一致性已经解决。

## 发现的真实 dtype 差异与影响

应单独修正模型声明中未显式保留的官方 FP32 参数例外：

- ratio > 1 的 compressor `wkv/wgate`。
- LM 和 Markov head。
- DSpark confidence `proj`。

现有 wrapper 在 forward 中会把这些权重转成 FP32，因此“参数存储 BF16”
不等于“乘加用了 BF16”。但如果源 checkpoint 包含 BF16 不能表示的数值，
先降精度再 `.float()` 无法找回丢失的信息。

本 tiny 的上述权重在 source 和 PTQ 文件中本来就是 BF16，转换为 FP32
只改变存储 dtype、不恢复额外精度。受控实验也确认不会改变这次结果，
所以它们不是本次 `0.01385` 差异的原因。

## 验证边界与下一步

本地模型目录含官方网络、kernel、配置和权重索引，但索引列出的 48 个权重
分片均不在本地。本轮没有读取原尺寸官方权重，也没有运行官方 CUDA/
TileLang kernel。因此结论是源码精度合同与 CPU tiny 受控实验的结论，
不能外推为官方 GPU 运行结果。

默认路径不应为了通过测试直接把整条 residual stream 改成 FP32。
先独立对齐上述 FP32 参数声明，再验证固定计算顺序的 mHC 参考路径及
更多 token/cache 边界。严格缓存比较失败继续保留。

复现命令：

```shell
.venv/bin/python tools/audit_tiny_dsv41_precision.py \
    .cache/tiny-models/DeepSeek-V4.1-Flash-Tiny \
    --reference-dir /Users/ang/models/DeepSeek-V4.1-Flash/inference
```

输出 `official-precision-audit.json`，记录源码 SHA256、行号、
各组结果和诊断后恢复状态。所有临时替换的函数、dtype 和参数值均恢复。
