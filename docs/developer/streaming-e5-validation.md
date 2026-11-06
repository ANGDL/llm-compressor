# E5 首批适配与验证

日期：2026-11-06。测试环境为 node16 的 `llm-quant-base` 容器，
代码路径 `/data/quant/llm-compressor`，Python 3.12、AutoRound 0.15.1。
设备测试使用容器提供的 `torch_xmlir` CUDA 兼容接口 `cuda:0`。

## 实现范围

新增独立 `streaming/modifier_adapters.py`，将能力契约、recipe 顺序、
工作集估算、target 所有权、算法初始化参数和故障清理集中在适配层。
沿用 pretrained streaming 的原有加载、校准、更新、压缩、事务和发布循环。
未修改 `modifiers/gptq/*`；E1 仍暂停。

| 方法 | 本批可执行范围 | 未纳入本批 |
| --- | --- | --- |
| QAD | `[QuantizationModifier, QADModifier]`；weight-only；CPU teacher/input/最佳权重快照；至少两个 batch；原算法训练与验证切分 | GPTQ + QAD 等组合、激活/KV 量化、optimizer step 级恢复 |
| AutoSmooth | `[AutoSmoothModifier, QuantizationModifier]`；weight-only；所有 smooth/balance/parent 都在一个 target 内 | 跨 target mapping、共享参数的跨 target 修改、任意 modifier 组合 |
| AutoRound | 独立 modifier；symmetric INT4/INT8 weight-only；Linear 两维均为 32 的倍数；compile 关闭；pipeline 独占 placement/output | 非对称/浮点量化、外部多设备 placement、cross-block q_input 恢复 |
| SpinQuant/QuIP | 提前拒绝，错误说明离线/在线变换适配器的缺口 | 尚未可执行；后续规划见 E5 主计划 |

AutoRound 通过原 modifier 中一个可选 State 回调接收 device/dtype/AMP/seed，
普通流水线没有这个回调时保持原路径。原有算法并未被替换为 RTN。
AutoRound 的 `q_input` 每个 target 清空，本批明确选择 independent-block 语义，
与普通 AutoRound 的跨 block 误差传播不等价。权重与 qparams 仍由 AutoRound
训练，CT 负责压缩，streaming writer 负责保存与提交。
AutoRound 的对称 INT scale 允许带符号，输出校验要求非零且有限；不能简单要求
scale 为正。外部优化临时将 BF16 参数提升为 FP32 时，适配层在压缩前恢复每个
weight/bias 的原 dtype 和设备，包含未量化的 norm。

适配器版本进入 modifier contract；AutoRound 的依赖版本也进入契约 fingerprint。
Python、NumPy 和 Torch 的 CPU/执行设备 RNG 按 target 设置、在离开时恢复。
QAD 的验证切分与 epoch 顺序继续使用原 modifier 的局部 seed。
外部算法调用 `manual_seed_all` 时，已初始化的其他 CUDA generator 也恢复原状态。
AutoRound 0.15.1 会开启进程级 deterministic algorithms 并设置
`CUBLAS_WORKSPACE_CONFIG`；适配器在正常返回或异常时恢复调用方的 enabled、
warn-only 和环境变量，防止影响后续模型初始化及 GPTQ 执行。

## 事务与正确性

CPU 测试使用本地构造的两层标准 Transformers Qwen3：hidden/intermediate=64，
2 个 attention heads、1 个 KV head、head_dim=32，4 个固定校准 batch。
模型从本地 safetensors 读取，无模型下载。QAD 使用 1 epoch，AutoSmooth 使用
2 个 grid 点，AutoRound 使用 2 次优化迭代；测试目的为接入和恢复正确性。

验收覆盖：

- 三种算法均真实执行两层，导出 checkpoint 后通过标准 Transformers reload，
  forward logits 有限，量化层具备实际 scale。
- AutoSmooth 确认逐层进入原算法的 grid search，并验证保存的 norm 与选中 scale
  一致；选中 identity scale 是合法搜索结果。
- QAD、AutoSmooth 与同 CPU/FP32 的普通 sequential oneshot 逐 tensor 对照，
  容差 `rtol=3e-3, atol=3e-5`。
- 三种算法分别注入“更新后、尚未提交”和“第一子图已经提交”的异常，重新运行
  相同 work_dir 后与未中断产物逐 tensor bitwise 相同。
- QAD batch 不足、AutoSmooth 跨 target mapping、AutoRound 不合法维度和丢失
  量化输出均明确报错；SpinQuant/QuIP 的范围限制在 preflight 报错。
- AutoRound 的 INT4 导出重载权重与算法训练后权重对照；INT8 独立执行并重载。
- 三种算法以 FP32/BF16 在 `cuda:0` 执行并重载，保存的 norm dtype 保持原值。

实测发现并修复了以下集成问题：AutoSmooth/AutoRound 的更新 forward 需要禁用
CT fake quant，避免未就绪 qparams 产生 NaN 或缺少 scale；恢复打包的 INT4
事务时还需恢复“原始 weight 已消费”的集合，避免最终静态复制再次加入浮点
weight。该修复沿用现有事务中的 quantized module 元数据。
另有 AutoRound 的临时 dtype 转换和进程级确定性设置泄漏，均在独立适配层处理。

## 验证命令

```bash
python -m pytest -q tests/llmcompressor/streaming/test_modifier_adapters.py
```

最终全量 PTQ 回归命令：

```bash
python -m pytest -q tests/llmcompressor/streaming \
  --ignore=tests/llmcompressor/streaming/inference
```

实测结果为 **384 passed, 2 skipped**，耗时 50.03 秒；其中 E5 专项的
32 个用例全部通过。两个跳过项分别需要显式配置本地 Kimi K3 checkpoint 和
完整 PTQ RSS 测量 checkpoint。本次没有排除 DeepSeek 或 GPTQ 回归用例。
新适配层及本次修改的 streaming Python 文件与测试通过 Ruff，`git diff --check`
通过。原 AutoRound modifier 仅增加三行可选回调。

定位过程中，AutoRound 遗留的 deterministic 设置曾令未初始化参数的旧 fixture
产生 NaN，并令后续 GPTQ 对照用例异常耗时。修复状态恢复后，同一进程的完整
回归通过；没有修改这些旧测试来绕过失败。

附加执行原 AutoSmooth/AutoRound 本地测试时，37 个原用例通过，2 个依赖模型
配置缓存的用例未执行；原 MXFP4 初始化测试仍受现有 `uint8` scale dtype 校验
限制。另行尝试的原 QAD NVFP4 CPU 用例遇到已安装 CT 的 Float8 promotion
限制。这些属于本批 INT4/INT8 接入验收以外的格式/后端限制，未修改原算法解决。

## 后续与限制

E5 尚未整体完成。SpinQuant/QuIP 的局部离线变换、全局 tensor 事务归属、
在线元数据导出与 reload 验证按主计划分阶段推进；不增加特化模型分支。
本次也未做真实大模型精度/吞吐验收或严格峰值上界验证。

适配器的 held/scratch/host 估算接入现有 admission，覆盖声明的训练权重、
优化器/最佳参数副本和主要缓存。它们仍属于保守工作集估算，不能推导任意模型
activation、attention/backend workspace、额外 kwargs 或任意外部依赖版本的
严格峰值保证。扩大 scheme、算法组合和设备范围前需要各自补齐验收。
