# 在仅支持原生 FP8 的 GPU 上验证 V4.1 算子

`tools/validate_dsv41_gpu_kernels.py` 不加载 checkpoint，不创建模型，也不依赖
Transformers。脚本直接加载指定的官方 `inference/kernel.py`，把官方 GPU
输出与独立 PyTorch reference、仓库中的适配实现进行比较。

环境要求以当前本地官方 `inference/requirements.txt` 为准：
`torch>=2.10.0`、`tilelang==0.1.12`，以及能运行官方 kernel 的 CUDA 环境。
脚本记录 GPU 型号、计算能力、PyTorch/CUDA/TileLang 版本和源码 SHA256。
原生 FP8 支持并不保证某个版本的 TileLang 能编译所有官方 kernel；
编译失败会作为 error 写入报告。

在仓库根目录执行，替换 GPU 机器上的官方代码路径：

```bash
python tools/validate_dsv41_gpu_kernels.py \
  --official-kernel /path/to/DeepSeek-V4.1-Flash/inference/kernel.py \
  --device cuda:0 \
  --quick \
  --report /tmp/dsv41-gpu-quick.json
```

完成快速检查后，去掉 `--quick` 执行完整形状矩阵：

```bash
python tools/validate_dsv41_gpu_kernels.py \
  --official-kernel /path/to/DeepSeek-V4.1-Flash/inference/kernel.py \
  --device cuda:0 \
  --report /tmp/dsv41-gpu-full.json
```

首次运行每个形状会触发 TileLang JIT 编译，不能把编译时间视为推理性能。
本脚本用于数值检查，不做性能基准。

默认覆盖：

| 项目 | 比较内容 |
| --- | --- |
| mHC/Sinkhorn | 官方 FP32 `pre/post/comb` 与独立参考、项目实现；hc=2/4 |
| 稀疏注意力 | 官方 64-KV 在线 softmax、BF16 权重转换、FP32 累加；额外比较项目全局 softmax 路径 |
| FP8 activation | FP32 scale、power-of-two FP32 scale、E8M0 scale；比较编码、scale 和 BF16 Q/DQ |
| FP8 GEMM | 原生 FP8 × FP8，与按 group 应用 scale 的 FP32 累加参考比较 |
| mHC 形状诊断 | 同一 token 在完整序列与单 token 投影时的系数、FP32 合并值、BF16 合并值 |
| FP4 软件语义 | E2M1 nearest-even、中点邻居、零组、E8M0/E4M3 scale，与项目软件 Q/DQ 比较 |

默认流程**不调用官方 `fp4_act_quant` 或 `fp4_gemm`，不编译原生 FP4
kernel**。FP4 软件检查使用 FP32 运算、查表和 FP8 scale cast，不要求硬件
支持 FP4。报告固定将原生 FP4 标记为 `skipped`；软件检查通过不能证明官方
FP4 kernel 对齐。

稀疏注意力覆盖随机索引、部分屏蔽、全部 `-1`、重复索引、最后一个有效
位置、跨 64-KV 边界，以及首个 block 完全屏蔽。包含生产模型的 head_dim=512，
并比较同一个 query 在整段执行与单独执行时的官方输出。

只跑某些算子时使用 `--operators`：

```bash
python tools/validate_dsv41_gpu_kernels.py \
  --official-kernel /path/to/DeepSeek-V4.1-Flash/inference/kernel.py \
  --operators hc sparse fp8 fp8_gemm mhc \
  --report /tmp/dsv41-without-fp4.json
```

每个 case 有独立且稳定的 seed，筛选算子或使用 `--quick` 不会改变同名
case 的输入。默认继续收集各算子的失败；需要首个失败即停止可加
`--fail-fast`。

项目 reference 从仓库的两个 Python 源文件直接加载，官方 backend dispatch
被禁用，避免实际比较两次同一个官方 kernel。如果只复制该脚本到 GPU
机器，官方与独立 reference 仍能测试，但报告会显示项目源码不可用。
要验证适配实现，请复制仓库或使用 `--project-root /path/to/llm-compressor`。
`--no-project` 可显式关闭项目比较。

报告中每项有 `actual_dtype/reference_dtype`、有限性、max abs、RMS、
relative L2、元素差异数和 `strict_allclose`。BF16 结果额外报告
`max_bf16_steps`；`largest_difference` 提供最大误差位置与数值。

- FP32 默认以 `atol=1e-6, rtol=1e-5` 检查。
- FP8 编码、scale、Q/DQ 和 FP4 软件语义按精确相等检查。
- BF16 attention/GEMM 默认预算为 `atol=0.002, rtol=0.015625`，同时保留严格
  比较结果。预算内通过不等于逐元素相等；可通过 `--bf16-atol`、
  `--bf16-rtol` 收紧预算。
- mHC 的 shape 比较属于 `diagnostic`，不是端到端 cache 验收项。
  关注 `diagnostics.bf16_hc_post_full_vs_decode` 的精确相等、
  `mismatched_elements` 和 `max_abs`；它不参与算子通过数。

退出码：`0` 表示所选检查执行完毕且验收项通过；`1` 表示至少一项检查失败
或算子报错；`2` 表示初始化失败、中断或 fail-fast 未完成。即使退出码为 0，
也要单独查看 mHC diagnostic 和原生 FP4 的 skipped 状态。报告在每个 case
结束时更新，编译错误会保留 traceback。

没有 CUDA 时可以预检 reference 和报告流程：

```bash
python tools/validate_dsv41_gpu_kernels.py \
  --self-test \
  --report /tmp/dsv41-reference-self-test.json
```

这不会执行 TileLang，不代表 GPU 算子验证通过。算子测试也不代替加载实际
checkpoint 后的 prefill/decode、KV cache、logits 和量化权重验收。
