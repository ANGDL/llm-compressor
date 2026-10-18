# 在仅支持原生 FP8 的 GPU 上验证 V4.1 算子

`tools/validate_dsv41_gpu_kernels.py` 不加载 checkpoint，不创建模型，也不依赖
Transformers。脚本直接加载指定的官方 `inference/kernel.py`，把官方 GPU
输出与独立 PyTorch reference、仓库中的适配实现进行比较。

如果只需要 kernel 级比较，可使用
`tools/compare_dsv41_kernel_level.py`。它直接调用官方
`hc_split_sinkhorn`、`sparse_attn`、`act_quant` 和 `fp8_gemm`，不经过
模型 forward、权重加载、cache 或 streaming pipeline。

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

kernel 级对照：

```bash
python tools/compare_dsv41_kernel_level.py \
  --official-kernel /path/to/DeepSeek-V4.1-Flash/inference/kernel.py \
  --project-root "$PWD" \
  --device cuda:0 \
  --report /tmp/dsv41-kernel-level.json
```

快速 smoke test：

```bash
python tools/compare_dsv41_kernel_level.py \
  --official-kernel /path/to/DeepSeek-V4.1-Flash/inference/kernel.py \
  --project-root "$PWD" \
  --device cuda:0 \
  --quick \
  --report /tmp/dsv41-kernel-level-quick.json
```

该脚本的 `project_vs_order_reference` 是项目实现和独立计算顺序参考的
严格比较；`official_vs_project` 则是官方 kernel 与项目实现的直接比较。
FP8 GEMM 当前项目没有独立的原生 FP8 GEMM kernel，因此该项只比较官方
kernel 与按 scale 分块累加的参考，并在报告中明确标记。

对 sparse attention 做逐块中间值对照：

```bash
python tools/compare_dsv41_kernel_level.py \
  --official-kernel /data/DeepSeek-V4.1-Flash-1017/inference/kernel.py \
  --project-root /data/llm-compressor \
  --device cuda:0 \
  --operators sparse \
  --trace-sparse \
  --save-sparse-tensors \
  --report /tmp/dsv41-sparse-trace.json
```

该命令保留 H=4/D=512/K=65、H=64/D=512/K=640 两个用例，seed 与此前
kernel 级报告相同。可加 `--quick` 只跑第一项。原有
`official_vs_project` 等检查仍使用 `atol=1e-6, rtol=1e-5`；
诊断输出不会覆盖失败结果或提高阈值。

`--trace-sparse` 在官方源码副本中插入只读中间值的全局内存写出，保留原有
GEMM、计算语句、线程数和 pipeline 配置；原始官方文件不修改。若官方函数
签名、64-KV block 或预期记录边界发生变化，会报告诊断错误。
工具逐块记录 QK、缩放后 score、running max、rescale、FP32 exp、
block sum、累计分母、BF16 权重、重缩放后的 accumulator 和 value GEMM
之后的 accumulator，最后记录加入 sink 的分母、转换前 FP32 输出和
最终 BF16 输出。各阶段保留匹配的正负无穷 mask，NaN 或不一致的
非有限值会被标记。

插入写出可能影响编译决策。诊断首先要求：

- `checks.instrumented_vs_official.exact` 为 true：插桩最终输出与原始官方
  输出逐元素相等。
- `checks.instrumented_normalized_cast_vs_output.exact` 为 true：记录的
  FP32 最终结果转 BF16 后与插桩实际输出相同。
- PyTorch trace 的最终输出与项目实现、独立参考分别精确相等，且记录的
  snapshot 不含未写入的 NaN。

未满足时，`sparse_trace.status` 为 `untrusted`，`attribution_valid` 为 false，
不会给出可信的首次差异结论。编译或执行异常保留原始比较结果与 traceback。
这些校验是使用中间值归因的必要条件；最终输出相同仍不证明插桩前后每条
内部浮点指令完全相同，需要结合生成代码判断。

重点读取：

| JSON 字段 | 含义 |
| --- | --- |
| `checks.official_vs_project.largest_bf16_step_difference` | 最大 BF16 表示值步数差的坐标、双方数值、绝对差和步数 |
| `sparse_trace.first_exact_divergence` | 按 block 和实际运算顺序扫描，首次逐元素不相等的位置 |
| `sparse_trace.first_strict_divergence` | 首次超出原严格阈值的位置 |
| `sparse_trace.first_bf16_weight_divergence` | 首次块内 BF16 exp 权重不相等的位置 |
| `sparse_trace.qk_backend_comparison` | 官方 QK snapshot 与不同 PyTorch GEMM 输入/累加 dtype 路径的逐 block 比较 |
| `sparse_trace.block_sum_backend_comparison` | 官方 block sum 与不同 FP32 reduction tree 候选的逐 block 比较 |
| `sparse_trace.value_gemm_backend_comparison` | 固定官方 BF16 权重和重缩放 accumulator 后，对比不同 value GEMM reduction 候选 |
| `sparse_trace.stages` | 所有 block/阶段的误差指标、最大差值坐标和轴含义 |
| `sparse_trace.output_points` | 最大绝对差和最大 BF16 步数差输出点对应的逐块 score/权重、分母和 accumulator 轨迹 |
| `sparse_trace.head_padding` | 项目 head 数与官方 kernel 实际补齐后的 head 数 |

坐标均从 0 开始，输出坐标为 `[batch, query, head, channel]`；
块内 score/权重的最后一维为 `key_in_block`，候选序号为
`block * 64 + key_in_block`。这与索引选中的 KV 位置有区别，后者保存在
原始 `indices` 张量中。官方会将不足 16 个 head 补齐到 16；逐块诊断的
PyTorch reference 也会先按 16 个 head 执行，再裁剪回公开输入 head 数，
避免把 GEMM 的矩阵形状差异误判为算子归约顺序差异。生产项目 reference
仍保留公开输入形状；如果 padded reference 与项目输出不再精确相等，诊断
会将其保留为独立的 head-padding 差异。

默认在 JSON 旁建立 `dsv41-sparse-trace-artifacts/`，写入插桩源码及其 hash；
若 TileLang 提供源码导出接口，也保存生成的 CUDA 源码。使用
`--save-sparse-tensors` 后，还会为每个用例保存包含输入、输出和完整中间值的
CPU `.pt` 文件；JSON 只保存指标和选中点的轨迹。可用 `--trace-dir` 指定目录。
报告在每个 case 后保存，并在诊断 JIT 前保存当前 case 的原始比较结果，
中断后可以查看部分报告。
诊断编译失败单独计入 `sparse_trace_summary.error`；原 case 的基础检查和
通过/失败结果仍保留。PyTorch 中间值来自计算顺序参考，并对项目最终输出做
精确校验；它不是对项目生产函数的直接插桩。全屏蔽 query 的最终归一化参考
遵循官方的 `0/inf`，项目校准路径为保持梯度有限采用 `0/1`，两者输出均为零。

`qk_backend_comparison` 当前包括：

- `fp32_einsum`：Q 和 KV 先转 FP32，再用 PyTorch einsum；
- `bf16_bmm_out_fp32`：保留 BF16 输入，用 `torch.bmm(..., out_dtype=float32)`；
- `bf16_bmm_default_cast`：BF16 bmm 输出后再转 FP32。

其中 `bf16_bmm_out_fp32` 依赖当前 PyTorch/CUDA 对 BF16 输入和 FP32 输出的
支持；不支持时只记录该 backend 的错误，不影响原始 sparse case 的状态。
该诊断用于判断 QK 归约路径是否接近 TileLang `T.gemm`，不会替换项目
生产实现，也不会改变原有通过阈值。

完整 sparse trace 还会记录两个隔离诊断：

- `block_sum_backend_comparison`：比较 `torch_sum`、相邻二叉树
  `pairwise_sum` 和逐元素 `sequential_sum`，用于判断官方
  `T.reduce_sum` 更接近哪种 FP32 归约树；
- `value_gemm_backend_comparison`：比较 FP32 einsum、BF16 输入/FP32
  输出 bmm、BF16 输出后转换，以及相邻二叉树的 BF16×BF16→FP32
  value GEMM。比较时固定使用官方 trace 中的 BF16 weights 和
  `rescaled_accumulator`，只隔离当前 block 的 value GEMM。

这些候选只用于定位归约顺序，不会自动替换生产实现，也不会把“误差最小”
自动解释为官方实现等价；仍需同时满足插桩 gate 和最终输出检查。

v3 GPU 报告已经给出决定性结果：`bf16_bmm_out_fp32` 与官方 QK
snapshot 逐元素精确一致，而 `fp32_einsum` 仍在 block 0 出现约
`6.48e-5` 的 FP32 差异，`bf16_bmm_default_cast` 的误差更大。因此项目
sparse attention 和独立 PyTorch reference 在 CUDA + BF16 输入时均采用
“BF16 输入、FP32 GEMM 输出”的契约；CPU 或不支持 `out_dtype` 的后端仍
回退到 FP32 einsum。value GEMM 继续使用 BF16 权重转换后的 FP32 累加。
这只是根据 v3 诊断修正 QK 计算路径，尚未替代 GPU v4 复测；在复测通过前，
原有 sparse 失败和 W8A8/W4A8 cache/recompute 失败必须继续保留。

v4 快速报告确认 QK 修正生效：QK、scaled scores、running max、rescale、
exp 和 BF16 权重均逐元素一致，`official_vs_project` 与
`project_vs_order_reference` 均精确通过，最终 BF16 输出也精确一致。当前
trace 的首次逐元素差异移动到 `block_sum`，只有 4 个元素，最大绝对差
`1.9073486328125e-6`，但仍在严格阈值内；这对应官方 `T.reduce_sum` 与
PyTorch `torch.sum` 的归约顺序差异。v4 还显示 value accumulator 存在同量级
的 FP32 差异，因此项目和 reference 已进一步改为 BF16 权重 × BF16 KV、
FP32 输出的 value GEMM；该修正需要下一轮 GPU trace 验证，不能用 v4 之前的
报告替代。

v5 快速报告确认 value GEMM 修正生效：block 0 的
`rescaled_accumulator` 和 `accumulator` 均逐元素一致；block 1 的
`rescaled_accumulator` 也精确一致，`accumulator` 仅有 127 个元素不同，
最大绝对差 `4.76837158203125e-7`。这说明剩余差异来自第二个 block 的
FP32 GEMM 归约顺序，而不是 BF16 权重、QK、running max 或在线 softmax
状态错误。`block_sum` 仍是首次逐元素差异（4 个元素、最大
`1.9073486328125e-6`），denominator 和 normalized 只继承该微小误差，
最终 BF16 output 仍逐元素精确一致。因而 v5 的 `status=passed` 代表最终
算子验收和严格阈值通过，不代表每个 FP32 中间张量 bitwise exact；这些
中间差异必须继续保留在报告中，不能通过放宽阈值抹除。

v6 完整矩阵显示 H=4/K=65 仍通过，但生产形状 H=64/D=512/K=640
失败：QK、running max、exp 和 BF16 weights 仍精确一致，首次严格分歧在
block 2 的 `accumulator`，最大绝对差 `4.76837158203125e-6`；误差随后
随 10 个在线 block 累积，最终有 56 个 BF16 输出元素不同，最大绝对差
`0.0009765625`，最大 BF16 步数差 10。这个失败必须保留，不能用
当前 `atol/rtol` 通过就标记为 bitwise 对齐完成。

v7 reduction 候选对照进一步确认，生产 value GEMM 当前选择已经正确：

- H64 用例中 `bf16_bmm_out_fp32` 的 value GEMM 候选最大绝对差为
  `5.7220458984375e-6`，总不一致元素数为 `1,440,099`；
- `fp32_einsum` 最大绝对差为 `7.62939453125e-6`，总不一致元素数为
  `1,614,212`；
- `bf16_pairwise_fp32` 最大绝对差同为 `7.62939453125e-6`；
- `bf16_bmm_default_cast` 最大绝对差约 `0.03117`，10 个 block 都超过
  严格阈值，明确不是官方路径。

block sum 候选中，H64 的 `torch_sum` 总不一致元素数为 `1,650`，
`pairwise_sum` 为 `1,837`，`sequential_sum` 为 `3,567`；三者最大绝对差
分别为 `1.9073486328125e-6`、`1.9073486328125e-6` 和
`5.7220458984375e-6`，都不能逐位复现官方 `T.reduce_sum`。因此当前不再
用候选 reduction tree 猜测性替换生产实现：QK/value 的 BF16 输入、FP32
输出 contract 已锁定，H64 剩余失败归因于 TileLang reduction tree 与
PyTorch/CUDA bmm 的实现差异，仍需保留为未完成的 bitwise 对齐项。

仅需本地 CPU 检查时，运行文末的 `--self-test`；它也校验 PyTorch trace
与独立参考最终输出相等，但不执行插桩 CUDA kernel。可选 GPU 回归：

```bash
DSV41_OFFICIAL_KERNEL=/data/DeepSeek-V4.1-Flash-1017/inference/kernel.py \
  python -m pytest -q tests/tools/test_compare_dsv41_kernel_level.py \
  -k official_gpu_trace
```

W8A8/W4A8 的 checkpoint cache/recompute 验证独立进行，例如：

```bash
python tools/validate_tiny_dsv41_inference.py \
  .cache/tiny-models/DeepSeek-V4.1-Flash-Tiny \
  --modes w8a8 w4a8 \
  --report /tmp/dsv41-cache-separate.json
```

该验证入口使用 CPU，复用已有量化 checkpoint，不重新量化。它与
checkpoint-free 的 GPU kernel 诊断分别出报告；当前缓存一致性失败继续保留。

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
| 稀疏注意力 | 官方 64-KV 在线 softmax、BF16 权重转换、FP32 累加；项目路径与独立在线参考额外做严格比较 |
| FP8 activation | FP32 scale、power-of-two FP32 scale、E8M0 scale；比较编码、scale 和 BF16 Q/DQ |
| FP8 GEMM | 原生 FP8 × FP8，与按 group 应用 scale 的 FP32 累加参考比较 |
| mHC 形状诊断 | 同一 token 在完整序列与单 token 投影时的系数、FP32 合并值、BF16 合并值 |
| FP4 软件语义 | E2M1 nearest-even、中点邻居、零组、E8M0/E4M3 scale，与项目软件 Q/DQ 比较 |

默认流程**不调用官方 `fp4_act_quant` 或 `fp4_gemm`，不编译原生 FP4
kernel**。FP4 软件检查使用 FP32 运算、查表和 FP8 scale cast，不要求硬件
支持 FP4。报告固定将原生 FP4 标记为 `skipped`；软件检查通过不能证明官方
FP4 kernel 对齐。

稀疏注意力覆盖随机索引、部分屏蔽、全部 `-1`、重复索引、最后一个有效
位置、跨 64-KV 边界，以及首个 block 完全屏蔽。完整矩阵包含生产模型的
64 heads、head_dim=512，以及 128/512/640 个候选（128 window + 512 index
top-k）。官方和项目实现都比较同一个 query 在整段执行与单独执行时的输出。

此前 H20 + TileLang 0.1.13 的报告中，官方 `head_dim=32` kernel 因
8-warp GEMM 布局无法编译。默认矩阵改用 `head_dim>=64`，保留小 head 数和
K=1；可加 `--include-small-head-dim` 显式复查 d=32。该开关直接调用原始官方
kernel，不修改线程数或用 PyTorch 替代官方输出；编译失败仍记录为 error。

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
- `project_vs_online_reference` 使用严格 `atol=1e-6, rtol=1e-5`，防止项目
  计算顺序偏差被较宽的 GPU BF16 容差掩盖。报告同时保留精确相等指标。
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
