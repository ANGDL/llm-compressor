# DeepSeek V4.1 raw checkpoint 导出与加载问题

记录日期：2026-10-20，Asia/Shanghai。
工作分支：`feat/deepseek-v41-native-quantization`。

## 任务与真实产物

修复 V4.1 streaming 量化产物与原始 checkpoint 命名约定不一致的问题，
复用 V4 示例的 `--save-raw-checkpoint-format` 机制。需要同时核对 tensor
名称、packed INT4 存储形状和量化配置中的模块选择器。

本次通过一个长期 SSH/容器终端，只读检查了 node16 上的以下内容：

| 内容 | 位置 |
| --- | --- |
| 推理容器 | `zhuang_xsgl_1015` |
| 2026-10-19 量化产物 | `/ssd2/model-quality/runs/20261019T093114Z_a619a1629881/deepseek_v41_wNa8-4c4ba690/model` |
| 2026-10-20 启动日志 | `/workspace/ds_v41/w4a8/log_20261020.log` |
| 用户提供的可用 INT4 对照 | `/ssd4/models/DeepSeek-V4.1-Flash-1017-INT4` |

产物目录以 `model` 结尾；日志目录为 `/workspace`。

## 已确认的问题

产物已经使用 raw tensor 名称：索引包含 96,040 个名称，其中没有
`model.` 前缀，使用 `head.weight` 和 `.scale`。V4.1 materializer 在本次
修复前就默认开启 raw 导出，因此仅添加命令行开关不能解决这次失败。

量化配置仍然使用 native 模型的选择器，例如：

```text
re:model\.(layers|mtp)\.\d+\.ffn\.experts\.\d+\.(w1|w2|w3)$
re:model\.mtp\.0\.main_proj$
```

容器内 xSGL 将 DSpark 模块构造在 `stages.*` 下。使用容器实际安装的
`pattern_utils.py` 和运行时模块重命名规则做 CPU 名称匹配时：

| 运行时模块 | 原导出配置 | 去掉选择器的内部 `model.` 前缀 |
| --- | --- | --- |
| `model.layers.0.mlp.experts` | 匹配 | 匹配 |
| `stages.0.mlp.experts` | 不匹配 | 匹配 |
| `stages.0.main_proj` | 不匹配 | 匹配 |
| `stages.0.self_attn.wq_a` | 不匹配 | 匹配 |

xSGL 的 W4A8 实现把 `config_groups[*].targets` 作为量化白名单。
DSpark 专家未命中时走浮点专家路径，没有创建量化 scale 参数，并按完整输入
宽度加载 packed INT4 数据。日志先出现 DSpark `.scale` 为 unexpected
weight 的警告，随后在专家权重加载处失败：

```text
Target sizes: [2304, 5120]. Tensor sizes: [2304, 2560]
```

两个 checkpoint 中 `layers.0.ffn.experts.0.w1.weight` 的实际存储形状
均为 `[2304, 2560]`、dtype 均为 INT8。相邻两个有符号 INT4 数存入一个
INT8 字节，最后一维减半符合预期，不能把这个存储形状差异视作权重损坏。
本次检查仅比较了选定 tensor 的头部信息，没有证明两个 checkpoint 的数值相同。

产物比对照少 45 个 scale 名称：43 个 `attn.wo_a.scale` 和两个
`engram.wkv.scale`。后续直接对比源 checkpoint 和量化 shard header，
确认这 45 个 scale 对应的权重均从 FP8 保存为 BF16，形状与源权重一致。
另观察到所检查的 PTQ scale 在产物中为 BF16、在对照中为 FP32；
本次没有改变 scale 精度策略。

对实际产物完成了全量名称检查：48,496 个非 scale tensor 的名称集合与源
checkpoint 完全一致；53 个输出 shard 共包含 96,040 个 tensor，全部 header
与输出 index 一致，没有缺失 shard、重复 tensor、索引错配或未登记 tensor。

`mtp.0.ffn.gate.bias_vl` 在源 checkpoint、量化产物和可用 INT4 对照中的
名称一致，均为 FP32、形状 `[128]`。读取实际 payload 后，量化产物的
512 字节与源 checkpoint 完全一致，可用 INT4 对照的该 tensor 也与源一致。
本地 xSGL 的 DSpark mapper 会将它转换为
`stages.0.mlp.gate.e_score_correction_bias_vl`。当前本地 DSpark stage 配置将
`vision_n_layers` 置为 0，导致 Gate 不注册这个目标参数；加载该 tensor 时
会打印 `unexpected weight` 并跳过。这项参数注册检查基于本地 xSGL，
不能据此认定远端部署代码版本相同，也不能将这条 warning 单独视为启动失败原因。

## 已完成的代码修改

1. 仅 V4.1 materializer 的 raw 导出同时转换 `config.json` 中的
   `config_groups[*].targets` 和 `ignore`，使配置与 tensor 使用相同命名
   空间。撤回此前错误放在 V4 基类中的转换，`deepseek_v4.py` 已恢复到当前
   HEAD 的内容；当前问题不能推广为 V4 也存在同一问题。映射保留正则锚点、
   类名 target、原始源量化配置，不改写正则内部任意 `lm_head` 子串，也不修改
   运行中 recipe 的对象。
2. V4.1 示例增加 `--save-raw-checkpoint-format` /
   `--no-save-raw-checkpoint-format`，默认开启，显式传给 materializer。
3. 通用 streaming 推理利用 materializer 已有的模块名称映射匹配量化配置。
   它同时接受逻辑名称和导出名称，没有加入 DSV4.1 专用分支，仍可读取旧产物。
4. 扩展现有缩小模型测试，覆盖 raw/native 导出、三种量化模式、DSpark、
   vision/aligner 非量化权重、Engram 原样保留以及回载。
5. tensor 保存继续继承 V4 的 `output_tensor_name()` → `_logical_to_raw()`。
   不新增源名称缓存、tensor 命名 override 或另一条保存路径。

## tensor 与 index 的一致性依据

此前 `_output_config_with_raw_names` 只转换配置，不能作为 tensor 命名正确的证明。
现已删除该额外方法，仅在 V4.1 的 `transform_output_config` 中处理配置。
命名链路为：

```text
源 index + 源 shard header
  -> 逻辑 tensor 名称
  -> 继承 V4 的 output_tensor_name/prepare_output_tensors
  -> _logical_to_raw
  -> 保存 shard
  -> finalize 读取实际 shard header
  -> 生成输出 index 的 weight_map
```

2026-10-20 本地原始
`/Users/ang/models/DeepSeek-V4.1-Flash/model.safetensors.index.json` 的
96,085 个名称经 raw → logical → raw 往返检查，96,085 个完全相同，
没有根级 `model.` 前缀。此项只验证本地索引的名称，不表示检查了未在本地
提供的全部真实权重 payload。

缩小模型端到端测试改为独立核对：

- raw 模式输出的非 scale 名称集合，直接等于源 index 的非 scale 名称集合，
  预期值不再调用待测的 `output_tensor_name`；
- 输出所有 shard 的 header 名称和所属文件构成的映射，严格等于输出
  `model.safetensors.index.json` 的 `weight_map`；
- 对物理输出模块名直接匹配保存的量化配置，所有 INT8/packed-INT4 模块均
  只命中一个 scheme，且对应的 scale 在输出 index 中；
- V4 配置选择器保持原行为；V4.1 native 导出保持 native 命名。

源 FP8/FP4 scale 会被目标 PTQ scale 替换，未量化模块的源 scale 也可能因
反量化而移除，所以不能要求输出与源 index 的全部 key、shape、dtype 或
`metadata.total_size` 相同。这里验收的是原有参数名称和模块前缀，以及输出
index 与实际 shard 的一致性。

核心修改文件：

- `examples/streaming_oneshot/deepseek_v41_wNa8.py`
- `examples/streaming_oneshot/README_deepseek_v41.md`
- `src/llmcompressor/streaming/materialization/deepseek_v41.py`
- `src/llmcompressor/streaming/inference/quantization.py`
- `src/llmcompressor/streaming/inference/api.py`
- `tests/llmcompressor/streaming/test_materialization.py`
- `tests/llmcompressor/modeling/test_deepseek_v41.py`

## 验证与交接状态

本地执行：

```bash
.venv/bin/pytest -q \
  tests/llmcompressor/streaming/test_materialization.py \
  tests/llmcompressor/streaming/test_pretrained.py \
  tests/llmcompressor/streaming/test_finalize.py \
  tests/llmcompressor/streaming/inference/test_inference.py \
  tests/llmcompressor/modeling/test_deepseek_v41.py \
  tests/examples/test_deepseek_v4_wna8.py
```

本次简化后结果：**135 passed，1 skipped**；跳过项需要 CUDA。
修改文件的 Ruff 检查及
`git diff --check` 通过。

提交前再次执行上述回归，结果相同。另执行 streaming calibration 输出复用的
定向回归，结果为 **3 passed，2 skipped**，跳过项需要 CUDA；设置
`DEEPSEEK_V41_REFERENCE_DIR=/Users/ang/models/DeepSeek-V4.1-Flash/inference`
后执行新增的 MoE 官方实现对照，普通路由和全专家校准两种模式均通过，
结果为 **2 passed**。该对照不代表完整模型或 GPU kernel 对齐验证通过。

本次没有修改远端已有产物、推理服务或量化进程。容器中执行了名称匹配、
实际产物的全量 index/header 检查和选定 tensor 的 payload 比较，
没有重新启动完整模型，因而尚不能宣布实际服务启动已通过。
本记录随导出修复及相关回归测试提交到上述功能分支，尚未推送。

MoE 性能、既有主干第 3 层参考数值分歧和量化 cache/recompute 精度问题，
仍是独立事项；本次导出修复没有验收这些问题。`observers/imatrix.py` 未修改。
