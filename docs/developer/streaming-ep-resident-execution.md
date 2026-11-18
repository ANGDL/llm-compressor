# Streaming EP：从路由统计到常驻窗口执行

本文描述实际 streaming pipeline 的执行路径。模型输入、校准样本和完整
sequence 传播语义保持不变。调度层根据真实专家输入安排计算，不改变 gate
选择的专家、路由权重、token 顺序或最终合并顺序。

## 1. 显存准入与执行读取使用相同口径

`BoundaryActivationStore.execution_metadata()` 返回：

- 实际执行会搬运的字段，包括与输入共享 tensor 对象的关联别名；
- 保持 tensor 身份关系的 meta 描述；
- 按 tensor 身份去重后的实际输入字节数。

`get_for_execution()` 与上述接口复用相同的字段选择逻辑。留在 CPU 的 carry
不计入执行设备的输入显存，同一个 tensor 的多个名字只计一次。磁盘准入仍
统计完整边界，因为持久化需要保留 carry。

内存边界缓存别名闭包；替换、提交或删除 batch 时使缓存失效。元数据读取
只构造 meta tensor，不克隆 CPU 激活。非内存存储可使用基类接口，其估算
与自身的执行读取保持一致。

## 2. 独立专家区域的识别

`parallel/region.py` 从现有 FX 子图识别可独立执行的专家区域，不要求模型
实现 `configure_streaming_ep()`，也不匹配某个模型的类名或层名。

区域可以跨过 trace 时固定的 target group 边界，但不能跨专家 collection。
下列条件必须成立：

1. 区域内的模块调用都是已发现的专家；
2. 每个专家在一个校准 boundary 中只调用一次；
3. 专家输入不依赖另一个专家的输出；
4. 专家输入的依赖切片不包含原地修改或其他副作用；
5. 路由输入来自已提交边界或可读取的常量，不需要额外加载模型权重；
6. 专家 tensor 参数共享二维输入的首维 token 轴，输出保持这个 token 轴。

含 attention、gate 权重计算、shared expert 或其他非独立模块的子图继续走
原来的 sequential 路径。这些子图发布路由前沿后，后续独立专家可形成一个
较大的区域。调度不假设所有模型的 router 采用相同 top-k 实现。

结构不满足独立执行条件时，保留原顺序。已选区域若不能从真实离散路由值
推导输入形状，应在加载专家权重前明确报错，避免使用猜测的路由负载。

## 3. 真实路由统计

`profile_expert_region()` 遍历当前区域的全部校准边界：

- 离散路由索引、offset、整数掩码保留真实 CPU 值；
- 浮点激活替换成 meta tensor；
- 只执行专家输入的无副作用依赖切片；
- 不执行专家，不生成 Hessian，不使用随机或均匀路由估计。

得到每个专家的总 routed token 数，以及每个原始 boundary 的 token 数。
空专家也保留在统计中，后续校准需要保持原始调用和 sample count 语义。

这条路径适用于显式 FX 路由和 fused/native 模型已经暴露的专家叶子。
无法暴露独立输入的 fused MoE 需要 adapter，不能通过更改 gate 语义绕过。

## 4. 放置与窗口显存模型

对专家 `e`：

```text
resident_bytes(e) =
    materialized_weight_bytes(e)
  + persistent_modifier_state(e)
  + conservative_update_and_observer_workspace(e)
```

专家按真实 routed work 进行稳定 greedy placement。每个 owner 随后调用
`plan_resident_windows()`，在独立设备的实际可用显存中生成常驻窗口。
当前实现保守地为同一窗口的专家累加各自 workspace，避免依赖 modifier
内部未知的 scratch 复用。

窗口的 token capacity 使用：

```text
capacity(owner, window) =
    floor(
      (usable_device_bytes - resident_bytes(window))
      / conservative_runtime_bytes_per_token
    )
```

显式 token 上限进一步约束 capacity。能容纳的专家保持常驻；只有权重和
modifier 状态装不下时才切换专家窗口。每个 rank 的容量可不同。

初始线性容量还需要通过 modifier 的实际 `estimate_batch_memory` 复核。
调度使用单调准入搜索选取可容纳的最大 token 数，额外 dispatch workspace
同时计入最终设备预留。因此新 modifier 可以声明非线性 workspace 估算，
不能直接复用 GPTQ 的每 token 常数。每个窗口的 dispatch 数按各专家分别
向上取整后求和；空专家也需要一个保留 sample 语义的调用。

窗口包含的总 routed token 超过 capacity 时，将同一专家跨原始 boundary
的输入合并成尽可能少的调用。单个原始调用仍超过 capacity 时按 token
切片。不同权重的专家保持独立 GEMM，不将它们当成共享权重的大矩阵。

## 5. 各 owner 独立推进

每个 owner 的执行顺序为：

```text
CPU prepare(window)
  -> admit + H2D + install(window)
  -> collect statistics from all calibration boundaries
  -> owner-local modifier update
  -> propagate with updated weights
  -> save quantized weights
  -> release hooks, statistics and device weights
  -> next window
```

`AsyncResidentWindowScheduler` 为每个 owner 建立独立流水。当前窗口计算时，
最多预取下一个窗口的 CPU 权重；不提前占用第二份设备常驻空间。一个 owner
不等待其他 owner 完成同一编号的窗口。

初始化 observer hooks 时使用短暂的 host 锁，保护 modifier 共用的 hook
集合。保存 shard 时序列化 writer 元数据操作。这些锁不包围专家计算或
Hessian 分解，不构成每窗口的设备全局 barrier。

预取结果在失败和取消路径中也必须释放。已经开始的 owner 清理自己的状态，
尚未开始的窗口停止推进。

## 6. 校准统计与原始 sample count

合并多个二维专家输入后，一次 forward 对应多个原始校准调用。拆分一个
原始调用时，多个 forward 又只对应一个原始 sample。

`calibration_batch_context(..., sample_count=...)` 显式保留这个关系：

- 合并调用贡献其中原始 sample 的总数；
- 一个原始调用的第一段贡献一次；
- 后续 token 段贡献零次；
- 空输入调用同样计入原始 sample 数。

Hessian 仍由 modifier 的原始 hook 累积。调度层只修正 sample counter，
不自行实现 GPTQ 统计公式。

传播期间使用 `HooksMixin.disable_hooks_local()`，只关闭当前执行上下文
的 hook。其他 owner 可能仍在校准，其统计收集必须继续。全局 hook 开关
不能用于这种异步窗口流水。

## 7. 输出合并与精度

各 owner 的传播结果以原始 boundary、专家和 token offset 标识，暂存 CPU。
路由边界读取、专家输入 gather、输入拼接、传播结果回传和结果拼接均在分配
前申请 host memory reservation，分配后转成 committed 记录，避免把实际
分配同时作为 promised capacity 重复计费。

最后的 combine FX 子图将专家调用替换为结果输入，并保留原图的合并操作和
原地写入顺序。合并在主执行设备上完成，保持 FP32 accumulator、路由权重
位置以及模型原有的 dtype 转换。

区域结束时需要等待结果可用，这是数据依赖。不同 owner 的窗口内部没有
对应窗口间的全局同步。当前 combine 仍由主设备执行，其性能应独立计量。
combine 搬运前单独检查输入与专家结果的设备容量，原始 carry 继续留在 CPU。

## 8. Modifier 扩展契约

窗口路径按能力选择 adapter，不根据 modifier 名字推断资源或可并行性。
adapter 必须显式提供：

- `supports_independent_expert_windows = True`；
- `expert_window(model, modules)`，管理当前窗口的 observer 和 hooks；
- `update_owner(device, modules)`，只更新该 owner 的统计和权重；
- 可验证的 held、update、observer 与 token workspace 估算。

共享 bookkeeping 可使用 `begin_owner_update()` 和
`finalize_owner_update()` 在区域边界整理。窗口不能提前释放另一个
modifier 尚需依赖的状态。

当前 GPTQ adapter 实现这套契约。其他 modifier 只有在声明并验证同等语义
后才进入该路径；没有契约时继续使用现有 pipeline。当前优化路径用于单进程
EP、直接写 checkpoint。事务恢复模式保持原有路径，不能把新的动态窗口
计划静默用于旧的恢复记录。

GPTQ 当前只为 weight-only 或逐 token 动态激活量化开启调用合并。静态激活
observer 或 local-dynamic observer 可能依赖原始调用边界，继续走原路径，
避免通过调度改变它们的统计口径。

## 9. 设备原生数值算子

主仓库通过 `llmcompressor.ops` 接口选择外部预编译 provider：

| 算子契约 | 目的 |
| --- | --- |
| `bmm_fp32` | 保持 BF16 输入值和 FP32 GEMM 累积，避免运行时隐式 CPU 回退 |
| `gptq_factorize_hessian` | 保持 dead-column、dampening、RTN fallback 和上三角 inverse-Hessian factor 语义 |

GPTQ 的分发桥放在 `streaming/numerics.py`，通过带引用计数的执行上下文
接入原有函数引用。退出上下文后恢复引用，不修改 GPTQ 源目录。

Kunlunxin provider 使用 FP32 diagonal-tile Cholesky、原生 triangular-solve
tile 和设备 GEMM。大三角系统通过分块前代求解，块间更新仍在设备上完成，
不依赖 CUDA BLAS handle。大矩阵不会传到 CPU 分解。factorization 以最多四个
矩阵为内部工作片，额外 scratch 计入 streaming admission。

精度验收必须包含 factor 与 FP32 reference 的对照、非正定矩阵的 RTN
标记、实际量化权重和 qparams、传播输出以及最终模型质量。FP32 算法在不同
实现之间不承诺逐位相同。算子 profile 必须检查 CPU fallback 事件，不能把
返回的设备 tensor 或线程并行误当成设备计算已生效。

## 10. 验证与记录

本地单测覆盖路由统计、跨 trace group 合并、空专家、token 切片、
Hessian/sample counter、权重/qparams、传播合并和上下文隔离。

设备测试使用显式选定的物理设备，先验证 tile、triangular solve、
factorization 和 GEMM，再验证完整 streaming pipeline 的短例。通过算子
和短 pipeline 验证后才扩大到完整量化。

运行编号、模型路径、profile、耗时实测和其他实验记录保留在私有目录，
不作为设计文档或代码提交内容。
