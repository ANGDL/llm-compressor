# 事务化 Streaming PTQ 实施计划

日期：2026-11-05。

状态：事务架构主体待实施，成功后工作目录清理已单独落实；
EP（Expert Parallelism，专家并行）已纳入前期架构与分阶段交付，运行时尚待实现。
本计划基于本地 `8bca0747a` 和当前工作区的审查，
下述未注明完成的能力不代表已经实现或通过测试。实施前应记录新的实际代码基线。

本计划中的 Streaming PTQ 包含 RTN、iMatrix、GPTQ、AWQ、SmoothQuant，
以及后续具有明确执行契约的其他方法。GPTQ 是其中一条算法扩展线。

## 1. 目标与交付边界

将现有逐子图量化流水线完善为资源受控、可验证恢复、能够扩展算法的执行系统。
核心交付是：

1. 进程在关键位置被强制终止后，可从最后一个完整事务继续；
2. 恢复与不中断执行保持相同的量化、传播和输出语义；
3. 正常执行、恢复、校验、发布均遵守内存和 I/O 预算；
4. 历史 activation 可安全回收，不随已完成模型深度长期累积；
5. 用能力协议接入 modifier，避免为每种算法复制整个 pipeline；
6. 在这些基础上扩展 GPTQ 统计管理、MoE 专家执行和多模态工作集；
7. 同一执行协议覆盖单设备与 EP，显式管理专家归属、通信、全局提交和恢复。

首个运行实现采用单进程、单执行设备、本地持久文件系统；
但从 M0 起定义并行拓扑、参数所有权和事务参与者，单设备作为 EP size=1 的适配。
单设备仍使用单个 work directory 的单写者；EP 扩展为一个全局协调者和
互不重叠的 rank 写入域，不允许多个 rank 竞争同一个事务文件。
首个 EP 实现限定单机多设备、固定拓扑、各进程可见的同一本地持久文件系统，
每个进程绑定一个执行设备。多机、独立本地盘、弹性重分片另行设计。
CPU 作为小模型正确性基线，目标加速设备承担资源和性能验收。
默认保持现有普通运行模式；恢复能力继续显式启用。

设备 API 兼容性和具体 kernel 可用性分别记录。已有 Triton/eager 保护与
预编译硬件算子分发策略保持原有含义，不通过切换后端偷偷改变数值算法。

### 1.1 三种变更类别

| 类别 | 示例 | 验收方式 |
| --- | --- | --- |
| 保持算法语义 | 逐 batch 恢复、历史扫描去重、boundary 回收、减少权重复制 | 同配置下与当前基线比较 tensor、qparams 和输出 |
| 改变数值执行顺序 | Hessian 累积重排、并行归约、专家调度顺序 | 预先声明容差，验证统计、权重和端到端质量 |
| 改变算法或校准语义 | true sequential、低秩 Hessian、token 抽样、路由加权 | 独立实验模式、独立指纹、独立质量对照 |

不得把第三类能力以“纯性能优化”名义混入默认路径。

### 1.2 文档关系

- [Streaming PTQ 使用和当前架构](../guides/streaming_ptq.md)
  继续描述实际实现与用户接口。
- [原逐 Subgraph 设计](streaming-ptq-coding-plan.md)
  保留作为设计历史；其中“目标能力”、历史测试数字和传播描述不能代替当前实现。
- 本文定义后续实施顺序、协议约束、实验扩展和验收门槛。

尤其需要区分：当前传播关闭 fake quantization，沿用 Sequential Pipeline；
GPTQ 可传播其原地更新的权重，统计型 RTN recipe 可复用校准输出。
第一阶段保持这些行为，不能把所有模式描述为“完整 W/A QDQ 传播”。

## 2. 当前基础与差距

下表是静态代码观察，既有测试的存在不代表本次重新执行通过。

| 领域 | 可复用实现 | 待解决问题 |
| --- | --- | --- |
| 主循环 | `streaming/pipeline.py` | 正常与恢复分支的资源和状态契约不完全一致 |
| 权重物化 | `loading/session.py`、`target.py`、`materialization/*` | 设备侧算法统计与 workspace 还需统一 admission |
| 事务 | `checkpoint/writer.py` | boundary 缺少与权重同等级的摘要、结构和持久化验证 |
| 恢复 | 已完成子图跳过与 boundary 恢复 | 历史事务反复全量校验，boundary 整体装入列表 |
| activation | 内存/磁盘两种 store | 历史事务保留 boundary 副本；回收需要新协议 |
| 输出 | 普通模式直接 shard，恢复模式 payload 后组装 | 恢复路径重复序列化、写入及组装成本 |
| 发布 | `finalize.py`、`finalize_only` | 删除 state、替换旧输出等位置的崩溃窗口需覆盖 |
| 身份 | manifest、recipe/source fingerprint | 实际 batch、图计划、传播语义、实现版本需更明确 |
| modifier | layerwise lifecycle、输出复用声明 | 当前 pretrained 入口仍有类型支持列表 |
| GPTQ | 现有 Hessian、分解、batched quantization、fallback | 子图内多个 Hessian 和量化副本仍可能形成峰值 |
| 并行 | GPTQ 已有多 rank 分配、统计 reduce 与结果 broadcast | 不能直接用于专家分片，需要显式 group、所有权和全局事务 |
| 测试 | writer、manifest、等价性及 RSS probe | 缺少完整进程强杀矩阵、深度扩展及恢复资源曲线 |

当前 `modifiers/gptq/base.py` 的分布式路径按默认进程组分配量化任务，
归约 Hessian 后将结果广播给各 rank；这不构成 Streaming PTQ 的 EP 支持证明。
工作区未跟踪的 `modeling/step5/native_model.py` 已有专家维分片及输出归约代码，
属于模型侧可审查的候选适配，不能据此认定 streaming 的加载、统计、恢复和导出
已经支持 EP。实施前分别记录模型侧和流水线侧的验证状态。

当前低层 boundary API 仍具有 collect/quantize/finalize 三阶段语义。
首期将其与 pretrained 子图闭环分别标记和测试，不自动改变已有调用者的行为。

## 3. 执行与恢复协议

### 3.1 子图事务内容

事务 `T_i` 表示当前子图的一次完整结果：

```text
T_i
  执行身份：run、子图、前驱事务、输入 boundary 摘要、传播模式
  模型修改：压缩权重、qparams、norm 等修改、tied-weight 元数据
  下一状态：后续仍使用的具名 activation 与辅助值
  可恢复状态：modifier 状态或可确定重建的声明，必要时包含 RNG
  完整性信息：对象名称、类型、shape、大小、摘要、batch 数和顺序
```

boundary 是图执行的 live state，可能包含跨多个子图使用的值，
不等价于单个 hidden tensor。对象别名和原有 dtype 必须正确表达。

共享参数与跨子图修改必须具有唯一所有者和显式依赖。不能把“已经写出”
误认为“后续不会再次修改”。不能证明修改范围时在 preflight 拒绝该 recipe，
或者在后续阶段合并为更大的执行/事务单元。

### 3.2 首个子图前的输入冻结

在第一个事务提交前，流式固化实际校准 batch 的身份：

- token、mask、position 和多模态输入摘要；
- batch 顺序、有效长度、实际样本数；
- tokenizer/processor、chat template、预处理参数和随机配置；
- trace 计划及输入/输出名称、消费关系、工作集和模型代码实现身份。

对于生成器或随机变换，必须固化可重放输入或明确拒绝恢复。
首期允许保存 tokenized batches 或实际首 boundary，逐 batch 计算摘要，
不能为了 hash 把整个数据集装入内存。初始化未完成时重建整个输入快照。

计划在完整输入摘要冻结后开始子图事务，避免同一 run 的身份在执行中变化。
首期恢复模式优先要求稳定的参数和依赖环境；版本不一致默认拒绝，
后续才通过明确的兼容规则允许迁移。

### 3.3 提交顺序

```text
准备当前子图输入和权重
  -> 校准与 modifier 更新
  -> 按声明策略生成下一 boundary
  -> 写事务对象到 pending
  -> 完整性检查及必要同步
  -> 写入并同步事务 manifest
  -> 发布 committed 事务并同步父目录
  -> 更新可重建的恢复位置缓存
  -> 生成可持久验证的回收授权
  -> 删除过期临时对象
```

提交前，事务引用的数据必须完整。计算中的对象不可被 writer 后台线程继续读取；
writer 只拥有不可变快照或明确转移所有权的 CPU 数据。

第一期承诺进程异常/强杀恢复。文件与目录同步也按持久化目标实现，
但主机掉电保证必须在指定文件系统、挂载参数和存储栈上单独验证；
网络文件系统和对象存储不沿用本地 rename 假设。

### 3.4 恢复规则

1. 获取 work directory 的操作系统级单写者锁，防止两个恢复进程共同写入。
2. 校验运行身份和计划。
3. 扫描一次 committed 元数据，确定从首子图开始的连续提交链；洞和分支报错。
4. frontier 索引只是缓存，丢失时可根据事务和回收记录重建。
5. 按校验策略检查历史模型产物，禁止每跳过一个子图就重新检查全历史。
6. 只加载最后连续提交点对应的有效 boundary，逐 batch 读取。
7. 清理当前 run 的未提交对象，重新执行当前未提交子图。
8. 已提交数据损坏时报告对象和事务，不自动覆盖；需要回滚时另建恢复分支。

“逻辑结果接受一次”允许未提交子图重算，不承诺底层计算只执行一次。
首期未提交 GPTQ 子图重新校准，不默认保存全模型 Hessian。

### 3.5 boundary 回收

保留策略以当前有效 frontier 和仍在使用的快照为根。
首期建议保留最近两个可恢复 boundary，具体数量是存储策略，不参与算法选择。
最早的输入快照若配置保留，应单独计入资源预算。

GC 协议必须解决以下问题：

- 只有 successor 事务已完整提交，旧 boundary 才可能失效；
- 先持久化 retirement/compaction 记录，再删除对象，崩溃后可补做删除；
- 原事务 manifest 保持不可变，历史引用通过回收记录解释；
- “合法回收”与“文件意外缺失”在校验中是不同状态；
- 不删除未来图节点仍需要的 live value，复用对象采用引用关系回收；
- 仍保留旧 boundary 不等于可以安全回滚模型修改，回滚能力单独设计；
- 输出权重一直保留到发布，不受 activation GC 影响。

### 3.6 EP 的执行拓扑与所有权

EP 将不同专家放到不同 rank；每个 rank 只加载自己负责的专家权重，
按真实路由执行后合并输出。它与单设备轮换专家组、将独立量化任务分给多卡
是三种不同能力，支持矩阵必须分别列出。

从单设备阶段就定义以下对象；名称为设计草案：

| 契约 | 必须表达的内容 |
| --- | --- |
| `ParallelContext` | rank/device、EP group、数据副本 group、控制组、协调者及通信适配器 |
| `PlacementPlan` | 全局模块/专家 ID、owner、复制或分片方式、参数与 qparams 的逻辑名称 |
| `StatisticsScope` | 哪些 rank 具有同一专家的独立样本，哪些只是同一输入的副本 |
| `BoundaryLayout` | batch/token 身份、复制或分片布局、持久化 owner、重放所需辅助值 |
| `CommitParticipants` | 当前事务必须参与的 rank、各自产物责任和零输出参与者 |

首个 EP 版本设 DP=TP=PP=1：dense、router 和 shared expert 在 EP rank 上复制，
路由专家按专家维切分，每个专家保持完整投影维度；分片映射必须显式记录。
模块所有权不靠 `rank * experts_per_rank` 隐含推导，支持专家数不能整除 EP size；
零专家或零 token 的 rank 也必须按计划参加控制与通信。
后续 EP×DP、EP×TP 使用不同 group/分片协议，不能将默认 world group 当作通用组。
现有 Step5 的 shared expert TP 分片与这一首期策略不同，需要独立适配与验收。

权重加载按逻辑专家切片读取，避免先把整个 MoE target 装到每张卡再切分。
若源格式为多个专家融合的 tensor，loader 要支持按专家维读取和恢复逻辑命名，
或在 preflight 明确给出受预算约束的转换方案。EP rank 内仍可叠加 E3 专家组
流式执行，两者共享专家计划，但独立控制设备驻留上限。

校准与通信分两步实现：

1. **复制输入的参考 EP**：各 rank 看到同一 batch、token 和路由，
   仅计算本地专家贡献，然后在 EP group 合并；用于先验证分片、统计和事务。
   输出合并的精度、top-k slot 顺序和容差必须声明，此模式仍有输入复制成本。
2. **token dispatch EP**：输入按声明的 token 布局分布，
   将 token 发往专家 owner，再按原 token/slot 合并返回；通信后端可用
   all-to-all 或其他等价实现，但执行器不绑定某一种通信库。

每条路由携带稳定的 batch/token/slot/expert 身份；padding token 排除规则、
top-k 多路分发、空 split、变长 batch、接收顺序和输出归位都必须验证。
首期不得为满足通信容量而丢 token、截断热专家或改变 router 选择；
超预算通过有界 microbatch、确定性分段或 preflight 拒绝处理。
通信 buffer、临时排序索引和在途请求计入每 rank 的 CPU/设备预算；
不能仅按专家权重均分估计 EP 内存，热专家可能成为单 rank 峰值。

统计必须绑定逻辑专家和输入集合：

- 本地专家只累计实际送达该专家的输入；down projection 使用对应 gated activation；
- 复制输入 EP 不将 dense/router 的重复样本按 world size 再计数；
- 专家统计只在持有同一逻辑专家且样本互补的数据副本间归约，
  首期 DP=1 时无需跨 EP group 归约不同专家的 Hessian；
- 后续数据副本归约需保留现有 H/count 的归一化语义，
  不直接求平均已归一化 H，也不把 token 数等同于当前 GPTQ 的样本计数；
- 量化结果保留在 owner，不向所有 EP rank 广播全部专家权重；
  复制模块由指定 owner 生成一致结果并仅在对应复制组同步；
- 自然路由中的冷专家与 `moe_calibrate_all_experts=True` 分别定义策略；
  无统计时明确沿用已有策略、显式 fallback 或拒绝，禁止伪造统计。

### 3.7 EP 全局事务、恢复与清理

事务单位首期仍为完整 decoder/subgraph，所有 rank 接受同一个全局 frontier。
rank-local prepared 是候选产物，不能让某个 rank 独自跳到下一子图。

```text
各 rank 完成本地计算、通信与持久写入
  -> 各自发布不可变 prepared manifest（允许零对象）
  -> 协调者验证预期参与者、对象摘要和下一 boundary 布局完整
  -> 协调者持久发布唯一 global commit
  -> 所有 rank 根据该 commit 进入下一子图
  -> 协调者发布全局 GC 授权
  -> 各 rank 只回收授权的本地对象
```

身份包含 run、attempt/generation、子图、前驱 global commit、
拓扑/placement 摘要、预期参与者、各 rank manifest 摘要及 boundary 布局。
global commit 只引用已经完成同步且可恢复访问的不可变数据。
通信 barrier 不能替代持久 commit；协调者也不能只收到内存 ACK 就提交。
即使复制 boundary 只保存一份，也要标记唯一持久化 owner 和完整性信息，
恢复时显式分发，不默认各 rank 的列表可以互相替代。

任一 rank 或协调者失败后，由 supervisor/launcher 超时终止本次 worker group，
确认旧进程退出、释放锁后以新 attempt 全组重启，读取最后一个 global commit。
不在受损通信组里依赖新 barrier 或捕获异常继续；不首期实现在线协调者选举。
首个版本保持 EP size、placement、数据布局和关键后端配置不变，
拓扑改变时拒绝原地恢复；更换 EP size 的已完成模型加载与未完成 run 恢复分开设计。
这是一套固定故障边界下的协调提交协议，不宣称提供通用分布式共识。
EP-B 首次交付要求 `checkpoint_progress=True`，以完整验证持久事务；
后续若增加无持久恢复的 EP 模式，单独定义其跨 rank 完成与失败语义，
不能把仅在内存中的协调状态称为可恢复提交。现有单设备普通模式继续保持原接口。

全局恢复锁由协调者持有，并覆盖整个 worker group 生命周期。
协调者退出但其他 rank 尚存活时，新 attempt 必须经 supervisor 确认旧组已终止；
generation 校验与独立写入域共同防止旧 worker 的迟到写入被新 attempt 接受。
故障上报、超时和取消采用独立控制路径；无法安全取消的在途设备通信随进程退出。

每 rank 直接写最终可复用的权重 chunk，协调者只汇总 metadata/index，
不把所有专家权重或 Hessian 收到 rank 0。最终 checkpoint 使用全局逻辑名称，
复制参数只写一份，融合专家张量通过显式输出适配满足目标格式；
无法有界拼接的格式在 preflight 拒绝，不用全模型 gather 补救。
首批验收包括标准加载器重载和改变部署 EP size 后的加载验证，
不要求所有输出格式都天然支持任意分片。

成功清理须等待全局输出发布完成、全部 writer/通信结束并释放文件引用，
再由协调者授权清理框架拥有的各 rank 临时目录，最后删除空 work 根目录。
单个 rank 完成不能触发整个 work 根目录清理；失败保留全局恢复链。
发布已成功但清理失败应可根据发布记录幂等补做，非框架文件继续保留。
新增 rank 数据应位于框架明确拥有的目录中，并扩展现有清理函数的覆盖测试。

### 3.8 EP 与底层分布式实现的边界

执行协议不绑定某个 MoE 推理引擎；先实现可检查的参考执行器，再评估通信内核。
进程组创建顺序、collective 调用顺序及重启行为按固定依赖版本核验。
参考 [PyTorch 2.9 进程组文档](https://docs.pytorch.org/docs/2.9/distributed.html#groups)
和 [torchrun 故障模式](https://docs.pytorch.org/docs/2.9/elastic/run.html#failure-modes)；
引用用于约束适配设计，不代表项目将依赖版本固定为 2.9。
launcher 重启进程不等于恢复了 PTQ 事务，后者仍由 global commit 协议负责。

## 4. 配置与模块演进

以下名称是待评审的设计草案，不是现有可调用接口。
首期只接通必要字段，避免一次引入所有配置。

| 配置对象 | 候选内容 | 兼容要求 |
| --- | --- | --- |
| `RecoveryOptions` | 校验等级、保留数、停止策略、环境兼容规则 | 从 `checkpoint_progress=True` 映射到严格恢复 |
| `ResourceLimits` | CPU、设备、workspace、临时磁盘、队列字节预算 | 无配置时保持现有普通模式 |
| `ActivationStorageOptions` | memory/disk/auto、staging、后续 spill 策略 | 首期不自动切换存储模式 |
| `ExecutionSemantics` | sequential-compatible、实验性 QDQ/true-sequential 等 | 默认固定为现有兼容模式 |
| `ParallelOptions` | EP size、placement、输入/路由布局、通信后端、超时 | 默认 EP=1；不隐式接管已有默认进程组 |

CLI 先提供只读 `inspect` 与 `resume --dry-run` 设计：
显示身份、最后提交点、校验策略、将重做的子图和估计资源，不消耗量化设备计算。
EP 额外显示参与者、全局与各 rank 准备状态、拓扑兼容性和每 rank 预算，
不能把 rank-local prepared 显示成全局完成。
真实执行仍复用 `streaming_oneshot()`，不在 CLI 再实现一次主循环。

建议代码职责：

| 位置 | 职责 |
| --- | --- |
| `artifacts/manifest.py`、`validation.py` | schema v2、身份与兼容策略 |
| 新 `recovery.py` | 连续提交链、恢复计划、锁、停止与重放 |
| 新 `compaction.py` | boundary 引用与可持久回收记录 |
| `checkpoint/writer.py` | 对象持久化、commit 与输出编码 |
| `activations/store.py` | batch 读取、结构校验、字节记账 |
| `loading/host_memory.py` 及预算适配 | CPU/设备/磁盘 admission 与预估 |
| 新 `modifier_contract.py` | 工作集、修改范围、状态和传播能力 |
| 新 `parallel/context.py`、`placement.py` | 单设备适配、group/专家所有权和拓扑验证 |
| 新 `parallel/executor.py`、`transport.py` | EP 输入/专家调度、dispatch/combine 和有界通信 |
| 新 `checkpoint/coordinator.py` | local/global 提交适配、参与者、失败与清理授权 |
| `pipeline.py` | 唯一执行主循环，调用上述组件 |
| `finalize.py` | 完整性验证、发布状态机与重试 |

schema v2 不原地升级旧 run。旧格式可使用已有 reader 检查；
不满足新恢复契约的旧数据不能伪装成 v2 继续执行。
新 schema 在 commit、GC、发布和 EP 所有权协议冻结后再确定具体字段；
M1 即表达 participant/layout，即使实际只有一个 rank。

## 5. 实施里程碑

所有里程碑必须产出可单独审查的改动、测试和结构化结果。
下列测试文件名是建议；先复用现有测试夹具，再根据范围拆文件。

### M0：基线、执行语义和资源观测

依赖：无。预计 2–3 个工作日。

实施：

- 记录 revision、dirty patch 身份、依赖、硬件和数据 manifest；
- 区分 pretrained 闭环与旧 boundary 三阶段入口；
- 明确 RTN/iMatrix 输出复用、GPTQ 权重传播、AWQ/SmoothQuant 修改范围；
- 固定 source、实际 batches、recipe、dtype、后端和传播策略；
- 增加每子图与全进程的时间、RSS、设备 allocated/reserved、I/O 和临时磁盘报告；
- 将普通 streaming、当前恢复模式、sequential oneshot 纳入基线对照；
- 冻结单设备/EP 的逻辑专家 ID、placement、统计范围与事务参与者契约；
- 区分现有模型侧分片、分布式 GPTQ 调度与 streaming EP 的实际支持状态。

涉及：`pipeline.py`、`pretrained.py`、`artifacts/*`、
`tests/llmcompressor/streaming/resource_probe.py` 和 guide。

验收：

- tiny dense 的 iMatrix+RTN、GPTQ 对照；AWQ/SmoothQuant 使用明确数值容差；
- 在固定输入、CPU FP32 eager 的确定性基线要求逻辑 tensor 内容相同；
- 目标设备的容差单独预注册，不在失败后临时放宽；
- 资源报告区分 source、最终输出和临时数据，不把必要输出增长当作泄漏。

### M1：事务完整性与进程故障矩阵

依赖：M0。预计 4–6 个工作日。

实施：

- boundary 记录结构、大小、摘要、batch ID/顺序、预期数量和 alias 元数据；
- 对 payload、manifest、事务目录执行明确的提交/同步协议；
- 以单参与者协调器实现首版，保留 local prepared 与全局接受的语义边界；
- 加入单写者锁，异常退出后新进程可以获得锁；
- 用受测试控制的故障点，在数据写入、manifest、rename、索引更新和清理间强杀；
- 错误分类为未提交、身份不兼容、已提交损坏、资源不足，禁止混为普通重试；
- 在安全事务边界处理主动停止请求，已启动的错误设备 kernel 不自动重跑。

涉及：`checkpoint/writer.py`、`activations/store.py`、`artifacts/*`、
新 `recovery.py`、新 `test_transaction_recovery.py` 和子进程夹具。

验收：

- 每个关键故障点都由父进程等待明确握手后触发强杀，避免靠 sleep 猜位置；
- 用全新子进程恢复，与无中断执行的逻辑 tensor/qparams/输出比较；
- 覆盖“仍可反序列化但数值被修改”的 boundary 损坏；
- 覆盖 ENOSPC/短写等写入失败；日志明确指出事务与对象；
- 两个进程同时启动同一 work directory 时，第二个在写入前失败；
- 进程故障测试与掉电验证分别记录，不互相替代。

### M2：有界恢复与历史 boundary 回收

依赖：M1。预计 4–6 个工作日。

实施：

- 单次元数据扫描构建 `ResumePlan`，移除跳过子图时的全历史重复校验；
- 以 batch reader 代替 `committed_boundary()` 的整列表返回；
- 恢复只物化 frontier 所需数据，不重新写出所有历史 boundary；
- 实现有引用关系和 retirement 记录的 GC，保留最近必要恢复根；
- 全量历史 payload 校验最多按策略执行一次，并记录读字节量；
- 记录 boundary 恢复峰值、检查耗时和清理耗时。

涉及：新 `recovery.py`、`compaction.py`、`pipeline.py`、
`checkpoint/writer.py`、`activations/store.py` 和对应测试。

验收：

- 固定宽度下以 8/32/128 子图比较恢复扫描计数和读字节量；
- 不出现每个已完成子图重复读取全历史输出的行为；
- 恢复活跃 CPU batch 数有界；元数据规模允许随事务数线性增长；
- 临时 activation 峰值符合保留 frontier、共享 live state 和未提交快照的字节预算；
- 在授权回收前后、删除过程中、恢复索引丢失时均可正确重建；
- 损坏当前有效 boundary 必须失败，不能因存在 retirement 记录而被误判合法。

### M3：恢复模式输出与发布优化

依赖：M2。预计 4–6 个工作日。

实施：

- 以子图为单位直接产生不可变 safetensors chunk，事务引用 chunk 和 boundary；
- 在格式可表达时避免 per-tensor bin -> 重新组装 -> 再复制整个结果；
- chunk 可直接进入最终 index；重分片是可选后处理，不决定事务执行单位；
- 引入有字节上限的异步快照与写入，队列满时 backpressure；
- 事务提交等待其输出和 boundary 都已完整持久化，不能把“已入队”当成完成；
- writer 失败向主线程传播，释放预留和快照，保留可诊断状态；
- chunk 使用全局逻辑参数名和 owner，index 汇总接口不要求权重集中到协调者；
- 发布使用不可变 generation、持久发布记录和可恢复的目录切换状态；
- 替换已有输出时保存旧版本直到新版本完整可见，覆盖重命名间窗口。

涉及：`checkpoint/writer.py`、`output.py`、`finalize.py`、
`pretrained.py`、`test_checkpoint_writer.py`、`test_finalize.py`。

验收：

- 已完成 chunk 可按内容身份重用，重复 finalize 不再量化；
- 在 metadata 验证、最终 marker、旧输出移动、新输出安装之间注入失败；
- 原有输出与新输出至少有一份可恢复，正式路径的可见性契约明确记录；
- 普通模式、恢复同步模式、恢复异步模式输出协议一致；
- I/O 统计中压缩 payload 的正常写入次数可解释，边界与输出分别核算。

### M4：统一预算与 modifier 能力协议

依赖：M0、M2；可与 M3 部分并行开发。预计 4–6 个工作日。

实施：

- 在加载前预算当前权重、算法持有统计、量化 scratch、预取及写入快照；
- CPU、设备、临时磁盘分别 admission，不用 CPU 可用量替代设备统计估算；
- 防止 prefetch 和 writer 争抢容量；保留当前子图完成提交的最低余量；
- 将工作集、修改范围、统计生命周期、传播策略、恢复状态、输出所有权定义为协议；
- 加入参数 placement、统计同步范围、collective 责任和每 rank 通信预算；
- 先为 RTN/iMatrix/GPTQ/AWQ/SmoothQuant 实现窄适配；
- 不支持的能力在 preflight 解释原因，不在执行中悄悄换算法或关闭恢复；
- 持续记录估算与实测偏差，无法可靠估算的适配器使用明确保守上界或拒绝。

涉及：`loading/*`、`modifier_contract.py`、`pretrained.py`、
`pipeline.py`、各 modifier 的窄接口及资源测试。

验收：

- 资源不足在读取下一大对象前报告；执行预算不替代 allocator/OOM 实测；
- 不能依赖某个 modifier 的类名自动推断安全状态；
- 跨子图 norm/共享 scale 修改有明确所有者和恢复测试；
- 分布式 modifier 显式接收统计/复制 group，不默认沿用 world-group reduce/broadcast；
- 新 modifier 只增加适配和专项测试，不复制加载/事务/发布主循环。

### M5：Streaming PTQ 扩展验证

依赖：M1–M4；按第 6 节逐项选择，不整体打包。每项独立立项和验收。

首批选择：

1. GPTQ 精确统计与副本优化；
2. activation 存储与恢复频率的成本策略；
3. MoE 专家组执行可行性验证。
4. E9 的参考 EP 与全局事务验证，先 RTN/iMatrix，再接入 GPTQ。

EP 从 M0 纳入契约，M5/E9 交付真实运行；实现前不得将“预留接口”标为支持。
QAD、AutoRound、变换、KV cache 和其他并行组合分别经过能力评审后接入。
普通 pipeline 的算法行为不因 streaming 实验自动改变。

## 6. Streaming PTQ 延伸路线

### E1：GPTQ 精确统计与内存优化

优先级：P1，M4 后第一项。目标是降低同一子图内 Hessian 和 scratch 峰值。

Hessian（海森矩阵）是损失对参数的二阶导数矩阵。对 GPTQ 使用的局部线性
输出重建损失，其每个输出行的 Hessian 与输入二阶矩 `X @ X.T` 成比例，
可用校准输入累积，不需要训练整个模型。它表达输入通道的共同变化如何影响
权重量化误差，而不只是每个权重单独的重要性。输入维度为 `d` 时，
单个 FP32 Hessian 的存储为 `4 * d * d` 字节；当前实际归一化以实现为准。

当前入口已有 `estimate_sequential_memory()` 和 batched quantization。
继续优化应复用这些接口，不重新实现 GPTQ 量化算法。

实施顺序：

1. 审计 `prepare_batch()` 的 normalize/stack 和量化函数的 W/H clone；
2. 对已转移所有权的可变 workspace 才允许复用，保留原调用语义兼容入口；
3. 对输入确实相同的 Q/K/V、gate/up 等模块，评估共享不可变原始 Hessian；
4. 累积相同输入只执行一次，消费者使用私有分解/排列 workspace；
5. refcount 归零才释放统计；不要共享会被分解、加 damping 或排列修改的 H；
6. 若设备仍不足，评估 CPU Hessian 与分组重放，明确记录传输和重复 forward 成本。

共享条件包含输入路径、实际输入、mask、累积次数、计数归一化、dtype、
传播模式和采样集合。shape 相同、模块名相似、同层专家均不是共享证据。

当前 `accumulate_hessian()` 的计数与上游输入重塑有关，不能未经对照就改成
“统一 token 平均”。先冻结当前统计语义，再研究新的 normalization。
同样需要审计原地输入操作，避免共享 hook 后改变其他消费者看到的输入。

验收：单独比较原始 H/count、qparams、fallback、packed tensors 和 logits；
确定性条件下保持等价，降低峰值与耗时的实际收益由实验决定。
仅减小 GPTQ `block_size` 不等于移除了完整输入维度平方级 Hessian。

### E2：子图内分阶段校准与 true sequential

优先级：P2，独立实验语义。

当前“逐 decoder 子图”与“decoder 内按投影顺序重新校准”是不同执行策略。
可实验 attention 输入投影 -> 输出投影 -> MLP gate/up -> down 等分组，
但分组必须来自真实依赖，不能硬编码适用于所有模型的名称序列。

每组量化后重新计算后续组需要的输入，可以减少同时存活的统计，
也可能改变校准分布与模型质量。建立 group-plan、额外 forward 次数和
误差传播声明，并纳入指纹。

首期仍在整个 decoder 结束时提交事务；组内失败重做当前 decoder。
只有单个 decoder 重做成本已被证明显著时，再评估组级持久化。
验收将统计、峰值、计算放大和独立数据质量一起比较，不要求与旧模式逐值相同。

### E3：已有 MoE 支持之上的专家组流式执行

优先级：P1 可行性验证，完整实现晚于 M4。

当前已增加独立的 `streaming.moe.ExpertGroupExecutor` 执行层骨架：
它按有限专家组调用现有权重 session 加载和回收线性化专家，并提供
组驻留前后 callback，使 GPTQ 等 modifier 可以在不修改
`modifiers/gptq/*` 的前提下挂载自己的 collector/lifecycle。该骨架只证明
专家分组、路由合并和设备驻留边界，不代表 pretrained streaming pipeline
已经接入专家组模式，也不代表任何具体 MoE 模型已经获得完整支持。

通用模型接口已经预留为 `MoEModelAdapter`。当前实现仅提供
`StandardTransformersMoEAdapter`，约束为线性化后的 `experts` 集合和
`gate`/`router` 返回 `(logits, top_k_weights, top_k_indices)` 的标准
Transformers 形态。DSV4.1、Kimi、GPT-OSS 等具有特殊路由或融合权重布局的
模型不进入本阶段；它们未来通过独立 adapter 注册，不修改通用 executor。

当前项目已经支持已适配 MoE 模型的校准、量化和 streaming 执行；
这里增加的是更细的设备驻留粒度，不是首次接入 MoE 模型。
目标是处理“单个 MoE decoder 的全部专家放不进设备”的情况。
这将扩展当前全 target 驻留契约，必须是显式 opt-in 的执行器，
不能用普通子图预取能力宣称已经支持。

实施：

- 先支持一种 tiny MoE 与一种已注册实际模型，固定路由和校准策略；
- 保存 router 输出、token-to-expert 映射和稳定 batch/token 身份；
- 只物化有限专家组，收集各专家真实输入统计并分别量化；
- down projection 统计使用对应专家实际 gated activation；
- 后续传播按既定语义使用原始/修改后的专家权重，稳定合并输出；
- expert 分组改变执行顺序时记录数值容差；
- 保持每个专家的统计独立，不因 router 热度或输入宽度相同就合并 H；
- 区分自然路由与 `moe_calibrate_all_experts=True`，不得静默切换。

首期保持 decoder 级事务。专家级 child transaction 需要记录 attention/dense
修改、已完成专家修改和聚合状态，作为后续独立协议设计。
专家组执行器使用 §3.6 的同一逻辑专家与 placement 接口；
单设备分组与 EP 可以分别交付，后续在每个 EP rank 内组合使用。

验收：与全专家驻留参考比较统计/路由/输出，专家组内存有界；
记录额外重放、磁盘和时间开销，确认单 target 原本不可执行的案例能够完成。

### E4：activation 分层存储与检查点频率

优先级：P1/P2，先实现可解释策略。

将“activation 放哪里”与“是否保存恢复点”逐步解耦：

- CPU memory：适合 boundary 较小、内存充足；
- local disk：适合较大校准集和长序列；
- mixed/spill：在显式预算下逐 batch 迁移，不能整 boundary 突然复制；
- pinned staging：只 pin 有界批次；异步 H2D 需明确增加的设备 batch 数量。

初期恢复模式继续每子图提交。稀疏恢复点是后续独立模式：
只持久化部分 activation frontier，故障后从最近可恢复点重放中间子图，
并根据状态契约重建 modifier/RNG/工作权重，验证与已接受模型产物一致。
不能直接使用最终量化权重代替重放时所需原始权重。

先用固定间隔和预算规则对比保存成本、重算窗口、恢复时间，再考虑自适应；
不存在无成本的高频恢复点，也不默认保存全部历史 boundary。

### E5：QAD、AutoRound 与变换适配

优先级：P2，分别实施。

| 方法 | 必须补齐的契约 | 首个最小范围 |
| --- | --- | --- |
| QAD | teacher/cache、train/validation 切分、RNG、优化器与浮点可训练权重生命周期 | 一个子图训练完成后提交；中途重做整个子图 |
| AutoRound | 外部算法的量化/保存所有权、qparams 导出、hook 和工作集限制 | 通过 output adapter 接入；不能假设其已有保存可直接事务化 |
| SpinQuant/QuIP | 离线修改、在线 transform 元数据、共享/tied tensor 与 norm 依赖 | 先支持已证明局部可表达的变换，明确后续 runtime 需求 |
| AutoSmooth | 搜索输入、输出修改、norm/scale 状态、统计重放 | 单个局部 mapping，验证跨边界修改范围 |

优先减少算法适配器的状态，再考虑把优化器内部步骤持久化。
新增 recipe 只有通过工作集、恢复和输出格式验收后才能进入支持矩阵。
组合 modifier 要验证顺序和所有权，不以“各自支持”推导“任意组合支持”。

### E6：多模态、MTP、跨层辅助分支

优先级：P1 回归覆盖，P2 新能力。

- 将视觉 encoder、projector、文本 decoder、MTP 的依赖放入统一计划；
- 保留 modality 与 processor 身份、真实图像/音频输入摘要；
- prefix 计算昂贵时可单独形成可恢复事务，避免恢复时重复编码；
- 只选择文本 decoder 量化时，未量化分支正确复制与原始 dtype 保持需要验收；
- 显式验证 embedding 作为参数传入、共享 head、跨层索引器等外部工作集；
- 变长图像 token、分辨率和输入组合纳入计划适用范围；
- 动态分支超出已验证执行计划时拒绝，不宣称任意模型自动可追踪。

### E7：校准统计复用与候选分支

优先级：P2，用于减少同一前缀下重复试验。

将可选统计缓存限定为少量显式目标的内容寻址对象，身份至少包含：
实际输入 boundary、模块原始权重、采样/计数语义、observer、变换、
dtype 和实现版本。不得恢复到旧的“默认保存全模型 H”架构。

只在输入和前缀相同、统计与候选算法兼容时复用。候选配置一旦改变前层量化、
旋转或输入传播，下游缓存失效。H/count 可用于兼容的 GPTQ 候选；
observer 搜索所需额外输入、全局 scale、路由信息仍需各自满足。

首个应用是在固定 boundary 的当前子图比较少量候选，不跨候选混用下游状态。
候选评分与最终独立质量评测分离。

### E8：KV cache、结构剪枝和其他分布式扩展

优先级：P3 探索，分别评审。

- KV cache：先定义 prefill/decode 校准、cache 状态和部署格式，
  不把关闭 cache 的静态 decoder 校准等同于 cache 量化；
- REAP/MoNe：模型结构和 config 修改会影响后续图计划及权重映射，
  需要结构版本/重规划协议后才能接入；
- EP 已提升为前期架构要求和 E9 独立交付，不属于本节的 P3 探索；
- EP×DP、EP×TP、PP、多机及弹性重分片单独定义统计/权重布局和恢复迁移；
- 数据并行归约数值顺序和 world size 纳入环境契约，不能复用单进程 bitwise 承诺；
- 对象存储：使用不可变对象与明确的 commit protocol，单独验证一致性；
  不把本地目录 rename 当作可移植实现。

这些能力不阻塞单设备核心交付，也不在首轮承诺时间表内。

### E9：EP 分阶段交付

优先级：P1 架构必选，运行实现按以下阶段验收。
依赖：EP-A 随 M0/M1，EP-B 依赖 M3/M4，EP-C 依赖 EP-B。
E3 的单设备专家组流式执行不是启动 EP-B 的前置条件。

| 阶段 | 交付 | 验收范围 |
| --- | --- | --- |
| EP-A：契约 | `ParallelContext`、placement、统计范围、参与者、EP=1 适配 | 单设备结果及普通路径成本基线不退化，schema 不依赖隐式 rank=0 |
| EP-B：参考 EP | 单机固定 EP、复制输入、本地专家、全局提交/恢复/导出/清理 | tiny MoE EP=1/2/4，RTN/iMatrix 后 GPTQ；rank/协调者强杀 |
| EP-C：dispatch | 分布 token、可变 token dispatch/combine、有界通信与回压 | 路由/统计/输出等价、热专家压力、目标设备通信及时间收益 |
| EP-D：组合 | rank 内 E3 专家组，后续 DP/TP、多机或更换拓扑恢复 | 每个组合独立能力评审，不由 EP-B 自动推导支持 |

EP-B 至少覆盖：

- 专家数非整除、rank 无专家或无 token、top-k 多路贡献、变长 batch；
- 冷专家及全专家校准的显式策略，复制 dense/shared 参数与统计不重复；
- 与 EP=1 比较路由、专家 H/count、qparams、packed tensor 和固定输入 logits；
- 故障发生在本地写入、prepared、global commit、发布、GC/清理时，
  全组重启只接受最后一条完整全局链，未提交专家结果不局部前进；
- 协调者先退出、其他 rank 迟到写入、通信超时、缺少参与者和摘要冲突；
- 导出无重复/遗漏逻辑 tensor，可经标准加载器验证，
  发布成功后不遗留框架临时目录，失败保留可恢复数据；
- 各 rank 记录设备/CPU/磁盘峰值、专家 token 负载、通信字节、
  collective 等待和全局 commit 延迟，报告最慢与峰值 rank。

EP 与单设备归约/合并顺序不同，使用预注册容差；固定 EP 拓扑下
“中断后恢复”应对照同拓扑无中断执行，不能只对照另一 EP size。
CPU 多进程测试用于协议与参考正确性，不替代目标设备 EP 通信和峰值验收。
先选择一种有可用输入且可显式适配的实际 MoE，再承诺模型支持范围；
现有默认进程组 GPTQ 路径必须完成 scope 改造后才能进入 EP-B 支持矩阵。

## 7. 性能目标与验证矩阵

### 7.1 必须满足的硬门槛

| 门槛 | 判定 |
| --- | --- |
| 结果一致性 | 固定确定性基线逐逻辑 tensor 一致；其他环境按预注册容差 |
| 故障恢复 | 所有已列关键故障点用新进程恢复后通过一致性与完整性验证 |
| 资源上界 | 不保留历史模型实体权重/H；activation 恢复批数和队列字节有界 |
| 临时磁盘 | 扣除输出、输入冻结快照后，历史 boundary 受保留策略约束 |
| 身份校验 | 输入、图计划、recipe、传播、dtype 或关键实现变化拒绝错误复用 |
| 正常模式兼容 | 继续走直接写出路径，恢复优化不强制增加中间持久化 |
| 损坏处理 | 已提交输出/有效 boundary 损坏可定位且不被静默重新计算覆盖 |
| 发布 | 重复 finalize 和目录切换中断均可恢复，不接受半成品为完成 |
| EP 提交 | 各 rank 只按 global commit 前进；任一参与者失败后全组恢复 |
| EP 所有权 | 专家统计/产物无误归约、重复或遗漏；输出不依赖训练时 rank 命名 |

### 7.2 性能目标的制定

M0 先采集基线，M1 前冻结每类模型的性能预算。
以下是评估维度，不是已经取得的结果或适用于所有硬件的固定阈值：

- normal / recovery-sync / recovery-async 的无故障总时间；
- 每阶段 source read、activation write/read、output write 和 hash bytes；
- 最坏及中位恢复时间、重做子图数量；
- 同宽度 8/32/128 子图的峰值和扫描曲线；
- 同深度下增加样本数/序列长度的 memory 与 disk 行为；
- 同规模输出的 writer 放大系数；
- 开启恢复的 GPU/设备空闲比例和正常运行成本；
- GPTQ 多 consumer 统计共享与不共享的峰值、耗时及数值比较；
- EP=1/2/4 的总时间、各 rank 峰值、负载偏斜、通信量和等待比例；
- 复制输入参考 EP 与 dispatch EP 分别报告，扩卡不能默认宣称线性加速。

不得用更少样本、更短序列、不同传播策略或 RTN fallback 增多来解释为系统加速。
硬件算子微基准与端到端节约分别记录，实际 backend/fallback 写入报告。

### 7.3 模型与算法矩阵

| 层级 | 模型/输入 | 算法 | 目的 |
| --- | --- | --- | --- |
| 单元与故障 | 人工 tensor、嵌套 boundary、跨 shard/tied tensor | writer/GC/manifest | 穷举协议窗口 |
| 确定性端到端 | tiny dense，固定 batches，CPU FP32 | RTN+iMatrix、GPTQ、AWQ、SmoothQuant | 与 sequential/reference 比较 |
| 深度资源 | 固定宽度、8/32/128 子图，独立进程 | RTN+iMatrix、GPTQ | 暴露历史保留与重复扫描 |
| 宽度资源 | 增加输入维度、相同深度 | GPTQ | H 与 workspace 估算 |
| 实际 dense | 一种本地可用完整模型，固定短/长序列 | W4A16、W4A8、W8A8 中已验证组合 | 性能与模型质量 |
| MoE | tiny MoE，再选择 DeepSeek/GLM/Kimi 一种现有适配 | RTN+iMatrix、GPTQ | 路由、专家工作集和恢复 |
| EP 协议 | tiny MoE，EP=1/2/4，非整除专家与空 rank，父进程强杀 | RTN+iMatrix，再 GPTQ | group/统计所有权、全局 commit、发布与清理 |
| EP 设备 | 单机 2/4 张目标设备，固定与偏斜路由 | 已通过参考 EP 的 recipe | dispatch/combine、通信预算、超时与真实峰值 |
| 多模态/MTP | 项目已有可用适配与样本 | 已支持的局部 recipe | prefix、共享参数和非文本状态 |
| 硬件 | CPU 参考 + 实际目标 GPU API 设备 | 同配置、记录 kernel provider | 内存/I/O/兼容与部署 reload |

具体大模型、数据与设备在执行前填写；未运行项标记 NOT_RUN/不适用原因，
不根据测试文件存在或历史文档数字标记 PASS。

### 7.4 质量评测

系统路径改变而算法语义不变时，先比较逻辑 tensor/qparams 和固定输入 logits，
再运行适用的 NLL/任务 smoke。实验模式另外比较独立数据上的 ΔNLL 与关键任务。
测试集不参与 recipe 或阈值选择。

容器、runtime、有效 activation 模式、weight packing、transform 和 KV 设置
必须匹配。参考 QDQ 与真实部署 kernel 验收分别记录。

### 7.5 结构化产物

建议在现有 artifact/report 约定上扩展以下内容，不另建一套模型质量平台：

```text
run-manifest.json
execution-semantics.json
parallel-plan.json
resource-profile.jsonl
resume-plan.json
fault-matrix.json
equivalence-report.json
io-summary.json
evaluation.json
```

每份结果携带 run/attempt、代码身份、模型/数据指纹和校验策略。
EP 结果增加拓扑/placement 身份、全局及 rank-local 事务标识；
资源与 I/O 按 rank 报告并汇总，不能只输出 rank 0 的观测。
CI 接入现有 `ci/model_quality`，压缩过程指标不能替代模型质量结果。

## 8. 改动拆分与排期

以下按一名主要开发者、已有可用目标设备估算；需要模型、驱动或算法接口排障时重估。
不包含大模型排队时间、主机掉电试验和所有 E 类扩展。
本计划不授权自动提交、发布模型或启动长时间远程任务。

| 顺序 | 可审查改动 | 依赖 | 粗估工作日 |
| --- | --- | --- | --- |
| A | 冻结语义、基线、EP-A 所有权与拓扑契约 | 无 | 3–4 |
| B | boundary 完整性、参与者、单写者与单参与者协调协议 | A | 3–4 |
| C | 子进程故障矩阵与错误分类 | B | 2–3 |
| D | 单次恢复规划、batch reader、历史校验去重 | C | 2–3 |
| E | 带回收记录的 boundary GC 与故障测试 | D | 2–3 |
| F | chunk writer、恢复异步写入与发布状态机 | E | 4–6 |
| G | 统一资源 admission 与 modifier 契约 | A、E | 4–6 |
| H | GPTQ 副本/共享统计的小范围验证 | G | 3–5 |
| I | MoE 专家组可行性样机 | E、G | 5–8 |
| J | 真实模型矩阵、报告、使用文档 | C 起持续，收敛于 F/G | 3–5 |
| K | EP-B：参考执行、全局事务与 RTN/iMatrix | F、G；不依赖 I | 7–10 |
| L | EP-B：GPTQ scope 适配及故障/资源/实际模型矩阵 | K | 5–8 |
| M | EP-C：有界 token dispatch/combine 与设备性能验证 | L | 7–12 |

A–G 是含 EP-A 架构的核心交付，约 5–7 周；H/I 是首批延伸，约再 2–3 周。
K/L 为首个完整参考 EP，约额外 3–4 周；M 约再 2–3 周，
具体通信后端/模型适配风险在 EP-A 评审后重估。上述阶段按一名开发者依次投入估算，
不能把总投入当作已经并行压缩的日历时间。
若 EP 优先，F/G 后先 K/L，再安排 H/I；普通单设备核心无需等待 dispatch 性能优化。
算法语义实验 E2、更多 modifier、EP-D 和 E8 不强行塞进核心交付。
每个里程碑达标再进入依赖它的步骤，不能用日历时间替代验收。

## 9. 首轮实现任务单

用户明确要求成功任务不遗留 `xxx.work` 一类工作目录。清理要求适用于普通模式、
checkpoint-progress 恢复模式及旧 boundary 入口：最终输出发布成功后删除框架
拥有的临时数据，并移除空的工作根目录；失败时保留恢复数据。
显式 work directory 中非框架文件保留，不递归删除调用者的任意文件。
源码和输出与工作目录的重叠必须在开始执行前拒绝，防止清理影响模型数据。
EP 延伸遵守 §3.7：全局发布完成后统一授权清理，不能由各 rank 独立删除工作根目录。

2026-11-05 清理要求已实现：普通 pretrained、checkpoint-progress 和旧 boundary
入口在成功发布后共用清理函数，失败时保留进度；完成后清理、失败保留、
恢复重试、路径重叠、既有 MoE 输出与 sequential 对照等相关本地测试
58 passed、3 skipped，Ruff 检查通过。跳过项目不计作目标设备验证。

开始编码时优先执行 A/B/C，暂不重写全部 streaming：

- [x] 固定当前传播语义与基线，补充实际实现和历史设计的差异说明。
- [x] 定义 `ParallelContext`、逻辑专家/参数 owner、统计 scope 和 EP=1 适配。
- [ ] 确定参考 EP 与 dispatch EP 的输入/路由布局，记录后续 DP/TP 的边界。
- [x] 固定 manifest v2 的必要身份、boundary schema 和兼容边界。
- [x] schema 表达拓扑、placement、参与者、local prepared 和 global commit。
- [x] 增加 boundary 摘要、数量/顺序、shape/dtype 及别名验证。
- [x] 增加 work directory 单写者锁。
- [x] 定义并实现文件/manifest/commit 的持久化顺序。
- [x] 提供可由父进程协调的内部故障点，避免将测试 hook 暴露为正式算法 API。
- [x] 覆盖写入前后、commit 前后和可反序列化损坏。
- [x] 以 tiny dense 执行事务强杀/恢复验证；RTN+iMatrix 和 GPTQ 的完整矩阵继续扩展。
- [ ] 记录真实执行结果，再更新 guide 支持范围。

M2 当前进度（2026-11-05）：

- [x] 已实现一次 committed subgraph 扫描和连续链校验。
- [x] 已实现逐 batch frontier reader，恢复不再装入历史 boundary 列表。
- [x] 已实现保留最近两个 boundary 的 retirement 记录和幂等删除。
- [x] 已覆盖历史 boundary 合法回收、损坏 boundary、强杀恢复和 subgraph 回归。
- [x] 已补充 8/32/128 子图扫描计数和读字节量测试。
- [ ] 继续生成独立进程资源曲线报告。

M3 当前进度（2026-11-05）：

- [x] 发布阶段持久化 `ready`、`backed_up`、`installed`、`cleaned` 状态。
- [x] finalize 重试可处理输出已备份、输出已安装和旧 backup 残留窗口。
- [x] 覆盖重复 finalize、direct-writer publish-in-place 和配置/索引校验回归。

M4 当前进度（2026-11-05）：

- [x] 增加显式 `ModifierContract`，表达工作集估算、统计生命周期、传播、恢复和输出所有权。
- [x] calibration output 复用通过 contract 判断，空 recipe 保持不复用。
- [x] modifier contract 指纹纳入 manifest 和 run fingerprint。
- [ ] 将 modifier contract 纳入资源 admission，补充各算法专项上界。

核心完成后的下一优先级是 D/E：恢复和 GC 的收益覆盖全部 PTQ 算法。
E1 的 GPTQ 优化和 E3 的 MoE 专家执行作为两个独立延伸验证，
依据峰值、重算和质量数据决定后续投入。
EP-B 不依赖这两项优化；在单设备事务和预算稳定后可优先交付。
