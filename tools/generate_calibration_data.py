#!/usr/bin/env python3
# ruff: noqa: E501
"""
Generate multi-turn conversation data using DeepSeek V4 API for quantization
calibration.

Generates 256 bilingual multi-turn conversations covering tool use, agent
workflows, code generation, animation production, mathematics, data analysis,
science, and creative writing. Chinese and English each account for 50% by
default, while simple daily-chat prompts remain a small minority. Includes both
thinking mode (reasoning_effort: max/high) and non-thinking mode data.

Output: JSONL file with {"messages": [...]} format compatible with
deepseek_v4_w8a8.py's preprocess function (encode_messages with
thinking_mode="thinking").

Usage:
    export DEEPSEEK_API_KEY="your-api-key"
    python generate_calibration_data.py [--output calibration_data.jsonl]
        [--num-samples 256]
        [--max-context-length 8192]
        [--english-ratio 0.5]
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI

DEFAULT_NUM_SAMPLES = 256
DEFAULT_MAX_CONTEXT_LENGTH = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 4096
MIN_OUTPUT_TOKENS = 256
MESSAGE_OVERHEAD_TOKENS = 4
DEFAULT_ENGLISH_RATIO = 0.5

LANGUAGE_INSTRUCTIONS = {
    "zh": "请使用中文回答，代码、API名称和必要的技术术语可以保留英文。",
    "en": "Respond in English. Keep code, API names, and necessary technical terms unchanged.",
}


# ===========================================================================
# Prompt Templates
# ===========================================================================

SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a knowledgeable tutor who explains concepts clearly and patiently.",
    "You are an expert data scientist skilled in analysis and visualization.",
    "You are a mathematician who solves problems with rigorous step-by-step reasoning.",
    "You are a senior software engineer helping with code and system design.",
    "You are a creative writer with a talent for vivid storytelling.",
    "You are a science educator who makes complex topics accessible.",
    "You are a history buff who provides rich historical context.",
    "You are a philosophical discussion partner exploring deep questions.",
    "You are a business strategy consultant providing actionable insights.",
    "You are a medical professional explaining health topics accurately.",
    "You are a financial advisor helping with personal finance decisions.",
    "You are a linguistics expert analyzing language and communication.",
    "You are a psychologist discussing human behavior and cognition.",
    "You are an environmental scientist discussing ecology and sustainability.",
    (
        "You are an AI agent that plans tasks, calls tools when needed, and "
        "reports verifiable results."
    ),
    (
        "You are a software engineer who writes production-ready code and "
        "tests, explaining important tradeoffs."
    ),
    (
        "You are a film and animation director who turns an idea into a "
        "technically feasible shot plan."
    ),
]

# Most calibration examples exercise tools, planning, code, or reasoning. Daily
# chat remains available, but its low weight prevents it from dominating the set.
ZH_SCENARIO_PROMPTS = {
    "tool_and_agent": [
        "你是一个研究型Agent。请规划一次联网检索，调用搜索、网页抓取和引用整理工具，最后给出可核验的结论。",
        "设计一个客服Agent处理退款请求的完整流程，说明何时调用订单查询、库存和人工转接工具。",
        "给出一个使用浏览器工具完成竞品价格调查的任务分解，并定义每一步的输入、输出和失败重试策略。",
        "模拟一个代码库维护Agent：先搜索相关文件，再运行测试，定位回归并提出最小补丁。请展示工具调用顺序。",
        "为旅行规划Agent设计多代理协作方案，让交通、住宿和预算代理互相校验后生成行程。",
        "一个数据分析Agent需要读取CSV、执行SQL、绘制图表并写报告。请定义工具接口和编排逻辑。",
        "设计一个能调用日历、邮件和待办事项工具的个人助理，讨论权限、确认操作和幂等性。",
        "模拟终端Agent排查服务不可用：检查日志、端口、进程和依赖，并在每一步说明依据。",
        "如何让RAG Agent在检索结果冲突时进行来源排序、事实核查和带引用回答？",
        "为多轮任务Agent设计状态机，覆盖规划、工具执行、观察、反思、交付和人工接管。",
        "给出MCP工具服务器的一个安全设计，包含参数校验、超时、审计日志和敏感信息脱敏。",
        "当工具返回空结果、超时或格式错误时，Agent应该如何恢复并向用户解释？请给出伪代码。",
        "设计一个财务报表Agent，要求它调用计算器和数据库工具，并防止未经确认的转账操作。",
        "让Agent阅读一组API文档后生成调用计划，说明如何处理分页、速率限制和版本兼容。",
        "比较ReAct、Plan-and-Execute和函数调用Agent在复杂任务中的优缺点，并给出选型建议。",
        "模拟一个多代理辩论：事实核查Agent、分析Agent和编辑Agent如何协作产出新闻摘要？",
        "为图像生成Agent设计提示词改写、生成、质量检查和重试的工具链。",
        "设计一个能操作表格的Agent，说明读取单元格、批量更新和撤销机制的工具协议。",
    ],
    "code_generation": [
        "用Python实现一个可取消、可重试且限速的asyncio任务池，并编写单元测试。",
        "请为FastAPI服务实现JWT认证、中间件、刷新令牌和权限测试，给出完整代码骨架。",
        "用TypeScript设计一个类型安全的事件总线，支持通配符订阅、异步处理和错误隔离。",
        "为React数据表格编写虚拟滚动、服务端分页和键盘导航组件，并解释状态管理。",
        "用SQL和Python实现用户留存漏斗分析，处理时区、重复事件和缺失日期。",
        "实现一个Rust多生产者多消费者队列，要求无锁读取并给出基准测试思路。",
        "为一个Python包设计插件系统，包含注册、发现、版本约束和懒加载。",
        "写一个Docker Compose部署的Web应用，包含健康检查、数据库迁移和优雅停机。",
        "实现LRU缓存并加入TTL、线程安全和命中率统计，比较不同锁策略。",
        "请审查下面这类SQL慢查询的优化方案：索引设计、执行计划、分页和连接顺序。",
        "用C++实现生产者消费者线程池，要求异常安全、可关闭和避免任务饥饿。",
        "为Transformer推理编写KV cache分页管理的伪代码，说明并发访问和内存回收。",
        "设计一个GitHub Actions CI流水线，覆盖格式化、类型检查、单测、构建和发布。",
        "把一个单体订单服务拆成微服务，给出API契约、事件模型、幂等和最终一致性代码示例。",
        "实现一个命令行工具：解析参数、读取配置、输出JSON，并为错误提供可操作提示。",
        "用Python写一个安全的文件批处理脚本，支持dry-run、断点续跑和原子写入。",
        "请为一个排序算法实现属性测试，说明生成数据、性质和反例缩减方法。",
        "实现WebSocket聊天室的心跳、重连、广播和背压处理，并列出边界测试。",
        "重构一段难以维护的继承层次为组合式设计，说明接口和迁移步骤。",
        "给出CUDA kernel性能优化的排查清单，覆盖内存合并、占用率和同步开销。",
    ],
    "math_reasoning": [
        "求解微分方程 dy/dx + 2xy = x 的通解，验证初值并讨论解的唯一性。",
        "计算矩阵 [[3,1,0],[1,2,1],[0,1,3]] 的特征值和特征向量，展示正交化过程。",
        "用反证法和无限递降法分别证明根号2是无理数，并比较两种证明。",
        "用泰勒展开近似计算 sin(0.1)，给出余项上界并证明误差小于10^-6。",
        "求解不定积分 ∫ x^2 * e^x dx，展示分部积分并通过求导验证。",
        "用梯度下降法求 f(x,y)=x^2+3y^2+2xy-4x-6y 的最小值，分析收敛条件。",
        "推导贝叶斯定理，构造一个医疗检测的数值例子并计算后验概率。",
        "证明圆内接正n边形面积的公式，并求n趋于无穷时的极限。",
        "给定一个转移矩阵，计算马尔可夫链的平稳分布并判断是否遍历。",
        "证明中心极限定理在样本均值上的应用，并解释有限样本时的误差来源。",
        "证明欧拉公式 e^(iπ)+1=0，分别使用幂级数和复数几何解释。",
        "用傅里叶变换求解一个简单热方程，说明边界条件如何影响系数。",
        "求解一个带整数约束的线性规划问题，比较单纯形法与分支定界法。",
        "解释信息论熵、互信息和KL散度的关系，并完成一个离散分布计算题。",
        "给出一个组合数学计数问题，使用容斥原理和生成函数两种方法求解。",
        "证明一个图论命题：树有n个顶点当且仅当有n-1条边且连通无环。",
        "分析递推式T(n)=2T(n/2)+n log n，给出递归树和主定理推导。",
        "构造一个反例说明‘相关不等于因果’，并用潜在结果框架形式化。",
        "求矩阵指数并用它求解线性常微分方程组，检查特征值重根情况。",
        "设计一道概率推理题，要求列出假设、条件独立性、计算过程和结果合理性检查。",
    ],
    "animation_and_visual": [
        "把‘光合作用’制作成60秒科普动画，给出分镜、旁白、镜头运动和视觉隐喻。",
        "用Three.js制作可交互的3D太阳系演示，规划场景、相机、动画循环和性能优化。",
        "用Manim编写傅里叶级数可视化动画，给出代码结构和逐段讲解。",
        "设计一个Blender Python脚本生成机械臂装配动画，包含关键帧、约束和渲染设置。",
        "为数学课程制作‘梯度下降寻找最低点’动画，说明坐标系、轨迹和逐帧数据。",
        "用HTML Canvas实现粒子烟花效果，讨论时间步长、碰撞和移动端性能。",
        "把一篇产品发布稿改编成30秒竖屏短视频，输出镜头表、字幕和音效节奏。",
        "设计一个角色表情动画系统，覆盖骨骼、blend shape、状态机和口型同步。",
        "用SVG和CSS制作可访问的流程图动画，要求支持暂停、键盘操作和减少动效设置。",
        "规划一段城市延时摄影的后期动画：转场、色彩、速度曲线和音乐节拍如何配合？",
        "为‘黑洞吸积盘’制作科学可视化，区分真实物理、艺术夸张和观众提示。",
        "设计一个交互式数据故事，让用户拖动时间轴观察销量变化并触发注释动画。",
    ],
    "data_and_automation": [
        "分析电商平台留存率下降的原因，给出SQL、指标口径、可视化和实验方案。",
        "用Python对销售数据进行季节性分解，处理缺失值、异常值和节假日效应。",
        "详细解释A/B测试的统计原理，设计样本量、随机化、停止规则和结果报告。",
        "如何用SQL进行用户行为漏斗分析？请给出可执行查询并说明去重逻辑。",
        "比较PCA、UMAP和自编码器降维，在可解释性、速度和数据规模上如何选择？",
        "设计推荐系统离线与在线评估，覆盖NDCG、覆盖率、延迟和长期留存。",
        "用统计学方法检测异常值，并讨论删除、截尾、稳健模型的风险。",
        "构建一个从日志到告警的可观测性流水线，说明采集、聚合、采样和隐私处理。",
        "给出一个数据质量检查框架，覆盖schema、范围、唯一性、时效性和血缘。",
        "解释因果推断中的倾向得分、双重差分和随机实验的适用条件。",
        "设计一个批量文件转换任务的自动化调度方案，包含依赖、重试、监控和回滚。",
        "请把一份杂乱的JSON数据清洗成分析表，列出字段映射、类型转换和校验规则。",
    ],
    "science_and_domain": [
        "详细解释量子纠缠的实验验证、不可超光速通信原因和潜在应用。",
        "说明CRISPR基因编辑的工作原理、脱靶检测、医学应用和伦理边界。",
        "解释黑洞形成、事件视界和霍金辐射，并区分观测证据与理论推测。",
        "用能量收支和反馈机制解释全球变暖证据及不确定性。",
        "比较狭义相对论的时间膨胀与长度收缩，给出具体计算例子。",
        "从分子层面说明DNA复制、转录、翻译及其错误校正机制。",
        "解释暗物质和暗能量的观测证据、候选模型和仍未解决的问题。",
        "比较先天免疫和适应性免疫，追踪一次疫苗接种后的时间线。",
        "解释超导的宏观现象、BCS理论和高温超导的挑战。",
        "说明地震波如何反演地球内部结构，并设计一个观测实验。",
    ],
    "creative_writing": [
        "构思一个人工智能觉醒的科幻短篇，给出人物弧线、冲突升级和开放式结局。",
        "写一段赛博朋克城市追逐戏的电影分镜，包含景别、运动、光线和声音。",
        "用冰山理论设计一个短篇小说，标出显性事件与未说出口的背景。",
        "创作一篇关于‘时间’的散文开头，并解释意象、节奏和叙述视角。",
        "塑造一个复杂反派角色，说明其目标、道德自洽、弱点和转变节点。",
        "把一个历史事件改写成三幕剧，区分史实、戏剧化和需要查证的内容。",
        "写一首关于星空和梦想的现代诗，使用具体意象而非抽象口号。",
        "为一款解谜游戏设计世界观、关卡线索和玩家逐步获得信息的节奏。",
    ],
    "daily_chat": [
        "推荐几道简单又营养的家常菜，并按准备时间排序。",
        "如何科学地改善睡眠质量？请区分证据充分和因人而异的建议。",
        "规划一个周末短途旅行，列出预算、交通、天气变化下的备选方案。",
        "怎样制定一个可执行的个人理财计划，并指出常见风险？",
        "推荐几本适合反复阅读的经典书籍，说明适合的阅读目的。",
    ],
}

EN_SCENARIO_PROMPTS = {
    "tool_and_agent": [
        "Act as a research agent. Plan a web search, call search and page-fetch tools, and produce a cited, verifiable conclusion.",
        "Design a customer-support agent for a refund request. Specify when it calls order lookup, inventory, and human-escalation tools.",
        "Break down a browser-tool workflow for competitor price research, including inputs, outputs, retries, and failure handling.",
        "Simulate a code-maintenance agent: search the repository, run tests, locate a regression, and propose the smallest patch. Show the tool order.",
        "Design a travel-planning multi-agent system where transport, lodging, and budget agents cross-check one another before producing an itinerary.",
        "A data agent must read CSV files, run SQL, create charts, and write a report. Define its tool interfaces and orchestration logic.",
        "Design a personal assistant that can call calendar, email, and task tools. Discuss permissions, confirmation gates, and idempotency.",
        "Simulate a terminal agent diagnosing an unavailable service by checking logs, ports, processes, and dependencies. Explain the evidence at every step.",
        "How should a RAG agent rank sources, resolve conflicting retrievals, fact-check claims, and answer with citations?",
        "Design a state machine for a multi-step task agent covering planning, tool execution, observation, reflection, delivery, and human takeover.",
        "Propose a secure MCP tool server with parameter validation, timeouts, audit logs, and secret redaction.",
        "How should an agent recover from an empty result, timeout, or malformed tool response? Include pseudocode and user-facing errors.",
        "Design a financial-reporting agent that calls calculator and database tools while preventing unconfirmed money transfers.",
        "Have an agent read API documentation and create an execution plan that handles pagination, rate limits, and version compatibility.",
        "Compare ReAct, Plan-and-Execute, and function-calling agents for complex tasks. Give selection criteria and a recommendation.",
        "Simulate a multi-agent editorial workflow in which fact-checking, analysis, and editing agents produce a sourced news brief.",
        "Design a tool chain for an image-generation agent: prompt rewriting, generation, quality checks, and bounded retries.",
        "Design a spreadsheet-manipulation agent with tools for reading cells, batch updates, validation, and undo.",
    ],
    "code_generation": [
        "Implement a cancellable, retryable, rate-limited asyncio task pool in Python and include unit tests.",
        "Build a FastAPI service with JWT authentication, middleware, refresh tokens, and permission tests. Provide a complete code skeleton.",
        "Design a type-safe TypeScript event bus with wildcard subscriptions, asynchronous handlers, and error isolation.",
        "Implement a React data table with virtual scrolling, server-side pagination, and keyboard navigation. Explain state management.",
        "Implement user-retention funnel analysis with SQL and Python, handling time zones, duplicate events, and missing dates.",
        "Implement a Rust multi-producer, multi-consumer queue with lock-free reads and describe a benchmark plan.",
        "Design a Python package plugin system with registration, discovery, version constraints, and lazy loading.",
        "Write a Docker Compose deployment for a web application with health checks, database migrations, and graceful shutdown.",
        "Implement an LRU cache with TTL, thread safety, and hit-rate metrics. Compare alternative locking strategies.",
        "Review a slow SQL query and propose improvements to indexes, query plans, pagination, and join order.",
        "Implement a C++ producer-consumer thread pool with exception safety, shutdown, and starvation avoidance.",
        "Write pseudocode for paged KV-cache management during Transformer inference, including concurrent access and reclamation.",
        "Design a GitHub Actions CI pipeline for formatting, type checking, tests, builds, and releases.",
        "Decompose a monolithic order service into microservices with API contracts, events, idempotency, and eventual consistency examples.",
        "Implement a CLI that parses arguments, reads configuration, emits JSON, and gives actionable error messages.",
        "Write a safe Python batch-file processor with dry-run mode, resumability, and atomic writes.",
        "Create property-based tests for a sorting algorithm. Explain data generation, invariants, and counterexample shrinking.",
        "Implement WebSocket chat heartbeats, reconnects, broadcasts, and backpressure. List boundary tests.",
        "Refactor a hard-to-maintain inheritance hierarchy into composition. Explain interfaces and migration steps.",
        "Give a CUDA-kernel performance checklist covering coalesced memory, occupancy, synchronization, and profiling evidence.",
    ],
    "math_reasoning": [
        "Solve dy/dx + 2xy = x, verify an initial condition, and discuss existence and uniqueness.",
        "Compute the eigenvalues and eigenvectors of [[3,1,0],[1,2,1],[0,1,3]], showing the orthogonalization steps.",
        "Prove that the square root of 2 is irrational using both contradiction and infinite descent, then compare the proofs.",
        "Approximate sin(0.1) with a Taylor expansion, bound the remainder, and prove the error is below 10^-6.",
        "Evaluate the indefinite integral of x^2 * e^x using integration by parts and verify it by differentiation.",
        "Use gradient descent to minimize f(x,y)=x^2+3y^2+2xy-4x-6y and analyze convergence conditions.",
        "Derive Bayes' theorem, construct a numerical medical-test example, and calculate the posterior probability.",
        "Derive the area of a regular n-gon inscribed in a circle and compute its limit as n approaches infinity.",
        "Given a transition matrix, compute a Markov chain's stationary distribution and determine whether it is ergodic.",
        "Explain how the central limit theorem applies to a sample mean and identify finite-sample error sources.",
        "Prove Euler's identity e^(i*pi)+1=0 using power series and a geometric interpretation.",
        "Use a Fourier transform to solve a simple heat equation and explain how boundary conditions affect coefficients.",
        "Solve an integer-constrained linear program and compare the simplex method with branch and bound.",
        "Explain entropy, mutual information, and KL divergence, then work through a discrete-distribution calculation.",
        "Create a combinatorics problem and solve it with both inclusion-exclusion and generating functions.",
        "Prove the graph-theory statement that a tree with n vertices has n-1 edges, and prove the converse.",
        "Analyze T(n)=2T(n/2)+n log n using a recursion tree and the Master theorem.",
        "Construct a counterexample showing that correlation does not imply causation, then formalize it with potential outcomes.",
        "Compute a matrix exponential and use it to solve a linear ODE system, including the repeated-eigenvalue case.",
        "Design a probability puzzle requiring explicit assumptions, conditional independence, calculations, and sanity checks.",
    ],
    "animation_and_visual": [
        "Turn photosynthesis into a 60-second educational animation with a shot list, narration, camera motion, and visual metaphors.",
        "Plan an interactive 3D solar-system demo in Three.js, including scene setup, camera, animation loop, and performance work.",
        "Write the structure of a Manim animation visualizing Fourier series and explain it section by section.",
        "Design a Blender Python script that creates a robotic-arm assembly animation with keyframes, constraints, and render settings.",
        "Create an animation for gradient descent finding a minimum. Specify axes, trajectory, annotations, and frame-by-frame data.",
        "Implement a particle-fireworks effect with HTML Canvas and discuss time steps, collisions, and mobile performance.",
        "Adapt a product launch message into a 30-second vertical video with a shot table, captions, and sound timing.",
        "Design a facial-animation system covering skeletal rigs, blend shapes, state machines, and lip synchronization.",
        "Create an accessible SVG and CSS flowchart animation with pause, keyboard controls, and reduced-motion support.",
        "Plan the post-production of an urban time-lapse: transitions, color, speed curves, and music synchronization.",
        "Design a scientific visualization of a black-hole accretion disk and distinguish physical evidence from artistic exaggeration.",
        "Build an interactive data story in which a timeline slider changes sales charts and reveals explanatory annotations.",
    ],
    "data_and_automation": [
        "Analyze a decline in e-commerce retention with SQL, metric definitions, visualizations, and an experiment plan.",
        "Use Python to decompose seasonal sales data while handling missing values, outliers, and holiday effects.",
        "Explain the statistics of A/B testing and design sample sizing, randomization, stopping rules, and reporting.",
        "Write an executable SQL query for a user-behavior funnel and explain its deduplication logic.",
        "Compare PCA, UMAP, and autoencoders for dimensionality reduction by interpretability, speed, and data scale.",
        "Design offline and online evaluation for a recommender system, covering NDCG, coverage, latency, and long-term retention.",
        "Detect outliers statistically and discuss the risks of deletion, winsorization, and robust models.",
        "Build an observability pipeline from logs to alerts, including collection, aggregation, sampling, and privacy controls.",
        "Define a data-quality framework covering schema, ranges, uniqueness, freshness, and lineage.",
        "Explain when to use propensity scores, difference-in-differences, or randomized experiments for causal inference.",
        "Design an automated batch-file conversion scheduler with dependencies, retries, monitoring, and rollback.",
        "Transform messy JSON into an analytics table and list field mappings, type conversions, and validation rules.",
    ],
    "science_and_domain": [
        "Explain experimental evidence for quantum entanglement, why it cannot enable faster-than-light communication, and its applications.",
        "Describe CRISPR gene editing, off-target detection, medical applications, and ethical boundaries.",
        "Explain black-hole formation, the event horizon, and Hawking radiation, separating evidence from theory.",
        "Use energy budgets and feedback mechanisms to explain evidence for global warming and key uncertainties.",
        "Compare time dilation and length contraction in special relativity with a numerical example.",
        "Trace DNA replication, transcription, translation, and error correction at the molecular level.",
        "Explain observational evidence for dark matter and dark energy, candidate models, and open questions.",
        "Compare innate and adaptive immunity by tracing the timeline after vaccination.",
        "Explain macroscopic superconductivity, BCS theory, and the challenges of high-temperature superconductors.",
        "Explain how seismic waves reveal Earth's internal structure and design an observation experiment.",
    ],
    "creative_writing": [
        "Outline a science-fiction short story about an awakened AI with character arcs, escalating conflict, and an open ending.",
        "Write a cyberpunk chase scene as a film storyboard with shot sizes, motion, lighting, and sound.",
        "Design a short story using the iceberg theory and label explicit events versus hidden backstory.",
        "Write the opening of an essay about time and explain its imagery, rhythm, and narrative viewpoint.",
        "Create a complex antagonist with goals, moral self-consistency, weaknesses, and turning points.",
        "Adapt a historical event into a three-act play and distinguish facts, dramatization, and claims needing research.",
        "Write a modern poem about stars and dreams using concrete images rather than abstract slogans.",
        "Design the world, clues, and information pacing for a mystery game.",
    ],
    "daily_chat": [
        "Recommend several simple and nutritious home-cooked meals, ordered by preparation time.",
        "How can someone improve sleep scientifically? Separate well-supported advice from person-dependent suggestions.",
        "Plan a weekend trip with a budget, transportation, and fallback options for changing weather.",
        "Create a practical personal-finance plan and point out common risks.",
        "Recommend classic books for repeated reading and explain the purpose each suits.",
    ],
}

SCENARIO_PROMPTS = {
    "zh": ZH_SCENARIO_PROMPTS,
    "en": EN_SCENARIO_PROMPTS,
}

SCENARIO_WEIGHTS = {
    "tool_and_agent": 0.20,
    "code_generation": 0.22,
    "math_reasoning": 0.18,
    "animation_and_visual": 0.12,
    "data_and_automation": 0.12,
    "science_and_domain": 0.08,
    "creative_writing": 0.05,
    "daily_chat": 0.03,
}

ZH_FOLLOWUP_PROMPTS = [
    "把方案拆成可执行步骤，并明确每一步的输入、输出和验收标准。",
    "请给出一个包含边界条件和失败处理的具体例子。",
    "如果资源、时间或权限减半，应该如何调整方案？",
    "请检查刚才的推导或代码，指出一个可能的错误并修正它。",
    "对比两个替代方案，给出选择依据和可量化的权衡。",
    "请把回答改写成可以直接交给团队执行的任务清单。",
    "哪些结论依赖额外假设？请列出假设并说明如何验证。",
    "补充测试、监控或评估指标，确保这个方案上线后可观察。",
    "请用一组小数据手算一遍关键步骤，并解释结果。",
    "如果前提条件改变，结论或工具调用顺序会如何变化？",
    "请给出更简洁的版本，同时保留最容易被忽略的风险。",
    "从反例或最坏情况出发，重新审视刚才的设计。",
]

ZH_THIRD_TURN_PROMPTS = [
    "根据上面的结果做一次复盘：哪些地方最可能失败，如何增加自动化检查？",
    "请输出最终交付物的结构（代码、表格、分镜或报告），并标注仍需人工确认的部分。",
    "现在加入一个新的约束，再完整走一遍关键推理或执行流程。",
    "请把这次讨论总结成一份短小的决策记录，包含背景、选择、证据和后续行动。",
    "如果需要让另一个Agent接手，哪些状态、上下文和工具结果必须传递？",
    "给出一个最小可运行或可验证的版本，以及从它扩展到生产版本的路线。",
]

EN_FOLLOWUP_PROMPTS = [
    "Break the proposal into executable steps and define the input, output, and acceptance criteria for each step.",
    "Give a concrete example that includes boundary conditions and failure handling.",
    "How would the plan change if the available resources, time, or permissions were cut in half?",
    "Review the reasoning or code above, identify one likely issue, and correct it.",
    "Compare two alternatives and give measurable tradeoffs and a decision rule.",
    "Rewrite the answer as a task list that a team could execute directly.",
    "Which conclusions depend on extra assumptions? List them and explain how to validate them.",
    "Add tests, monitoring, or evaluation metrics so the solution is observable in production.",
    "Work through the key steps with a small dataset and explain whether the result is sensible.",
    "If a premise changes, how do the conclusion or tool-call order change?",
    "Give a shorter version while retaining the most easily missed risks.",
    "Reconsider the design from a counterexample or worst-case scenario.",
]

EN_THIRD_TURN_PROMPTS = [
    "Review the result: what is most likely to fail, and how can automated checks catch it?",
    "Output the final deliverable structure and mark the parts that still need human confirmation.",
    "Add a new constraint and walk through the key reasoning or execution flow again.",
    "Summarize this discussion as a short decision record with context, choice, evidence, and next actions.",
    "If another agent takes over, which state, context, and tool results must be transferred?",
    "Give a minimal runnable or verifiable version and a path from it to production quality.",
]

FOLLOWUP_PROMPTS = {"zh": ZH_FOLLOWUP_PROMPTS, "en": EN_FOLLOWUP_PROMPTS}
THIRD_TURN_PROMPTS = {
    "zh": ZH_THIRD_TURN_PROMPTS,
    "en": EN_THIRD_TURN_PROMPTS,
}


# ===========================================================================
# Conversation Generator
# ===========================================================================

class ConversationGenerator:
    """Generate multi-turn conversations using DeepSeek V4 API."""

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
    ):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY environment variable not set.\n"
                "Usage: export DEEPSEEK_API_KEY='your-api-key'"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self.model = model
        if max_context_length <= MIN_OUTPUT_TOKENS:
            raise ValueError(
                f"max_context_length must be greater than {MIN_OUTPUT_TOKENS}"
            )
        self.max_context_length = max_context_length

    def generate_one(
        self, sample_id: int, language: Optional[str] = None
    ) -> Optional[dict]:
        """Generate a single multi-turn conversation.

        Returns:
            dict with "messages" (list of role/content dicts) and "meta" (metadata),
            or None if generation failed.
        """
        language = language or random.choice(list(SCENARIO_PROMPTS))
        if language not in SCENARIO_PROMPTS:
            raise ValueError(f"Unsupported language: {language}")
        scenario = random.choices(
            list(SCENARIO_PROMPTS[language]),
            weights=[
                SCENARIO_WEIGHTS[name] for name in SCENARIO_PROMPTS[language]
            ],
            k=1,
        )[0]
        system_prompt = (
            f"{random.choice(SYSTEM_PROMPTS)}\n{LANGUAGE_INSTRUCTIONS[language]}"
        )
        user_prompt = random.choice(SCENARIO_PROMPTS[language][scenario])

        # Reasoning-heavy examples are useful for calibration; keep a small
        # non-thinking slice for ordinary instruction-following behavior.
        mode_rand = random.random()
        if mode_rand < 0.5:
            thinking_mode = "thinking"
            reasoning_effort = "max"
        elif mode_rand < 0.8:
            thinking_mode = "thinking"
            reasoning_effort = "high"
        else:
            thinking_mode = "non-thinking"
            reasoning_effort = None

        # Multi-turn context is more representative of tool and coding tasks.
        num_turns = 3 if random.random() < 0.65 else 2

        messages = [{"role": "system", "content": system_prompt}]

        try:
            # --- First turn ---
            messages.append({"role": "user", "content": user_prompt})
            response = self._call_api(messages, thinking_mode, reasoning_effort)
            if response is None:
                return None
            assistant_msg = self._build_assistant_msg(response, thinking_mode)
            messages.append(assistant_msg)

            # --- Second turn ---
            followup = self._make_followup(user_prompt, language)
            messages.append({"role": "user", "content": followup})
            response = self._call_api(messages, thinking_mode, reasoning_effort)
            if response is None:
                return None
            assistant_msg2 = self._build_assistant_msg(response, thinking_mode)
            messages.append(assistant_msg2)

            # --- Third turn (optional) ---
            if num_turns == 3:
                third_prompt = random.choice(THIRD_TURN_PROMPTS[language])
                messages.append({"role": "user", "content": third_prompt})
                response = self._call_api(messages, thinking_mode, reasoning_effort)
                if response is None:
                    return None
                assistant_msg3 = self._build_assistant_msg(response, thinking_mode)
                messages.append(assistant_msg3)

            return {
                "messages": messages,
                "meta": {
                    "thinking_mode": thinking_mode,
                    "reasoning_effort": reasoning_effort,
                    "num_turns": num_turns,
                    "language": language,
                    "scenario": scenario,
                    "max_context_length": self.max_context_length,
                    "sample_id": sample_id,
                },
            }

        except Exception as e:
            print(f"[Sample {sample_id}] Error: {e}")
            return None

    def _call_api(
        self, messages: list, thinking_mode: str, reasoning_effort: Optional[str]
    ) -> Optional[dict]:
        """Make a single chat completion API call with retries.

        Args:
            messages: List of message dicts (system/user/assistant).
            thinking_mode: "thinking" or "non-thinking".
            reasoning_effort: "max", "high", or None.

        Returns:
            dict with "content" and optionally "reasoning_content", or None on failure.
        """
        request_messages = self._fit_context(messages)
        input_tokens = self._estimate_tokens(request_messages)
        available_output_tokens = self.max_context_length - input_tokens
        if available_output_tokens < MIN_OUTPUT_TOKENS:
            raise ValueError(
                "Unable to fit request within max_context_length after truncation"
            )

        kwargs: dict = {
            "model": self.model,
            "messages": request_messages,
            "stream": False,
            "max_tokens": min(DEFAULT_MAX_OUTPUT_TOKENS, available_output_tokens),
        }

        if thinking_mode == "thinking":
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = reasoning_effort

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                result = {"content": choice.message.content}
                reasoning = getattr(choice.message, "reasoning_content", None)
                if reasoning:
                    result["reasoning_content"] = reasoning
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  [Sample {messages[1].get('content', '')[:30]}...] "
                          f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise

        return None

    @staticmethod
    def _estimate_tokens(messages: list) -> int:
        """Conservatively estimate tokens without requiring a model tokenizer.

        A character-based upper bound avoids undercounting Chinese, source code,
        and JSON punctuation. The estimate is used only to keep API requests
        below the configured context limit; the model tokenizer may use fewer
        tokens for ordinary English text.
        """
        text_length = sum(
            len(str(message.get("content", "")))
            + len(str(message.get("reasoning_content", "")))
            for message in messages
        )
        # Account for role/separator tokens added by the chat template.
        return max(1, text_length + MESSAGE_OVERHEAD_TOKENS * len(messages))

    def _fit_context(self, messages: list) -> list:
        """Keep system/latest messages and newest history within the context cap."""
        if (
            self._estimate_tokens(messages) + MIN_OUTPUT_TOKENS
            <= self.max_context_length
        ):
            return messages

        system = messages[:1]
        latest = messages[-1:]
        middle = messages[1:-1]
        budget = self.max_context_length - MIN_OUTPUT_TOKENS

        kept = list(system)
        if self._estimate_tokens(kept) > budget - MESSAGE_OVERHEAD_TOKENS:
            system_content = str(kept[0].get("content", ""))
            max_chars = max(1, budget - 2 * MESSAGE_OVERHEAD_TOKENS - 1)
            system_message = {
                **kept[0],
                "content": system_content[:max_chars],
            }
            system_message.pop("reasoning_content", None)
            kept = [system_message]
        kept_tokens = self._estimate_tokens(kept)
        latest_tokens = self._estimate_tokens(latest)
        if latest_tokens > budget - kept_tokens:
            content = str(latest[0].get("content", ""))
            max_chars = max(1, budget - kept_tokens - MESSAGE_OVERHEAD_TOKENS)
            latest_message = {**latest[0], "content": content[-max_chars:]}
            latest_message.pop("reasoning_content", None)
            latest = [latest_message]
            latest_tokens = self._estimate_tokens(latest)

        remaining = budget - kept_tokens - latest_tokens
        selected = []
        for message in reversed(middle):
            message_tokens = self._estimate_tokens([message])
            if message_tokens <= remaining:
                selected.append(message)
                remaining -= message_tokens
                continue
            if remaining > 0:
                content = str(message.get("content", ""))
                max_chars = remaining - MESSAGE_OVERHEAD_TOKENS
                if max_chars <= 0:
                    break
                partial = {**message, "content": content[-max_chars:]}
                # Old reasoning traces are less useful than preserving the
                # latest turns and can otherwise defeat the context budget.
                partial.pop("reasoning_content", None)
                selected.append(partial)
            break

        kept.extend(reversed(selected))
        kept.extend(latest)
        return kept

    @staticmethod
    def _build_assistant_msg(response: dict, thinking_mode: str) -> dict:
        """Build an assistant message dict from API response.

        Always includes "content". Includes "reasoning_content" only for
        thinking-mode responses that have it (so encode_messages with
        thinking_mode="thinking" can render <think> blocks).
        """
        msg = {"role": "assistant", "content": response.get("content", "")}
        if thinking_mode == "thinking" and response.get("reasoning_content"):
            msg["reasoning_content"] = response["reasoning_content"]
        return msg

    @staticmethod
    def _make_followup(original_prompt: str, language: str) -> str:
        """Generate a context-aware followup prompt."""
        followup = random.choice(FOLLOWUP_PROMPTS[language])
        # Occasionally prepend topic reference for continuity (~30% chance)
        if random.random() < 0.3:
            topic = original_prompt[:30].rstrip("？?。.")
            if language == "zh":
                followup = f"关于'{topic}'这个，{followup}"
            else:
                followup = f'Regarding the original topic "{topic}": {followup}'
        return followup


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn conversation data for DeepSeek V4 "
                    "quantization calibration."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file path. Defaults to 'calibration_data_<model>.jsonl'.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
        help="Number of conversation samples to generate.",
    )
    parser.add_argument(
        "--max-context-length", type=int, default=DEFAULT_MAX_CONTEXT_LENGTH,
        help=(
            "Maximum context length in tokens for each API request, including "
            "the generated response (default: 8192)."
        ),
    )
    parser.add_argument(
        "--english-ratio", type=float, default=DEFAULT_ENGLISH_RATIO,
        help=(
            "Fraction of samples whose prompts and responses should be in "
            "English (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Number of parallel API workers.",
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-v4-pro",
        help="Model name for the API (e.g., 'deepseek-v4-pro').",
    )
    args = parser.parse_args()

    # Default output filename based on model name
    if args.output is None:
        model_slug = args.model.replace("/", "-").replace(":", "-")
        args.output = f"calibration_data_{model_slug}.jsonl"

    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.max_context_length <= MIN_OUTPUT_TOKENS:
        parser.error(
            f"--max-context-length must be greater than {MIN_OUTPUT_TOKENS}"
        )
    if not 0.0 <= args.english_ratio <= 1.0:
        parser.error("--english-ratio must be between 0.0 and 1.0")

    generator = ConversationGenerator(
        model=args.model,
        max_context_length=args.max_context_length,
    )
    results: list = []
    failed: int = 0

    print(f"Generating {args.num_samples} conversations "
          f"with {args.max_workers} workers...")
    print(
        f"Output: {args.output}\n"
        f"Max context length: {args.max_context_length} tokens\n"
        f"Language mix: {args.english_ratio:.0%} English, "
        f"{1 - args.english_ratio:.0%} Chinese"
    )

    english_count = round(args.num_samples * args.english_ratio)
    chinese_count = args.num_samples - english_count
    language_plan = ["en"] * english_count + ["zh"] * chinese_count
    random.shuffle(language_plan)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(generator.generate_one, i, language_plan[i]): i
            for i in range(args.num_samples)
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append((sample_id, result))
                    if len(results) % 50 == 0:
                        print(f"Progress: {len(results)}/{args.num_samples}")
                else:
                    failed += 1
            except Exception as e:
                print(f"[Sample {sample_id}] Unexpected error: {e}")
                failed += 1

    # Sort by sample_id
    results.sort(key=lambda x: x[0])

    # Write JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for _, result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*50}")
    print(f"Done! Generated {len(results)} conversations, {failed} failed.")
    print(f"Output: {args.output}")

    thinking_max = sum(
        1 for _, r in results if r["meta"]["reasoning_effort"] == "max"
    )
    thinking_high = sum(
        1 for _, r in results if r["meta"]["reasoning_effort"] == "high"
    )
    non_thinking = sum(
        1 for _, r in results if r["meta"]["thinking_mode"] == "non-thinking"
    )
    two_turn = sum(1 for _, r in results if r["meta"]["num_turns"] == 2)
    three_turn = sum(1 for _, r in results if r["meta"]["num_turns"] == 3)
    language_counts = {
        language: sum(1 for _, r in results if r["meta"]["language"] == language)
        for language in ("zh", "en")
    }
    scenario_counts = {
        name: sum(1 for _, r in results if r["meta"]["scenario"] == name)
        for name in SCENARIO_WEIGHTS
    }

    print("\nDistribution:")
    print(f"  Thinking (max):  {thinking_max}")
    print(f"  Thinking (high): {thinking_high}")
    print(f"  Non-thinking:    {non_thinking}")
    print(f"  2-turn:          {two_turn}")
    print(f"  3-turn:          {three_turn}")
    print(f"  Chinese:         {language_counts['zh']}")
    print(f"  English:         {language_counts['en']}")
    print("  Scenarios:")
    for name, count in scenario_counts.items():
        print(f"    {name}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
