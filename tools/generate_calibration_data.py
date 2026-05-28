#!/usr/bin/env python3
"""
Generate multi-turn conversation data using DeepSeek V4 API for quantization calibration.

Generates 512 diverse multi-turn conversations covering daily chat, data analysis,
math, programming, science, and creative writing. Includes both thinking mode
(reasoning_effort: max/high) and non-thinking mode data.

Output: JSONL file with {"messages": [...]} format compatible with
deepseek_v4_w8a8.py's preprocess function (encode_messages with thinking_mode="thinking").

Usage:
    export DEEPSEEK_API_KEY="your-api-key"
    python generate_calibration_data.py [--output calibration_data.jsonl] [--num-samples 512]
"""

import os
import json
import random
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI


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
]

USER_PROMPTS = [
    # === Daily Chat & Lifestyle (15) ===
    "今天天气真好，适合出去做什么户外活动？",
    "推荐几道简单又营养的家常菜，要详细的做法步骤。",
    "如何养好一盆多肉植物？从浇水到光照都说说。",
    "最近有什么值得看的电影或电视剧推荐吗？",
    "教我几个科学有效的提高睡眠质量的方法。",
    "怎么挑选新鲜的水果和蔬菜？有什么小技巧？",
    "推荐一个适合周末短途旅行的地方，说说理由。",
    "如何高效地整理房间并保持长期整洁？",
    "咖啡和茶各有什么健康益处和注意事项？",
    "怎样科学地培养早起的习惯？",
    "有哪些简单有效的居家健身方法？",
    "如何制定一个合理的个人理财计划？",
    "推荐几本值得反复阅读的经典书籍。",
    "如何挑选适合自己的运动鞋？",
    "怎样做出一杯好喝的手冲咖啡？",

    # === Data Analysis & Statistics (15) ===
    "分析一下电商平台用户留存率下降的可能原因及解决方案。",
    "如何用Python对销售数据进行季节性分解？请给出代码示例。",
    "详细解释A/B测试的统计学原理和实际应用步骤。",
    "什么是混淆矩阵？如何用它全面评估分类模型的性能？",
    "处理数据集中缺失值的常用方法有哪些？各有什么适用场景？",
    "详细解释PCA降维的数学原理、计算步骤和应用场景。",
    "如何使用SQL进行用户行为漏斗分析？写出具体查询。",
    "解释时间序列分析中的ARIMA模型及其参数选择方法。",
    "如何全面评估推荐系统的效果？列出所有关键指标。",
    "数据可视化中箱线图和直方图各适合什么场景？举例说明。",
    "解释随机森林和梯度提升树的区别及各自优缺点。",
    "什么是过拟合？在机器学习中如何检测和防止过拟合？",
    "如何用统计学方法检测数据中的异常值？",
    "解释贝叶斯统计与频率统计的核心区别。",
    "什么是因果推断？介绍几种常用的因果推断方法。",

    # === Mathematics (15) ===
    "求解微分方程 dy/dx + 2xy = x 的通解，写出详细步骤。",
    "计算矩阵 [[3,1,0],[1,2,1],[0,1,3]] 的特征值和特征向量。",
    "用多种方法证明根号2是无理数。",
    "用泰勒展开近似计算 sin(0.1)，要求误差小于10^-6。",
    "解释贝叶斯定理的推导过程并用两个实际例子说明。",
    "求解不定积分 ∫ x^2 * e^x dx，展示分部积分法的运用。",
    "圆内接正n边形的面积公式推导及其极限情况。",
    "什么是马尔可夫链？详细解释其数学定义和实际应用。",
    "用梯度下降法求 f(x,y)=x^2+3y^2+2xy-4x-6y 的最小值。",
    "详细解释中心极限定理及其在统计推断中的重要性。",
    "证明欧拉公式 e^(iπ) + 1 = 0。",
    "解释傅里叶变换的数学原理和工程应用。",
    "什么是群论？它在数学和物理中有什么应用？",
    "求解线性规划问题的单纯形法步骤。",
    "解释信息论中的熵概念及其数学定义。",

    # === Programming & Software Engineering (15) ===
    "Python中装饰器的底层原理是什么？给出几个实用例子。",
    "如何用Docker和Docker Compose容器化一个完整的Web应用？",
    "详细解释RESTful API的设计原则和最佳实践。",
    "设计模式中的工厂模式和单例模式分别适用于什么场景？",
    "Git分支管理的最佳实践是什么？如何处理合并冲突？",
    "Python的asyncio事件循环是如何工作的？",
    "详细解释数据库索引的B+树结构和查询优化原理。",
    "微服务架构的优缺点及何时应该使用单体架构？",
    "React中状态管理方案（Context、Redux、Zustand）的对比。",
    "解释TCP/IP协议栈各层的功能及TCP拥塞控制算法。",
    "什么是CAP定理？在分布式系统设计中如何权衡？",
    "解释垃圾回收机制在不同编程语言中的实现。",
    "如何进行有效的代码审查？列出关键检查点。",
    "OAuth 2.0和JWT的区别及各自适用场景。",
    "解释Kubernetes的核心概念和架构设计。",

    # === Science & Nature (15) ===
    "详细解释量子纠缠的概念、实验验证和应用前景。",
    "CRISPR基因编辑技术的原理、应用和伦理问题。",
    "黑洞的形成过程和霍金辐射的物理意义。",
    "全球气候变暖的科学证据和主要影响。",
    "解释狭义相对论中的时间膨胀和长度收缩效应。",
    "DNA复制、转录和翻译的完整分子生物学过程。",
    "暗物质和暗能量的观测证据及理论假设。",
    "光合作用中光反应和暗反应的详细生化过程。",
    "板块构造学说如何解释地震和火山的分布？",
    "人体免疫系统的先天免疫和适应性免疫机制。",
    "什么是超导现象？解释BCS理论。",
    "核聚变与核裂变的区别及可控核聚变的挑战。",
    "解释进化论的自然选择机制和现代综合进化论。",
    "什么是干细胞？它们在医学上有什么应用前景？",
    "地震波的类型及其如何用于探测地球内部结构。",

    # === Creative & Writing (10) ===
    "写一首关于秋天的七言律诗，要有意境。",
    "帮我构思一个关于人工智能觉醒的科幻短篇情节。",
    "如何写出一篇逻辑严密、有说服力的议论文？",
    "创作一段描写未来赛博朋克城市的文字。",
    "讲一个关于勇气与自我成长的寓言故事。",
    "如何塑造一个令人难忘的小说反派角色？",
    "写一篇关于'时间'的散文开头。",
    "如何用'冰山理论'写一个短篇小说？",
    "创作一首关于星空和梦想的现代诗。",
    "为一段紧张的动作场景写一个电影分镜描述。",
]

FOLLOWUP_PROMPTS = [
    "能更详细地解释一下吗？",
    "给我一个具体的例子来说明。",
    "这个和之前讨论的有什么内在联系？",
    "还有其他替代方法或观点吗？",
    "用更简单通俗的方式重新解释一遍。",
    "这个在实际生活或工作中怎么应用？",
    "对比一下不同方案的优缺点。",
    "有没有相关的历史背景或发展历程？",
    "这个结论有什么局限性或前提条件？",
    "推荐一些进一步深入学习的资源。",
    "从相反的角度来分析一下这个问题。",
    "请用具体数据或研究来支撑你的观点。",
    "如果前提条件改变了，结论会怎么变？",
    "初学者应该从哪里入手学习这个领域？",
    "总结一下最关键的3-5个要点。",
    "这个领域目前最新的研究进展是什么？",
    "能给我一个循序渐进的学习路线吗？",
    "在极端情况下这个理论还成立吗？",
    "结合当前热点新闻，你怎么看这个问题？",
    "如果我想深入研究，需要掌握哪些前置知识？",
]

THIRD_TURN_PROMPTS = [
    "感谢你的详细解答！我还有一个相关的问题想请教。",
    "你刚才提到的观点很有意思，能再展开讲讲吗？",
    "我按照你的建议尝试了一下，遇到一个问题想请教。",
    "那如果我想更进一步深入学习，应该怎么做？",
    "你讲的内容和另一个概念有什么区别和联系？",
    "能不能帮我总结一个具体的行动计划或步骤？",
    "这个领域有哪些常见的误区需要特别避免？",
    "未来3-5年这个领域会怎么发展？你怎么看趋势？",
    "有没有适合新手的实践项目可以推荐？",
    "最后一个问题：这个领域的终极目标或前沿是什么？",
]


# ===========================================================================
# Conversation Generator
# ===========================================================================

class ConversationGenerator:
    """Generate multi-turn conversations using DeepSeek V4 API."""

    def __init__(self, model: str = "deepseek-v4-pro"):
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

    def generate_one(self, sample_id: int) -> Optional[dict]:
        """Generate a single multi-turn conversation.

        Returns:
            dict with "messages" (list of role/content dicts) and "meta" (metadata),
            or None if generation failed.
        """
        system_prompt = random.choice(SYSTEM_PROMPTS)
        user_prompt = random.choice(USER_PROMPTS)

        # Mode distribution: 40% thinking-max, 30% thinking-high, 30% non-thinking
        mode_rand = random.random()
        if mode_rand < 0.4:
            thinking_mode = "thinking"
            reasoning_effort = "max"
        elif mode_rand < 0.7:
            thinking_mode = "thinking"
            reasoning_effort = "high"
        else:
            thinking_mode = "non-thinking"
            reasoning_effort = None

        # Turn distribution: 50% 2-turn, 50% 3-turn
        num_turns = 3 if random.random() < 0.5 else 2

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
            followup = self._make_followup(user_prompt)
            messages.append({"role": "user", "content": followup})
            response = self._call_api(messages, thinking_mode, reasoning_effort)
            if response is None:
                return None
            assistant_msg2 = self._build_assistant_msg(response, thinking_mode)
            messages.append(assistant_msg2)

            # --- Third turn (optional) ---
            if num_turns == 3:
                third_prompt = random.choice(THIRD_TURN_PROMPTS)
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
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": 4096,
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
    def _make_followup(original_prompt: str) -> str:
        """Generate a context-aware followup prompt."""
        followup = random.choice(FOLLOWUP_PROMPTS)
        # Occasionally prepend topic reference for continuity (~30% chance)
        if random.random() < 0.3:
            topic = original_prompt[:30].rstrip("？?。.")
            followup = f"关于'{topic}'这个，{followup}"
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
        "--num-samples", type=int, default=512,
        help="Number of conversation samples to generate.",
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

    generator = ConversationGenerator(model=args.model)
    results: list = []
    failed: int = 0

    print(f"Generating {args.num_samples} conversations "
          f"with {args.max_workers} workers...")
    print(f"Output: {args.output}\n")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(generator.generate_one, i): i
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

    print(f"\nDistribution:")
    print(f"  Thinking (max):  {thinking_max}")
    print(f"  Thinking (high): {thinking_high}")
    print(f"  Non-thinking:    {non_thinking}")
    print(f"  2-turn:          {two_turn}")
    print(f"  3-turn:          {three_turn}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
