# main.py

import re
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig, AgentConfig, WorldModelConfig
from llm_client import LLMClient
from parser import parse_response, extract_user_message, is_terminal_action
from prompts import build_agent_prompt
from world_model_context import WorldModelContext
from trajectory_logger import TrajectoryLogger
from initial_states import PAGES, DEFAULT_PAGE


_DEFAULT = AppConfig()

# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def sanitize_filename(text: str, max_length: int = 30) -> str:
    """将文本转为安全的文件名"""
    for ch in r'/\:*?"<>|\n\r\t':
        text = text.replace(ch, "_")
    return text[:max_length].strip() or "trajectory"


def print_banner(title: str, char: str = "=", width: int = 60):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


# ══════════════════════════════════════════════════════════════
# 命令行参数
# ══════════════════════════════════════════════════════════════

BACKEND_CHOICES = ["dashscope", "dlc", "oai", "openai", "huggingface", "wm"]

def parse_cli_args():
    d_a = _DEFAULT.agent
    d_w = _DEFAULT.world_model

    parser = argparse.ArgumentParser(
        description="Agent ↔ World Model 交互模拟器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ag = parser.add_argument_group("Agent 配置")
    ag.add_argument("--agent-backend",     type=str,   default=d_a.backend,     choices=BACKEND_CHOICES)
    ag.add_argument("--agent-model",       type=str,   default=d_a.model)
    ag.add_argument("--agent-temperature", type=float, default=d_a.temperature)
    ag.add_argument("--agent-max-tokens",  type=int,   default=d_a.max_tokens)

    wg = parser.add_argument_group("World Model 配置")
    wg.add_argument("--wm-backend",     type=str,   default=d_w.backend,     choices=BACKEND_CHOICES)
    wg.add_argument("--wm-model",       type=str,   default=d_w.model)
    wg.add_argument("--wm-temperature", type=float, default=d_w.temperature)
    wg.add_argument("--wm-max-tokens",  type=int,   default=d_w.max_tokens)

    gg = parser.add_argument_group("通用配置")
    gg.add_argument("--page",          type=str,   default=DEFAULT_PAGE, choices=list(PAGES.keys()))
    gg.add_argument("--task",          type=str,   default=None)
    gg.add_argument("--max-steps",     type=int,   default=_DEFAULT.max_steps)
    gg.add_argument("--step-delay",    type=float, default=_DEFAULT.step_delay)
    gg.add_argument("--wm-window",     type=int,   default=_DEFAULT.wm_context_window)
    gg.add_argument("--agent-history", type=int,   default=_DEFAULT.agent_history_window)

    return parser.parse_args()


def build_config_from_args(args) -> AppConfig:
    return AppConfig(
        agent=AgentConfig(
            backend=args.agent_backend,
            model=args.agent_model,
            temperature=args.agent_temperature,
            max_tokens=args.agent_max_tokens,
        ),
        world_model=WorldModelConfig(
            backend=args.wm_backend,
            model=args.wm_model,
            temperature=args.wm_temperature,
            max_tokens=args.wm_max_tokens,
        ),
        max_steps=args.max_steps,
        wm_context_window=args.wm_window,
        agent_history_window=args.agent_history,
        step_delay=args.step_delay,
    )


# ══════════════════════════════════════════════════════════════
# 核心交互循环
# ══════════════════════════════════════════════════════════════

def run(config: AppConfig, initial_page: str, task: str):
    """Agent 决策 → 终止检测 → World Model 预测 → 循环"""

    # ── 初始化组件 ────────────────────────────────────
    agent = LLMClient(
        backend=config.agent.backend,
        model=config.agent.model,
        temperature=config.agent.temperature,
        max_tokens=config.agent.max_tokens,
    )
    world_model = LLMClient(
        backend=config.world_model.backend,
        model=config.world_model.model,
        temperature=config.world_model.temperature,
        max_tokens=config.world_model.max_tokens,
    )
    wm_context = WorldModelContext(initial_page, config.wm_context_window)

    base_name = sanitize_filename(task)
    logger = TrajectoryLogger(base_name)

    current_obs = initial_page
    history = []

    # ── 记录初始状态 ──────────────────────────────────
    logger.title("Agent ↔ World Model Trajectory")
    logger.section("Task: {}".format(task))
    logger.section("Initial Page State")
    logger.observation(current_obs)

    print_banner("CONFIGURATION")
    print("  🤖 Agent:       {}".format(agent))
    print("  🌍 World Model: {}".format(world_model))
    print("  📋 Task:        {}".format(task))

    print_banner("INITIAL PAGE STATE")
    print(current_obs)

    # ── 交互循环 ──────────────────────────────────────
    for step in range(1, config.max_steps + 1):
        print_banner("STEP {}".format(step))
        logger.section("Step {}".format(step))

        # ─────────────────────────────────────────────
        # 阶段 1: Agent 决策
        # ─────────────────────────────────────────────
        print("🤖 Agent thinking...")

        prompt = build_agent_prompt(
            goal=task,
            observation=current_obs,
            history_items=history,
            max_history=config.agent_history_window,
        )

        raw_agent = agent.query(prompt)
        if not raw_agent:
            print("❌ Agent failed to respond. Stopping.")
            logger.finish(task, "Agent failed to respond")
            break


        parsed_agent = parse_response(raw_agent)

        if parsed_agent.reasoning:
            print("\n\033[96m[Agent Reasoning]\n{}\033[0m".format(parsed_agent.reasoning))
            logger.agent_reasoning(parsed_agent.reasoning)

        action = parsed_agent.content
        if not action:
            print("❌ Could not extract action from response:")
            print("   {}".format(raw_agent))
            logger.finish(task, "Failed to extract action")
            break

        print("\n\033[92m[Action] {}\033[0m".format(action))
        logger.action(action)

        # ─────────────────────────────────────────────
        # 阶段 2: 终止检测
        # ─────────────────────────────────────────────
        if is_terminal_action(action):
            if "infeasible" in action:
                msg = "Task declared infeasible by agent"
            else:
                msg = extract_user_message(action) or "Task completed"

            print_banner("🎉 TASK COMPLETED", char="*")
            print("  📝 Task:   {}".format(task))
            print("  🤖 Result: {}".format(msg))
            print_banner("END", char="*")

            logger.finish(task, msg)
            break

        # ── 死循环检测 ────────────────────────────────
        if len(history) >= 3:
            last_actions = [h["action"] for h in history[-3:]]
            if len(set(last_actions)) == 1 and last_actions[0] == action:
                print("\n⚠️ Detected repeated action loop (same action 4 times)!")
                current_obs = current_obs + \
                    "\n\n[SYSTEM WARNING] You have repeated the same action " \
                    "'{}' 4 times with no page change. ".format(action) + \
                    "Please try a DIFFERENT action, or use send_msg_to_user() " \
                    "to report what you've found so far."

        # 记录到 Agent 历史
        history.append({
            "reasoning": parsed_agent.reasoning or "",
            "action": action,
        })

        # ─────────────────────────────────────────────
        # 阶段 3: World Model 预测
        # ─────────────────────────────────────────────
        print("\n🌍 World Model simulating...")

        wm_messages = wm_context.build_next_input(action)
        raw_wm = world_model.query(wm_messages)

        if not raw_wm:
            print("❌ World Model failed to respond. Stopping.")
            logger.finish(task, "World Model failed to respond")
            break


        parsed_wm = parse_response(raw_wm)
        wm_context.record_response(raw_wm)

        if parsed_wm.reasoning:
            print("\n\033[93m[WM Reasoning]\n{}\033[0m".format(parsed_wm.reasoning))
            logger.wm_reasoning(parsed_wm.reasoning)

        # 更新当前页面状态
        current_obs = parsed_wm.content or current_obs

        print("\n\033[90m[New Page State]\n{}\033[0m".format(current_obs))
        logger.observation(current_obs)
        logger.step_divider()

        time.sleep(config.step_delay)

    else:
        print_banner("⚠️ MAX STEPS REACHED")
        logger.finish(task, "Max steps ({}) reached".format(config.max_steps))

    # ── 保存轨迹文件 ──────────────────────────────────
    logger.save()


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_cli_args()
    config = build_config_from_args(args)

    initial_page = PAGES.get(args.page, PAGES[DEFAULT_PAGE])

    if args.task:
        task = args.task
    else:
        task = input("\n>>> 请输入任务 (回车使用默认): ").strip()
        if not task:
            task = "在搜索框输入'春节'并点击百度一下"

    run(config, initial_page, task)


if __name__ == "__main__":
    main()
