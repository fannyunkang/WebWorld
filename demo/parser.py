# parser.py — 响应解析

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedResponse:
    """解析后的 LLM 响应"""
    reasoning: Optional[str] = None
    content: Optional[str] = None  # action 或 page state


def parse_response(raw_text: str) -> ParsedResponse:
    if not raw_text:
        return ParsedResponse()

    reasoning = None
    for tag in ["reason", "think"]:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", raw_text, re.DOTALL | re.IGNORECASE)
        if m:
            reasoning = m.group(1).strip()
            break

    # 优先提取 <action>（Agent 场景）
    m = re.search(r"<action>(.*?)</action>", raw_text, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        content = re.sub(r"^```(?:python)?", "", content, flags=re.MULTILINE)
        content = re.sub(r"```$", "", content, flags=re.MULTILINE)
        return ParsedResponse(reasoning=reasoning, content=content.strip())

    # World Model 场景：<reason>...</reason> 之后的所有内容就是页面状态
    content = raw_text

    # 移除所有 reasoning 标签及其内容
    for tag in ["reason", "think"]:
        content = re.sub(rf"<{tag}>.*?</{tag}>", "", content, flags=re.DOTALL | re.IGNORECASE)

    # 移除可能残留的标签（如未闭合的 <reason>）
    for tag in ["reason", "think"]:
        content = re.sub(rf"</?{tag}>", "", content, flags=re.IGNORECASE)

    content = content.strip()

    # 如果内容被 ```  包裹，去掉代码围栏
    content = re.sub(r"^```\w*\n?", "", content)
    content = re.sub(r"\n?```$", "", content)
    content = content.strip()

    return ParsedResponse(reasoning=reasoning, content=content if content else None)


def extract_user_message(action_text: str) -> Optional[str]:
    """从 send_msg_to_user('...') 中提取消息文本"""
    m = re.search(r"send_msg_to_user\s*$(?:'|\")(.*?)(?:'|\")$", action_text, re.DOTALL)
    return m.group(1) if m else None


def is_terminal_action(action_text: str) -> bool:
    """判断是否为终止动作"""
    return "send_msg_to_user" in action_text or "infeasible" in action_text
