# /mnt/data/zikai/WebWorld_Bench/demo/llm_client.py

import sys
from pathlib import Path
from typing import Optional

# demo/ 的上一层就是项目根目录 WebWorld_Bench/
PROJECT_ROOT = Path(__file__).parent.parent  # demo/ → WebWorld_Bench/
sys.path.insert(0, str(PROJECT_ROOT))

from core.serve.unified_api import unified_call


class LLMClient:
    """
    对 unified_call 的薄封装。
      - query("prompt string")         → 单轮，给 Agent 用
      - query([{role, content}, ...])   → 多轮，给 World Model 用
    """

    def __init__(self, backend: str, model: str,
                 temperature: float = 0.7, max_tokens: int = 4096,
                 max_retries: int = 3):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def query(self, messages) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                if isinstance(messages, str):
                    prompt = messages
                else:
                    prompt = self._messages_to_prompt(messages)

                resp = unified_call(
                    backend=self.backend,
                    model=self.model,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    parse_reason=False
                )

                if resp:
                    return resp.strip() if isinstance(resp, str) else str(resp).strip()

            except Exception as e:
                print(f"  ⚠️ Attempt {attempt+1}/{self.max_retries} "
                      f"({self.backend}/{self.model}): {e}")

        return None

    @staticmethod
    def _messages_to_prompt(messages: list) -> str:
        """多轮 messages 拼成单个 prompt"""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"[Previous Response]\n{content}")
        return "\n\n".join(parts)

    def __repr__(self):
        return f"LLMClient({self.backend}/{self.model})"
