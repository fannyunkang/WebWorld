# world_model_context.py — 世界模型的上下文管理（滑动窗口）

import re
from prompts import WORLD_MODEL_SYSTEM, WORLD_MODEL_FIRST_STEP, WORLD_MODEL_NEXT_STEP
from parser import parse_response


class WorldModelContext:
    """
    管理世界模型的多轮对话上下文，并在超长时自动滑动窗口。

    上下文结构：
        messages[0] (user):      第一步的完整 prompt（含 system + 初始状态 + 第一个动作）
        messages[1] (assistant): 世界模型对第一步的回复
        messages[2] (user):      第二步 prompt
        messages[3] (assistant): 第二步回复
        ...
    """

    def __init__(self, initial_observation: str, window_size: int = 10):
        self.initial_observation = initial_observation
        self.window_size = window_size
        self.messages: list[dict] = []
        self._step_count = 0

    def build_next_input(self, action: str) -> list[dict]:
        """
        添加下一步的 user 消息，返回当前完整的 messages 列表。
        """
        if self._step_count == 0:
            content = WORLD_MODEL_FIRST_STEP.format(
                system=WORLD_MODEL_SYSTEM,
                observation=self.initial_observation,
                action=action,
            )
        else:
            content = WORLD_MODEL_NEXT_STEP.format(action=action)

        self.messages.append({"role": "user", "content": content})
        return self.messages

    def record_response(self, raw_response: str):
        """记录世界模型的回复，并在需要时执行滑动窗口压缩。"""
        self.messages.append({"role": "assistant", "content": raw_response})
        self._step_count += 1
        self._maybe_trim()

    def _maybe_trim(self):
        """当上下文过长时，将最早的 (state, action) 对合并为新的首条消息。"""
        max_messages = self.window_size * 2  # 每步 = 1 user + 1 assistant
        if len(self.messages) <= max_messages:
            return

        # 取第二轮的 assistant 回复作为新的 base state
        second_response = self.messages[1]["content"]
        parsed = parse_response(second_response)
        base_state = parsed.content or self.initial_observation

        # 取第三条 user 消息中的 action
        third_user = self.messages[2]["content"]
        action_match = re.search(r"Action: '(.*?)'", third_user)
        base_action = action_match.group(1) if action_match else "noop()"

        # 重建首条消息
        new_first = WORLD_MODEL_FIRST_STEP.format(
            system=WORLD_MODEL_SYSTEM,
            observation=base_state,
            action=base_action,
        )

        # 裁剪：保留新的首条 + 第四条之后的所有消息
        self.messages = [
            {"role": "user", "content": new_first}
        ] + self.messages[3:]

        print("  ✂️ World model context trimmed (sliding window)")
