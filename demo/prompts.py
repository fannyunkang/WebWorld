# prompts.py — 所有提示词模板

WORLD_MODEL_SYSTEM = """You are a web world model. I will provide you with an \
initial page state and a sequence of actions.
You must first analyze how the action changes the page state inside \
<reason>...</reason> tags.
Then, strictly predict the resulting full page state.

## Output Format:
First, analyze the action inside <reason>...</reason> tags.
Then output the full predicted page state DIRECTLY.
"""

WORLD_MODEL_FIRST_STEP = """{system}

Initial Page State:
{observation}

First Action: '{action}'

Next Page State:"""

WORLD_MODEL_NEXT_STEP = """Continue the trajectory. Given the previous state, \
predict the next page state after this action.

Action: '{action}'

Next Page State:"""

# ─────────────────────────────────────────────────────────────

AGENT_HEADER = """\
You are an agent trying to solve a web task based on the content of the page and
user instructions. You can interact with the page and explore, and send messages
to the user. Each time you submit an action it will be sent to the browser and
you will receive a new page.

# Goal:
{goal}

# Current page state:
{observation}

# Interaction history:
{history}
"""

AGENT_ACTION_SPACE = """\
# Action space:
Note: This action set allows you to interact with your environment. Most of them
are python function executing playwright code. The primary way of referring to
elements in the page is through bid which are specified in your observations.

21 different types of actions are available.

noop(wait_ms: float = 1000)
mouse_move(x: float, y: float)
mouse_click(x: float, y: float, button: Literal['left', 'middle', 'right'] = 'left')
mouse_dblclick(x: float, y: float, button: Literal['left', 'middle', 'right'] = 'left')
mouse_down(x: float, y: float, button: Literal['left', 'middle', 'right'] = 'left')
mouse_up(x: float, y: float, button: Literal['left', 'middle', 'right'] = 'left')
scroll(delta_x: float, delta_y: float)
click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list = [])
keyboard_press(key: str)
keyboard_type(text: str)
fill(bid: str, value: str, enable_autocomplete_menu: bool = False)
hover(bid: str)
tab_focus(index: int)
new_tab()
go_back()
go_forward()
goto(url: str)
tab_close()
select_option(bid: str, options: str | list[str])
send_msg_to_user(text: str)
report_infeasible(reason: str)

Only a single action can be provided at once. Example:
fill('b534', 'Montre', True)

# Response format:
<reason>
Think step by step. Describe what you see and what you plan to do.
</reason>

<action>
One single action to be executed.
</action>
"""


def build_agent_prompt(goal: str, observation: str,
                       history_items: list, max_history: int = 10) -> str:
    """组装 Agent 的完整提示词"""
    recent = history_items[-max_history:]
    offset = max(0, len(history_items) - max_history)

    lines = []
    for i, item in enumerate(recent):
        lines.append(f"## step {offset + i}")
        if item.get("reasoning"):
            lines.append(f"**Reasoning:**\n{item['reasoning']}")
        lines.append(f"**Action:**\n{item['action']}\n")

    header = AGENT_HEADER.format(
        goal=goal,
        observation=observation,
        history="\n".join(lines),
    )
    return header + AGENT_ACTION_SPACE
