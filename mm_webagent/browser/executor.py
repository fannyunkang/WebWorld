"""Execute validated browser actions on a Playwright page."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from mm_webagent.browser.element_registry import ElementRegistry
from mm_webagent.data.action_schema import ParsedAction


@dataclass
class ExecutionResult:
    success: bool
    message: str = ""
    terminal: bool = False


@dataclass
class PlaywrightActionExecutor:
    page: Any
    element_registry: ElementRegistry = field(default_factory=dict)

    def execute(self, action: ParsedAction) -> ExecutionResult:
        try:
            args, kwargs = _parse_call(action.raw)
            handler = getattr(self, f"_do_{action.name}", None)
            if not handler:
                return ExecutionResult(False, f"Unsupported executable action: {action.name}")
            handler(*args, **kwargs)
            return ExecutionResult(True, terminal=action.is_terminal)
        except Exception as exc:
            return ExecutionResult(False, str(exc), terminal=action.is_terminal)

    def _locator_for_bid(self, bid: str):
        info = self.element_registry.get(str(bid))
        selector = info.selector if info and info.selector else f'[data-mm-bid="{bid}"]'
        return self.page.locator(selector).first

    def _do_click(self, bid: str, button: str = "left", modifiers: list[str] | None = None) -> None:
        self._locator_for_bid(bid).click(button=button, modifiers=modifiers or [])

    def _do_fill(self, bid: str, text: str, press_enter: bool = False) -> None:
        locator = self._locator_for_bid(bid)
        locator.fill(text)
        if press_enter:
            self.page.keyboard.press("Enter")

    def _do_select_option(self, bid: str, options):
        self._locator_for_bid(bid).select_option(options)

    def _do_hover(self, bid: str) -> None:
        self._locator_for_bid(bid).hover()

    def _do_scroll(self, dx: float, dy: float) -> None:
        self.page.mouse.wheel(dx, dy)

    def _do_goto(self, url: str) -> None:
        self.page.goto(url)

    def _do_go_back(self) -> None:
        self.page.go_back()

    def _do_go_forward(self) -> None:
        self.page.go_forward()

    def _do_keyboard_press(self, key: str) -> None:
        self.page.keyboard.press(key)

    def _do_keyboard_type(self, text: str) -> None:
        self.page.keyboard.type(text)

    def _do_tab_new(self) -> None:
        self.page.context.new_page()

    def _do_tab_close(self) -> None:
        self.page.close()

    def _do_send_msg_to_user(self, text: str) -> None:
        return None

    def _do_infeasible(self, reason: str) -> None:
        return None


def _parse_call(raw: str) -> tuple[list[Any], dict[str, Any]]:
    expr = ast.parse(raw, mode="eval").body
    if not isinstance(expr, ast.Call):
        raise ValueError(f"Expected function-style action: {raw}")
    args = [ast.literal_eval(arg) for arg in expr.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords if kw.arg}
    return args, kwargs
