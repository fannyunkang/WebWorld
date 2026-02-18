# trajectory_logger.py — 轨迹记录（HTML 为主，PDF 可选）

import os
from typing import Optional


class TrajectoryLogger:
    """将交互轨迹同时写入 HTML 和 PDF"""

    def __init__(self, base_name: str = "trajectory"):
        self.html_path = f"{base_name}.html"
        self.pdf_path = f"{base_name}.pdf"
        self._html_parts: list[str] = [self._html_head()]
        self._pdf_story = []  # 延迟初始化 PDF 依赖

    # ── 公开方法 ─────────────────────────────────────

    def title(self, text: str):
        self._html_parts.append(f"<h1>{self._esc(text)}</h1>")

    def section(self, text: str):
        self._html_parts.append(f"<h2>{self._esc(text)}</h2>")

    def agent_reasoning(self, text: str):
        self._html_parts.append(self._card("🤖 Agent Reasoning",
                                           text, "agent-reasoning"))

    def action(self, text: str):
        self._html_parts.append(self._card("⚡ Action",
                                           f"<code>{self._esc(text)}</code>",
                                           "action"))

    def wm_reasoning(self, text: str):
        self._html_parts.append(self._card("🌍 World Model Reasoning",
                                           text, "wm-reasoning"))

    def observation(self, text: str):
        self._html_parts.append(self._card("📄 Page State",
                                           self._esc(text), "observation",
                                           use_pre=True))

    def finish(self, task: str, message: str):
        self._html_parts.append(f"""
        <div class="final-message">
            <h3>🎉 TASK COMPLETED</h3>
            <p><strong>Task:</strong> {self._esc(task)}</p>
            <p><strong>Result:</strong> {self._esc(message)}</p>
        </div>""")

    def step_divider(self):
        self._html_parts.append('<hr class="step-divider">')

    def save(self):
        self._html_parts.append("</div></body></html>")
        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._html_parts))
        print(f"✅ HTML saved: {self.html_path}")

    # ── 私有方法 ─────────────────────────────────────

    def _esc(self, text: str) -> str:
        if not text:
            return ""
        return (str(text)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>"))

    def _card(self, title: str, body: str, css_class: str,
              use_pre: bool = False) -> str:
        tag = "pre" if use_pre else "p"
        return f"""
        <div class="{css_class}">
            <h4>{title}</h4>
            <{tag}>{body}</{tag}>
        </div>"""

    def _html_head(self) -> str:
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent-World Model Trajectory</title>
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     "Microsoft YaHei", sans-serif;
        max-width: 1100px; margin: 0 auto; padding: 20px;
        background: #f5f5f5; line-height: 1.6;
    }
    .container {
        background: #fff; padding: 30px;
        border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,.1);
    }
    h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
    h2 { color: #555; padding: 10px; background: #f0f0f0;
         border-left: 4px solid #2196F3; margin-top: 30px; }
    .agent-reasoning {
        background: #E3F2FD; border-left: 4px solid #2196F3;
        padding: 15px; margin: 10px 0; border-radius: 4px;
    }
    .agent-reasoning h4 { margin-top: 0; color: #1976D2; }
    .action {
        background: #E8F5E9; border-left: 4px solid #4CAF50;
        padding: 15px; margin: 10px 0; border-radius: 4px;
        font-family: 'Courier New', monospace;
    }
    .action h4 { margin-top: 0; color: #388E3C; }
    .wm-reasoning {
        background: #FFF9C4; border-left: 4px solid #FBC02D;
        padding: 15px; margin: 10px 0; border-radius: 4px;
    }
    .wm-reasoning h4 { margin-top: 0; color: #F57F17; }
    .observation {
        background: #FAFAFA; border: 1px solid #ddd;
        padding: 15px; margin: 10px 0; border-radius: 4px;
        font-family: 'Courier New', monospace; font-size: 12px;
        overflow-x: auto; white-space: pre-wrap; word-wrap: break-word;
    }
    .observation h4 {
        margin-top: 0; color: #666;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .final-message {
        background: #FFEBEE; border: 2px solid #F44336;
        padding: 20px; margin: 20px 0; border-radius: 8px; text-align: center;
    }
    .final-message h3 { color: #C62828; margin-top: 0; }
    .step-divider { border-top: 2px dashed #ccc; margin: 30px 0; }
    pre { white-space: pre-wrap; word-wrap: break-word; margin: 0; }
</style>
</head>
<body><div class="container">"""
