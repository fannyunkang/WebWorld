"""Qwen-VL style multimodal dataset formatting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def vl_user_text(row: dict[str, Any]) -> str:
    observation = row.get("observation", {})
    history = row.get("history", [])
    return (
        "You are a multimodal web operation Agent. Select exactly one valid browser action.\n\n"
        f"Instruction:\n{row.get('instruction', '')}\n\n"
        f"URL:\n{observation.get('url', '')}\n\n"
        f"Page State:\n{observation.get('page_state', '')}\n\n"
        f"History:\n{json.dumps(history, ensure_ascii=False)}\n\n"
        "Return <reason>brief plan</reason> and <action>one_valid_action(...)</action>."
    )


def to_qwen_vl_messages(row: dict[str, Any]) -> dict[str, Any]:
    observation = row.get("observation", {})
    screenshot = observation.get("screenshot")
    content = []
    if screenshot:
        content.append({"type": "image", "image": screenshot})
    content.append({"type": "text", "text": vl_user_text(row)})
    return {
        "id": row.get("id"),
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": f"<action>{row.get('target_action', '')}</action>"},
        ],
    }


def convert_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    count = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_path).open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            dst.write(json.dumps(to_qwen_vl_messages(json.loads(line)), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert SFT JSONL to Qwen-VL multimodal messages.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count = convert_jsonl(args.input, args.output)
    print(json.dumps({"output": args.output, "records": count}, indent=2))


if __name__ == "__main__":
    main()
