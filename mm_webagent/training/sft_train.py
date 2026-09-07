"""SFT entry point for multimodal web action prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_webagent.training.lora_config import LoraConfigSpec


def inspect_dataset(path: str | Path, limit: int = 3) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qwen-VL LoRA SFT for web actions.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/sft_lora")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without launching training.")
    args = parser.parse_args()

    preview = inspect_dataset(args.data)
    lora = LoraConfigSpec()
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA: {lora.to_peft_kwargs()}")
    print(f"Preview records: {len(preview)}")

    if args.dry_run:
        return

    raise SystemExit(
        "Install transformers, peft, trl, accelerate, and a Qwen-VL processor, "
        "then replace this dry-run scaffold with the project-specific training loop."
    )


if __name__ == "__main__":
    main()
