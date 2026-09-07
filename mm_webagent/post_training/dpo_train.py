"""DPO validation scaffold for preferred vs rejected web actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_preference_file(path: str | Path) -> dict[str, int]:
    valid = 0
    invalid = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("prompt") and row.get("chosen") and row.get("rejected"):
                valid += 1
            else:
                invalid += 1
    return {"valid": valid, "invalid": invalid}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DPO preference data for TRL DPOTrainer.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    stats = validate_preference_file(args.data)
    print(f"Model: {args.model}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
