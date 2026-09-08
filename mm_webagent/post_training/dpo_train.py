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
    parser.add_argument("--output-dir", default="outputs/dpo_lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    stats = validate_preference_file(args.data)
    print(f"Model: {args.model}")
    print(json.dumps(stats, indent=2))

    if not args.train:
        return

    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit("Install transformers, peft, trl, accelerate, and datasets before --train.") from exc

    dataset = load_dataset("json", data_files=args.data, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, device_map="auto")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
