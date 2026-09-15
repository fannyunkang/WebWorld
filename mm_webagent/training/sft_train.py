"""SFT entry point for multimodal web action prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_webagent.training.dataset_format import load_jsonl, sft_text
from mm_webagent.training.lora_config import LoraConfigSpec
from mm_webagent.training.vl_dataset_format import to_qwen_vl_messages


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
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--train", action="store_true", help="Launch TRL SFT training.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without launching training.")
    parser.add_argument("--format", choices=["text", "qwen-vl"], default="text")
    args = parser.parse_args()

    preview = inspect_dataset(args.data)
    lora = LoraConfigSpec()
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA: {lora.to_peft_kwargs()}")
    print(f"Preview records: {len(preview)}")
    if args.format == "qwen-vl" and preview:
        print(f"Qwen-VL preview: {json.dumps(to_qwen_vl_messages(preview[0]), ensure_ascii=False)[:1000]}")

    if args.dry_run or not args.train:
        return

    if args.format == "qwen-vl":
        raise SystemExit(
            "Qwen-VL multimodal records are validated in this scaffold. "
            "Use --dry-run or export with `python -m mm_webagent.training.vl_dataset_format`; "
            "wire the exported messages into your Qwen-VL processor/trainer for GPU training."
        )

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install transformers, peft, trl, accelerate, and datasets before --train.") from exc

    rows = load_jsonl(args.data)
    dataset = Dataset.from_dict({"text": [sft_text(row) for row in rows]})
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, device_map="auto")
    peft_config = LoraConfig(task_type="CAUSAL_LM", **lora.to_peft_kwargs())
    train_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_length,
        dataset_text_field="text",
        logging_steps=1,
        save_strategy="epoch",
    )
    trainer = SFTTrainer(
        model=model,
        args=train_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
