"""LoRA defaults for Qwen-VL policy fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LoraConfigSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def to_peft_kwargs(self) -> dict:
        values = asdict(self)
        values["lora_alpha"] = values.pop("alpha")
        values["lora_dropout"] = values.pop("dropout")
        values["target_modules"] = list(values["target_modules"])
        return values
