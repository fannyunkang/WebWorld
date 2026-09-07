"""Generate vLLM launch commands for the trained policy model."""

from __future__ import annotations

import argparse


def build_vllm_command(
    model: str,
    served_model_name: str = "mm-webagent",
    port: int = 8000,
    max_model_len: int = 8192,
) -> str:
    return (
        "vllm serve "
        f"{model} "
        f"--served-model-name {served_model_name} "
        f"--port {port} "
        f"--max-model-len {max_model_len}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the vLLM command for policy serving.")
    parser.add_argument("--model", default="outputs/sft_lora")
    parser.add_argument("--served-model-name", default="mm-webagent")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()
    print(build_vllm_command(args.model, args.served_model_name, args.port, args.max_model_len))


if __name__ == "__main__":
    main()
