"""Project command line entry point for the multimodal web-agent scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_webagent.data.trajectory_builder import load_webworld_records, to_grpo_rollout_seed, to_sft_records, write_jsonl
from mm_webagent.eval.retrieval_eval import compare, load_documents, load_queries
from mm_webagent.post_training.pipeline import describe_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal WebAgent project runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-data")
    build.add_argument("--input", default="examples/sample_trajectory.json")
    build.add_argument("--sft-output", default="data/mm_webagent/sft_train.jsonl")
    build.add_argument("--grpo-output", default="data/mm_webagent/grpo_rollout_seed.jsonl")

    rag = sub.add_parser("eval-graphrag")
    rag.add_argument("--documents", default="examples/operation_memory_docs.json")
    rag.add_argument("--queries", default="examples/operation_memory_eval.json")
    rag.add_argument("--top-k", type=int, default=3)

    sub.add_parser("pipeline")
    args = parser.parse_args()

    if args.command == "build-data":
        records = load_webworld_records(args.input)
        write_jsonl(to_sft_records(records), args.sft_output)
        write_jsonl(to_grpo_rollout_seed(records), args.grpo_output)
        print(f"Built {args.sft_output} and {args.grpo_output}")
    elif args.command == "eval-graphrag":
        report = compare(load_documents(args.documents), load_queries(args.queries), args.top_k)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command == "pipeline":
        print(describe_pipeline())


if __name__ == "__main__":
    main()
