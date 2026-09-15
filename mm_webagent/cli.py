"""Project command line entry point for the multimodal web-agent scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_webagent.data.training_corpus_builder import build_training_corpus
from mm_webagent.data.trajectory_builder import load_webworld_records, to_grpo_rollout_seed, to_sft_records, write_jsonl
from mm_webagent.eval.agent_benchmark import run_benchmark
from mm_webagent.eval.retrieval_eval import compare, load_documents, load_queries
from mm_webagent.post_training.pipeline import describe_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal WebAgent project runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-data")
    build.add_argument("--input", default="examples/sample_trajectory.json")
    build.add_argument("--sft-output", default="data/mm_webagent/sft_train.jsonl")
    build.add_argument("--grpo-output", default="data/mm_webagent/grpo_rollout_seed.jsonl")

    corpus = sub.add_parser("build-training-corpus")
    corpus.add_argument("--repeats-per-class", type=int, default=10)
    corpus.add_argument("--trajectory-output", default="data/mm_webagent/training_trajectories.json")
    corpus.add_argument("--sft-output", default="data/mm_webagent/sft_train.jsonl")
    corpus.add_argument("--dpo-output", default="data/mm_webagent/dpo_train.jsonl")
    corpus.add_argument("--grpo-output", default="data/mm_webagent/grpo_rollout_seed.jsonl")

    rag = sub.add_parser("eval-graphrag")
    rag.add_argument("--documents", default="examples/operation_memory_docs.json")
    rag.add_argument("--queries", default="examples/operation_memory_eval.json")
    rag.add_argument("--top-k", type=int, default=3)

    benchmark = sub.add_parser("eval-agent-benchmark")
    benchmark.add_argument("--output", default="experiments/agent_benchmark_20class_results.json")

    sub.add_parser("pipeline")
    args = parser.parse_args()

    if args.command == "build-data":
        records = load_webworld_records(args.input)
        write_jsonl(to_sft_records(records), args.sft_output)
        write_jsonl(to_grpo_rollout_seed(records), args.grpo_output)
        print(f"Built {args.sft_output} and {args.grpo_output}")
    elif args.command == "build-training-corpus":
        stats = build_training_corpus(
            repeats_per_class=args.repeats_per_class,
            trajectory_output=args.trajectory_output,
            sft_output=args.sft_output,
            dpo_output=args.dpo_output,
            grpo_output=args.grpo_output,
        )
        print(json.dumps(stats, indent=2))
    elif args.command == "eval-graphrag":
        report = compare(load_documents(args.documents), load_queries(args.queries), args.top_k)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command == "eval-agent-benchmark":
        report = run_benchmark()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"summary": report["summary"], "deltas_vs_direct_policy": report["deltas_vs_direct_policy"]}, indent=2))
    elif args.command == "pipeline":
        print(describe_pipeline())


if __name__ == "__main__":
    main()
