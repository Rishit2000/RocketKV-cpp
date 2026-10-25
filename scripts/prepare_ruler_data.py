#!/usr/bin/env python3
"""
Pre-generate static JSONL dataset files for the complete NVIDIA RULER benchmark suite.
Generates all 13 official RULER tasks (500 samples per task) into data/ruler/.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_suite.datasets import generate_ruler_dataset
from eval_suite.engine import ALL_RULER_TASKS

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-generate static JSONL datasets for NVIDIA RULER benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--save-dir", type=str, default="data/ruler",
                        help="Target directory to save RULER JSONL files")
    parser.add_argument("--context-len", type=int, default=8192,
                        help="Context length (number of input tokens) for the benchmark")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of samples to generate per task (official RULER uses 500)")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated tasks to generate or 'all'")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.tasks == "all":
        tasks_to_generate = ALL_RULER_TASKS
    else:
        tasks_to_generate = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print("=" * 80)
    print("NVIDIA RULER DATASET PREPARATION")
    print(f"Target Directory: {os.path.abspath(args.save_dir)}")
    print(f"Context Length: {args.context_len} tokens | Samples per Task: {args.num_samples}")
    print(f"Tasks ({len(tasks_to_generate)}): {', '.join(tasks_to_generate)}")
    print("=" * 80)

    total_samples = 0
    for task in tasks_to_generate:
        print(f"  Generating {task} ({args.num_samples} samples at {args.context_len} tokens)...", end="", flush=True)
        samples = generate_ruler_dataset(
            task=task,
            target_tokens=args.context_len,
            num_samples=args.num_samples
        )

        out_path = os.path.join(args.save_dir, f"{task}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                record = {
                    "index": s["sample_id"] - 1,
                    "input": s["prompt"],
                    "outputs": s["answers"],
                    "task": task,
                    "depth": s.get("depth", 0.5),
                    "metric_type": s.get("metric_type", "em"),
                    "context_length_chars": s.get("context_length_chars", len(s["prompt"]))
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_samples += len(samples)
        file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f" Done. ({file_size_mb:.2f} MB saved to {os.path.basename(out_path)})")

    print("\n" + "=" * 80)
    print(f"Successfully generated {total_samples} total RULER benchmark samples in '{args.save_dir}'.")
    print("=" * 80)

if __name__ == "__main__":
    main()

