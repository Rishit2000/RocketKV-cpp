#!/usr/bin/env python3
"""
RocketKV-llama Evaluation Runner
Streamlined CLI driver that loads all experiment parameters from a single JSON configuration file.
Usage:
    python eval.py configs/longbench.json
"""

import os
import sys
import argparse
from eval_suite import (
    load_config_file,
    build_attention_modes,
    run_evaluation
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RocketKV benchmark evaluation using a single JSON configuration file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/longbench.json",
        help="Path to JSON configuration file (e.g. configs/longbench.json, configs/ruler.json)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config_file(args.config)

    model_path = config["model"]["path"]
    runner_path = config["model"]["runner"]
    max_tokens = int(config["model"]["max_tokens"])
    n_ctx = int(config["model"].get("n_ctx", 9216))
    benchmark = config["benchmark"]["suite"]
    tasks = config["benchmark"]["tasks"]
    context_len = int(config["benchmark"]["context_len"])
    num_samples = int(config["benchmark"]["num_samples"])
    output_dir = config["management"]["output_dir"]
    data_dir = config["management"].get("data_dir", "data")

    # Validate execution binary and model file
    if not os.path.exists(runner_path):
        print(f"Error: Runner binary not found at '{runner_path}'. Run 'make runner' first.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"\nError: Model file '{model_path}' not found.")
        sys.exit(1)

    attn_modes = build_attention_modes(
        attn_type=config["attention"]["attn_type"],
        kv_type=config["attention"]["kv_type"],
        budget=config["attention"]["budget"],
        window_size=int(config["attention"].get("window_size", 32)),
        kernel_size=int(config["attention"].get("kernel_size", 63))
    )

    run_evaluation(
        model_path=model_path,
        benchmark=benchmark,
        task=tasks,
        attn_modes=attn_modes,
        context_len=context_len,
        num_samples=num_samples,
        max_tokens=max_tokens,
        runner_path=runner_path,
        output_dir=output_dir,
        n_ctx=n_ctx,
        data_dir=data_dir
    )

if __name__ == "__main__":
    main()
