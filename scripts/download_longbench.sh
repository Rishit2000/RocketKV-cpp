#!/usr/bin/env bash
# ==============================================================================
# Helper script to download LongBench JSONL datasets from Hugging Face
# ==============================================================================

set -e

DATA_DIR="data/longbench"
mkdir -p "$DATA_DIR"

echo "Downloading official LongBench dataset files into $DATA_DIR..."

if command -v python3 &> /dev/null; then
    python3 - << 'EOF'
import os
import json
from datasets import load_dataset

TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
]

save_dir = "data/longbench"
os.makedirs(save_dir, exist_ok=True)

print(f"Fetching {len(TASKS)} tasks from Hugging Face (THUDM/LongBench)...")
for task in TASKS:
    target_file = os.path.join(save_dir, f"{task}.jsonl")
    if os.path.exists(target_file):
        print(f"  - {task}: already exists, skipping.")
        continue
    print(f"  - Downloading {task}...", end="", flush=True)
    try:
        ds = load_dataset("THUDM/LongBench", task, split="test")
        with open(target_file, "w", encoding="utf-8") as f:
            for item in ds:
                f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
        print(" done.")
    except Exception as e:
        print(f" failed: {e}")

print("LongBench dataset preparation complete.")
EOF
else
    echo "Python 3 is required to download datasets."
    exit 1
fi

