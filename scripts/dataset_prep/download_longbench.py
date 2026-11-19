#!/usr/bin/env python3
"""Prepare LongBench data in the upstream RocketKV layout (dataset/longbench/<task>).

Also keeps a JSONL copy under data/longbench for the legacy eval_suite path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval_suite.engine import LONGBENCH_EN_TASKS

URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"


def write_hf_disk(task: str, rows: list, dest: Path):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit("Install datasets to write HF disk format: pip install datasets") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(dest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded official data.zip")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/longbench", help="JSONL output directory")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "dataset/longbench",
        help="Upstream-compatible HF disk output directory",
    )
    parser.add_argument("--hf-hub", action="store_true", help="Use datasets.load_dataset('THUDM/LongBench') like upstream")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_hub:
        from datasets import load_dataset
        for task in LONGBENCH_EN_TASKS:
            data = load_dataset("THUDM/LongBench", task, split="test")
            data.save_to_disk(str(args.dataset_dir / task))
            rows = [dict(row) for row in data]
            (args.data_dir / f"{task}.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            print(f"{task}: {len(rows)} samples")
        return

    archive = args.archive or args.data_dir.parent / "longbench-data.zip"
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        partial = archive.with_suffix(".partial")
        urllib.request.urlretrieve(URL, partial)
        partial.replace(archive)

    counts = {}
    with zipfile.ZipFile(archive) as source:
        for task in LONGBENCH_EN_TASKS:
            matches = [name for name in source.namelist() if name == f"{task}.jsonl" or name.endswith(f"/{task}.jsonl")]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one dataset member for {task}: {matches}")
            content = source.read(matches[0])
            rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
            if not rows or any(not row.get("answers") for row in rows):
                raise ValueError(f"Empty dataset or missing answers: {task}")
            (args.data_dir / f"{task}.jsonl").write_bytes(content)
            write_hf_disk(task, rows, args.dataset_dir / task)
            counts[task] = len(rows)
            print(f"{task}: {len(rows)} samples")
    (args.data_dir / "source.json").write_text(
        json.dumps({"url": URL, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "tasks": counts}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
