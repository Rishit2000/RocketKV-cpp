#!/usr/bin/env python3
"""Generate RULER inputs with the official NVIDIA RULER code vendored in NVlabs/RocketKV.

For every (context length, task) in the config this calls RocketKV's
eval/ruler_utils/data/prepare.py exactly as pipeline/baseline/eval_ruler.py does
(HF tokenizer, model chat template, synthetic benchmark) and writes
<data_dir>/ruler/<context_len>/<task>/validation.jsonl.
prepare.py reports generator failures without a non-zero exit code, so every
output file is checked for the requested sample count.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_suite.config import load_config_file
from eval_suite.datasets import ruler_file

# Source files each RULER task family reads from ruler_utils/data/synthetic/json/.
# (synthetic.yaml: type_haystack=essay)
ESSAY_TASKS = {"niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multivalue", "niah_multiquery"}
SOURCES = {"qa_1": "squad.json", "qa_2": "hotpotqa.json"}


def required_source(task):
    if task in SOURCES:
        return SOURCES[task]
    return "PaulGrahamEssays.json" if task in ESSAY_TASKS else None


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ruler-dir", required=True, type=Path, help="NVlabs/RocketKV eval/ruler_utils")
    args = parser.parse_args()
    config = load_config_file(args.config)
    bench = config["benchmark"]
    if bench["suite"] != "ruler":
        raise ValueError("Config suite must be 'ruler'")
    tasks, lengths, count = bench["tasks"], bench["context_lens"], bench["num_samples"]
    if count < 1:
        raise ValueError("num_samples must be positive for RULER generation")
    template = bench.get("chat_template", "llama3")
    tokenizer = Path(config["model"]["tokenizer"])
    tokenizer = tokenizer if tokenizer.is_absolute() else ROOT / tokenizer
    data_dir = Path(config["management"]["data_dir"])
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    workdir = args.ruler_dir.resolve() / "data"
    if not (workdir / "prepare.py").is_file():
        raise FileNotFoundError(f"{workdir}/prepare.py not found; is --ruler-dir RocketKV's eval/ruler_utils?")
    if not (tokenizer / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Tokenizer not found in {tokenizer}; run scripts/data/download_model.sh")
    for task in tasks:
        source = required_source(task)
        if source and not (workdir / "synthetic/json" / source).is_file():
            raise FileNotFoundError(f"{task} needs synthetic/json/{source}; its download failed")

    # prepare.py and the task generators it spawns call `python`; use this interpreter.
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")
    for length in lengths:
        for task in tasks:
            target = Path(ruler_file(str(data_dir), length, task))
            if target.is_file() and count_lines(target) == count:
                print(f"skip {length}/{task}: {count} samples present", flush=True)
                continue
            target.unlink(missing_ok=True)
            command = [sys.executable, "prepare.py", "--save_dir", str(data_dir / "ruler" / str(length)),
                       "--benchmark", "synthetic", "--task", task,
                       "--tokenizer_path", str(tokenizer), "--tokenizer_type", "hf",
                       "--max_seq_length", str(length), "--model_template_type", template,
                       "--num_samples", str(count)]
            result = subprocess.run(command, cwd=workdir, env=env, capture_output=True, text=True)
            found = count_lines(target) if target.is_file() else 0
            if result.returncode != 0 or found != count:
                sys.stderr.write(result.stdout[-4000:] + result.stderr[-4000:])
                raise RuntimeError(f"{length}/{task}: expected {count} samples, found {found}")
            with open(target, encoding="utf-8") as f:
                lens = [json.loads(line)["length"] for line in f]
            print(f"wrote {length}/{task}: {count} samples, tokens {min(lens)}-{max(lens)}", flush=True)


if __name__ == "__main__":
    main()
