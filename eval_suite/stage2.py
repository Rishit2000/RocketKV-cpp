#!/usr/bin/env python3
"""Resumable LongBench / RULER comparison: full KV, Stage 1, and Stage 1 + Metal HSA.

Prompts and scoring follow the RocketKV paper code (NVlabs/RocketKV):
  LongBench  raw prompt middle-truncated to context_len HF tokens, llama3 chat template
             except few-shot/code tasks, scored by eval_long_bench.scorer().
  RULER      official RULER inputs used as-is (llama3 template and answer prefix included),
             max_new_tokens for every task, string_match_all / string_match_part.

Each sample is a separate runner invocation, so completed predictions survive an
interruption; re-running the same command resumes. Model loading is in wall time only.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_suite.config import load_config_file
from eval_suite.datasets import apply_longbench_protocol, load_longbench_dataset, load_ruler_dataset
from eval_suite.metrics import score_longbench_sample, score_ruler_sample

MODES = ("off", "stage1", "hsa")
N_CTX_MARGIN = 256  # chat template tokens and HF -> llama.cpp re-tokenization drift


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (no git checkout)"


def build_units(config):
    """Return [(group, task, samples)]; group is "" for LongBench and the context length for RULER."""
    bench = config["benchmark"]
    suite, tasks, count = bench["suite"], bench["tasks"], bench["num_samples"]
    data_dir = str(resolve(config["management"]["data_dir"]))
    if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks) or (count != -1 and count < 1):
        raise ValueError("Use an explicit task list and a positive sample count or -1 for all samples")
    units = []
    if suite == "longbench":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(resolve(config["model"]["tokenizer"])))
        for task in tasks:
            rows = load_longbench_dataset(task, count, data_dir=data_dir)
            if not rows or (count != -1 and len(rows) != count):
                raise ValueError(f"{task}: expected {count} samples, found {len(rows)}")
            rows = apply_longbench_protocol(rows, tokenizer, bench["context_len"], bench.get("chat_template", "llama3"))
            units.append(("", task, rows, bench["context_len"]))
    elif suite == "ruler":
        for length in bench["context_lens"]:
            for task in tasks:
                rows = load_ruler_dataset(task, length, count, data_dir=data_dir,
                                          max_new_tokens=bench.get("max_new_tokens", 128))
                units.append((str(length), task, rows, length))
    else:
        raise ValueError(f"Unsupported suite: {suite}")
    return units


def score(suite, task, record, sample):
    if suite == "ruler":
        return score_ruler_sample(task, record["generated"], sample["answers"])
    return score_longbench_sample(task, record["generated"], sample["answers"], sample.get("all_classes"))


def check_backends(m, mode, stage1_backend):
    if m["effective_mode"] != mode:
        raise RuntimeError("Runner used the wrong mode")
    if mode != "off" and stage1_backend == "metal":
        if m["requested_stage1_backend"] != "metal":
            raise RuntimeError("Runner did not accept Metal Stage 1")
        if m["kept_tokens"] < m["prompt_tokens"] and (
            m["stage1_backend"] != "metal" or m["stage1_capture"] != "metal"
            or m["stage1_capture_metal_calls"] < 1 or m["stage1_capture_host_bytes"] != 0
        ):
            raise RuntimeError("Stage 1 eviction did not use GPU-resident Metal capture and scoring")
    if mode == "hsa" and (m["stage2_backend"] != "metal" or m["model_backend"] != "metal" or
                          m["stage2_layer_calls"] != m["model_layers"] * m["decode_forward_calls"]):
        raise RuntimeError("Metal HSA did not execute in every layer")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--budget", type=int, nargs="+",
                        help="Token budgets for stage1/hsa (default: attention.budget in the config)")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--stage1-backend", choices=("cpu", "metal"), default="cpu",
                        help="Stage 1 scoring backend for stage1 and hsa runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load and format all inputs and print the plan; writes nothing")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = load_config_file(args.config)
    suite = config["benchmark"]["suite"]
    attention = config["attention"]
    budgets = args.budget or attention["budget"]
    budgets = budgets if isinstance(budgets, list) else [budgets]
    modes = MODES if args.mode == "all" else (args.mode,)
    if not budgets or min(budgets) < 1 or args.threads < 1:
        raise ValueError("Budgets and thread count must be positive")
    # One run per (mode, budget); full KV ignores the budget and runs once.
    runs = [(mode, None if mode == "off" else b) for mode in modes for b in ([None] if mode == "off" else budgets)]

    units = build_units(config)
    model = resolve(config["model"]["path"])
    runner = resolve(config["model"]["runner"])
    if not args.dry_run:
        # Require the official scoring libraries rather than any fallback.
        if suite == "longbench":
            for name in ("rouge", "fuzzywuzzy", "python-Levenshtein"):
                importlib.metadata.version(name)
        if not model.is_file() or not runner.is_file():
            raise FileNotFoundError(f"Missing model ({model}) or runner ({runner}); "
                                    "run scripts/data/download_model.sh and scripts/stage2.sh build")
    if args.dry_run:
        n = sum(len(rows) for _, _, rows, _ in units)
        print(f"{suite} dry run: {n} samples x {len(runs)} runs = {n * len(runs)} runner calls", flush=True)
        for g, t, rows, length in units:
            max_tokens = max(r["tokens_to_generate"] for r in rows)
            chars = sorted(len(r["prompt"]) for r in rows)
            print(f"  {g + '/' if g else ''}{t}: {len(rows)} samples, n_ctx={length + max_tokens + N_CTX_MARGIN}, "
                  f"max_tokens={max_tokens}, prompt chars median={chars[len(chars) // 2]} max={chars[-1]}")
        return
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Prevent two processes from writing the same experiment concurrently.
    lock = (output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    samples = {f"{g}/{t}" if g else t: rows for g, t, rows, _ in units}
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    manifest = {
        "suite": suite, "config": {k: v for k, v in config.items() if k != "model"},
        "model": str(model), "model_size": model.stat().st_size if model.is_file() else None,
        "driver_sha256": digest(Path(__file__)),
        "datasets_sha256": digest(ROOT / "eval_suite/datasets.py"),
        "scoring_sha256": digest(ROOT / "eval_suite/metrics.py"),
        "runner": str(runner), "runner_sha256": digest(runner),
        "git_commit": git_commit(),
        "units": [f"{g}/{t}" if g else t for g, t, _, _ in units],
        "samples_per_unit": {k: len(v) for k, v in samples.items()},
        "runs": [f"{m}" if b is None else f"{m}_b{b}" for m, b in runs],
        "stage1_backend": args.stage1_backend, "hsa_backend": "metal",
        "threads": args.threads, "gpu_layers": 99, "kv_type": "f16",
        "window_size": attention.get("window_size", 32), "kernel_size": attention.get("kernel_size", 63),
        "n_ctx_rule": f"context_len + max_tokens + {N_CTX_MARGIN}; runner truncation disabled (--context-len 0)",
        "inputs_sha256": hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
        "checkpoint_policy": "one subprocess and one scored checkpoint per sample",
    }
    manifest = json.loads(json.dumps(manifest))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if {k: v for k, v in old.items() if k != "runs"} != {k: v for k, v in manifest.items() if k != "runs"}:
            raise ValueError("Existing experiment has different inputs, binary, or settings; use a new output directory")
        manifest["runs"] = old["runs"] + [r for r in manifest["runs"] if r not in old["runs"]]
    save(manifest_path, manifest)
    save(output / "samples.json", samples)
    total = sum(len(rows) for _, _, rows, _ in units) * len(runs)
    print(f"{suite}: {len(units)} task groups, {total // len(runs)} samples x {len(runs)} runs "
          f"({', '.join(manifest['runs'])}) = {total} runner calls -> {output}", flush=True)

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    completed = 0
    started = time.time()
    status = {"state": "running", "pid": os.getpid(), "started_at": started, "completed": 0, "total": total}
    save(output / "status.json", status)
    try:
        for mode, budget in runs:
            run = mode if budget is None else f"{mode}_b{budget}"
            summary.setdefault(run, {})
            for group, task, rows, length in units:
                unit = f"{group}/{task}" if group else task
                directory = output / run / unit
                directory.mkdir(parents=True, exist_ok=True)
                records = []
                for sample in rows:
                    key = f"{sample['sample_id']:03d}"
                    checkpoint = directory / f"{key}.json"
                    status.update(run=run, unit=unit, sample_id=sample["sample_id"], completed=completed, updated_at=time.time())
                    save(output / "status.json", status)
                    if checkpoint.exists():
                        record = json.loads(checkpoint.read_text())
                    else:
                        n_ctx = length + sample["tokens_to_generate"] + N_CTX_MARGIN
                        request = [{"sample_id": sample["sample_id"], "prompt": sample["prompt"], "max_tokens": sample["tokens_to_generate"]}]
                        source, target = directory / "input.json", directory / "pending.json"
                        save(source, request)
                        command = [str(runner), "--model", str(model), "--batch", str(source), "--output", str(target),
                                   "--rocket-mode", mode, "--stage2-backend", "metal" if mode == "hsa" else "cpu-reference",
                                   "--kv-type", "f16", "--gpu-layers", "99", "--threads", str(args.threads),
                                   "--budget", str(budget or budgets[0]), "--n-ctx", str(n_ctx), "--context-len", "0",
                                   "--window-size", str(manifest["window_size"]), "--kernel-size", str(manifest["kernel_size"])]
                        if mode != "off":
                            command.extend(["--stage1-backend", args.stage1_backend])
                        begin = time.monotonic()
                        with (directory / f"{key}.log").open("w") as log:
                            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
                        wall = time.monotonic() - begin
                        results = json.loads(target.read_text())
                        if len(results) != 1 or results[0]["sample_id"] != sample["sample_id"] or results[0]["status"] != "ok":
                            raise RuntimeError("Runner returned a failed or mismatched sample")
                        record = results[0]
                        check_backends(record["metrics"], mode, args.stage1_backend)
                        record.update(task=task, group=group, budget=budget, references=sample["answers"], wall_seconds=wall,
                                      score=score(suite, task, record, sample))
                        save(checkpoint, record)
                        target.unlink()
                        source.unlink()
                    records.append(record)
                    completed += 1
                    inference = sum((r["metrics"]["ttft_ms"] + r["metrics"]["ms_per_token"] * r["metrics"]["decode_tokens"]) / 1000 for r in records)
                    summary[run][unit] = {"samples": len(records), "score_pct": 100 * sum(r["score"] for r in records) / len(records),
                                          "wall_seconds": sum(r["wall_seconds"] for r in records), "inference_seconds": inference}
                    save(summary_path, summary)
                    status.update(completed=completed, updated_at=time.time(), elapsed_seconds=time.time() - started)
                    save(output / "status.json", status)
                    print(f"[{completed}/{total}] {run} {unit}/{key}: score={record['score']:.4f} wall={record['wall_seconds']:.1f}s", flush=True)
        status.update(state="complete", finished_at=time.time())
        save(output / "status.json", status)
        print(f"Completed {total} evaluations. Results: {output}", flush=True)
    except BaseException as error:
        status.update(state="failed", error=str(error), finished_at=time.time())
        save(output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
