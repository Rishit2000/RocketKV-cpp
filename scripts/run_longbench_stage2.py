#!/usr/bin/env python3
"""Resumable LongBench comparison: full KV, Stage 1, and Metal HSA.

Each sample is a separate runner invocation so completed predictions survive an
interruption. Model loading overhead is included in wall time, not inference time.
Scoring reuses this repository's LongBench metrics and prediction cleanup.
"""
import argparse
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
from eval_suite.datasets import load_longbench_dataset
from eval_suite.metrics import score_longbench_sample

MODES = ("off", "stage1", "hsa")


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/longbench.json")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = load_config_file(args.config)
    tasks = config["benchmark"]["tasks"]
    count = config["benchmark"]["num_samples"]
    if not isinstance(tasks, list) or len(set(tasks)) != len(tasks) or count < 1:
        raise ValueError("Use an explicit task list and a positive sample count")
    if args.budget < 1 or args.threads < 1:
        raise ValueError("Budget and thread count must be positive")
    # Require the scoring libraries rather than silently using fallback metrics.
    versions = {name: importlib.metadata.version(name) for name in ("rouge", "fuzzywuzzy", "python-Levenshtein")}
    model = Path(config["model"]["path"]).resolve()
    runner = Path(config["model"]["runner"]).resolve()
    if not model.is_file() or not runner.is_file():
        raise FileNotFoundError("Build the runner and download the model first")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Prevent two processes from writing the same experiment concurrently.
    import fcntl
    lock = (output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    samples = {}
    for task in tasks:
        samples[task] = load_longbench_dataset(task, count, data_dir=config["management"]["data_dir"])
        if len(samples[task]) != count:
            raise ValueError(f"{task}: expected {count} samples, found {len(samples[task])}")
    manifest = {
        "model": str(model), "model_size": model.stat().st_size, "model_mtime_ns": model.stat().st_mtime_ns,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scoring_sha256": hashlib.sha256((ROOT / "eval_suite/metrics.py").read_bytes()).hexdigest(),
        "runner": str(runner), "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "tasks": tasks, "samples_per_task": count, "modes": MODES,
        "budget": args.budget, "threads": args.threads, "gpu_layers": 99, "kv_type": "f16",
        "n_ctx": config["model"]["n_ctx"], "context_len": config["benchmark"]["context_len"],
        "window_size": config["attention"].get("window_size", 32),
        "kernel_size": config["attention"].get("kernel_size", 63),
        "prompt_format": "repository LongBench templates, no additional chat wrapper",
        "scoring": "eval_suite.metrics.score_longbench_sample (repository prediction cleanup)",
        "scoring_packages": versions,
        "inputs_sha256": hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
        "checkpoint_policy": "one subprocess and one scored checkpoint per sample",
    }
    # JSON normalization makes tuple/list comparisons stable when resuming.
    manifest = json.loads(json.dumps(manifest))
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing experiment has different inputs, binary, or settings; use a new output directory")
    save(manifest_path, manifest)
    save(output / "samples.json", samples)
    total = len(tasks) * count * len(MODES)
    completed = 0
    summary = {}
    started = time.time()
    status = {"state": "running", "pid": os.getpid(), "started_at": started, "completed": 0, "total": total}
    save(output / "status.json", status)
    try:
        for mode in MODES:
            summary[mode] = {}
            for task in tasks:
                directory = output / mode / task
                directory.mkdir(parents=True, exist_ok=True)
                records = []
                for sample in samples[task]:
                    key = f"{sample['sample_id']:03d}"
                    checkpoint = directory / f"{key}.json"
                    status.update(mode=mode, task=task, sample_id=sample["sample_id"], completed=completed, updated_at=time.time())
                    save(output / "status.json", status)
                    if checkpoint.exists():
                        record = json.loads(checkpoint.read_text())
                    else:
                        request = [{"sample_id": sample["sample_id"], "prompt": sample["prompt"], "max_tokens": sample["tokens_to_generate"]}]
                        source, target = directory / "input.json", directory / "pending.json"
                        save(source, request)
                        command = [str(runner), "--model", str(model), "--batch", str(source), "--output", str(target),
                                   "--rocket-mode", mode, "--stage2-backend", "metal" if mode == "hsa" else "cpu-reference",
                                   "--kv-type", "f16", "--gpu-layers", "99", "--threads", str(args.threads),
                                   "--budget", str(args.budget), "--n-ctx", str(manifest["n_ctx"]),
                                   "--context-len", str(manifest["context_len"]),
                                   "--window-size", str(manifest["window_size"]), "--kernel-size", str(manifest["kernel_size"])]
                        begin = time.monotonic()
                        with (directory / f"{key}.log").open("w") as log:
                            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
                        wall = time.monotonic() - begin
                        results = json.loads(target.read_text())
                        if len(results) != 1 or results[0]["sample_id"] != sample["sample_id"] or results[0]["status"] != "ok":
                            raise RuntimeError("Runner returned a failed or mismatched sample")
                        record = results[0]
                        m = record["metrics"]
                        if m["effective_mode"] != mode:
                            raise RuntimeError("Runner used the wrong mode")
                        if mode == "hsa" and (m["stage2_backend"] != "metal" or m["model_backend"] != "metal" or
                                              m["stage2_layer_calls"] != m["model_layers"] * m["decode_forward_calls"]):
                            raise RuntimeError("Metal HSA did not execute in every layer")
                        record.update(task=task, references=sample["answers"], wall_seconds=wall,
                                      score=score_longbench_sample(task, record["generated"], sample["answers"], sample.get("all_classes")))
                        save(checkpoint, record)
                        target.unlink()
                        source.unlink()
                    records.append(record)
                    completed += 1
                    inference = sum((r["metrics"]["ttft_ms"] + r["metrics"]["ms_per_token"] * r["metrics"]["decode_tokens"]) / 1000 for r in records)
                    summary[mode][task] = {"samples": len(records), "score_pct": 100 * sum(r["score"] for r in records) / len(records),
                                           "wall_seconds": sum(r["wall_seconds"] for r in records), "inference_seconds": inference}
                    save(output / "summary.json", summary)
                    status.update(completed=completed, updated_at=time.time(), elapsed_seconds=time.time()-started)
                    save(output / "status.json", status)
                    print(f"[{completed}/{total}] {mode} {task}/{key}: score={record['score']:.4f} wall={record['wall_seconds']:.1f}s", flush=True)
        status.update(state="complete", finished_at=time.time())
        save(output / "status.json", status)
        print(f"Completed {total} evaluations. Results: {output}", flush=True)
    except BaseException as error:
        status.update(state="failed", error=str(error), finished_at=time.time())
        save(output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
