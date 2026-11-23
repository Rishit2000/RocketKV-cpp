#!/usr/bin/env python3
"""Resumable latency benchmark: fixed prompts, fixed output length, all configurations interleaved.

For each repeat, length and prompt, every configuration in the config runs once (in config order),
so slow drift affects all configurations alike. One discarded warm-up run precedes the measurements. Each run generates exactly max_tokens tokens
(--ignore-eos) from a RULER prompt of the given length and records the runner's timing split
(prefill, Stage 1 eviction, Stage 2 init, per-step decode) plus peak resident memory.

summary.json reports, per (configuration, length), medians over prompts and repeats of:
TTFT and its split, first decode step, inter-token latency p50/p90/p99 (decode steps after the
first), decode tokens/s, end-to-end latency (TTFT + all decode steps), peak RSS, logical KV size.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_suite.config import load_config_file
from eval_suite.datasets import load_ruler_dataset
from eval_suite.stage2 import check_backends, resolve, save

N_CTX_MARGIN = 256


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))]


def summarize(records):
    steps = lambda r: r["metrics"]["decode_step_ms"]
    rows = {}
    for r in records:
        rows.setdefault((r["config"], r["length"]), []).append(r)
    out = []
    for (config, length), group in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
        med = lambda f: statistics.median(f(r) for r in group)
        itl = [x for r in group for x in steps(r)[1:]]
        out.append({
            "config": config, "length": length, "runs": len(group),
            "prompt_tokens": med(lambda r: r["metrics"]["prompt_tokens"]),
            "kept_tokens": med(lambda r: r["metrics"]["kept_tokens"]),
            "ttft_ms": med(lambda r: r["metrics"]["ttft_ms"]),
            "prefill_ms": med(lambda r: r["metrics"]["prefill_ms"]),
            "evict_ms": med(lambda r: r["metrics"]["evict_ms"]),
            "stage2_init_ms": med(lambda r: r["metrics"]["stage2_init_ms"]),
            "first_decode_ms": med(lambda r: steps(r)[0]),
            "itl_p50_ms": percentile(itl, 50), "itl_p90_ms": percentile(itl, 90), "itl_p99_ms": percentile(itl, 99),
            "decode_tokens_per_s": med(lambda r: 1000 * len(steps(r)) / sum(steps(r))),
            "e2e_ms": med(lambda r: r["metrics"]["ttft_ms"] + sum(steps(r))),
            "peak_rss_mb": med(lambda r: r["peak_rss_bytes"] / 2**20),
            "kv_cache_mb": med(lambda r: r["metrics"]["kv_cache_mb"]),
            "uncompressed_kv_cache_mb": med(lambda r: r["metrics"]["uncompressed_kv_cache_mb"]),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; writes nothing")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = load_config_file(args.config)
    bench, attention = config["latency"], config["attention"]
    model, runner = resolve(config["model"]["path"]), resolve(config["model"]["runner"])
    data_dir = str(resolve(config["management"]["data_dir"]))
    prompts = {n: load_ruler_dataset(bench["task"], n, bench["prompts"], data_dir=data_dir)
               for n in bench["lengths"]}
    configs = bench["configs"]
    plan = [(rep, n, sample, c) for rep in range(bench["repeats"]) for n in bench["lengths"]
            for sample in prompts[n] for c in configs]
    print(f"latency: {len(configs)} configs x {len(bench['lengths'])} lengths x {bench['prompts']} prompts "
          f"x {bench['repeats']} repeats = {len(plan)} runs, {bench['max_tokens']} tokens each", flush=True)
    if args.dry_run:
        return
    if not model.is_file() or not runner.is_file():
        raise FileNotFoundError(f"Missing model ({model}) or runner ({runner})")
    output = Path(args.output).resolve()
    (output / "runs").mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"config": {k: v for k, v in config.items() if k != "model"}, "model": str(model),
                "runner": str(runner), "runner_sha256": digest(runner), "driver_sha256": digest(Path(__file__)),
                "threads": args.threads, "gpu_layers": 99, "kv_type": "f16"}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Existing latency run has different settings or binary; use a new output directory")
    save(manifest_path, manifest)

    def run(rep, n, sample, c, target):
        source = output / "runs" / "input.json"
        save(source, [{"sample_id": sample["sample_id"], "prompt": sample["prompt"], "max_tokens": bench["max_tokens"]}])
        command = ["/usr/bin/time", "-l", str(runner), "--model", str(model), "--batch", str(source),
                   "--output", str(target.with_suffix(".pending")), "--rocket-mode", c["mode"],
                   "--stage2-backend", c.get("stage2_backend", "cpu-reference"),
                   "--kv-type", "f16", "--gpu-layers", "99", "--threads", str(args.threads),
                   "--budget", str(c.get("budget", attention["budget"])), "--n-ctx",
                   str(n + bench["max_tokens"] + N_CTX_MARGIN), "--context-len", "0", "--ignore-eos",
                   "--window-size", str(attention["window_size"]), "--kernel-size", str(attention["kernel_size"])]
        if c["mode"] != "off":
            command += ["--stage1-backend", c["stage1_backend"]]
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        (output / "runs" / f"{target.stem}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{target.name}: runner failed\n{result.stderr[-3000:]}")
        rss = re.search(r"(\d+)\s+maximum resident set size", result.stderr)
        row = json.loads(target.with_suffix(".pending").read_text())[0]
        check_backends(row["metrics"], c["mode"], c.get("stage1_backend"), c.get("stage2_backend"))
        if row["metrics"]["decode_forward_calls"] != bench["max_tokens"] - 1:
            raise RuntimeError(f"{target.name}: expected {bench['max_tokens'] - 1} decode steps")
        save(target, {"config": c["name"], "length": n, "sample_id": sample["sample_id"], "repeat": rep,
                      "peak_rss_bytes": int(rss.group(1)) if rss else None, "metrics": row["metrics"]})
        target.with_suffix(".pending").unlink()
        source.unlink()

    rep0, n0, sample0, c0 = plan[0]
    warmup = output / "runs" / "warmup.json"
    run(rep0, n0, sample0, c0, warmup)
    warmup.unlink()
    records = []
    for i, (rep, n, sample, c) in enumerate(plan):
        target = output / "runs" / f"{c['name']}_{n}_s{sample['sample_id']:03d}_r{rep}.json"
        if not target.exists():
            run(rep, n, sample, c, target)
        record = json.loads(target.read_text())
        records.append(record)
        m = record["metrics"]
        print(f"[{i + 1}/{len(plan)}] r{rep} {n:6} {c['name']:12} ttft {m['ttft_ms']:9.1f} ms  "
              f"itl {statistics.median(m['decode_step_ms'][1:]):7.3f} ms", flush=True)
        if (i + 1) % len(configs) == 0 or i + 1 == len(plan):
            save(output / "summary.json", summarize(records))
    print(f"Completed {len(plan)} runs. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
