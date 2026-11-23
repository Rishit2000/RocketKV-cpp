#!/usr/bin/env python3
"""Paired Stage 1 latency measurements; short generation, no task scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True, help="Runner-format JSON sample array")
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--host-runner", type=Path,
                        help="Compare an older host-capture Metal runner, current resident Metal and native full KV")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1 or any(n < 1 for n in args.contexts):
        parser.error("repeats, max-tokens and contexts must be positive")
    model = args.model.resolve()
    runner = ROOT / "build/runner"
    if not model.is_file() or not runner.is_file():
        parser.error("Model and built runner must exist")
    variants = {"cpu": (runner, "hsa", "cpu"), "metal": (runner, "hsa", "metal")}
    reference = "cpu"
    if args.host_runner:
        host = args.host_runner.resolve()
        if not host.is_file():
            parser.error("host-runner must exist")
        variants = {"host": (host, "hsa", "metal"), "metal": (runner, "hsa", "metal"),
                    "off": (runner, "off", "cpu")}
        reference = "host"
    source = json.loads(args.batch.read_text())
    if not isinstance(source, list) or not source or any(
        not isinstance(s, dict) or not isinstance(s.get("prompt"), str) for s in source
    ):
        parser.error("batch must be a nonempty array of prompts")
    samples = [
        {"sample_id": i + 1, "prompt": s["prompt"], "max_tokens": args.max_tokens}
        for i, s in enumerate(source)
    ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "model": str(model), "model_size": model.stat().st_size,
        "model_mtime_ns": model.stat().st_mtime_ns,
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "variants": {name: {"binary": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                            "mode": mode, "stage1_backend": backend,
                            "stage2_backend": "metal" if mode == "hsa" else "cpu-reference"}
                     for name, (binary, mode, backend) in variants.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contexts": args.contexts, "repeats": args.repeats, "max_tokens": args.max_tokens,
        "samples": len(samples),
        "budget": 1024, "window": 32, "kernel": 63, "threads": 4,
        "gpu_layers": 99, "ubatch": 1024, "kv_type": "f16",
        "measurement": "Fresh process per sample; TTFT includes capture allocation/dispatch, excludes model and pipeline initialization",
        "quality": "Short greedy-output comparison only; not task-standard LongBench scores",
    }
    save(output / "manifest.json", manifest)
    save(output / "samples.json", samples)
    records = []
    for repeat in range(args.repeats):
        for context in args.contexts:
            for index, sample in enumerate(samples):
                order = list(variants)
                shift = (repeat + index) % len(order)
                order = order[shift:] + order[:shift]
                for variant in order:
                    binary, mode, backend = variants[variant]
                    name = f"{context}-{index + 1}-r{repeat}-{variant}"
                    request = output / "input.json"
                    target = output / f"{name}.json"
                    save(request, [sample])
                    command = [
                        str(binary), "--model", str(model), "--batch", str(request),
                        "--output", str(target), "--rocket-mode", mode,
                        "--stage1-backend", backend, "--stage2-backend", "metal" if mode == "hsa" else "cpu-reference",
                        "--gpu-layers", "99", "--kv-type", "f16", "--threads", "4",
                        "--budget", "1024", "--window-size", "32", "--kernel-size", "63",
                        "--context-len", str(context), "--n-ctx", str(context + args.max_tokens),
                        "--ubatch", "1024",
                    ]
                    begin, wall = time.monotonic(), time.time()
                    with (output / f"{name}.log").open("w") as log:
                        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                    elapsed, wall_elapsed = time.monotonic() - begin, time.time() - wall
                    rows = json.loads(target.read_text())
                    if len(rows) != 1 or rows[0]["status"] != "ok" or rows[0]["sample_id"] != sample["sample_id"]:
                        raise RuntimeError(f"Mismatched runner output: {name}")
                    row = rows[0]
                    m = row["metrics"]
                    if (m["requested_stage1_backend"] != backend or m["effective_mode"] != mode
                            or m["requested_gpu_layers"] != 99
                            or (mode == "hsa" and (m["stage1_backend"] != backend
                                or m["stage2_backend"] != "metal" or m["model_backend"] != "metal"
                                or m["stage2_layer_calls"] != m["model_layers"] * m["decode_forward_calls"]))
                            or (mode == "off" and (m["stage1_backend"] != "none" or m["stage2_layer_calls"] != 0))):
                        raise RuntimeError(f"Expected both stages to execute on the requested backends: {name}")
                    if variant == "metal" and (m["stage1_capture"] != "metal" or m["stage1_capture_host_bytes"] != 0):
                        raise RuntimeError(f"Resident path performed a host Q/K read: {name}")
                    m["generation_wall_ms"] = m["ms_per_token"] * m["decode_tokens"]
                    m["total_inference_ms"] = m["ttft_ms"] + m["generation_wall_ms"]
                    row.update(context=context, repeat=repeat, backend=variant, command=command,
                               active_seconds=elapsed, wall_seconds=wall_elapsed)
                    save(target, row)
                    # Preserve anomalous raw results, but never summarize them as valid speedups.
                    if abs(wall_elapsed - elapsed) > 1 or m["ttft_ms"] > elapsed * 1000 + 100:
                        raise RuntimeError(f"Inconsistent clocks: {name}")
                    records.append(row)
                    print(f"{name}: eviction={m['evict_ms']:.3f} ms TTFT={m['ttft_ms']:.3f} ms", flush=True)
    summary = {}
    for context in args.contexts:
        pairs = {}
        for row in records:
            if row["context"] == context:
                pairs.setdefault((row["repeat"], row["sample_id"]), {})[row["backend"]] = row
        result = {"pairs": len(pairs), "generated_equal": 0, "decode_counts_equal": 0}
        for pair in pairs.values():
            a, b = pair[reference], pair["metal"]
            if (a["metrics"]["prompt_tokens"], a["metrics"]["kept_tokens"]) != (
                b["metrics"]["prompt_tokens"], b["metrics"]["kept_tokens"]
            ):
                raise RuntimeError("CPU/Metal prompt or retained-token counts differ")
            result["generated_equal"] += a["generated"] == b["generated"]
            result["decode_counts_equal"] += a["metrics"]["decode_tokens"] == b["metrics"]["decode_tokens"]
        for metric in ("evict_ms", "prefill_ms", "ttft_ms", "decode_ms_per_forward", "total_inference_ms"):
            result[metric] = {
                backend: (statistics.median(pair[backend]["metrics"][metric] for pair in pairs.values())
                          if all(pair[backend]["metrics"][metric] is not None for pair in pairs.values()) else None)
                for backend in variants
            }
            for other in (reference, "off") if "off" in variants else (reference,):
                key = "paired_reduction_pct" if other == reference else "reduction_vs_native_pct"
                result[metric][key] = (
                    100 * (1 - statistics.median(
                        pair["metal"]["metrics"][metric] / pair[other]["metrics"][metric] for pair in pairs.values()
                    )) if all(pair[other]["metrics"][metric] and pair["metal"]["metrics"][metric] is not None
                              for pair in pairs.values()) else None
                )
        if "off" in variants:
            result["native_decode_counts_equal"] = sum(
                pair["off"]["metrics"]["decode_tokens"] == pair["metal"]["metrics"]["decode_tokens"]
                for pair in pairs.values()
            )
        summary[context] = result
    save(output / "summary.json", summary)
    (output / "input.json").unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
