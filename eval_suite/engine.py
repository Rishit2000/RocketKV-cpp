"""
Benchmark engine module: orchestrates task loops across attention modes,
computes aggregate metrics, renders summary ASCII tables, and persists JSON reports.
Supports evaluating on entire datasets (e.g. all 200-500 samples) or custom subsets.
"""

import os
import json
import time
from typing import List, Dict, Any, Union
from .metrics import score_sample
from .datasets import load_evaluation_samples
from .runner import execute_runner_batch

LONGBENCH_EN_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
]

LONGBENCH_ZH_TASKS = [
    "multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh"
]

ALL_LONGBENCH_TASKS = LONGBENCH_EN_TASKS + LONGBENCH_ZH_TASKS

ALL_RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2"
]

def print_task_comparison_table(task: str, task_results: Dict[str, Any]):
    """Print an ASCII comparison table across all evaluated attention modes matching midpoint specs."""
    headers = [
        "Attention Mode",
        "KV Cache (MB)",
        "Memory Saved",
        "Retained Tokens",
        "TTFT (ms)",
        "Prefill Speed",
        "Decode Speed",
        "Evict Overhead",
        "Accuracy"
    ]
    width = 160
    print("\n" + "=" * width)
    print(f"Task Summary: {task.upper()} (Midpoint Evaluation)")
    print("=" * width)
    print(f"{headers[0]:<24} | {headers[1]:<13} | {headers[2]:<12} | {headers[3]:<18} | {headers[4]:<12} | {headers[5]:<14} | {headers[6]:<22} | {headers[7]:<15} | {headers[8]:<10}")
    print("-" * width)

    # Find full-kv baseline for accurate relative memory savings
    baseline_mb = 0.0
    if "full-kv" in task_results:
        baseline_mb = task_results["full-kv"]["avg_kv_cache_mb"]
    if baseline_mb <= 0.0:
        baseline_mb = max((res.get("uncompressed_kv_cache_mb", 0.0) for res in task_results.values()), default=0.0)

    for mode_name, res in task_results.items():
        kv_mb = f"{res['avg_kv_cache_mb']:.1f} MB"
        actual = res["avg_kv_cache_mb"]
        saved = f"{(1.0 - actual / baseline_mb) * 100.0:.1f}%" if baseline_mb > 0 else "0.0%"

        # Retained tokens vs total prompt tokens
        avg_kept = res.get("avg_kept_tokens", 0.0)
        avg_prompt = res.get("avg_prompt_tokens", 0.0)
        if avg_prompt > 0:
            pct_kept = (avg_kept / avg_prompt) * 100.0
            retained_str = f"{int(round(avg_kept))} / {int(round(avg_prompt))} ({pct_kept:.0f}%)"
        else:
            retained_str = "N/A"

        ttft_str = f"{res.get('avg_ttft_ms', 0.0):.1f} ms"
        pref_spd = f"{res.get('avg_prefill_tok_per_sec', 0.0):.1f} tok/s"
        dec_tok_s = res.get('avg_tokens_per_sec', 0.0)
        dec_ms = res.get('avg_decode_ms_per_token', 0.0)
        dec_spd = f"{dec_tok_s:.1f} tok/s ({dec_ms:.1f} ms)" if dec_tok_s > 0 else "N/A"
        evict = f"{res.get('avg_evict_ms', 0.0):.2f} ms" if res.get('avg_evict_ms', 0.0) > 0 else "0.00 ms (N/A)"
        acc = f"{res['accuracy_pct']:.1f}%"

        print(f"{mode_name:<24} | {kv_mb:<13} | {saved:<12} | {retained_str:<18} | {ttft_str:<12} | {pref_spd:<14} | {dec_spd:<22} | {evict:<15} | {acc:<10}")

    print("=" * width)

def run_evaluation(
    model_path: str,
    benchmark: str,
    task: Union[str, List[str]],
    attn_modes: List[Dict[str, Any]],
    context_len: int,
    num_samples: int,
    max_tokens: int,
    runner_path: str,
    output_dir: str,
    n_ctx: int = 9216,
    data_dir: str = "data"
):
    """Run specified attention modes across datasets and calculate comprehensive metrics."""
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"eval_results_{benchmark}_{context_len}.json")
    all_results = {}

    tasks_to_run = []
    if isinstance(task, list):
        tasks_to_run.extend(task)
    elif isinstance(task, str) and "," in task:
        tasks_to_run.extend([t.strip() for t in task.split(",") if t.strip()])
    else:
        if benchmark in ["ruler", "all"]:
            if task == "all":
                tasks_to_run.extend(ALL_RULER_TASKS)
            elif benchmark == "ruler":
                tasks_to_run.append(task)
        if benchmark in ["longbench", "all"]:
            if task in ["all", "longbench"]:
                tasks_to_run.extend(ALL_LONGBENCH_TASKS)
            elif task in ["en", "english"]:
                tasks_to_run.extend(LONGBENCH_EN_TASKS)
            elif task in ["zh", "chinese"]:
                tasks_to_run.extend(LONGBENCH_ZH_TASKS)
            elif benchmark == "longbench" and task not in tasks_to_run:
                tasks_to_run.append(task)
        if not tasks_to_run:
            tasks_to_run.append(task)

    sample_str = "Full Dataset" if (num_samples is None or num_samples <= 0) else f"{num_samples} samples"
    print("\n" + "=" * 80)
    print("EVALUATION: CONSUMER-EDGE KV COMPRESSION BENCHMARK")
    print(f"Model: {os.path.basename(model_path)} | Context: {context_len} tokens (n_ctx: {n_ctx}) | Dataset Scale: {sample_str}")
    print(f"Benchmarks: {benchmark.upper()} | Tasks ({len(tasks_to_run)}): {', '.join(tasks_to_run)}")
    print("=" * 80)

    for current_task in tasks_to_run:
        print(f"\n>>> Running Task: [{current_task.upper()}] <<<")

        # Load samples (entire dataset if num_samples <= 0)
        try:
            samples = load_evaluation_samples(
                benchmark=benchmark,
                task=current_task,
                context_len=context_len,
                num_samples=num_samples,
                data_dir=data_dir
            )
        except Exception as e:
            print(f"  [ERROR] Failed to load dataset for {current_task}: {e}")
            continue

        n_samples = len(samples)
        print(f"  Loaded {n_samples} evaluation samples for {current_task.upper()}.")

        task_results = {}
        task_max_tokens = samples[0].get("tokens_to_generate", max_tokens) if samples else max_tokens

        for mode in attn_modes:
            mode_name = mode["name"]
            kv_type = mode["kv_type"]
            enable_rocket = mode["rocket"]
            budget = mode["budget"]

            print(f"\n  Evaluating Mode: {mode_name:<16} (KV: {kv_type}, RocketKV: {enable_rocket}, Budget: {budget})")

            start_mode_time = time.time()

            batch_samples = [
                {
                    "sample_id": sample.get("sample_id", idx + 1),
                    "prompt": sample["prompt"],
                    "max_tokens": task_max_tokens
                }
                for idx, sample in enumerate(samples)
            ]

            results_list = execute_runner_batch(
                runner_path=runner_path,
                model_path=model_path,
                kv_type=kv_type,
                samples=batch_samples,
                token_budget=budget,
                enable_rocket=enable_rocket,
                n_ctx=n_ctx,
                context_len=context_len,
                window_size=mode.get("window_size", 32),
                kernel_size=mode.get("kernel_size", 63)
            )

            mode_records = []
            total_score = 0.0
            total_ttft = 0.0
            total_kv_mb = 0.0
            total_uncompressed_mb = 0.0
            total_evict_ms = 0.0
            total_prompt_tokens = 0
            total_prefill_ms = 0.0
            total_kept_tokens = 0
            total_decode_spd = 0.0
            decode_count = 0

            for idx, (res_item, sample) in enumerate(zip(results_list, samples)):
                generated = res_item.get("generated", "")
                metrics = res_item.get("metrics", {})
                gt = sample.get("answers", [])
                score = score_sample(
                    benchmark=benchmark,
                    task=current_task,
                    prediction=generated,
                    ground_truths=gt,
                    all_classes=sample.get("all_classes")
                )

                p_tokens = metrics.get("prompt_tokens", len(sample["prompt"]) // 4)
                pref_ms = metrics.get("prefill_ms", metrics.get("ttft_ms", 0.0))
                dec_tokens = metrics.get("decode_tokens", 0)
                tok_per_sec = metrics.get("tokens_per_sec", 0.0)

                total_score += score
                total_ttft += metrics.get("ttft_ms", 0.0)
                total_prompt_tokens += p_tokens
                total_prefill_ms += pref_ms
                total_kept_tokens += metrics.get("kept_tokens", p_tokens)
                total_kv_mb += metrics.get("kv_cache_mb", 0.0)
                total_uncompressed_mb += metrics.get("uncompressed_kv_cache_mb", 0.0)
                total_evict_ms += metrics.get("evict_ms", 0.0)

                if dec_tokens > 0 and tok_per_sec > 0:
                    total_decode_spd += tok_per_sec
                    decode_count += 1

                mode_records.append({
                    "sample_id": sample.get("sample_id", idx + 1),
                    "generated": generated,
                    "ground_truth": gt,
                    "score": score,
                    "metrics": metrics
                })

            n = len(samples)
            avg_score = (total_score / n) * 100.0 if n > 0 else 0.0
            avg_ttft = total_ttft / n if n > 0 else 0.0
            avg_prefill_spd = (total_prompt_tokens / (total_prefill_ms / 1000.0)) if total_prefill_ms > 0 else 0.0
            avg_tokens_per_sec = (total_decode_spd / decode_count) if decode_count > 0 else 0.0
            avg_decode_ms = (1000.0 / avg_tokens_per_sec) if avg_tokens_per_sec > 0 else 0.0
            avg_kv_mb = total_kv_mb / n if n > 0 else 0.0
            avg_uncompressed_mb = total_uncompressed_mb / n if n > 0 else 0.0
            avg_evict_ms = total_evict_ms / n if n > 0 else 0.0
            avg_kept = total_kept_tokens / n if n > 0 else 0.0
            avg_prompt = total_prompt_tokens / n if n > 0 else 0.0
            elapsed = time.time() - start_mode_time

            print(f"  Completed {mode_name} in {elapsed:.1f}s. Final Accuracy: {avg_score:.2f}%")

            task_results[mode_name] = {
                "config": mode,
                "accuracy_pct": avg_score,
                "avg_ttft_ms": avg_ttft,
                "avg_prefill_tok_per_sec": avg_prefill_spd,
                "avg_decode_ms_per_token": avg_decode_ms,
                "avg_tokens_per_sec": avg_tokens_per_sec,
                "avg_evict_ms": avg_evict_ms,
                "avg_kv_cache_mb": avg_kv_mb,
                "uncompressed_kv_cache_mb": avg_uncompressed_mb,
                "avg_kept_tokens": avg_kept,
                "avg_prompt_tokens": avg_prompt,
                "total_samples": n,
                "records": mode_records
            }
            all_results[current_task] = task_results

            # Incrementally save results after each mode
            with open(out_file, "w") as f:
                json.dump(all_results, f, indent=2)

        print_task_comparison_table(current_task, task_results)

    # Final persistent save
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed evaluation results saved to: {out_file}\n")
