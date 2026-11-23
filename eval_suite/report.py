#!/usr/bin/env python3
"""Benchmark tables (docs/xiang/benchmark_plan.md) from one benchmark output root.

Expected subdirectories (any may be missing; its tables are skipped):
  ruler/         eval_suite.stage2, RULER, Metal backends: off, stage1_b*, hsa_b*      (E0, E2, E4)
  ruler_cpu/     same samples subset with CPU backends: stage1_b1024, hsa_b1024          (E1, E3)
  ruler_budget/  hsa_b* extra budgets at long contexts                                   (E4 sweep)
  longbench/     eval_suite.stage2, LongBench: off, stage1_b1024, hsa_b1024             (E0, E2, E4)
  latency/       eval_suite.latency summary.json                                         (E0-E4)
  latency_long/  eval_suite.latency summary.json and runs/ with 1024-token generations
Writes ROOT/report.md and prints it.
"""
import argparse
import json
from pathlib import Path
import statistics

LONGBENCH_GROUPS = {
    "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Few-shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}
NAMES = {"off": "Full KV (E0)", "stage1": "Stage 1 (E2)", "hsa": "Stage 1+2 (E4)"}
LATENCY_NAMES = {"E0_full": "Full KV (E0)", "E1_s1_cpu": "Stage 1 · CPU (E1)", "E2_s1_metal": "Stage 1 · Metal (E2)",
                 "E3_hsa_cpu": "Stage 1+2 · CPU (E3)", "E4_hsa_metal": "Stage 1+2 · Metal (E4)"}


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(x, digits=1):
    return "—" if x is None else f"{x:.{digits}f}"


def run_key(run):
    mode, _, budget = run.partition("_b")
    return (list(NAMES).index(mode), int(budget or 0))


def run_label(run):
    mode, _, budget = run.partition("_b")
    return NAMES[mode], budget or "—"


def records(directory):
    """{run: {unit: {sample_id: record}}} from per-sample checkpoints."""
    out = {}
    for path in sorted(directory.glob("*/**/[0-9][0-9][0-9].json")):
        rel = path.relative_to(directory).parts
        out.setdefault(rel[0], {}).setdefault("/".join(rel[1:-1]), {})[path.stem] = json.loads(path.read_text())
    return out


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def ruler_by_length(recs):
    """{run: {length: mean task score %}}; each task weighted equally."""
    out = {}
    for run, units in recs.items():
        lengths = {}
        for unit, samples in units.items():
            lengths.setdefault(int(unit.split("/")[0]), []).append(100 * mean(r["score"] for r in samples.values()))
        out[run] = {n: mean(v) for n, v in lengths.items()}
    return out


def accuracy_section(root):
    ruler = records(root / "ruler") if (root / "ruler").is_dir() else {}
    budget = records(root / "ruler_budget") if (root / "ruler_budget").is_dir() else {}
    longbench = records(root / "longbench") if (root / "longbench").is_dir() else {}
    if not ruler and not longbench:
        return []
    by_length = ruler_by_length(ruler)
    lengths = sorted({n for v in by_length.values() for n in v})
    header = ["Method", "Budget"] + [f"RULER {n // 1024}k" for n in lengths] + ["RULER avg"]
    header += [f"LB {g}" for g in LONGBENCH_GROUPS] + ["LB avg"]
    rows = []
    for run in sorted(set(ruler) | set(longbench), key=run_key):
        r = by_length.get(run, {})
        row = list(run_label(run)) + [fmt(r.get(n)) for n in lengths] + [fmt(mean(r.values()) if r else None)]
        lb = {task: 100 * mean(x["score"] for x in s.values()) for task, s in longbench.get(run, {}).items()}
        groups = [mean(lb[t] for t in tasks if t in lb) if any(t in lb for t in tasks) else None
                  for tasks in LONGBENCH_GROUPS.values()]
        rows.append(row + [fmt(g) for g in groups] + [fmt(mean(lb.values()) if lb else None)])
    out = ["## Table A: accuracy (%)", "",
           "RULER: mean over 5 tasks per length (string_match_all / string_match_part for qa). "
           "LongBench: official metrics, group = mean of its tasks, LB avg = mean over tasks.", "",
           table(header, rows), ""]
    if ruler:
        tasks = sorted({u.split("/")[1] for v in ruler.values() for u in v})
        rows = []
        for run in sorted(ruler, key=run_key):
            per_task = {t: mean(100 * mean(r["score"] for r in s.values())
                                for u, s in ruler[run].items() if u.endswith("/" + t)) for t in tasks}
            rows.append(list(run_label(run)) + [fmt(per_task[t]) for t in tasks])
        out += ["RULER per task (mean over lengths):", "", table(["Method", "Budget"] + tasks, rows), ""]
    if budget:
        sweep = {**{k: v for k, v in ruler.items() if k.startswith("hsa") or k == "off"}, **budget}
        by = ruler_by_length(sweep)
        lens = sorted({n for v in ruler_by_length(budget).values() for n in v})
        rows = [list(run_label(run)) + [fmt(by[run].get(n)) for n in lens] for run in sorted(sweep, key=run_key)]
        out += ["## Budget sweep: RULER accuracy (%) of Stage 1+2 (E4)", "",
                table(["Method", "Budget"] + [f"{n // 1024}k" for n in lens], rows), ""]
    return out


def consistency_section(root):
    if not (root / "ruler_cpu").is_dir() or not (root / "ruler").is_dir():
        return []
    cpu, metal = records(root / "ruler_cpu"), records(root / "ruler")
    rows = []
    for run, label in (("stage1_b1024", "E1 vs E2 (Stage 1)"), ("hsa_b1024", "E3 vs E4 (Stage 1+2)")):
        pairs = [(c, metal[run][u][k]) for u, s in cpu.get(run, {}).items() for k, c in s.items()
                 if k in metal.get(run, {}).get(u, {})]
        if not pairs:
            continue
        lens = sorted({int(c["group"]) for c, _ in pairs})
        rows.append([label, "/".join(f"{n // 1024}k" for n in lens), len(pairs),
                     f"{100 * mean(c['generated'] == m['generated'] for c, m in pairs):.1f}%",
                     fmt(100 * mean(c["score"] for c, _ in pairs)), fmt(100 * mean(m["score"] for _, m in pairs))])
    return ["## Table B: CPU / Metal consistency (budget 1024, same RULER samples)", "",
            table(["Comparison", "Lengths", "Samples", "Identical output", "Accuracy CPU (%)", "Accuracy Metal (%)"],
                  rows), ""] if rows else []


def latency_section(root):
    path = root / "latency" / "summary.json"
    if not path.is_file():
        return []
    summary = json.loads(path.read_text())
    base = {s["length"]: s for s in summary if s["config"] == "E0_full"}
    order = list(LATENCY_NAMES)
    summary.sort(key=lambda s: (order.index(s["config"]) if s["config"] in order else 99, s["length"]))
    rows, ttft, memory = [], [], []
    for s in summary:
        name, n, b = LATENCY_NAMES.get(s["config"], s["config"]), f"{s['length'] // 1024}k", base.get(s["length"])
        rows.append([name, n, fmt(s["ttft_ms"], 0), fmt(s["first_decode_ms"], 2), fmt(s["itl_p50_ms"], 2),
                     fmt(s["itl_p99_ms"], 2), fmt(s["e2e_ms"] / 1000, 2), fmt(s["decode_tokens_per_s"], 1),
                     f"{b['itl_p50_ms'] / s['itl_p50_ms']:.2f}×" if b else "—",
                     f"{b['e2e_ms'] / s['e2e_ms']:.2f}×" if b else "—"])
        ttft.append([name, n, fmt(s["prefill_ms"], 0), fmt(s["evict_ms"], 1), fmt(s["stage2_init_ms"], 1),
                     fmt(s["ttft_ms"], 0)])
        memory.append([name, n, int(s["prompt_tokens"]), int(s["kept_tokens"]), fmt(s["kv_cache_mb"], 0),
                       fmt(s["uncompressed_kv_cache_mb"], 0),
                       f"{s['uncompressed_kv_cache_mb'] / s['kv_cache_mb']:.1f}×" if s["kv_cache_mb"] else "—",
                       fmt(s["peak_rss_mb"], 0)])
    runs = summary[0]["runs"] if summary else 0
    return ["## Table C: latency (budget 1024, 128 generated tokens, medians)", "",
            f"Each cell: {runs} runs (prompts × repeats). ITL = per decode step after the first; "
            "speedups are Full KV / method for ITL p50 and end-to-end time.", "",
            table(["Method", "Length", "TTFT (ms)", "First decode (ms)", "ITL p50 (ms)", "ITL p99 (ms)",
                   "End-to-end (s)", "Decode (tok/s)", "ITL speedup", "E2E speedup"], rows), "",
            "## Table D: TTFT breakdown (ms, budget 1024)", "",
            table(["Method", "Length", "Prefill", "Stage 1 eviction", "Stage 2 init", "TTFT"], ttft), "",
            "## Table E: memory (budget 1024)", "",
            "KV sizes are the runner's logical estimates of the prompt KV (F16); peak RSS from /usr/bin/time -l "
            "(Metal buffers in unified memory are not all counted in RSS).", "",
            table(["Method", "Length", "Prompt tokens", "Kept tokens", "KV (MB)", "Full KV (MB)", "Compression",
                   "Peak RSS (MB)"], memory), ""]


def long_generation_section(root):
    runs = root / "latency_long" / "runs"
    if not runs.is_dir():
        return []
    groups = {}
    for path in runs.glob("E*_r*.json"):
        r = json.loads(path.read_text())
        groups.setdefault((r["config"], r["length"]), []).append(r["metrics"]["decode_step_ms"])
    windows = [(1, 128), (128, 512), (512, 1023)]
    rows = []
    for (config, n), series in sorted(groups.items(), key=lambda x: (x[0][1], list(LATENCY_NAMES).index(x[0][0]))):
        rows.append([LATENCY_NAMES.get(config, config), f"{n // 1024}k"] +
                    [fmt(statistics.median(x for s in series for x in s[a:b]), 2) for a, b in windows] +
                    [fmt(statistics.median(sum(s) for s in series) / 1000, 2)])
    return ["## Long generation: median ITL (ms) by decode step window, 1024 tokens", "",
            table(["Method", "Length"] + [f"steps {a + 1}-{b}" for a, b in windows] + ["Total decode (s)"], rows), ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    sections = ["# RocketKV-cpp benchmark results", ""]
    for part in (accuracy_section, consistency_section, latency_section, long_generation_section):
        sections += part(args.root)
    text = "\n".join(sections)
    (args.root / "report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
