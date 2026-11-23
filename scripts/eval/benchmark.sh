#!/usr/bin/env bash
# Full benchmark (docs/xiang/benchmark_plan.md, reduced scale), resumable, into OUTPUT_ROOT:
#   ruler/         E0, E2, E4 at budgets 512/1024/2048: RULER 5 tasks x 10 x 2k-32k
#   ruler_cpu/     E1, E3 at budget 1024 on the first 4 samples per task at 2k-8k (CPU/Metal consistency)
#   ruler_budget/  E4 at budgets 256/4096: RULER 16k/32k, 5 samples per task
#   longbench/     E0, E2, E4 at budget 1024: LongBench 16 tasks x 20, 32k
#   latency/       E0-E4 latency, latency_long/ 1024-token decode scaling
# then writes OUTPUT_ROOT/report.md. Re-run the same command to resume.
# Usage: scripts/eval/benchmark.sh OUTPUT_ROOT
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:?usage: scripts/eval/benchmark.sh OUTPUT_ROOT}"
eval_dir="$root/scripts/eval"
bash "$eval_dir/latency.sh" "$out/latency"
bash "$eval_dir/latency_long.sh" "$out/latency_long"
bash "$eval_dir/ruler.sh" "$out/ruler"
bash "$eval_dir/ruler.sh" "$out/ruler_cpu" --mode stage1 hsa --budget 1024 --context-lens 2048 4096 8192 \
    --num-samples 4 --stage1-backend cpu --stage2-backend cpu-reference
bash "$eval_dir/ruler.sh" "$out/ruler_budget" --mode hsa --budget 256 4096 --context-lens 16384 32768 --num-samples 5
bash "$eval_dir/longbench.sh" "$out/longbench" --budget 1024 --num-samples 20
bash "$eval_dir/report.sh" "$out"
