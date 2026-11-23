#!/usr/bin/env bash
# Tables A-E of docs/xiang/benchmark_plan.md from a benchmark output root (see scripts/eval/benchmark.sh).
# Usage: scripts/eval/report.sh OUTPUT_ROOT   (writes OUTPUT_ROOT/report.md)
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:?usage: scripts/eval/report.sh OUTPUT_ROOT}"
cd "$root"
exec "${PYTHON:-python3}" -m eval_suite.report "$out"
