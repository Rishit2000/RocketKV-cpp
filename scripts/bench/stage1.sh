#!/usr/bin/env bash
# Time Stage 1 capture and scoring separately from task scoring (eval_suite/bench_stage1.py).
# Usage: scripts/bench/stage1.sh --model M.gguf --batch B.json --output DIR [--contexts ...]
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
exec "${PYTHON:-python3}" -m eval_suite.bench_stage1 "$@"
