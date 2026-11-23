#!/usr/bin/env bash
# LongBench, 16 English tasks, all samples, middle-truncated to 32k (configs/longbench_32k.json):
# full KV once, then Stage 1 and Stage 1 + Metal HSA for each budget in the config.
# Usage: scripts/eval/longbench.sh OUTPUT_DIR [--mode off|stage1|hsa] [--budget N ...] [--dry-run]
# Re-run the same command to resume.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:?usage: scripts/eval/longbench.sh OUTPUT_DIR [eval_suite.stage2 options]}"; shift
cd "$root"
exec "${PYTHON:-python3}" -m eval_suite.stage2 --config configs/longbench_32k.json \
    --output "$out" --stage1-backend metal "$@"
