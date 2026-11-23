#!/usr/bin/env bash
# RULER, 5 tasks x 10 samples x 2k/4k/8k/16k/32k (configs/ruler_5task.json):
# full KV once, then Stage 1 and Stage 1 + Metal HSA for each budget in the config.
# Usage: scripts/eval/ruler.sh OUTPUT_DIR [--mode off|stage1|hsa] [--budget N ...] [--dry-run]
# Re-run the same command to resume.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:?usage: scripts/eval/ruler.sh OUTPUT_DIR [eval_suite.stage2 options]}"; shift
cd "$root"
exec "${PYTHON:-python3}" -m eval_suite.stage2 --config configs/ruler_5task.json \
    --output "$out" --stage1-backend metal "$@"
