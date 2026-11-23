#!/usr/bin/env bash
# Decode scaling over 1024 generated tokens for E0, E2, E4 at 8k and 32k (configs/latency_long.json).
# Usage: scripts/eval/latency_long.sh OUTPUT_DIR [--dry-run]   (re-run to resume)
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:?usage: scripts/eval/latency_long.sh OUTPUT_DIR [eval_suite.latency options]}"; shift
cd "$root"
exec "${PYTHON:-python3}" -m eval_suite.latency --config configs/latency_long.json --output "$out" "$@"
