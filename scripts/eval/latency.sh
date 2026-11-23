#!/usr/bin/env bash
# Latency of E0-E4 (configs/latency.json): TTFT split, first decode step, inter-token latency,
# end-to-end time and peak memory at 2k-32k, 128 output tokens, configurations interleaved.
# Usage: scripts/eval/latency.sh OUTPUT_DIR [--dry-run]   (re-run to resume)
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:?usage: scripts/eval/latency.sh OUTPUT_DIR [eval_suite.latency options]}"; shift
cd "$root"
exec "${PYTHON:-python3}" -m eval_suite.latency --config configs/latency.json --output "$out" "$@"
