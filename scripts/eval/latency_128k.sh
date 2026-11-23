#!/usr/bin/env bash
# Latency at 64k/128k (configs/latency_128k.json, 128 tokens) and a 1024-token decode at 128k
# (configs/latency_long_128k.json) for E0, E2, E4, into OUTPUT_ROOT/latency_128k and
# OUTPUT_ROOT/latency_long_128k. Needs data/ruler/{65536,131072} (scripts/data/prepare_ruler.sh
# --config configs/ruler_128k.json). Usage: scripts/eval/latency_128k.sh OUTPUT_ROOT  (re-run to resume)
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:?usage: scripts/eval/latency_128k.sh OUTPUT_ROOT}"
cd "$root"
"${PYTHON:-python3}" -m eval_suite.latency --config configs/latency_128k.json --output "$out/latency_128k"
exec "${PYTHON:-python3}" -m eval_suite.latency --config configs/latency_long_128k.json --output "$out/latency_long_128k"
