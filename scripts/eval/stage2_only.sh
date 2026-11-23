#!/usr/bin/env bash
# Ablation E5: Stage 2 (HSA) alone, no Stage 1 eviction (--rocket-mode stage2), budget 1024.
# RULER 2k-32k (configs/ruler_5task.json) into OUTPUT_ROOT/ruler_stage2, RULER 64k/128k
# (configs/ruler_128k.json) into OUTPUT_ROOT/ruler_128k_stage2, and latency against E4
# (configs/latency_stage2.json) into OUTPUT_ROOT/latency_stage2.
# Usage: scripts/eval/stage2_only.sh OUTPUT_ROOT [accuracy|latency]  (default both; re-run to resume)
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:?usage: scripts/eval/stage2_only.sh OUTPUT_ROOT [accuracy|latency]}"
part="${2:-all}"
cd "$root"
if [[ "$part" != latency ]]; then
    scripts/eval/ruler.sh "$out/ruler_stage2" --mode stage2 --budget 1024
    scripts/eval/ruler.sh "$out/ruler_128k_stage2" --config configs/ruler_128k.json --mode stage2 --budget 1024
fi
if [[ "$part" != accuracy ]]; then
    "${PYTHON:-python3}" -m eval_suite.latency --config configs/latency_stage2.json --output "$out/latency_stage2"
fi
