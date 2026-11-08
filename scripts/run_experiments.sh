#!/usr/bin/env bash
# ==============================================================================
# RocketKV-llama: Systematic Benchmarking Script for Apple Silicon
# All benchmark parameters are maintained in declarative JSON configs in configs/
# ==============================================================================

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH=${1:-"models/Llama-3.2-3B-Instruct-Q8_0.gguf"}
EXTRA_ARGS=${2:-""}

PYTHON=${PYTHON:-python3}
if ! "$PYTHON" --version >/dev/null 2>&1; then
  PYTHON="/usr/bin/python3"
fi

echo "================================================================================"
echo " RocketKV on Consumer Edge Hardware - Systematic Benchmark Suite"
echo " Target Model: $MODEL_PATH"
echo " Python Interpreter: $PYTHON"
echo "================================================================================"

# 1. LongBench Benchmark Suite (Full-KV vs Quantized-KV vs RocketKV)
echo ""
echo "=== Running LongBench Benchmark Suite (configs/longbench.json) ==="
"$PYTHON" eval.py configs/longbench.json

echo ""
echo "================================================================================"
echo " Benchmarks completed. Results stored in results/ directory."
echo "================================================================================"

