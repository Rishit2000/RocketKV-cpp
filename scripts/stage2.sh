#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'HELP'
Usage: scripts/stage2.sh <command> [arguments]
  build                         Build the pinned, patched llama.cpp and runner.
  test [MODEL.gguf]              Run algorithm, model, and UBSan tests.
  run MODEL.gguf [BATCH.json] [OUTPUT_DIR]
                                Compare off / stage1 / hsa on CPU with F16 KV.

test downloads a verified 1.1 MB model when no model is supplied.
run creates a demonstration batch when BATCH.json is omitted.
Results default to results/stage2/. This is not a Metal speed benchmark.

Build: JOBS=4 CXX=c++ METAL=ON|OFF (default: ON on macOS)
Test:  SANITIZERS=undefined (ASan requires a compatible runtime)
Run:   BUDGET=128 N_CTX=2048 CONTEXT_LEN=1024 MAX_TOKENS=32
       WINDOW_SIZE=32 KERNEL_SIZE=63 THREADS=4
HELP
}

fail() { echo "Error: $*" >&2; exit 1; }

abs_path() {
    python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$1"
}

build_stage2() (
    [[ $# -eq 0 ]] || fail "Usage: scripts/stage2.sh build"
    LLAMA="$ROOT/llama.cpp"
    REVISION="$(cat "$ROOT/patches/llama.cpp-revision")"
    PATCH="$ROOT/patches/llama-stage2.patch"
    JOBS="${JOBS:-4}"
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "JOBS must be positive."

    if [[ ! -d "$LLAMA/.git" ]]; then
        [[ ! -e "$LLAMA" ]] || fail "$LLAMA is not a standalone Git checkout; leaving it untouched."
        git init -q "$LLAMA"
        git -C "$LLAMA" remote add origin https://github.com/ggml-org/llama.cpp.git
    fi
    if ! git -C "$LLAMA" rev-parse --verify HEAD >/dev/null 2>&1; then
        git -C "$LLAMA" fetch --quiet --depth 1 origin "$REVISION"
        git -C "$LLAMA" checkout --quiet --detach "$REVISION"
    fi
    [[ "$(git -C "$LLAMA" rev-parse HEAD)" == "$REVISION" ]] ||
        fail "llama.cpp must be at $REVISION. Existing checkout was not changed."

    if git -C "$LLAMA" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
        echo "Rocket patch is already applied."
    elif git -C "$LLAMA" apply --check "$PATCH"; then
        git -C "$LLAMA" apply "$PATCH"
    else
        fail "Rocket patch conflicts with the local llama.cpp checkout."
    fi

    if [[ "$(uname -s)" == "Darwin" ]]; then METAL="${METAL:-ON}"; else METAL="${METAL:-OFF}"; fi
    [[ "$METAL" == "ON" || "$METAL" == "OFF" ]] || fail "METAL must be ON or OFF."
    cmake -S "$LLAMA" -B "$LLAMA/build" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DGGML_METAL="$METAL" -DGGML_METAL_EMBED_LIBRARY=OFF \
        -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    cmake --build "$LLAMA/build" --target llama -j "$JOBS"
    make -C "$ROOT" runner test_rocket CXX="${CXX:-c++}"
    echo "Built runner with llama.cpp $REVISION and CPU-reference HSA."
)

test_stage2() (
    [[ $# -le 1 ]] || fail "Usage: scripts/stage2.sh test [MODEL.gguf]"
    MODEL="$(abs_path "${1:-$ROOT/models/stories260K.gguf}")"
    if [[ $# -eq 1 && ! -f "$MODEL" ]]; then fail "model not found: $MODEL"; fi
    build_stage2
    if [[ $# -eq 0 ]]; then
        mkdir -p "$ROOT/models"
        checked_model="$MODEL"
        if [[ ! -f "$MODEL" ]]; then
            download="$(mktemp "$ROOT/models/.stories260K.XXXXXX")"
            checked_model="$download"
            trap 'rm -f -- "$download"' EXIT
            curl --fail --location --retry 2 --output "$download" \
                "https://huggingface.co/ggml-org/models/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories260K.gguf"
        fi
        python3 - "$checked_model" <<'PY'
import hashlib
import sys
from pathlib import Path
expected = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != expected:
    raise SystemExit("Test model checksum mismatch; refusing to use this file")
PY
        if [[ "$checked_model" != "$MODEL" ]]; then
            mv "$checked_model" "$MODEL"
            trap - EXIT
        fi
    fi
    export ROCKET_TEST_MODEL="$MODEL"
    cd "$ROOT"
    make test
    mkdir -p build
    SANITIZER_FLAGS=(-std=c++17 -O1 -g -Wall -Wextra -Werror -Wno-unknown-pragmas
        "-fsanitize=${SANITIZERS:-undefined}" -fno-sanitize-recover=all -fno-omit-frame-pointer)
    "${CXX:-c++}" "${SANITIZER_FLAGS[@]}" test_rocket.cpp rocket.cpp -o build/test_rocket_sanitized
    echo "Sanitizers: ${SANITIZERS:-undefined}"
    python3 - "$ROOT/build/test_rocket_sanitized" <<'PY'
import subprocess
import sys
try:
    subprocess.run([sys.argv[1]], check=True, timeout=120)
except subprocess.TimeoutExpired:
    raise SystemExit("Sanitizer timed out. Check runtime compatibility; some Apple ASan runtimes hang before main().")
PY
    bash -n "$ROOT/scripts/stage2.sh"
    python3 -m unittest discover -s tests -p 'test_*.py' -v
    "${CXX:-c++}" "${SANITIZER_FLAGS[@]}" \
        -I. -Illama.cpp/include -Illama.cpp/ggml/include -Illama.cpp/vendor \
        runner.cpp rocket.cpp rocket_adapter.cpp -o build/runner_sanitized \
        -Lllama.cpp/build/bin -Wl,-rpath,"$ROOT/llama.cpp/build/bin" \
        -lllama -lggml -lggml-base -lggml-cpu
    ROCKET_RUNNER="$ROOT/build/runner_sanitized" \
        python3 "$ROOT/tests/test_stage2_integration.py" Stage2IntegrationTests -v
    echo "Stage 2 tests passed."
)

run_stage2() (
    [[ $# -ge 1 && $# -le 3 ]] || fail "Usage: scripts/stage2.sh run MODEL.gguf [BATCH.json] [OUTPUT_DIR]"
    MODEL="$(abs_path "$1")"
    [[ -f "$MODEL" ]] || fail "model not found: $MODEL"
    INPUT="${2:-}"
    if [[ -n "$INPUT" ]]; then
        INPUT="$(abs_path "$INPUT")"
        [[ -f "$INPUT" ]] || fail "batch not found: $INPUT"
    fi
    OUTPUT="$(abs_path "${3:-$ROOT/results/stage2}")"
    build_stage2
    mkdir -p "$OUTPUT"
    if [[ -z "$INPUT" ]]; then
        INPUT="$OUTPUT/input.json"
        python3 - "$INPUT" "${MAX_TOKENS:-32}" <<'PY'
import json
import sys
prompt = ("Alice keeps the blue key in a wooden box. Bob has a red bicycle. "
          "The bird found a golden coin beside the river. ") * 12
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump([{"sample_id": 1, "prompt": prompt + "Where does Alice keep the key?",
                "max_tokens": int(sys.argv[2])}], f)
PY
    fi
    for mode in off stage1 hsa; do
        "$ROOT/runner" --model "$MODEL" --batch "$INPUT" --output "$OUTPUT/$mode.json" \
            --rocket-mode "$mode" --kv-type f16 --gpu-layers 0 \
            --budget "${BUDGET:-128}" --n-ctx "${N_CTX:-2048}" \
            --context-len "${CONTEXT_LEN:-1024}" --max-tokens "${MAX_TOKENS:-32}" \
            --window-size "${WINDOW_SIZE:-32}" --kernel-size "${KERNEL_SIZE:-63}" \
            --threads "${THREADS:-4}"
    done
    python3 - "$ROOT" "$MODEL" "$INPUT" "$OUTPUT" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
root, model, source, output = map(Path, sys.argv[1:])
samples = json.loads(source.read_text())
print("\nCPU reference comparison (includes selection and packing overhead)")
print(f"{'mode':<10} {'forwards':>10} {'ms/forward':>12} {'HSA layers':>12} {'init ms':>10}")
for mode in ("off", "stage1", "hsa"):
    rows = json.loads((output / f"{mode}.json").read_text())
    if len(rows) != len(samples):
        raise SystemExit(f"{mode}: sample count mismatch")
    for i, (row, sample) in enumerate(zip(rows, samples)):
        if row.get("status") != "ok" or row["sample_id"] != sample.get("sample_id", i + 1):
            raise SystemExit(f"{mode}: failed or mismatched sample")
        m = row["metrics"]
        if m["effective_mode"] != mode:
            raise SystemExit(f"{mode}: wrong execution path")
        if mode == "hsa" and (m["stage2_backend"] != "cpu-reference"
                or m["stage2_layer_calls"] != m["model_layers"] * m["decode_forward_calls"]):
            raise SystemExit("HSA was not executed in every layer")
    steps = sum(row["metrics"]["decode_forward_calls"] for row in rows)
    elapsed = sum(row["metrics"]["decode_forward_ms"] for row in rows)
    calls = sum(row["metrics"]["stage2_layer_calls"] for row in rows)
    setup = sum(row["metrics"]["stage2_init_ms"] for row in rows)
    latency = f"{elapsed / steps:.4f}" if steps else "n/a"
    print(f"{mode:<10} {steps:>10} {latency:>12} {calls:>12} {setup:>10.3f}")
    if mode == "hsa" and not steps:
        raise SystemExit("No decode forward ran; use a different prompt or larger max_tokens")
manifest = {
    "model": str(model), "batch": str(source), "stage2_backend": "cpu-reference",
    "llama_revision": (root / "patches/llama.cpp-revision").read_text().strip(),
    "runner_base_commit": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
}
(output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Results: {output}")
PY
)

command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
    help|-h|--help)
        [[ $# -eq 0 ]] || fail "help does not accept arguments"
        usage
        exit 0 ;;
    build|test|run)
        if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
            [[ $# -eq 1 ]] || fail "help does not accept arguments"
            usage
            exit 0
        fi ;;
    *) fail "Unknown command: $command. Use scripts/stage2.sh --help." ;;
esac
case "$command" in
    build) build_stage2 "$@" ;;
    test) test_stage2 "$@" ;;
    run) run_stage2 "$@" ;;
esac
