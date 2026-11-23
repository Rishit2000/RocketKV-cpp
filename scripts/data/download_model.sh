#!/usr/bin/env bash
# Download the evaluated model into models/:
#   Llama-3.2-3B-Instruct-Q8_0.gguf          GGUF weights run by build/runner
#   Llama-3.2-3B-Instruct-tokenizer/         HF tokenizer for LongBench truncation/chat
#                                            template and RULER generation
# The tokenizer comes from unsloth's ungated copy of meta-llama/Llama-3.2-3B-Instruct.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
models="$root/models"
mkdir -p "$models"

fetch() {  # fetch URL DEST: download atomically, skip if present
    [[ -s "$2" ]] && { echo "present: ${2#$root/}"; return; }
    curl -fL --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
    echo "downloaded: ${2#$root/}"
}

fetch "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf" \
      "$models/Llama-3.2-3B-Instruct-Q8_0.gguf"

tok="$models/Llama-3.2-3B-Instruct-tokenizer"
mkdir -p "$tok"
for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do
    fetch "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct/resolve/main/$f" "$tok/$f"
done
