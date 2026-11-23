#!/usr/bin/env bash
# Download the official LongBench release (THUDM/LongBench data.zip, 34 JSONL files
# including the LongBench-E variants) into data/longbench/<task>.jsonl.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
dest="$root/data/longbench"
if [[ -s "$dest/repobench-p.jsonl" && -s "$dest/narrativeqa.jsonl" ]]; then
    echo "LongBench already present in ${dest#$root/}"; exit 0
fi
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
curl -fL --retry 3 -o "$tmp/data.zip" https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip -q "$tmp/data.zip" -d "$tmp"
mkdir -p "$dest"
mv "$tmp"/data/*.jsonl "$dest/"
echo "$(ls "$dest" | wc -l | tr -d ' ') files in ${dest#$root/}"
