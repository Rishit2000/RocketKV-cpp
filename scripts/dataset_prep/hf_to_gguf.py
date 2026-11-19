#!/usr/bin/env python3
"""Download original Llama safetensors, then run llama.cpp's HF-to-GGUF converter."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Llama-3.2-3B-Instruct")
    parser.add_argument("--revision", default="main", help="Resolved to an immutable HF commit before download")
    parser.add_argument("--check-only", action="store_true", help="Check gated access and disk space without downloading weights")
    args = parser.parse_args()
    from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_url, snapshot_download

    info = HfApi().model_info(MODEL, revision=args.revision, files_metadata=True)
    shards = [f for f in info.siblings if f.rfilename.endswith(".safetensors") and "/" not in f.rfilename]
    if not shards or any(f.size is None for f in shards):
        raise RuntimeError("Model metadata did not include safetensors shard sizes")
    source_bytes = sum(f.size for f in shards)
    directory = args.model_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Safetensors are BF16; F16 output needs approximately the same space.
    existing = sum((directory / f.rfilename).stat().st_size for f in shards if (directory / f.rfilename).is_file())
    needed = max(0, source_bytes - existing) + source_bytes + 1024**3
    available = shutil.disk_usage(directory).free
    print(f"Source: {MODEL}@{info.sha}", flush=True)
    print(f"Safetensors: {source_bytes / 1024**3:.2f} GiB; additional space needed: {needed / 1024**3:.2f} GiB; free: {available / 1024**3:.2f} GiB", flush=True)
    # A public model-info response alone does not establish access to gated files.
    get_hf_file_metadata(hf_hub_url(MODEL, "config.json", revision=info.sha), token=get_token())
    if available < needed:
        raise RuntimeError("Insufficient space for safetensors and F16 GGUF; choose another --model-dir or free disk space")
    if args.check_only:
        return
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("Install converter dependencies first: python -m pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
    output = directory / "Llama-3.2-3B-Instruct-F16.gguf"
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}; preserve or move it before converting again")
    snapshot_download(MODEL, revision=info.sha, local_dir=directory,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt"],
                      ignore_patterns=["original/*"])
    partial = output.with_suffix(".partial.gguf")
    subprocess.run([sys.executable, str(ROOT / "llama.cpp/convert_hf_to_gguf.py"), str(directory),
                    "--outtype", "f16", "--outfile", str(partial)], check=True)
    partial.replace(output)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    manifest = {"model_id": MODEL, "revision": info.sha, "outtype": "f16",
                "gguf": output.name, "gguf_bytes": output.stat().st_size, "gguf_sha256": digest.hexdigest(),
                "converter_revision": subprocess.check_output(["git", "-C", str(ROOT / "llama.cpp"), "rev-parse", "HEAD"], text=True).strip()}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Converted: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.exit(f"Model preparation failed: {exc}")
