"""
Runner client module: handles subprocess execution of the C++ runner binary,
passes parameters, streams prompts via stdin, and parses structured METRICS_JSON output.
"""

import subprocess
import json
import os
import tempfile
from typing import Tuple, Dict, Any, List

def execute_runner_batch(
    runner_path: str,
    model_path: str,
    kv_type: str,
    samples: List[Dict[str, Any]],
    token_budget: Any = 1024,
    enable_rocket: bool = True,
    n_ctx: int = 9216,
    context_len: int = 8192,
    window_size: int = 32,
    kernel_size: int = 63,
) -> List[Dict[str, Any]]:
    """
    Invoke the compiled C++ runner in batch mode to process all samples sequentially
    within a single process lifecycle, eliminating the per-sample process spawn and
    shader compilation bottleneck.
    """
    in_fd, in_file = tempfile.mkstemp(suffix=".json", prefix="batch_in_")
    out_fd, out_file = tempfile.mkstemp(suffix=".json", prefix="batch_out_")
    os.close(in_fd)
    os.close(out_fd)

    try:
        with open(in_file, "w", encoding="utf-8") as f:
            json.dump(samples, f)

        cmd = [
            runner_path,
            "--model", model_path,
            "--kv-type", kv_type,
            "--rocket", "1" if enable_rocket else "0",
            "--n-ctx", str(n_ctx),
            "--context-len", str(context_len),
            "--batch", in_file,
            "--output", out_file
        ]

        if enable_rocket and token_budget not in (None, "N/A", 0, "0"):
            cmd.extend([
                "--budget", str(token_budget),
                "--window-size", str(window_size),
                "--kernel-size", str(kernel_size)
            ])

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for line in process.stderr:
            print(line, end="", flush=True)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Runner process exited with code {process.returncode}")

        with open(out_file, "r", encoding="utf-8") as f:
            batch_results = json.load(f)

        return batch_results
    finally:
        if os.path.exists(in_file):
            try:
                os.remove(in_file)
            except OSError:
                pass
        if os.path.exists(out_file):
            try:
                os.remove(out_file)
            except OSError:
                pass
