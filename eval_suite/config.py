"""
Configuration management module for RocketKV evaluation suite.
Loads structured JSON config files with defaults and path resolution.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "path": "models/Llama-3.2-3B-Instruct-Q8_0.gguf",
        "runner": "./build/runner",
        "max_tokens": 32,
        "n_ctx": 9216
    },
    "benchmark": {
        "suite": "longbench",
        "tasks": "all",
        "context_len": 8192,
        "num_samples": 50
    },
    "attention": {
        "attn_type": "all",
        "kv_type": "q8_0",
        "budget": 1024,
        "window_size": 32,
        "kernel_size": 63
    },
    "management": {
        "output_dir": "results",
        "data_dir": "data"
    }
}

def load_config_file(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration dictionary from JSON file with smart path resolution and defaults."""
    config = json.loads(json.dumps(DEFAULT_CONFIG)) # Deep copy

    resolved_path = None
    if config_path:
        candidates = [
            config_path,
            os.path.join(os.path.dirname(__file__), "..", config_path),
            os.path.join(os.path.dirname(__file__), "../configs", os.path.basename(config_path)),
            os.path.join("configs", os.path.basename(config_path)),
            os.path.join("RocketKV-llama", config_path)
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break
        if not resolved_path:
            raise FileNotFoundError(f"Configuration file not found: '{config_path}' (searched: {candidates})")
    else:
        # Default fallback candidates
        candidates = [
            "configs/default.json",
            os.path.join(os.path.dirname(__file__), "../configs/default.json"),
            "config.json"
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break

    if resolved_path and os.path.exists(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            for section, vals in loaded.items():
                if isinstance(vals, dict) and section in config:
                    config[section].update(vals)
                else:
                    config[section] = vals

    # Post-process & Normalize configurations
    bench = config["benchmark"]
    if isinstance(bench.get("num_samples"), str):
        val_str = bench["num_samples"].strip().lower()
        if val_str in ["all", "full", "-1"]:
            bench["num_samples"] = -1
        else:
            bench["num_samples"] = int(val_str)

    if isinstance(bench.get("tasks"), str) and "," in bench["tasks"]:
        bench["tasks"] = [t.strip() for t in bench["tasks"].split(",") if t.strip()]

    # Resolve runner path if relative
    runner = config["model"].get("runner", "./build/runner")
    if not os.path.isabs(runner):
        candidates = [
            runner,
            os.path.join(os.path.dirname(__file__), "..", runner),
            os.path.join(os.getcwd(), runner),
            os.path.join(os.getcwd(), "RocketKV-llama", runner)
        ]
        for c in candidates:
            if os.path.exists(c):
                config["model"]["runner"] = os.path.abspath(c)
                break

    # Resolve model path if relative
    model_path = config["model"].get("path", "")
    if model_path and not os.path.isabs(model_path):
        candidates = [
            model_path,
            os.path.join(os.path.dirname(__file__), "..", model_path),
            os.path.join(os.getcwd(), model_path),
            os.path.join(os.getcwd(), "RocketKV-llama", model_path)
        ]
        for c in candidates:
            if os.path.exists(c):
                config["model"]["path"] = os.path.abspath(c)
                break

    return config

def build_attention_modes(
    attn_type: str,
    kv_type: str,
    budget: Union[int, List[int], str],
    window_size: int = 32,
    kernel_size: int = 63
) -> List[Dict[str, Any]]:
    """Build list of attention mode dictionaries based on requested type and budget(s)."""
    if isinstance(budget, str):
        budgets = [int(b.strip()) for b in budget.split(",") if b.strip()]
    elif isinstance(budget, list):
        budgets = [int(b) for b in budget]
    else:
        budgets = [int(budget)]

    modes = []
    if attn_type in ["full-kv", "all"]:
        modes.append({
            "name": "full-kv",
            "kv_type": "f16",
            "rocket": False,
            "budget": "N/A",
            "window_size": "N/A",
            "kernel_size": "N/A"
        })
    if attn_type in ["quantized-kv", "all"]:
        modes.append({
            "name": f"quantized-kv ({kv_type})",
            "kv_type": kv_type,
            "rocket": False,
            "budget": "N/A",
            "window_size": "N/A",
            "kernel_size": "N/A"
        })
    if attn_type in ["rocket-kv", "all"]:
        for b in budgets:
            modes.append({
                "name": f"rocket-kv (budget={b})",
                "kv_type": "f16",
                "rocket": True,
                "budget": b,
                "window_size": window_size,
                "kernel_size": kernel_size
            })
    return modes
