"""C++ runner backend used by the upstream RocketKV evaluation scripts."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from eval_suite.runner import execute_runner_batch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Llama-3.2-3B-Instruct" / "Llama-3.2-3B-Instruct-F16.gguf"
DEFAULT_RUNNER = ROOT / "runner"

# Llama-3.2-3B-Instruct attention layout (needed by upstream compress()).
LLAMA32_3B_CONFIG = SimpleNamespace(num_attention_heads=24, num_key_value_heads=8)


class RunnerModel:
    """Stand-in for the Hugging Face model object expected by upstream scripts."""

    def __init__(self, model_path: Path, runner_path: Path, pipeline_params: Dict[str, Any]):
        self.model_path = Path(model_path)
        self.runner_path = Path(runner_path)
        self.pipeline_params = pipeline_params
        self.config = LLAMA32_3B_CONFIG

    def eval(self):
        return self


def resolve_model_path(pipeline_params: Optional[Dict[str, Any]] = None) -> Path:
    env = os.environ.get("ROCKETKV_MODEL")
    if env:
        return Path(env)
    return DEFAULT_MODEL


def resolve_runner_path() -> Path:
    env = os.environ.get("ROCKETKV_RUNNER")
    if env:
        return Path(env)
    return DEFAULT_RUNNER


def resolve_tokenizer_dir(pipeline_params: Dict[str, Any], model_path: Path) -> Path:
    env = os.environ.get("ROCKETKV_TOKENIZER")
    if env:
        return Path(env)
    # Prefer the local conversion directory next to the GGUF.
    if (model_path.parent / "tokenizer.json").exists() or (model_path.parent / "tokenizer_config.json").exists():
        return model_path.parent
    name = pipeline_params.get("tokenizer_name") or pipeline_params.get("model_name")
    return Path(name)


def initialize_model_tokenizer(pipeline_config: Dict[str, Any]):
    from transformers import AutoTokenizer

    model_path = resolve_model_path(pipeline_config)
    runner_path = resolve_runner_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"GGUF model not found at {model_path}. "
            "Run scripts/dataset_prep/hf_to_gguf.py or set ROCKETKV_MODEL."
        )
    if not runner_path.is_file():
        raise FileNotFoundError(f"Runner not found at {runner_path}. Build with: make runner (or set ROCKETKV_RUNNER).")
    tokenizer_dir = resolve_tokenizer_dir(pipeline_config, model_path)
    local_only = (tokenizer_dir / "tokenizer.json").exists() or (tokenizer_dir / "tokenizer_config.json").exists()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=local_only, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = RunnerModel(model_path, runner_path, pipeline_config)
    return model, tokenizer


def strip_bos(tokenizer, prompt: str) -> str:
    # runner.cpp adds BOS during tokenization; avoid adding it twice.
    if getattr(tokenizer, "bos_token", None) and prompt.startswith(tokenizer.bos_token):
        return prompt[len(tokenizer.bos_token) :]
    return prompt


def run_prompts(
    model: RunnerModel,
    prompts: List[str],
    max_new_tokens: int,
    pipeline_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    context_len = int(os.environ.get("ROCKETKV_CONTEXT_LEN", pipeline_params.get("model_max_len", 8192)))
    n_ctx = context_len + 1024
    method = pipeline_params.get("method", "rocket")
    enable_rocket = method != "baseline"
    token_budget = pipeline_params.get("token_budget", 1024)
    window_size = int(pipeline_params.get("window_size", 32) or 32)
    kernel_size = int(pipeline_params.get("kernel_size", 63) or 63)
    samples = [
        {"sample_id": idx + 1, "prompt": prompt, "max_tokens": max_new_tokens}
        for idx, prompt in enumerate(prompts)
    ]
    return execute_runner_batch(
        str(model.runner_path.resolve()),
        str(model.model_path.resolve()),
        "f16",
        samples,
        token_budget=token_budget if enable_rocket else "N/A",
        enable_rocket=enable_rocket,
        n_ctx=n_ctx,
        context_len=context_len,
        window_size=window_size,
        kernel_size=kernel_size,
    )
