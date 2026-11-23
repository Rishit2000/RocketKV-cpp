"""
Dataset loading for LongBench and RULER.

LongBench: official THUDM/LongBench JSONL files in <data_dir>/longbench/<task>.jsonl.
RULER: inputs produced by the official NVIDIA RULER generator vendored in NVlabs/RocketKV
(eval/ruler_utils), stored as <data_dir>/ruler/<context_len>/<task>/validation.jsonl.
Create them with scripts/data/prepare_ruler.sh; nothing is generated on the fly.
"""

import os
import json
from typing import List, Dict, Any

def load_prompt_templates() -> Dict[str, str]:
    """Load official LongBench prompt templates."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, "config/dataset2prompt.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}

def load_task_max_lengths() -> Dict[str, int]:
    """Load official LongBench generation max lengths."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, "config/dataset2maxlen.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}

from .metrics import DATASET2METRIC

# ==============================================================================
# Official LongBench Dataset Loader
# ==============================================================================

LONGBENCH_TASKS = set(DATASET2METRIC.keys())

def load_longbench_dataset(task: str, num_samples: int = -1, context_len: int = -1, data_dir: str = "data") -> List[Dict[str, Any]]:
    """
    Load real evaluation instances from official LongBench JSONL files.
    If num_samples <= 0, returns the entire dataset for this task.
    Exact token-level middle-truncation is handled downstream by runner.cpp.
    """
    templates = load_prompt_templates()
    max_lens = load_task_max_lengths()

    # Search paths for the JSONL dataset
    candidates = [
        os.path.join(data_dir, "longbench", f"{task}.jsonl"),
        os.path.join(data_dir, f"{task}.jsonl"),
        os.path.join("data", "longbench", f"{task}.jsonl"),
        os.path.join("data", f"{task}.jsonl"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../data/longbench/{task}.jsonl"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../../data/longbench/{task}.jsonl"),
    ]

    found_file = None
    for p in candidates:
        if os.path.exists(p):
            found_file = p
            break

    if not found_file:
        raise FileNotFoundError(
            f"Official LongBench dataset file for task '{task}' not found in '{data_dir}/longbench'.\n"
            f"Searched: {candidates}\n"
            f"Please ensure official JSONL files are unpacked in data/longbench/."
        )

    template = templates.get(task, "Context:\n{context}\n\nQuestion: {input}\nAnswer:")
    tokens_to_generate = max_lens.get(task, 32)

    samples = []
    with open(found_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if num_samples > 0 and idx >= num_samples:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            context = item.get("context", "")
            inp = item.get("input", "")
            prompt = template.format(context=context, input=inp)
            answers = item.get("answers", [])
            all_classes = item.get("all_classes")

            samples.append({
                "task": task,
                "sample_id": idx + 1,
                "prompt": prompt,
                "answers": answers,
                "all_classes": all_classes,
                "tokens_to_generate": tokens_to_generate,
                "context_length_chars": len(prompt),
                "is_official": True
            })

    return samples


# ==============================================================================
# RocketKV LongBench prompt protocol
# ==============================================================================

# RocketKV (pipeline/baseline/eval_longbench.py) feeds these tasks without a chat wrapper.
LONGBENCH_NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
# Llama 3.2's chat template stamps the current date unless one is given. Fix it to the
# value Llama 3.1's template (used by RocketKV) hard-codes, so prompts are reproducible.
CHAT_DATE_STRING = "26 Jul 2024"


def apply_longbench_protocol(samples: List[Dict[str, Any]], tokenizer, max_len: int,
                             chat_template: str = "llama3") -> List[Dict[str, Any]]:
    """Match RocketKV get_pred(): middle-truncate the raw prompt to max_len HF tokens
    (BOS included, decoded without special tokens), then apply the model chat template
    except for LONGBENCH_NO_CHAT_TASKS. The runner tokenizes with add_special=true and
    parses special tokens, so chat prompts start with two BOS tokens, as in RocketKV."""
    if chat_template not in (None, "llama3"):
        raise ValueError(f"Unsupported chat_template: {chat_template}")
    out = []
    for sample in samples:
        prompt = sample["prompt"]
        ids = tokenizer(prompt, truncation=False).input_ids
        if max_len > 0 and len(ids) > max_len:
            half = int(max_len / 2)
            prompt = (tokenizer.decode(ids[:half], skip_special_tokens=True)
                      + tokenizer.decode(ids[-half:], skip_special_tokens=True))
        if chat_template and sample["task"] not in LONGBENCH_NO_CHAT_TASKS:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                   add_generation_prompt=True, date_string=CHAT_DATE_STRING)
        out.append({**sample, "prompt": prompt, "raw_prompt_tokens": len(ids)})
    return out

# ==============================================================================
# Official RULER inputs
# ==============================================================================

def ruler_file(data_dir: str, context_len: int, task: str) -> str:
    return os.path.join(data_dir, "ruler", str(context_len), task, "validation.jsonl")


def load_ruler_dataset(task: str, context_len: int, num_samples: int = -1, data_dir: str = "data",
                       max_new_tokens: int = 128) -> List[Dict[str, Any]]:
    """Load official RULER samples. `input` already ends with the llama3 chat template and the
    task answer prefix; like RocketKV it is used as-is and every task generates max_new_tokens."""
    path = ruler_file(data_dir, context_len, task)
    if not os.path.exists(path):
        raise FileNotFoundError(f"RULER file not found: {path}. Run scripts/data/prepare_ruler.sh first.")
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if num_samples > 0 and idx >= num_samples:
                break
            item = json.loads(line)
            samples.append({
                "task": task,
                "sample_id": idx + 1,
                "ruler_index": item["index"],
                "context_len": context_len,
                "prompt": item["input"],
                "answers": item["outputs"],
                "tokens_to_generate": max_new_tokens,
                "ruler_length": item.get("length"),
            })
    if num_samples > 0 and len(samples) != num_samples:
        raise ValueError(f"{path}: expected {num_samples} samples, found {len(samples)}")
    return samples

# ==============================================================================
# Master Loader Function
# ==============================================================================

def load_evaluation_samples(benchmark: str, task: str, context_len: int, num_samples: int = -1,
                            data_dir: str = "data") -> List[Dict[str, Any]]:
    """Dispatch to the LongBench or official RULER loader."""
    if benchmark == "longbench" or (benchmark == "all" and task in LONGBENCH_TASKS):
        return load_longbench_dataset(task=task, num_samples=num_samples, context_len=context_len, data_dir=data_dir)
    return load_ruler_dataset(task=task, context_len=context_len, num_samples=num_samples, data_dir=data_dir)
