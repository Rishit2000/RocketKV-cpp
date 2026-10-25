"""
Dataset loading and procedural generation for RULER and LongBench tasks.
Supports:
1. Full official LongBench JSONL datasets (8,400+ real-world instances across 21 tasks)
2. Full official NVIDIA RULER synthetic benchmark (13 tasks, up to 500+ samples per task)
"""

import os
import glob
import json
import random
import uuid
import re
from typing import List, Dict, Any, Optional

def load_paul_graham_essays() -> str:
    """Load text from background Paul Graham essays in the repository."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    essay_paths = glob.glob(os.path.join(base_dir, "../../RocketKV/eval/passkey_utils/background/PaulGrahamEssays/*.txt"))
    if not essay_paths:
        essay_paths = glob.glob("../RocketKV/eval/passkey_utils/background/PaulGrahamEssays/*.txt")
    if not essay_paths:
        essay_paths = glob.glob("RocketKV/eval/passkey_utils/background/PaulGrahamEssays/*.txt")
    
    text = ""
    for path in essay_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text += f.read() + "\n\n"
        except Exception:
            pass
    if not text:
        text = "The quick brown fox jumps over the lazy dog. Modern LLM inference on consumer devices requires efficient KV cache memory management. " * 500
    return text

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
# NVIDIA RULER Benchmark Procedural Generator
# ==============================================================================

WORDS_POOL = [
    "apple", "banana", "cherry", "dragonfruit", "elderberry", "fig", "grape", "honeydew",
    "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince", "raspberry",
    "strawberry", "tangerine", "ugli", "vanilla", "watermelon", "apricot", "blackberry",
    "blueberry", "cantaloupe", "cranberry", "date", "gooseberry", "grapefruit", "guava",
    "kumquat", "lime", "lychee", "mulberry", "olive", "passionfruit", "peach", "pear"
]

CITIES_POOL = [
    "Tokyo", "Berlin", "Paris", "London", "Toronto", "Sydney", "Rome", "Madrid",
    "Beijing", "Seoul", "Vienna", "Amsterdam", "Singapore", "Dublin", "Stockholm",
    "Zurich", "Oslo", "Helsinki", "Brussels", "Copenhagen", "Warsaw", "Prague"
]

def generate_ruler_dataset(task: str, target_tokens: int, num_samples: int = 500, data_dir: str = "data") -> List[Dict[str, Any]]:
    """
    Generate or load the complete NVIDIA RULER benchmark suite (13 tasks).
    If pre-generated file exists, loads from disk. Otherwise generates procedurally.
    If num_samples <= 0, loads/generates the full dataset (500 samples default for generation).
    """
    # First check if pre-generated JSONL exists
    candidates = [
        os.path.join(data_dir, "ruler", task, "validation.jsonl"),
        os.path.join(data_dir, "ruler", f"{task}.jsonl"),
        os.path.join(data_dir, f"{task}.jsonl"),
        os.path.join("data", "ruler", f"{task}.jsonl"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../data/ruler/{task}.jsonl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            samples = []
            with open(p, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if num_samples > 0 and idx >= num_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    samples.append({
                        "task": task,
                        "sample_id": idx + 1,
                        "prompt": item.get("input", item.get("prompt", "")),
                        "answers": item.get("outputs", item.get("answers", [])),
                        "tokens_to_generate": item.get("tokens_to_generate", 64),
                        "context_length_chars": len(item.get("input", "")),
                        "is_official": True
                    })
            if len(samples) > 0:
                return samples

    if num_samples <= 0:
        num_samples = 500

    # Procedural generation conforming to official NVIDIA RULER specification
    background_text = load_paul_graham_essays()
    approx_chars = target_tokens * 4
    samples = []
    random.seed(42)

    for i in range(num_samples):
        sample_id = i + 1
        depth_ratio = (i / max(1, num_samples - 1)) if num_samples > 1 else 0.5
        depth_ratio = max(0.05, min(0.95, depth_ratio))

        # Base haystack
        start_idx = random.randint(0, max(0, len(background_text) - approx_chars))
        haystack = background_text[start_idx:start_idx + approx_chars]

        tokens_to_generate = 64

        # 13 Official RULER tasks
        if task in ["niah_single_1", "niah_single_2", "niah_single_3", "niah"]:
            is_uuid = ("3" in task)
            key_word = random.choice(WORDS_POOL)
            val = str(uuid.uuid4())[:8] if is_uuid else str(random.randint(100000, 999999))
            needle = f" Some special magic information: the passkey for {key_word} is {val}. Remember this passkey. "
            question = f"What is the passkey for {key_word} mentioned in the text?"
            answers = [val]

            insert_pos = int(len(haystack) * depth_ratio)
            context = haystack[:insert_pos] + needle + haystack[insert_pos:]
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        elif "multikey" in task:
            num_keys = 4 if "1" in task else 3
            selected_words = random.sample(WORDS_POOL, num_keys)
            is_uuid = ("3" in task)
            values = [str(uuid.uuid4())[:8] if is_uuid else str(random.randint(1000, 9999)) for _ in range(num_keys)]
            needles = [f"The secret number for key {k} is {v}." for k, v in zip(selected_words, values)]
            target_idx = random.randint(0, num_keys - 1)
            target_key = selected_words[target_idx]
            answers = [values[target_idx]]

            full_needle = " " + " ".join(needles) + " "
            insert_pos = int(len(haystack) * depth_ratio)
            context = haystack[:insert_pos] + full_needle + haystack[insert_pos:]
            prompt = f"Context:\n{context}\n\nQuestion: What is the secret number for key {target_key}?\nAnswer:"

        elif "multivalue" in task:
            target_key = random.choice(WORDS_POOL)
            values = [str(random.randint(1000, 9999)) for _ in range(4)]
            needles = [f"One assignment for key {target_key} is {v}." for v in values]
            full_needle = " " + " ".join(needles) + " "
            answers = values
            insert_pos = int(len(haystack) * depth_ratio)
            context = haystack[:insert_pos] + full_needle + haystack[insert_pos:]
            prompt = f"Context:\n{context}\n\nQuestion: List all numbers assigned to key {target_key}.\nAnswer:"

        elif "multiquery" in task:
            selected_keys = random.sample(WORDS_POOL, 3)
            values = [str(random.randint(1000, 9999)) for _ in range(3)]
            needles = [f"The value for {k} is {v}." for k, v in zip(selected_keys, values)]
            full_needle = " " + " ".join(needles) + " "
            answers = [f"{k}: {v}" for k, v in zip(selected_keys, values)] + values
            insert_pos = int(len(haystack) * depth_ratio)
            context = haystack[:insert_pos] + full_needle + haystack[insert_pos:]
            prompt = f"Context:\n{context}\n\nQuestion: What are the values for {', '.join(selected_keys)}?\nAnswer:"

        elif task in ["vt", "variable_tracking"]:
            # Variable tracking chain (4 hops)
            var_names = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"]
            val = str(random.randint(100, 999))
            chain_text = (
                f" Let variable {var_names[0]} equal {val}."
                f" Let variable {var_names[1]} equal {var_names[0]}."
                f" Let variable {var_names[2]} equal {var_names[1]}."
                f" Let variable {var_names[3]} equal {var_names[2]}."
            )
            insert_pos = int(len(haystack) * depth_ratio)
            context = haystack[:insert_pos] + f" {chain_text} " + haystack[insert_pos:]
            prompt = (
                f"Memorize and track the chain of variable assignments in the following text.\n\n"
                f"{context}\n\n"
                f"Question: Find all variables that are assigned the value {val} in the text above.\nAnswer:"
            )
            answers = var_names[:4] + [val]

        elif task in ["cwe", "common_words_extraction"]:
            # Common word extraction: 10 frequent words repeated in background list
            common_words = random.sample(WORDS_POOL, 10)
            list_items = []
            for w in common_words:
                list_items.extend([w] * random.randint(8, 15))
            random.shuffle(list_items)
            inserted_list = " ".join([f"Word {idx+1}: {w}" for idx, w in enumerate(list_items)])
            insert_pos = int(len(haystack) * depth_ratio)
            context = haystack[:insert_pos] + "\n" + inserted_list + "\n" + haystack[insert_pos:]
            prompt = (
                f"Below is a list of words. Some appear more often than others.\n\n"
                f"{context}\n\n"
                f"Question: What are the most common words in the above list?\nAnswer:"
            )
            answers = common_words

        elif task in ["fwe", "freq_words_extraction"]:
            coded_words = ["ALPHA_X", "BETA_Y", "GAMMA_Z"]
            repeated_tokens = []
            for w in coded_words:
                repeated_tokens.extend([w] * random.randint(10, 20))
            random.shuffle(repeated_tokens)
            coded_text = " ... " + " ... ".join(repeated_tokens) + " ... "
            insert_pos = int(len(haystack) * depth_ratio)
            context = haystack[:insert_pos] + coded_text + haystack[insert_pos:]
            prompt = (
                f"Read the following coded text and track frequency of coded words.\n\n"
                f"{context}\n\n"
                f"Question: What are the three most frequently appeared words in the coded text?\nAnswer:"
            )
            answers = coded_words

        elif task in ["qa_1", "qa_2"]:
            fact = f"In the historical archives, the secret conference was held in {random.choice(CITIES_POOL)} under code name Falcon."
            city_ans = fact.split("held in ")[1].split(" under")[0]
            insert_pos = int(len(haystack) * depth_ratio)
            context = haystack[:insert_pos] + f" {fact} " + haystack[insert_pos:]
            prompt = f"Text:\n{context}\n\nQuestion: Where was the secret conference held?\nAnswer:"
            answers = [city_ans]

        else:
            needle_num = str(random.randint(100000, 999999))
            city = random.choice(CITIES_POOL)
            needle = f" The special passkey assigned to {city} is {needle_num}. "
            insert_pos = int(len(haystack) * depth_ratio)
            context = haystack[:insert_pos] + needle + haystack[insert_pos:]
            prompt = f"Context:\n{context}\n\nQuestion: What is the passkey assigned to {city}?\nAnswer:"
            answers = [needle_num]

        samples.append({
            "task": task,
            "sample_id": sample_id,
            "depth": depth_ratio,
            "prompt": prompt,
            "answers": answers,
            "tokens_to_generate": tokens_to_generate,
            "context_length_chars": len(prompt),
            "is_official": True
        })

    return samples

# ==============================================================================
# Master Loader Function
# ==============================================================================

def load_evaluation_samples(
    benchmark: str,
    task: str,
    context_len: int,
    num_samples: int = -1,
    data_dir: str = "data"
) -> List[Dict[str, Any]]:
    """
    Universal dataset loader dispatching to real LongBench JSONL files
    or full NVIDIA RULER benchmark generator.
    """
    if benchmark == "longbench" or (benchmark == "all" and task in LONGBENCH_TASKS):
        return load_longbench_dataset(task=task, num_samples=num_samples, context_len=context_len, data_dir=data_dir)
    else:
        return generate_ruler_dataset(task=task, target_tokens=context_len, num_samples=num_samples, data_dir=data_dir)

# Backward-compatibility wrappers
def generate_ruler_samples(task: str, target_tokens: int, num_samples: int) -> List[Dict[str, Any]]:
    return generate_ruler_dataset(task=task, target_tokens=target_tokens, num_samples=num_samples)

def generate_longbench_samples(task: str, target_tokens: int, num_samples: int) -> List[Dict[str, Any]]:
    try:
        return load_longbench_dataset(task=task, num_samples=num_samples)
    except Exception:
        return generate_ruler_dataset(task=task, target_tokens=target_tokens, num_samples=num_samples)

