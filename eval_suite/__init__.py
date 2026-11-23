"""
eval_suite package initialization.
Exports metrics, dataset loaders, runner client, and evaluation engine.
"""

from .metrics import (
    normalize_text,
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    compute_edit_similarity,
    compute_retrieval_score,
    compute_count_score,
    score_longbench_sample,
    score_ruler_sample,
    score_sample
)

from .datasets import (
    load_longbench_dataset,
    load_ruler_dataset,
    load_evaluation_samples
)

from .runner import execute_runner_batch

from .config import (
    DEFAULT_CONFIG,
    load_config_file,
    build_attention_modes
)

from .engine import (
    run_evaluation,
    print_task_comparison_table,
    ALL_LONGBENCH_TASKS,
    LONGBENCH_EN_TASKS,
    LONGBENCH_ZH_TASKS,
    ALL_RULER_TASKS
)

__all__ = [
    "normalize_text",
    "compute_exact_match",
    "compute_f1",
    "compute_rouge_l",
    "compute_edit_similarity",
    "compute_retrieval_score",
    "compute_count_score",
    "score_longbench_sample",
    "score_sample",
    "score_ruler_sample",
    "load_longbench_dataset",
    "load_ruler_dataset",
    "load_evaluation_samples",
    "execute_runner_batch",
    "DEFAULT_CONFIG",
    "load_config_file",
    "build_attention_modes",
    "run_evaluation",
    "print_task_comparison_table",
    "ALL_LONGBENCH_TASKS",
    "LONGBENCH_EN_TASKS",
    "LONGBENCH_ZH_TASKS",
    "ALL_RULER_TASKS"
]

