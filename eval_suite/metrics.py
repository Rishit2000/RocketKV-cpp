"""
Comprehensive metrics calculation module faithful to official LongBench & RULER evaluations.
LongBench metrics follow RocketKV/eval/longbench_utils/metrics.py and eval_long_bench.py;
the English metrics require rouge and fuzzywuzzy (no silent fallbacks).
"""

import re
import string
from collections import Counter
from typing import List, Optional, Dict, Any

# ==============================================================================
# Normalization Functions
# ==============================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace (English)."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_zh_answer(s: str) -> str:
    """Lower text and remove punctuation, extra whitespace (Chinese)."""
    def white_space_fix(text: str) -> str:
        return "".join(text.split())

    def remove_punc(text: str) -> str:
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

normalize_text = normalize_answer  # Backward compatibility alias

# ==============================================================================
# QA F1 Metrics
# ==============================================================================

def f1_score(prediction_tokens: List[str], ground_truth_tokens: List[str], **kwargs) -> float:
    """Token-level F1 calculation."""
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0 or len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return float(f1)

def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Official LongBench QA F1 score for English QA tasks."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Official LongBench QA F1 score for Chinese QA tasks."""
    try:
        import jieba
        prediction_tokens = list(jieba.cut(prediction, cut_all=False))
        ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    except ImportError:
        prediction_tokens = list(prediction)
        ground_truth_tokens = list(ground_truth)

    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)

# ==============================================================================
# Counting & Retrieval Metrics
# ==============================================================================

def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Official LongBench Counting Score: extracts integer counts via regex r'\\d+'.
    Penalizes guessing multiple numbers: right_num / len(numbers).
    """
    numbers = re.findall(r"\d+", prediction)
    gt_numbers = re.findall(r"\d+", str(ground_truth))
    gt_target = gt_numbers[0] if gt_numbers else str(ground_truth).strip()
    right_num = 0
    for number in numbers:
        if str(number) == gt_target:
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Official LongBench Retrieval Score (English):
    Extracts 'Paragraph (\\d+)' from ground truth and checks occurrences in prediction.
    """
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth, re.IGNORECASE)
    if matches:
        ground_truth_id = matches[0]
    else:
        num_matches = re.findall(r'\d+', ground_truth)
        ground_truth_id = num_matches[0] if num_matches else ground_truth.strip()

    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Official LongBench Retrieval Score (Chinese):
    Extracts '段落(\\d+)' from ground truth and checks occurrences in prediction.
    """
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    if matches:
        ground_truth_id = matches[0]
    else:
        num_matches = re.findall(r'\d+', ground_truth)
        ground_truth_id = num_matches[0] if num_matches else ground_truth.strip()

    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

# ==============================================================================
# Code Similarity Metric
# ==============================================================================

def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Official LongBench code similarity (LCC, RepoBench-P): first line without `, # or //."""
    from fuzzywuzzy import fuzz
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100

# ==============================================================================
# Classification Metric
# ==============================================================================

def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Official LongBench classification score (TREC, LSHT), kept bug-for-bug: removal while
    iterating can skip items, and `em_match_list != 0` is always true, so an output naming
    no class scores 0 instead of falling back to the closest class."""
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0

# ==============================================================================
# ROUGE-L Metrics
# ==============================================================================

def _rouge_l_pure_python(prediction: str, ground_truth: str) -> float:
    """Pure-Python Longest Common Subsequence (LCS) ROUGE-L matching pltrdy/rouge."""
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        text = "".join(ch if ch.isalnum() else " " for ch in text)
        return text.split()

    x = tokenize(prediction)
    y = tokenize(ground_truth)
    if not x or not y:
        return 0.0

    n, m = len(x), len(y)
    dp = [0] * (m + 1)
    for i in range(n):
        cur = [0] * (m + 1)
        for j in range(m):
            if x[i] == y[j]:
                cur[j + 1] = dp[j] + 1
            else:
                cur[j + 1] = max(dp[j + 1], cur[j])
        dp = cur

    lcs_len = dp[m]
    if lcs_len == 0:
        return 0.0
    p = lcs_len / n
    r = lcs_len / m
    f = (2.0 * p * r) / (p + r)
    return float(f)

def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Official LongBench ROUGE-L F1 (pltrdy/rouge); any rouge error scores 0."""
    from rouge import Rouge
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return float(scores["rouge-l"]["f"])

def rouge_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Official LongBench ROUGE-L score for Chinese summarization."""
    try:
        import jieba
        pred_cut = " ".join(list(jieba.cut(prediction, cut_all=False)))
        gt_cut = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    except ImportError:
        pred_cut = " ".join(list(prediction))
        gt_cut = " ".join(list(ground_truth))
    return rouge_score(pred_cut, gt_cut)

# ==============================================================================
# Exact Match & RULER Metrics
# ==============================================================================

def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Exact Match (EM) / substring match for RULER tasks."""
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if norm_gt and (norm_gt in norm_pred or norm_pred == norm_gt):
            return 1.0
        if gt.strip().lower() in prediction.strip().lower():
            return 1.0
    return 0.0

# Backward compatibility wrappers
def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    return max((qa_f1_score(prediction, gt) for gt in ground_truths), default=0.0)

def compute_rouge_l(prediction: str, ground_truths: List[str]) -> float:
    return max((rouge_score(prediction, gt) for gt in ground_truths), default=0.0)

def compute_edit_similarity(prediction: str, ground_truths: List[str]) -> float:
    return max((code_sim_score(prediction, gt) for gt in ground_truths), default=0.0)

def compute_retrieval_score(prediction: str, ground_truths: List[str]) -> float:
    return max((retrieval_score(prediction, gt) for gt in ground_truths), default=0.0)

def compute_count_score(prediction: str, ground_truths: List[str]) -> float:
    return max((count_score(prediction, gt) for gt in ground_truths), default=0.0)

# ==============================================================================
# Task-to-Metric Mapping and Official Scorer
# ==============================================================================

DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

FEW_SHOT_TASKS = {"trec", "triviaqa", "samsum", "lsht"}

def score_longbench_sample(
    task: str,
    prediction: str,
    ground_truths: List[str],
    all_classes: Optional[List[str]] = None
) -> float:
    """
    Same as RocketKV eval/longbench_utils/eval_long_bench.py scorer() for one sample.
    RocketKV's post_process() leaves llama3 outputs untouched, so the only cleanup is
    keeping the first line for few-shot tasks; the score is the max over references.
    """
    if task in FEW_SHOT_TASKS:
        prediction = prediction.lstrip('\n').split('\n')[0]
    metric_fn = DATASET2METRIC[task]
    score = 0.0
    for gt in ground_truths:
        score = max(score, metric_fn(prediction, gt, all_classes=all_classes))
    return float(score)

# ==============================================================================
# RULER metrics (RocketKV eval/ruler_utils/eval, i.e. official NVIDIA RULER)
# ==============================================================================

# qa_* use string_match_part; every other synthetic task uses string_match_all.
RULER_PART_MATCH_TASKS = {"qa_1", "qa_2"}

def postprocess_ruler_prediction(prediction: str) -> str:
    """RULER evaluate.py postprocess_pred(): strip, map control characters to newlines."""
    return re.sub(r'[\x00-\x1f]', '\n', prediction.strip()).strip()

def score_ruler_sample(task: str, prediction: str, references: List[str]) -> float:
    """Per-sample RULER score in [0, 1]; the task score is the mean x 100."""
    if not references:
        raise ValueError("RULER sample has no references")
    pred = postprocess_ruler_prediction(prediction).lower()
    hits = [1.0 if r.lower() in pred else 0.0 for r in references]
    return max(hits) if task in RULER_PART_MATCH_TASKS else sum(hits) / len(hits)

def score_sample(
    benchmark: str,
    task: str,
    prediction: str,
    ground_truths: List[str],
    all_classes: Optional[List[str]] = None
) -> float:
    """Dispatch to the RULER or LongBench scorer."""
    if benchmark.lower() == "ruler":
        return score_ruler_sample(task, prediction, ground_truths)
    return score_longbench_sample(task, prediction, ground_truths, all_classes=all_classes)
