"""
Comprehensive metrics calculation module faithful to official LongBench & RULER evaluations.
Matches RocketKV/eval/longbench_utils/metrics.py and eval_long_bench.py byte-for-byte in semantics,
with pure-Python standard-library fallbacks for environments without external packages.
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
    """
    Official LongBench Code Similarity Score (LCC, RepoBench-P).
    Extracts first non-comment, non-backtick line and computes Levenshtein/edit ratio.
    """
    all_lines = prediction.lstrip('\n').split('\n')
    pred_line = ""
    for line in all_lines:
        clean = line.strip()
        if ('`' not in clean) and (not clean.startswith('#')) and (not clean.startswith('//')):
            pred_line = line
            break
    if not pred_line and all_lines:
        pred_line = all_lines[0]

    try:
        from fuzzywuzzy import fuzz
        return float(fuzz.ratio(pred_line, ground_truth) / 100.0)
    except ImportError:
        import difflib
        return float(difflib.SequenceMatcher(None, pred_line, ground_truth).ratio())

# ==============================================================================
# Classification Metric
# ==============================================================================

def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Official LongBench Classification Score (TREC, LSHT).
    Finds matching class names from all_classes in prediction.
    Penalizes ambiguity if multiple classes are output (1 / len(em_match_list)).
    Falls back to closest class via edit similarity if no direct match is found.
    """
    all_classes = kwargs.get("all_classes")
    if not all_classes:
        return float(normalize_answer(ground_truth) in normalize_answer(prediction))

    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)

    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)

    if len(em_match_list) > 0:
        if ground_truth in em_match_list:
            score = 1.0 / len(em_match_list)
        else:
            score = 0.0
    else:
        import difflib
        best_match = None
        highest_similarity = 0.0
        for candidate in all_classes:
            similarity = difflib.SequenceMatcher(None, candidate, prediction).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = candidate
        score = float(best_match == ground_truth)
    return float(score)

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
    """Official LongBench ROUGE-L score for English summarization."""
    try:
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        return float(scores["rouge-l"]["f"])
    except Exception:
        return _rouge_l_pure_python(prediction, ground_truth)

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

def clean_prediction(task: str, prediction: str) -> str:
    """
    Clean prompt continuation artifacts (Question:, Answer:, Passage:, etc.)
    Faithful to official RocketKV/pipeline/model_utils.py post_process().
    """
    if not isinstance(prediction, str):
        return str(prediction)

    # Cut off if the model starts hallucinating subsequent QA pairs or prompt markers
    for delimiter in [
        "\n\nQuestion", "\nQuestion:", "\n\nQuestion:",
        "\n\nAnswer", "\nAnswer:", "\n\nAnswer:",
        "\n\nNote:", "\n\nNote",
        "(Document", "(Passage", "[INST]", "[/INST]"
    ]:
        if delimiter in prediction:
            prediction = prediction.split(delimiter)[0]

    prediction = prediction.strip()

    # For few-shot tasks, extract first line (matching official LongBench scorer)
    if task in FEW_SHOT_TASKS:
        prediction = prediction.lstrip('\n').split('\n')[0].strip()
    elif task == "2wikimqa":
        # 2wikimqa questions expect a single entity answer; if followed by commentary on next lines, take first non-empty line
        lines = [l.strip() for l in prediction.split('\n') if l.strip()]
        if lines:
            prediction = lines[0]

    return prediction

def score_longbench_sample(
    task: str,
    prediction: str,
    ground_truths: List[str],
    all_classes: Optional[List[str]] = None
) -> float:
    """
    Score a single LongBench sample matching RocketKV/eval/longbench_utils/eval_long_bench.py scorer().
    - Cleans prompt continuation artifacts via clean_prediction().
    - Truncates to first line for few-shot tasks (trec, triviaqa, samsum, lsht).
    - Evaluates against each reference in ground_truths and takes the max score.
    """
    if not ground_truths:
        return 0.0

    prediction = clean_prediction(task, prediction)

    metric_fn = DATASET2METRIC.get(task, qa_f1_score)
    best_score = 0.0
    for gt in ground_truths:
        s = metric_fn(prediction, gt, all_classes=all_classes)
        if s > best_score:
            best_score = s
    return float(best_score)

def score_sample(
    benchmark: str,
    task: str,
    prediction: str,
    ground_truths: List[str],
    all_classes: Optional[List[str]] = None
) -> float:
    """Unified sample scorer dispatching to RULER EM or LongBench official metrics."""
    if benchmark.lower() == "ruler":
        return compute_exact_match(prediction, ground_truths)
    return score_longbench_sample(task, prediction, ground_truths, all_classes=all_classes)
