"""Task-level metrics shared by automatic training validation and the API."""

from __future__ import annotations

from collections import Counter
from typing import Any


def metric_tokens(text: str) -> list[str]:
    """Normalize answer text into lowercase word and number tokens."""
    import re

    return re.findall(r"[a-z]+|[+-]?\d+(?:\.\d+)?", text.casefold().replace(",", ""))


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Return the longest common subsequence length using linear memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def calculate_answer_metrics(cases: list[dict[str, str]], split: str,
                             requested_cases: int, available_cases: int) -> dict[str, Any]:
    """Calculate reference-overlap scores for held-out free-form answers.

    These lexical scores compare generated answers with the supplied reference;
    they do not measure factuality or whether a response is safe to deploy.
    """
    token_precision_scores: list[float] = []
    token_recall_scores: list[float] = []
    token_f1_scores: list[float] = []
    rouge_l_f1_scores: list[float] = []
    exact_matches = 0
    numeric_examples = 0
    numeric_within_tolerance = 0

    def numeric_value(text: str) -> float | None:
        import re

        matches = re.findall(r"(?<![A-Za-z0-9])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
        if not matches:
            return None
        try:
            return float(matches[-1].replace(",", ""))
        except ValueError:
            return None

    for item in cases:
        expected_tokens = metric_tokens(item["expected"])
        actual_tokens = metric_tokens(item["actual"])
        exact_matches += int(expected_tokens == actual_tokens)
        expected_number = numeric_value(item["expected"])
        if expected_number is not None:
            numeric_examples += 1
            actual_number = numeric_value(item["actual"])
            if actual_number is not None:
                tolerance = max(abs(expected_number) * 0.01, 1e-9)
                numeric_within_tolerance += int(abs(actual_number - expected_number) <= tolerance)
        expected_counts = Counter(expected_tokens)
        actual_counts = Counter(actual_tokens)
        overlap = sum((expected_counts & actual_counts).values())
        precision = overlap / len(actual_tokens) if actual_tokens else 0.0
        recall = overlap / len(expected_tokens) if expected_tokens else 0.0
        token_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        lcs = _lcs_length(expected_tokens, actual_tokens)
        rouge_precision = lcs / len(actual_tokens) if actual_tokens else 0.0
        rouge_recall = lcs / len(expected_tokens) if expected_tokens else 0.0
        rouge_l_f1 = (2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall)
                      if rouge_precision + rouge_recall else 0.0)
        token_precision_scores.append(precision)
        token_recall_scores.append(recall)
        token_f1_scores.append(token_f1)
        rouge_l_f1_scores.append(rouge_l_f1)

    count = len(cases)
    mean = lambda scores: sum(scores) / len(scores) if scores else 0.0
    return {
        "metric_version": 3,
        "task_type": "answer_generation",
        "split": split,
        "examples_evaluated": count,
        "available_examples": available_cases,
        "requested_cases": requested_cases,
        "normalized_exact_match": exact_matches / count if count else 0.0,
        "exact_match_count": exact_matches,
        "numeric_examples": numeric_examples,
        "numeric_accuracy_within_1pct": numeric_within_tolerance / numeric_examples if numeric_examples else None,
        "numeric_within_1pct_count": numeric_within_tolerance,
        "rouge_l_f1": mean(rouge_l_f1_scores),
        "token_precision": mean(token_precision_scores),
        "token_recall": mean(token_recall_scores),
        "token_f1": mean(token_f1_scores),
        "confusion_matrix": None,
        "confusion_matrix_note": "Not applicable to free-form answer generation; this evaluation dataset has no discrete class labels.",
    }
