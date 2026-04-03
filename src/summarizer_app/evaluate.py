from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .text_utils import split_sentences, tokenize


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    counter: Counter = Counter()
    if n <= 0 or len(tokens) < n:
        return counter
    for i in range(len(tokens) - n + 1):
        counter[tuple(tokens[i : i + n])] += 1
    return counter


def rouge_n(reference: str, hypothesis: str, n: int = 1) -> dict:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    ref_ngrams = ngrams(ref_tokens, n)
    hyp_ngrams = ngrams(hyp_tokens, n)
    overlap = sum((ref_ngrams & hyp_ngrams).values())
    ref_total = max(1, sum(ref_ngrams.values()))
    hyp_total = max(1, sum(hyp_ngrams.values()))
    precision = overlap / hyp_total
    recall = overlap / ref_total
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def rouge_l(reference: str, hypothesis: str) -> dict:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    lcs = lcs_length(ref_tokens, hyp_tokens)
    ref_total = max(1, len(ref_tokens))
    hyp_total = max(1, len(hyp_tokens))
    precision = lcs / hyp_total
    recall = lcs / ref_total
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def length_accuracy(prediction: str, target_length: int, tolerance: float = 0.2) -> bool:
    if target_length <= 0:
        return True
    low = int(target_length * (1 - tolerance))
    high = int(target_length * (1 + tolerance))
    return low <= len(prediction) <= high


def evaluate_pairs(pairs: Iterable[tuple[str, str]], target_lengths: Iterable[int] | None = None) -> dict:
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    length_hits = []
    target_lengths = list(target_lengths) if target_lengths is not None else None

    for idx, (reference, hypothesis) in enumerate(pairs):
        rouge1_scores.append(rouge_n(reference, hypothesis, 1)["f1"])
        rouge2_scores.append(rouge_n(reference, hypothesis, 2)["f1"])
        rougel_scores.append(rouge_l(reference, hypothesis)["f1"])
        if target_lengths is not None and idx < len(target_lengths):
            length_hits.append(length_accuracy(hypothesis, target_lengths[idx]))

    total = max(1, len(rouge1_scores))
    report = {
        "rouge_1_f1": sum(rouge1_scores) / total,
        "rouge_2_f1": sum(rouge2_scores) / total,
        "rouge_l_f1": sum(rougel_scores) / total,
    }
    if length_hits:
        report["length_hit_rate"] = sum(1 for item in length_hits if item) / len(length_hits)
    return report
