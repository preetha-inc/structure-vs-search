
"""Answer normalization and evaluation metrics for HotpotQA-style QA."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalize answers using the standard HotpotQA convention."""
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(char for char in value if char not in string.punctuation)

    def normalize_whitespace(value: str) -> str:
        return " ".join(value.split())

    text = text.lower()
    text = remove_punctuation(text)
    text = remove_articles(text)
    return normalize_whitespace(text)


def exact_match_score(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)

    common = Counter(predicted_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_metrics(prediction: str, gold: str) -> dict[str, float]:
    return {
        "em": exact_match_score(prediction, gold),
        "f1": f1_score(prediction, gold),
    }


def supporting_passage_recall(
    retrieved_ids: list[str],
    gold_support_ids: list[str],
) -> float:
    """Recall over passage IDs, not answer strings."""
    gold = set(map(str, gold_support_ids))
    if not gold:
        return 0.0
    retrieved = set(map(str, retrieved_ids))
    return len(gold & retrieved) / len(gold)
