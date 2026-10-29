
"""BM25 sparse retrieval over a collection of text passages."""

import re
from collections.abc import Sequence
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercase text and split it into alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Retrieve passages ranked by BM25 relevance to a query."""

    def __init__(
        self,
        passages: Sequence[dict[str, Any]],
    ) -> None:
        if not passages:
            raise ValueError("The passage collection must not be empty.")

        self.passages = []
        tokenized_corpus = []
        seen_ids = set()

        for index, passage in enumerate(passages):
            text = passage.get("text", passage.get("contents", ""))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Passage at index {index} has no valid text."
                )

            passage_id = str(
                passage.get("id", passage.get("passage_id", index))
            )
            if passage_id in seen_ids:
                raise ValueError(f"Duplicate passage ID: {passage_id}")

            seen_ids.add(passage_id)

            normalized = {
                **passage,
                "id": passage_id,
                "title": str(passage.get("title", "")),
                "text": text,
            }
            self.passages.append(normalized)
            tokenized_corpus.append(tokenize(text))

        self.index = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the top-k passages, sorted by descending BM25 score."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (-float(scores[i]), i),
        )

        results = []
        for index in ranked_indices:
            score = float(scores[index])

            # Stop when all remaining passages have zero relevance.
            if score <= 0:
                break

            results.append(
                {
                    **self.passages[index],
                    "score": score,
                    "rank": len(results) + 1,
                }
            )

            if len(results) >= top_k:
                break

        return results
