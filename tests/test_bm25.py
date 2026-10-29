
import pytest

from structure_vs_search.retrieval.bm25 import BM25Retriever


@pytest.fixture
def retriever():
    passages = [
        {
            "id": "p1",
            "title": "Paris",
            "text": "Paris is the capital of France.",
        },
        {
            "id": "p2",
            "title": "Berlin",
            "text": "Berlin is the capital of Germany.",
        },
        {
            "id": "p3",
            "title": "Python",
            "text": "Python is a programming language.",
        },
    ]
    return BM25Retriever(passages)


def test_bm25_ranks_relevant_passage_first(retriever):
    results = retriever.search("capital of France", top_k=2)

    assert results
    assert results[0]["id"] == "p1"
    assert results[0]["rank"] == 1
    assert results[0]["score"] > 0


def test_bm25_returns_at_most_top_k(retriever):
    results = retriever.search("capital", top_k=1)

    assert len(results) <= 1


def test_bm25_empty_query_returns_no_results(retriever):
    assert retriever.search("!!!") == []


def test_bm25_rejects_invalid_top_k(retriever):
    with pytest.raises(ValueError):
        retriever.search("France", top_k=0)


def test_bm25_rejects_duplicate_ids():
    passages = [
        {"id": "same", "text": "First passage"},
        {"id": "same", "text": "Second passage"},
    ]

    with pytest.raises(ValueError, match="Duplicate passage ID"):
        BM25Retriever(passages)
