import pytest

from structure_vs_search.evaluation.budget import (
    passage_text,
    render_evidence,
    select_passages_within_budget,
)


def test_selects_passages_within_combined_budget():
    passages = [
        {"id": "p1", "title": "A", "text": "one"},
        {"id": "p2", "title": "B", "text": "two"},
        {"id": "p3", "title": "C", "text": "three"},
    ]

    # This test tokenizer counts whitespace-separated tokens in the whole block.
    count_tokens = lambda text: len(text.split())

    result = select_passages_within_budget(
        passages,
        token_cap=8,
        count_tokens=count_tokens,
    )

    assert result["evidence_token_count"] <= 8
    assert result["evidence_token_count"] == count_tokens(
        render_evidence(result["passages"])
    )


def test_skips_oversized_passage_and_continues():
    passages = [
        {"id": "long", "text": "this passage is definitely too long"},
        {"id": "short", "text": "ok"},
    ]

    result = select_passages_within_budget(
        passages,
        token_cap=3,
        count_tokens=lambda text: len(text.split()),
    )

    assert [p["id"] for p in result["passages"]] == ["short"]
    assert result["skipped_passage_ids"] == ["long"]


def test_rendering_includes_title_and_text():
    passage = {"title": "Paris", "text": "Capital of France"}
    assert passage_text(passage) == "Title: Paris\nPassage: Capital of France"


def test_combined_evidence_includes_separator_tokens():
    passages = [
        {"id": "p1", "text": "alpha"},
        {"id": "p2", "text": "beta"},
    ]

    result = select_passages_within_budget(
        passages,
        token_cap=4,
        count_tokens=lambda text: len(text.split()),
    )

    assert len(result["passages"]) == 2
    assert result["evidence_token_count"] == 4


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        select_passages_within_budget([], -1, len)
