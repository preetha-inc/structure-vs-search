
from structure_vs_search.evaluation.metrics import (
    exact_match_score,
    f1_score,
    normalize_answer,
    supporting_passage_recall,
)


def test_normalization():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"


def test_exact_match_ignores_case_punctuation_and_articles():
    assert exact_match_score("The Eiffel Tower!", "eiffel tower") == 1.0


def test_partial_answer_f1():
    assert f1_score("Paris France", "Paris") == 2 / 3


def test_no_token_overlap():
    assert f1_score("Tokyo", "Paris") == 0.0


def test_empty_answers():
    assert exact_match_score("", "") == 1.0
    assert f1_score("", "") == 1.0


def test_yes_no_answers():
    assert exact_match_score("Yes", "yes") == 1.0
    assert f1_score("No", "Yes") == 0.0


def test_supporting_passage_recall():
    assert supporting_passage_recall(["p1", "p3"], ["p1", "p2"]) == 0.5


def test_no_gold_supporting_passages():
    assert supporting_passage_recall(["p1"], []) == 0.0
