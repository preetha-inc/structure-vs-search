import json

import pytest

from structure_vs_search.evaluation.cost_logger import CostLogger


def test_logger_appends_records(tmp_path):
    path = tmp_path / "runs" / "pilot.jsonl"
    logger = CostLogger(path)

    logger.log({"question_id": "q1", "method": "A0", "exact_match": 1.0})
    logger.log({"question_id": "q2", "method": "A0", "exact_match": 0.0})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["question_id"] == "q1"
    assert records[1]["question_id"] == "q2"
    assert all("timestamp_utc" in record for record in records)


def test_logger_requires_question_id_and_method(tmp_path):
    logger = CostLogger(tmp_path / "runs.jsonl")

    with pytest.raises(ValueError, match="Missing required log fields"):
        logger.log({"prediction": "answer"})
