
"""Load question and passage records without losing their source identifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON array/object or JSONL file into records."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # JSONL: one JSON object per line.
        records = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on line {line_number} in {path}")
            records.append(record)
        return records

    if isinstance(parsed, list):
        records = parsed
    elif isinstance(parsed, dict):
        # Common wrappers used by dataset exports.
        for key in ("data", "questions", "passages", "documents", "records"):
            if isinstance(parsed.get(key), list):
                records = parsed[key]
                break
        else:
            records = [parsed]
    else:
        raise ValueError(f"Expected a JSON array or object in {path}")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Expected every record in {path} to be a JSON object")

    return records


def load_questions(
    path: str | Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load QA examples while preserving original IDs and gold metadata."""
    records = _read_records(path)
    questions = []

    for index, record in enumerate(records):
        question_id = record.get("_id", record.get("id", record.get("question_id")))
        question_text = record.get("question")

        if question_id is None:
            raise ValueError(f"Question record {index} has no source ID")
        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(f"Question {question_id!r} has no valid question text")

        # Retain all original fields, including answer and supporting facts.
        # Retrieval code must explicitly select only permitted input fields.
        item = dict(record)
        item["id"] = str(question_id)
        item["question"] = question_text.strip()
        questions.append(item)

    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative or None")
        questions = questions[:limit]

    return questions


def load_corpus(path: str | Path) -> list[dict[str, str]]:
    """Load a flat passage corpus with stable source IDs and text."""
    records = _read_records(path)
    passages = []
    seen_ids: set[str] = set()

    for index, record in enumerate(records):
        passage_id = record.get(
            "passage_id",
            record.get(
                "id",
                record.get("doc_id", record.get("_id", record.get("idx"))),
            ),
        )
        title = record.get("title", "")
        text = record.get("text", record.get("contents", record.get("content")))

        if passage_id is None:
            raise ValueError(f"Passage record {index} has no source ID")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Passage {passage_id!r} has no valid text")

        passage_id = str(passage_id)
        if passage_id in seen_ids:
            raise ValueError(f"Duplicate passage ID: {passage_id}")
        seen_ids.add(passage_id)

        passages.append(
            {
                "id": passage_id,
                "title": str(title),
                "text": text.strip(),
            }
        )

    return passages
