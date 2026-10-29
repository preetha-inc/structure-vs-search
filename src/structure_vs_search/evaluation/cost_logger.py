"""Append per-question experiment records as JSON Lines."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class CostLogger:
    """Write reproducible per-question results and cost measurements to JSONL."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        """Append one question/run record without overwriting earlier records."""
        entry = dict(record)
        entry.setdefault(
            "timestamp_utc",
            datetime.now(timezone.utc).isoformat(),
        )

        required = ("question_id", "method")
        missing = [key for key in required if key not in entry]
        if missing:
            raise ValueError(f"Missing required log fields: {missing}")

        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
