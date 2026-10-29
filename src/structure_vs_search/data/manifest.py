
"""Create a reproducibility manifest for local dataset files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(paths: list[str | Path]) -> dict:
    files = []

    for input_path in paths:
        path = Path(input_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cannot manifest missing file: {path}")

        files.append(
            {
                "filename": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": "RECORD THE EXACT OFFICIAL SOURCE AND VERSION",
        "files": files,
    }


def write_manifest(paths: list[str | Path], output_path: str | Path) -> None:
    manifest = build_manifest(paths)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest: {output_path}")
