"""Evidence selection under a shared token budget."""

from collections.abc import Callable, Iterable, Mapping
from typing import Any


def passage_text(passage: Mapping[str, Any]) -> str:
    """Render a passage in the same format used in the answer prompt."""
    title = str(passage.get("title") or "").strip()
    text = str(passage.get("text") or "").strip()

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if text:
        parts.append(f"Passage: {text}")
    return "\n".join(parts)


def render_evidence(passages: Iterable[Mapping[str, Any]]) -> str:
    """Render the complete evidence block in prompt order."""
    return "\n\n".join(
        text for passage in passages if (text := passage_text(passage))
    )


def select_passages_within_budget(
    passages: Iterable[Mapping[str, Any]],
    token_cap: int,
    count_tokens: Callable[[str], int],
) -> dict[str, Any]:
    """Select ranked passages whose combined rendered evidence fits the cap.

    Passages are indivisible. If one does not fit, it is skipped and selection
    continues with the next passage. Token counts are measured on the combined
    evidence string, including the separators between passages.
    """
    if token_cap < 0:
        raise ValueError("token_cap must be non-negative")

    selected = []
    skipped_ids = []

    for index, passage in enumerate(passages):
        passage_id = passage.get("id", passage.get("passage_id", str(index)))
        if not passage_text(passage):
            skipped_ids.append(str(passage_id))
            continue

        candidate = [*selected, dict(passage)]
        candidate_text = render_evidence(candidate)
        tokens = count_tokens(candidate_text)

        if (
            not isinstance(tokens, int)
            or isinstance(tokens, bool)
            or tokens < 0
        ):
            raise ValueError("count_tokens must return a non-negative integer")

        if tokens <= token_cap:
            selected = candidate
        else:
            skipped_ids.append(str(passage_id))

    evidence = render_evidence(selected)
    used_tokens = count_tokens(evidence) if evidence else 0

    if (
        not isinstance(used_tokens, int)
        or isinstance(used_tokens, bool)
        or used_tokens < 0
    ):
        raise ValueError("count_tokens must return a non-negative integer")

    if used_tokens > token_cap:
        raise RuntimeError("Selected evidence exceeds the configured token cap")

    return {
        "passages": selected,
        "evidence_token_count": used_tokens,
        "token_cap": token_cap,
        "skipped_passage_ids": skipped_ids,
    }
