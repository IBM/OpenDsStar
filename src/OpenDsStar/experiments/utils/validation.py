"""Validation utilities."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from ..core.types import ProcessedBenchmark


def ensure_unique_question_ids(benchmarks: Sequence[ProcessedBenchmark]) -> None:
    """
    Ensure all question IDs are unique.

    Args:
        benchmarks: Sequence of processed benchmarks

    Raises:
        ValueError: If duplicate question IDs are found
    """
    seen: set[str] = set()
    dups: list[str] = []
    for b in benchmarks:
        if b.question_id in seen:
            dups.append(b.question_id)
        seen.add(b.question_id)
    if dups:
        raise ValueError(
            f"Duplicate question_id(s) in processed benchmarks: {sorted(set(dups))}"
        )


def index_by_question_id(items: Sequence[Any], id_attr: str) -> Dict[str, Any]:
    """
    Index items by their question_id attribute.

    Args:
        items: Sequence of items with question_id attribute
        id_attr: Name of the attribute containing question_id

    Returns:
        Dictionary mapping question_id to item

    Raises:
        ValueError: If duplicate question IDs are found
    """
    out: Dict[str, Any] = {}
    for x in items:
        qid = getattr(x, id_attr)
        if qid in out:
            raise ValueError(f"Duplicate {id_attr}={qid}")
        out[qid] = x
    return out
