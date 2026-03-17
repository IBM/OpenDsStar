"""Core data types for the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, Mapping, Sequence


@dataclass(frozen=True)
class Document:
    """Raw document input with stream factory for safe re-reading."""

    document_id: str
    path: str
    mime_type: str | None
    extra_metadata: Mapping[str, Any]
    stream_factory: Callable[[], BinaryIO]  # stream-safe: fresh stream each time


@dataclass(frozen=True)
class GroundTruth:
    """Ground truth data for a benchmark question."""

    answers: Sequence[Any] = ()
    context_ids: Sequence[Any] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkEntry:
    """Raw benchmark question with ground truth."""

    question_id: str
    question: str
    ground_truth: GroundTruth
    additional_information: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessedBenchmark:
    """Processed benchmark ready for agent execution."""

    question_id: str
    question: str
    ground_truth: GroundTruth
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentOutput:
    """Output from agent execution."""

    question_id: str
    answer: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    """Evaluation result for a single question."""

    question_id: str
    score: float
    passed: bool
    details: Mapping[str, Any] = field(default_factory=dict)
