from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class AnalyzerState:
    """State for the analyzer agent."""

    filename: str
    code: str | None = None
    logs: str | None = None
    outputs: dict[str, any] = field(default_factory=dict)
    execution_error: str | None = None
    debug_tries: int = 0
    max_debug_tries: int = 3
    final_answer: str | None = None
    fatal_error: str | None = None
    trajectory: list[dict[str, any]] = field(default_factory=list)
    token_usage: list[dict[str, any]] = field(default_factory=list)

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __delitem__(self, k):
        setattr(self, k, None)

    def __iter__(self):
        return (f.name for f in fields(self))

    def __len__(self):
        return len(list(fields(self)))

    def keys(self):
        return [f.name for f in fields(self)]
