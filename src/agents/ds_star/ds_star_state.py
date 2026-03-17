from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


@dataclass
class DSStep:
    plan: str
    code: Optional[str] = None
    logs: Optional[str] = None
    outputs: Optional[Dict[str, Any]] = field(default_factory=dict)
    execution_error: Optional[str] = None
    verifier_sufficient: Optional[bool] = None
    verifier_explanation: Optional[str] = None
    router_action: Optional[str] = None
    router_fix_index: Optional[int] = None
    router_explanation: Optional[str] = None
    debug_tries: int = 0
    failed_code_attempts: List[Dict[str, str]] = field(
        default_factory=list
    )  # Track all failed code attempts with their errors

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __delitem__(self, k):
        setattr(self, k, None)

    def __iter__(self):  # yield keys
        return (f.name for f in fields(self))

    def __len__(self):
        return len(list(fields(self)))

    def keys(self):
        return [f.name for f in fields(self)]


class CodeMode(str, Enum):
    STEPWISE = "stepwise"
    FULL = "full"


@dataclass
class DSState:
    user_query: str
    tools: Dict[str, Callable[..., Any]]
    output_max_length: int = 500
    logs_max_length: int = 1000
    steps: List[DSStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    fatal_error: Optional[str] = None
    steps_used: int = 0
    max_steps: int = 5
    max_debug_tries: int = 5
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: List[Dict[str, Any]] = field(default_factory=list)
    code_mode: CodeMode = CodeMode.STEPWISE
    trajectory_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    _emitted_events: set = field(
        default_factory=set
    )  # Track (node, step_idx) to prevent duplicates

    # making the state behave like a dictionary, since Langgraph requires it to be a dictionary
    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __delitem__(self, k):
        setattr(self, k, None)

    def __iter__(self):  # yield keys
        return (f.name for f in fields(self))

    def __len__(self):
        return len(list(fields(self)))

    def keys(self):
        return [f.name for f in fields(self)]
