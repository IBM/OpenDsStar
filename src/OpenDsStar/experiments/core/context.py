"""Pipeline context and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .config import AgentConfig


class CacheStore(Protocol):
    """Cache interface for storing intermediate results."""

    def get(self, key: str) -> Any | None: ...
    def put(self, key: str, value: Any) -> None: ...


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the experiment pipeline."""

    fail_fast: bool = True
    continue_on_error: bool = False  # if True, collect failures and proceed
    run_id: str = "run"
    output_dir: Path | None = None  # Directory for experiment outputs
    cache_dir: Path | None = None  # Directory for experiment cache
    agent_config: AgentConfig | None = None  # Agent configuration


@dataclass
class PipelineContext:
    """Context object passed through the pipeline."""

    config: PipelineConfig
    cache: CacheStore | None = None
