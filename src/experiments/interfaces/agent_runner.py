"""Interface for running agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..core.context import PipelineContext
from ..core.types import AgentOutput, ProcessedBenchmark


class AgentRunner(ABC):
    """Interface for running an agent over benchmarks."""

    @abstractmethod
    def run_batch(
        self,
        ctx: PipelineContext,
        agent: Any,
        benchmarks: Sequence[ProcessedBenchmark],
    ) -> Sequence[AgentOutput]:
        """
        Run the agent on a batch of benchmarks.

        Args:
            ctx: Pipeline context with config and logger
            agent: The configured agent to run
            benchmarks: Processed benchmarks to run on

        Returns:
            Sequence of agent outputs
        """
        raise NotImplementedError
