"""Interface for evaluating agent outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.context import PipelineContext
from ..core.types import AgentOutput, BenchmarkEntry, EvalResult


class Evaluator(ABC):
    """Interface for evaluating agent outputs against ground truth."""

    @abstractmethod
    def evaluate_one(
        self,
        ctx: PipelineContext,
        output: AgentOutput,
        benchmark: BenchmarkEntry,
    ) -> EvalResult:
        """
        Evaluate a single agent output against ground truth.

        Args:
            ctx: Pipeline context with config and logger
            output: Agent's output for the question
            benchmark: Original benchmark with ground truth

        Returns:
            Evaluation result with score and details
        """
        raise NotImplementedError
