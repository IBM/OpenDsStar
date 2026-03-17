"""Evaluator implementations."""

from __future__ import annotations

from ...core.context import PipelineContext
from ...core.types import AgentOutput, BenchmarkEntry, EvalResult
from ...interfaces.evaluator import Evaluator


class NumericExactEvaluator(Evaluator):
    """
    Evaluator for numeric exact match.

    This is a placeholder implementation. Replace with your actual
    numeric evaluation logic (e.g., tolerance-based matching, etc.).
    """

    def evaluate_one(
        self,
        ctx: PipelineContext,
        output: AgentOutput,
        benchmark: BenchmarkEntry,
    ) -> EvalResult:
        """
        Evaluate a single numeric answer.

        Args:
            ctx: Pipeline context
            output: Agent's output
            benchmark: Original benchmark with ground truth

        Returns:
            Evaluation result
        """
        # Placeholder: always fail
        # In real implementation, parse and compare numbers
        try:
            predicted = float(str(output.answer))
            ground_truth_values = [
                float(str(a)) for a in benchmark.ground_truth.answers
            ]
            passed = any(abs(predicted - gt) < 1e-6 for gt in ground_truth_values)
            score = 1.0 if passed else 0.0
        except (ValueError, TypeError):
            passed = False
            score = 0.0

        return EvalResult(
            question_id=output.question_id,
            score=score,
            passed=passed,
            details={
                "predicted": output.answer,
                "ground_truth": benchmark.ground_truth.answers,
            },
        )


class TextExactEvaluator(Evaluator):
    """
    Evaluator for text exact match.

    This is a placeholder implementation. Replace with your actual
    text evaluation logic (e.g., F1, EM, fuzzy matching, etc.).
    """

    def evaluate_one(
        self,
        ctx: PipelineContext,
        output: AgentOutput,
        benchmark: BenchmarkEntry,
    ) -> EvalResult:
        """
        Evaluate a single text answer.

        Args:
            ctx: Pipeline context
            output: Agent's output
            benchmark: Original benchmark with ground truth

        Returns:
            Evaluation result
        """
        # Placeholder: case-insensitive exact match
        # In real implementation, use more sophisticated matching
        predicted = str(output.answer).strip().lower()
        ground_truth_values = [
            str(a).strip().lower() for a in benchmark.ground_truth.answers
        ]
        passed = predicted in ground_truth_values
        score = 1.0 if passed else 0.0

        return EvalResult(
            question_id=output.question_id,
            score=score,
            passed=passed,
            details={
                "predicted": output.answer,
                "ground_truth": benchmark.ground_truth.answers,
            },
        )
