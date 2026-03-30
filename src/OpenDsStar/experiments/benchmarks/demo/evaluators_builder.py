"""Demo evaluators builder implementation."""

from __future__ import annotations

from typing import List

from ...evaluators.ragbench_llm_judge import RagbenchLLMJudgeEvaluator
from ...interfaces.evaluator import Evaluator
from .evaluators import NumericExactEvaluator, TextExactEvaluator


class DemoEvaluatorsBuilder:
    """
    Configures evaluators for demo experiments.

    Provides both traditional exact match evaluators and LLM-based evaluation.
    """

    @staticmethod
    def build_evaluators() -> List[Evaluator]:
        """
        Build and return list of evaluators for demo.

        Returns:
            List of configured evaluators
        """
        return [
            RagbenchLLMJudgeEvaluator(),
            TextExactEvaluator(),
            NumericExactEvaluator(),
        ]
