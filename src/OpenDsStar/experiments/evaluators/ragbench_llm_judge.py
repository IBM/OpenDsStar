from __future__ import annotations

import logging
from typing import Any, Dict, List

from unitxt.operator import MultiStream, SequentialOperator

from ..core.context import PipelineContext
from ..core.types import AgentOutput, BenchmarkEntry, EvalResult
from ..interfaces.evaluator import Evaluator

logger = logging.getLogger(__name__)


class RagbenchLLMJudgeEvaluator(Evaluator):
    """
    Evaluator using a ragbench metric artifact (operator pipeline style),
    similar to your working project.
    """

    def __init__(
        self,
        metric_id: str = "metrics.rag.external_rag.answer_correctness.llama_3_3_70b_instruct_watsonx_judge",
        sub_scores: List[str] | None = None,
    ):
        self.metric_id = metric_id
        self.sub_scores = sub_scores or []

        # Build the operator once
        self._metrics_operator = SequentialOperator(steps=[self.metric_id])

    def evaluate_one(
        self,
        ctx: PipelineContext,
        output: AgentOutput,
        benchmark: BenchmarkEntry,
    ) -> EvalResult:
        try:
            # Construct a single-item dataset entry.
            # IMPORTANT: keys must match what the metric expects.
            # The RAGBench answer_correctness metric expects:
            #   "question", "answer" (prediction), "ground_truths" (not "references")
            entry: Dict[str, Any] = {
                "q_id": output.question_id,
                "question": benchmark.question,
                "answer": str(output.answer),
                "ground_truths": [str(a) for a in benchmark.ground_truth.answers],
            }

            multi_stream = MultiStream.from_iterables({"test": [entry]}, copying=True)

            # Run the metric
            instance = list(self._metrics_operator(multi_stream)["test"])[0]
            instance_scores = instance["score"]["instance"]

            # Pick the score
            # If you have sub_scores configured, use them; otherwise use "score".
            if self.sub_scores:
                # Example: pick the first requested sub-score
                raw = instance_scores.get(self.sub_scores[0], 0.0)
            else:
                raw = instance_scores.get("score", 0.0)

            score = self._normalize_score(float(raw))
            passed = score >= 0.5

            return EvalResult(
                question_id=output.question_id,
                score=score,
                passed=passed,
                details={
                    "predicted": output.answer,
                    "ground_truth": benchmark.ground_truth.answers,
                    "metric_id": self.metric_id,
                    "raw_instance_scores": instance_scores,
                },
            )

        except Exception as e:
            logger.error(
                "Error evaluating with ragbench operator LLM-as-Judge", exc_info=True
            )
            return self._create_error_result(
                question_id=output.question_id,
                predicted=output.answer,
                ground_truth=benchmark.ground_truth.answers,
                error_msg=str(e),
            )

    def _normalize_score(self, score: float) -> float:
        # keep same logic you had
        if score <= 0.0:
            return 0.0
        if 0.0 <= score <= 1.0:
            return score
        if 1.0 <= score <= 5.0:
            return (score - 1.0) / 4.0
        return max(0.0, min(1.0, score))

    def _create_error_result(
        self,
        question_id: str,
        predicted: Any,
        ground_truth: Any,
        error_msg: str,
    ) -> EvalResult:
        return EvalResult(
            question_id=question_id,
            score=0.0,
            passed=False,
            details={
                "predicted": predicted,
                "ground_truth": ground_truth,
                "error": error_msg,
                "metric_id": self.metric_id,
            },
        )
