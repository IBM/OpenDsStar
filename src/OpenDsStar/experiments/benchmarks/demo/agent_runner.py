"""Simple agent runner implementation."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ...core.context import PipelineContext
from ...core.types import AgentOutput, ProcessedBenchmark
from ...interfaces.agent_runner import AgentRunner

logger = logging.getLogger(__name__)


class SimpleAgentRunner(AgentRunner):
    """
    Simple agent runner that returns the prompt as answer.

    This is a placeholder implementation. Replace with your actual
    agent execution logic (e.g., LangChain invoke, LangGraph stream, etc.).
    """

    def run_batch(
        self,
        ctx: PipelineContext,
        agent: Any,
        benchmarks: Sequence[ProcessedBenchmark],
    ) -> Sequence[AgentOutput]:
        """
        Run the agent on a batch of benchmarks.

        Args:
            ctx: Pipeline context
            agent: The configured agent
            benchmarks: Processed benchmarks to run on

        Returns:
            Sequence of agent outputs
        """
        logger.info(f"running_agent benchmark_count={len(benchmarks)}")

        outputs = []
        for b in benchmarks:
            # Placeholder: just return the question as answer
            # In real implementation, you would invoke the agent here
            answer = f"(stub answer for: {b.question})"

            outputs.append(
                AgentOutput(
                    question_id=b.question_id,
                    answer=answer,
                    metadata={
                        "tools_available": getattr(agent, "tool_names", []),
                        "question": b.question,
                    },
                )
            )

        return outputs
