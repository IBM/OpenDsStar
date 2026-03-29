"""Generic agent runner for agents with invoke() method."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from tqdm import tqdm

from ..core.context import PipelineContext
from ..core.types import AgentOutput, ProcessedBenchmark
from ..interfaces.agent_runner import AgentRunner

logger = logging.getLogger(__name__)


class InvokeAgentRunner(AgentRunner):
    """
    Generic agent runner that invokes agents with an invoke() method.

    This runner works with any agent that implements an invoke(query: str) method
    that returns a dictionary with at least an 'answer' key. It's designed to work
    with OpenDsStarAgent and similar agent implementations.

    The runner:
    - Calls agent.invoke(prompt) for each benchmark
    - Extracts the answer from the result dictionary
    - Captures metadata like steps_used, llm_calls, tokens, etc.
    - Handles errors gracefully and logs them
    - Shows progress with tqdm
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
            agent: The configured agent (must have invoke() method)
            benchmarks: Processed benchmarks to run on

        Returns:
            Sequence of agent outputs
        """
        logger.info(f"running_agent benchmark_count={len(benchmarks)}")

        outputs = []

        # Use tqdm for progress tracking
        for b in tqdm(benchmarks, desc="Running agent on questions", unit="question"):
            try:
                # Always invoke the agent (no caching)
                result = agent.invoke(b.question)

                # Extract answer (required field)
                answer = result.get("answer", "")

                # Build metadata from agent result
                # Start with basic info
                metadata = {
                    "tools_available": (
                        [t.name for t in agent.tools] if hasattr(agent, "tools") else []
                    ),
                    "question": b.question,
                }

                # Add optional fields if present in result
                # Ensure all values are serializable
                optional_fields = [
                    "steps_used",
                    "max_steps",
                    "num_llm_calls",
                    "input_tokens",
                    "output_tokens",
                    "verifier_sufficient",
                    "fatal_error",
                    "execution_error",
                    "trajectory",
                    "plan",
                ]

                for field in optional_fields:
                    if field in result:
                        value = result[field]
                        # Ensure value is serializable
                        if isinstance(
                            value, (str, int, float, bool, type(None), list, dict)
                        ):
                            metadata[field] = value
                        else:
                            # Convert non-serializable objects to strings
                            metadata[field] = str(value)

                # Log success
                logger.info(
                    f"agent_completed question_id={b.question_id} "
                    f"steps_used={metadata.get('steps_used', 'N/A')} "
                    f"llm_calls={metadata.get('num_llm_calls', 'N/A')}"
                )

            except Exception as e:
                # Handle errors gracefully
                logger.error(f"agent_failed question_id={b.question_id} error={str(e)}")
                answer = f"ERROR: {str(e)}"
                metadata = {
                    "tools_available": (
                        [t.name for t in agent.tools] if hasattr(agent, "tools") else []
                    ),
                    "question": b.question,
                    "error": str(e),
                }

            outputs.append(
                AgentOutput(
                    question_id=b.question_id,
                    answer=answer,
                    metadata=metadata,
                )
            )

        return outputs
