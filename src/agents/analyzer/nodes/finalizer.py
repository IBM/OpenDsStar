"""Finalizer node for analyzer agent."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class FinalizerNode:
    """Finalizes the analysis and prepares the final answer."""

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create final answer from analysis results."""
        logger.info("Finalizing analysis")

        if state["fatal_error"]:
            state["final_answer"] = f"Analysis failed: {state['fatal_error']}"
        elif state["execution_error"]:
            state["final_answer"] = (
                f"Analysis failed after {state['debug_tries']} debug attempts.\n"
                f"Error: {state['execution_error']}"
            )
        else:
            # Success - return the logs as the answer
            state["final_answer"] = (
                state["logs"] if state["logs"] else "Analysis completed (no output)"
            )

        state["trajectory"].append(
            {
                "node": "n_finalizer",
                "success": not bool(state["execution_error"] or state["fatal_error"]),
            }
        )

        logger.info("Analysis finalized")
        return state
