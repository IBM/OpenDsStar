"""Executor adapter for analyzer agent - uses DS*Star ExecutorNode."""

import logging
from typing import Any, Dict

from agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from agents.ds_star.nodes.executer import ExecutorNode as DSStarExecutorNode

logger = logging.getLogger(__name__)


class ExecutorNode:
    """
    Adapter that uses DS*Star ExecutorNode for code execution.
    Converts AnalyzerState to DSState, calls DS*Star executor, and converts back.
    """

    def __init__(self, code_timeout: int = 30):
        self.code_timeout = code_timeout
        # Initialize DS*Star executor
        self.ds_executor = DSStarExecutorNode(
            tools={}, code_timeout=self.code_timeout  # No tools for analyzer
        )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the analysis code."""
        if state["fatal_error"]:
            state["trajectory"].append(
                {
                    "node": "n_execute",
                    "skipped": True,
                    "fatal_error": state["fatal_error"],
                }
            )
            return state

        if not state["code"]:
            state["logs"] = "[SKIPPED] No code to execute."
            state["outputs"] = {"_skipped": True}
            state["trajectory"].append(
                {"node": "n_execute", "skipped": True, "logs": state["logs"]}
            )
            return state

        logger.info("Executing analysis code")

        # Convert AnalyzerState to DSState
        ds_state = DSState(
            user_query=f"Analyze the file: {state['filename']}",
            tools={},
            steps=[
                DSStep(
                    plan=f"Analyze the file: {state['filename']}",
                    code=state["code"],
                )
            ],
            code_mode=CodeMode.STEPWISE,
        )

        # Call DS*Star executor (pass the DSState object, not dict)
        ds_state_result = self.ds_executor(ds_state)

        # Extract results
        if ds_state_result.steps:
            last_step = ds_state_result.steps[-1]
            state["logs"] = last_step.logs or ""
            state["outputs"] = last_step.outputs or {}
            state["execution_error"] = last_step.execution_error

            if state["execution_error"]:
                logger.warning(f"Execution error: {state['execution_error']}")
            else:
                logger.info("Code executed successfully")

        state["trajectory"].append(
            {
                "node": "n_execute",
                "had_error": bool(state["execution_error"]),
                "logs": state["logs"][:500] if state["logs"] else None,
            }
        )

        return state
