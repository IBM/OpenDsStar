import logging
from typing import Any, Callable, Dict, Optional

from agents.ds_star.ds_star_execute_env import execute_user_code
from agents.ds_star.ds_star_utils import add_event_to_trajectory
from agents.ds_star.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ExecutorNode(BaseNode):
    tools: Optional[Dict[str, Callable[..., Any]]] = None
    code_timeout: int = 30

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code for the current (latest) step."""
        if state.fatal_error:
            add_event_to_trajectory(
                state, "n_execute", skipped=True, fatal_error=state.fatal_error
            )
            return state

        last_step = state.steps[-1]
        i = len(state.steps) - 1
        if not last_step.code:
            last_step.logs = "[SKIPPED] No code for this step."
            last_step.outputs = {"_skipped": True}
            last_step.execution_error = None
            add_event_to_trajectory(
                state,
                "n_execute",
                skipped=True,
                logs=last_step.logs,
                outputs=last_step.outputs,
            )
            return state

        logs, outputs = execute_user_code(
            last_step.code, state, self.tools or {}, self.code_timeout
        )
        last_step.logs = logs
        last_step.outputs = outputs

        if isinstance(outputs, dict) and "_error" in outputs:
            last_step.execution_error = (
                "execution error: "
                + str(outputs.get("_error", "unknown"))
                + "\n\ntraceback:"
                + str(outputs.get("_traceback", "unknown"))
            )
            logger.warning(f"Execution error in step {i}: {last_step.execution_error}")
        else:
            last_step.execution_error = None
            keys = list(outputs.keys()) if isinstance(outputs, dict) else ["<non-dict>"]
            logger.info(f"Executed step {i} successfully. Outputs keys={keys}")

        add_event_to_trajectory(
            state,
            "n_execute",
            had_error=bool(last_step.execution_error),
            logs=last_step.logs,
            outputs=last_step.outputs,
        )
        return state
