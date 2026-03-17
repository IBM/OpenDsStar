import logging
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel, Field, ValidationError

from agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from agents.ds_star.ds_star_utils import (
    add_event_to_trajectory,
    invoke_structured_with_usage,
    truncate_text,
)
from agents.ds_star.nodes.base_node import BaseNode
from agents.utils.agents_utils import print_once

logger = logging.getLogger(__name__)


class AnswerOutput(BaseModel):
    answer: str = Field(..., description="Final answer for the user.")


def _collect_all_outputs(state: DSState) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for step in state.steps:
        if isinstance(getattr(step, "outputs", None), dict):
            merged.update(step.outputs)
    return merged


def _collect_all_logs(state: DSState, logs_max_length: int) -> str:
    """
    Collect logs from all steps. Each step's logs are truncated individually to logs_max_length.
    """
    blocks = []
    for i, step in enumerate(state.steps):
        logs = getattr(step, "logs", None)
        if logs:
            blocks.append(
                f"Step {i} logs:\n{truncate_text(str(logs), logs_max_length)}"
            )
    if not blocks:
        return "(no logs captured)"
    return "\n\n".join(blocks)


def build_finalizer_prompt(state: DSState) -> Tuple[str, str]:
    """
    Returns (system_message, user_message) so the caller can send a proper system role.

    Visibility rules requested:
    - STEPWISE: include outputs + logs aggregated across ALL steps.
    - FULL: include outputs + logs from LAST step only.
    """
    code_mode = getattr(state, "code_mode", CodeMode.STEPWISE)

    system_instructions = (
        "You are the Finalizer agent in a DS-STAR-style system.\n"
        "Your task is to produce the final natural-language answer for the user.\n\n"
        "General rules:\n"
        "- Prefer returning a PARTIAL but accurate answer over saying you are unable to answer.\n"
        "- If the system found fewer results than requested (for example, 3 deals instead of 5), "
        "  you MUST clearly state this and still list the available results.\n"
        "- Only say you are unable to answer if there is no meaningful data at all "
        "  or a fatal error is reported and prevents any useful result.\n"
        "- Be concise and factual; do not invent data.\n"
    )

    user_block = f"User query:\n{state.user_query.strip()}\n"

    plan_lines = [f"{i}. {s.plan.strip()}" for i, s in enumerate(state.steps)]
    plan_block = "Current plan (0-based steps):\n" + (
        "\n".join(plan_lines) if plan_lines else "(none)"
    )

    output_max_length = state.output_max_length
    logs_max_length = state.logs_max_length

    # -------- Outputs --------
    if not state.steps:
        outputs_block = "Relevant analysis outputs:\n(no steps executed; no outputs)\n"
    elif code_mode == CodeMode.STEPWISE:
        all_outputs = _collect_all_outputs(state)
        if not all_outputs:
            outputs_block = (
                "Relevant analysis outputs (all steps):\n(no outputs produced)\n"
            )
        else:
            lines = [
                f"{k} = {truncate_text(str(v), output_max_length)}"
                for k, v in all_outputs.items()
            ]
            outputs_block = (
                "Relevant analysis outputs (all steps):\n" + "\n".join(lines) + "\n"
            )
    else:
        last_step: DSStep = state.steps[-1]
        last_outputs = (
            str(last_step.outputs)
            if isinstance(getattr(last_step, "outputs", None), dict)
            and last_step.outputs
            else "No outputs produced."
        )
        last_outputs = truncate_text(last_outputs, output_max_length)
        outputs_block = (
            "Relevant analysis outputs (last step only):\n" f"{last_outputs}\n"
        )

    # -------- Logs --------
    if not state.steps:
        logs_block = "Execution logs:\n(no steps executed; no logs)\n"
    elif code_mode == CodeMode.STEPWISE:
        all_logs = _collect_all_logs(state, logs_max_length=logs_max_length)
        logs_block = "Execution logs (all steps):\n" + f"{all_logs}\n"
    else:
        last_step = state.steps[-1]
        last_logs = getattr(last_step, "logs", None)
        last_logs = (
            truncate_text(str(last_logs), logs_max_length)
            if last_logs
            else "(no logs captured)"
        )
        logs_block = "Execution logs (last step only):\n" + f"{last_logs}\n"

    fatal_block = f"Fatal error flag: {state.fatal_error}\n"

    user_message = (
        f"{user_block}\n\n"
        f"{plan_block}\n\n"
        f"{outputs_block}\n"
        f"{logs_block}\n"
        f"{fatal_block}\n"
        "Using the information above, write the final answer for the user.\n"
        "- If the data is partial (for example, fewer items than requested), "
        "explicitly mention that only a subset was found and present that subset clearly.\n"
        "- If there is a fatal error but some useful outputs/logs are still available, "
        "explain the limitation and use what is available.\n"
        "- Only say you are unable to answer if there is no meaningful data at all.\n"
        "Return AnswerOutput(answer=<final answer>) as a structured response.\n"
    )

    logger.info(
        "finalizer prompt length: %d", len(system_instructions) + len(user_message)
    )
    return system_instructions, user_message


class FinalizerNode(BaseNode):
    structured_output_schema: Optional[Type[BaseModel]] = AnswerOutput

    def __call__(self, state: DSState) -> DSState:
        """Generate final answer summary."""
        if not isinstance(state, DSState):
            raise TypeError(
                f"FinalizerNode expected DSState, got {type(state).__name__}. "
                "Fix the caller to pass DSState, or refactor this node to use dict access."
            )

        # If fatal_error and nothing to work with, return a direct error message.
        if state.fatal_error and not state.steps:
            state.final_answer = (
                f"Process terminated due to an error: {state.fatal_error}"
            )
            add_event_to_trajectory(
                state,
                "n_finalizer",
                fatal_error=state.fatal_error,
                answer=state.final_answer,
            )
            return state

        i = len(state.steps) - 1 if state.steps else -1

        system_msg, user_msg = build_finalizer_prompt(state)
        print_once("ds_star_finalizer_prompt", f"DS Star Finalizer prompt:\n{user_msg}")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            response, usage = invoke_structured_with_usage(
                self.llm_with_output, messages, run_name="n_finalizer"
            )
            state.token_usage.append(usage)
            state.final_answer = response.answer

            # if any outputs look like images (data URI or direct URL), append them
            # directly to the final answer string.  This happens *after* the LLM
            # produces its text; the model itself never needs to inspect or emit
            # the URI.  We simply check state.outputs for figures/URIs and tack
            # them on so the chat UI can render them.
            def _append_images(answer: str) -> str:
                all_out = _collect_all_outputs(state)
                img_lines = []

                # helper for figure -> data uri
                def _fig_to_uri(fig):
                    try:
                        img_bytes = fig.to_image(format="png")
                        if isinstance(img_bytes, str):
                            img_bytes = img_bytes.encode("utf-8")
                        import base64

                        return f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
                    except Exception:
                        return None

                for v in all_out.values():
                    # Plotly figure object?
                    try:
                        import plotly.graph_objects as _go
                    except ImportError:
                        _go = None
                    if _go is not None and isinstance(v, _go.Figure):
                        uri = _fig_to_uri(v)
                        if uri:
                            img_lines.append(uri)
                            continue

                    if isinstance(v, str):
                        if (
                            v.startswith("data:image")
                            or v.startswith("http")
                            and v.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
                        ):
                            img_lines.append(v)
                    elif isinstance(v, list) or isinstance(v, tuple):
                        for item in v:
                            if isinstance(item, str) and item.startswith("data:image"):
                                img_lines.append(item)
                if img_lines:
                    # append separated by two newlines to keep markdown/image parsing
                    return answer + "\n\n" + "\n\n".join(img_lines)
                return answer

            state.final_answer = _append_images(state.final_answer)

            add_event_to_trajectory(
                state,
                "n_finalizer",
                answer=state.final_answer,
            )
        except ValidationError as ve:
            state.fatal_error = f"Finalizer schema validation failed: {ve}"
            logger.error("Fatal error finalizing at step %s: %s", i, state.fatal_error)
            state.final_answer = "Unable to answer."

            add_event_to_trajectory(
                state,
                "n_finalizer",
                fatal_error=state.fatal_error,
                answer=state.final_answer,
            )
        except Exception as e:
            state.fatal_error = f"Finalizer invocation failed: {e}"
            logger.error("Fatal error finalizing at step %s: %s", i, state.fatal_error)
            state.final_answer = "Unable to answer."
            add_event_to_trajectory(
                state,
                "n_finalizer",
                fatal_error=state.fatal_error,
                answer=state.final_answer,
            )

        logger.info("Finalizer produced final answer: %s", state.final_answer)
        return state
