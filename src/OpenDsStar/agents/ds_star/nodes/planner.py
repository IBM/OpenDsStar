import logging
from typing import List, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from OpenDsStar.agents.ds_star.ds_star_utils import (
    add_event_to_trajectory,
    build_execution_environment_instructions,
    invoke_structured_with_usage,
    truncate_text,
)
from OpenDsStar.agents.ds_star.nodes.base_node import BaseNode
from OpenDsStar.agents.utils.agents_utils import print_once

logger = logging.getLogger(__name__)


class PlanOneStepOutput(BaseModel):
    step: str = Field(
        ..., description="A single, concise executable step (one sentence)."
    )


def summarize_step_for_planner(
    step: DSStep,
    index: int,
    include_outputs: bool,
    output_max_length: int,
    logs_max_length: int,
) -> str:
    """
    Summarize a step for the Planner.

    include_outputs:
      - True  -> include execution_error / outputs / logs (STEPWISE mode).
      - False -> only include plan + verifier/router info (FULL mode).
    """
    parts: List[str] = [f"Step {index} – Planned action:\n{step.plan.strip()}"]

    if include_outputs:
        if step.logs:
            logs_text = truncate_text(step.logs.strip(), logs_max_length)
            parts.append(f"  Logs (print outputs):\n{logs_text}")

        if step.execution_error:
            parts.append(f"  Execution error: {step.execution_error.strip()}")
        elif step.outputs:
            out_summaries: List[str] = []
            for k, v in step.outputs.items():
                v_repr = truncate_text(repr(v), output_max_length)
                out_summaries.append(f"{k} = {v_repr}")
            if out_summaries:
                parts.append("  Outputs:\n    " + "\n    ".join(out_summaries))

    if step.verifier_sufficient is not None:
        suff = "sufficient" if step.verifier_sufficient else "NOT sufficient"
        parts.append(f"  Verifier: {suff}")
        if step.verifier_explanation:
            parts.append(
                f"    Verifier explanation: {step.verifier_explanation.strip()}"
            )

    if step.router_action:
        parts.append(f"  Router action: {step.router_action}")
        if step.router_fix_index is not None:
            parts.append(f"    Fix index: {step.router_fix_index}")
        if step.router_explanation:
            parts.append(f"    Router explanation: {step.router_explanation.strip()}")

    return "\n".join(parts)


def summarize_last_outputs_for_planner(
    state: DSState,
    output_max_length: int,
) -> str:
    """Summarize ONLY the last step's outputs (used mainly in FULL mode)."""
    if not state.steps:
        return "No steps exist yet."

    last_step = state.steps[-1]
    if not last_step.outputs:
        return "The last step did not produce any outputs."

    lines: List[str] = []
    for k, v in last_step.outputs.items():
        v_str = repr(v)
        v_str = truncate_text(v_str, output_max_length)
        lines.append(f"{k} = {v_str}")
    return "\n".join(lines)


def build_planner_prompt(
    state: DSState,
    tools_specs: str,
    max_history_steps: int = 6,
) -> tuple[str, str]:
    """
    Returns (system_message, user_message).

    - STEPWISE: history includes per-step outputs/errors/logs.
    - FULL: history focuses on plans + verifier/router info, and also includes last step outputs explicitly.
    """
    # Build execution environment restrictions for planner context
    # Use the mode from state to ensure consistency with actual execution
    env_restrictions = build_execution_environment_instructions(
        state=state,
        mode=state.code_mode.value,  # Use state.code_mode instead of hardcoded "stepwise"
        include_prev_step_outputs=True,
        include_tools_instructions=True,
        logs_max_length=state.logs_max_length,
    )

    system_instructions = (
        "You are the Planner agent in a DS-STAR-style data science system.\n"
        "Your role is to propose the NEXT high-level analysis step in natural language.\n"
        "The Coder agent will later translate your step into Python code that calls tools.\n"
        "You NEVER write code yourself; you ONLY describe the next step.\n\n"
        "EXECUTION ENVIRONMENT RESTRICTIONS YOU SHOULD TAKE INTO ACCOUNT:\n"
        f"{env_restrictions}\n"
        "OUTPUT FORMAT:\n"
        "- Your response will be parsed into a structured object: PlanOneStepOutput.\n"
        "- You MUST provide ONLY the natural-language description of the step.\n"
        "- Do NOT wrap the output in JSON, markdown, or any other structure.\n"
        "- The content you output must be valid for: PlanOneStepOutput(step=<your sentence>).\n"
        "- The step MUST be one concise, concrete, executable instruction.\n\n"
        "Guidelines:\n"
        "- Steps must be concrete and executable using the available tools, OR focus on preparing the final answer using existing outputs.\n"
        "- Refer to tools by name when relevant (e.g., 'use tool_x to do Y'), but do NOT write code.\n"
        "- When using the `search_files` tool, review the summaries/descriptions of all returned files instead of assuming the first result is the most relevant; base your next step on the best match.\n"
        "- Use the user query and previous outcomes to decide what comes next.\n"
        "- If existing outputs already contain enough information to answer the query (even partially), "
        "you may propose a step like: 'Prepare a final answer summarizing the available results, clearly noting any limitations or missing data.'\n"
        "- Avoid repeating successful steps; move the analysis forward.\n"
        "- If a step was flawed, propose a corrected replacement instead of adding more flawed steps.\n"
        "- Never output more than one step.\n"
    )

    user_block = f"User query:\n{state.user_query.strip()}\n"
    tools_section = f"Available tools (usable later by the Coder):\n{tools_specs}\n"

    # ---------- FIRST STEP ----------
    if not state.steps:
        user_message = (
            f"{user_block}\n"
            f"{tools_section}\n"
            "There are no previous steps yet.\n"
            "Propose the FIRST high-level analysis step that should be executed using the tools.\n"
            "Examples of step types:\n"
            "- Load or fetch relevant data using a tool.\n"
            "- Inspect or summarize data using a tool.\n"
            "- Transform or filter data using a tool.\n"
            "- Train or evaluate a model using a tool.\n\n"
            "Respond with ONLY the step text. No explanations. No code."
        )
        return system_instructions, user_message

    # ---------- SUBSEQUENT STEPS ----------
    include_outputs_in_history = state.code_mode == CodeMode.STEPWISE
    output_max_length = state.output_max_length
    logs_max_length = state.logs_max_length

    total_steps = len(state.steps)
    start_idx = max(0, total_steps - max_history_steps)
    visible_steps = list(enumerate(state.steps[start_idx:], start=start_idx))

    history_lines = [
        summarize_step_for_planner(
            step,
            i,
            include_outputs=include_outputs_in_history,
            output_max_length=output_max_length,
            logs_max_length=logs_max_length,
        )
        for i, step in visible_steps
    ]

    history_header = (
        f"(Only the last {max_history_steps} steps are shown in detail out of "
        f"{total_steps} total steps.)\n"
        if start_idx > 0
        else ""
    )

    history_block = (
        "Previous steps and their outcomes:\n"
        f"{history_header}" + "\n\n".join(history_lines) + "\n"
    )

    last_outputs_block = ""
    if state.code_mode == CodeMode.FULL:
        last_outputs_block = (
            "Latest available outputs from the last step (current results):\n"
            f"{summarize_last_outputs_for_planner(state, output_max_length=output_max_length)}\n"
        )

    last_step = state.steps[-1]

    # FIX mode
    if last_step.router_action == "fix_step" and last_step.router_fix_index is not None:
        mode_instructions = (
            f"The Router determined that Step {last_step.router_fix_index} is incorrect or needs revision.\n"
            f"Your task is to propose a NEW corrected version of Step {last_step.router_fix_index}.\n"
            "Do NOT add a new step; overwrite this one.\n\n"
            "Respond with ONLY the corrected step text. No explanations. No code."
        )
    else:
        mode_instructions = (
            "The Router determined that the plan should continue with a NEW step.\n"
            "Propose the NEXT step that moves closer to answering the user query.\n"
            "This may be either:\n"
            "- Another analysis / retrieval / transformation step using tools, OR\n"
            "- A step that prepares the final answer using the results obtained so far "
            "(e.g., summarizing, formatting, and clearly explaining any limitations such as partial results).\n"
            "Do NOT restate earlier steps. Do NOT output more than one step.\n\n"
            "Respond with ONLY the next step text. No explanations. No code."
        )

    user_message = (
        f"{user_block}\n"
        f"{tools_section}\n"
        f"{history_block}\n"
        f"{last_outputs_block}\n"
        f"{mode_instructions}"
    )

    logger.info(
        "planner prompt length: %d", len(system_instructions) + len(user_message)
    )
    return system_instructions, user_message


class PlannerNode(BaseNode):
    structured_output_schema: Optional[Type[BaseModel]] = PlanOneStepOutput

    def __call__(self, state: DSState) -> DSState:
        """Plan exactly ONE next step, or replace a specific step when fixing."""
        # Defensive: if something upstream passed a dict, fail loudly and early.
        if not isinstance(state, DSState):
            raise TypeError(
                f"PlannerNode expected DSState, got {type(state).__name__}. "
                "Fix the caller to pass DSState, or refactor this node to use dict access."
            )

        if state.fatal_error:
            add_event_to_trajectory(
                state, "n_plan_one", skipped=True, fatal_error=state.fatal_error
            )
            return state

        steps: List[DSStep] = state.steps
        i = len(steps) - 1

        system_msg, user_msg = build_planner_prompt(state, self.tools_spec)
        print_once("ds_star_planner_prompt", f"DS Star Planner prompt:\n{user_msg}")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            response, usage = invoke_structured_with_usage(
                self.llm_with_output, messages, run_name="n_plan_one"
            )
            state.token_usage.append(usage)
            step_text = (response.step or "").strip()
        except ValidationError as ve:
            state.fatal_error = f"Planner schema validation failed: {ve}"
            logger.error("Fatal error planning at step %s: %s", i, state.fatal_error)
            add_event_to_trajectory(state, "n_plan_one", fatal_error=state.fatal_error)
            return state
        except Exception as e:
            state.fatal_error = f"Planner invocation failed: {e}"
            logger.error("Fatal error planning at step %s: %s", i, state.fatal_error)
            add_event_to_trajectory(state, "n_plan_one", fatal_error=state.fatal_error)
            return state

        if not step_text:
            state.fatal_error = "Planner produced empty step."
            add_event_to_trajectory(state, "n_plan_one", fatal_error=state.fatal_error)
            return state

        # ---- Apply: replace or append ----
        new_step = DSStep(plan=step_text)

        fix_idx: Optional[int] = None
        if steps:
            last = steps[-1]
            if last.router_action == "fix_step" and last.router_fix_index is not None:
                fix_idx = int(last.router_fix_index)

        if fix_idx is not None:
            # Clamp to valid range instead of silently doing append.
            fix_idx = max(0, min(fix_idx, len(steps) - 1))
            steps[fix_idx] = new_step
            del steps[fix_idx + 1 :]
            action = "replace"
            planned_index = fix_idx
        else:
            steps.append(new_step)
            action = "append"
            planned_index = len(steps) - 1

        state.steps = steps
        state.steps_used = int(state.steps_used) + 1

        if 0 <= planned_index < len(steps):
            logger.info("Planned step %s: %s", planned_index, steps[planned_index].plan)

        add_event_to_trajectory(
            state, "n_plan_one", planned_step=step_text, action=action
        )
        return state
