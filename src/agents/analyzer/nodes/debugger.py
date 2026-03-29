"""Debugger for analyzer agent - uses structured output like DS*Star."""

import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agents.ds_star.ds_star_utils import build_execution_environment_instructions

logger = logging.getLogger(__name__)


class CodeOutput(BaseModel):
    """Schema for structured code output."""

    code: str = Field(
        ..., description="Corrected Python code that fixes the execution error."
    )


class DebuggerNode:
    """Simple debugger that uses structured output to fix code errors."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        # Create structured output LLM
        if hasattr(llm, "with_structured_output"):
            self.llm_with_output = llm.with_structured_output(CodeOutput)
        else:
            raise TypeError(
                "Provided llm does not support `with_structured_output(...)`."
            )

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Debug and fix the code."""
        if state["fatal_error"]:
            state["trajectory"].append(
                {
                    "node": "n_debug",
                    "skipped": True,
                    "fatal_error": state["fatal_error"],
                }
            )
            return state

        if not state["execution_error"]:
            state["trajectory"].append(
                {"node": "n_debug", "skipped": True, "note": "No error to debug"}
            )
            return state

        state["debug_tries"] += 1
        logger.info(
            f"Debugging attempt {state['debug_tries']}/{state['max_debug_tries']}"
        )

        filename = state["filename"]

        # Build execution environment instructions
        env_instructions = build_execution_environment_instructions(
            state=None,
            mode="stepwise",
            include_prev_step_outputs=False,
            include_tools_instructions=False,
            logs_max_length=0,  # No print output tracking for analyzer
        )

        # Build debugger prompt directly
        debugger_prompt = f"""You are an expert data analyst debugging Python code.

User query:
Generate a Python code that loads and describes the content of {filename}.

# Requirements
- The file can be both unstructured or structured data.
- If there are too many structured data, print out just few examples.
- Print out essential information. For example, print out all the column names.
- The Python code should print out the content of {filename}.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Important: You should not include dummy contents since we will debug if error occurs.
- Do not use try: and except: to prevent error. I will debug it later.

# {env_instructions}

# Store any important results in the global 'outputs' dictionary for later retrieval.

Execution error:
{state['execution_error']}

Previous (failing) Python code:
{state['code']}

Output format:
- Your response will be parsed into CodeOutput.code.
- You MUST output ONLY the COMPLETE corrected Python script as plain text.
- Do NOT include backticks, markdown fences, JSON, or any surrounding text.
- Output ONLY raw Python code. No backticks, no markdown."""

        messages = [{"role": "user", "content": debugger_prompt}]

        try:
            # Call LLM with structured output
            response = self.llm_with_output.invoke(messages)

            # Extract code from structured output and remove any import statements
            code_without_imports = re.sub(
                r"^\s*(import|from)\s+.*$", "", response.code, flags=re.MULTILINE
            )
            state["code"] = code_without_imports
            logger.info(
                f"Debugger produced corrected code for step 0.\nCode: {state['code']}"
            )

            state["trajectory"].append(
                {
                    "node": "n_debug",
                    "debug_try": state["debug_tries"],
                    "code_length": len(state["code"]) if state["code"] else 0,
                }
            )

        except Exception as e:
            logger.error(f"Debugging failed: {e}")
            state["fatal_error"] = f"Debugging failed: {e}"
            state["trajectory"].append({"node": "n_debug", "error": str(e)})

        return state
