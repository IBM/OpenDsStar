import ast
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from agents.ds_star.ds_star_state import DSState, DSStep

logger = logging.getLogger(__name__)

# Constants
INPUT_TOKENS_KEY = "input_tokens"
OUTPUT_TOKENS_KEY = "output_tokens"
PROMPT_TOKENS_KEY = "prompt_tokens"
COMPLETION_TOKENS_KEY = "completion_tokens"


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""



class _UsageCaptureHandler(BaseCallbackHandler):
    """Capture token usage from LLM calls (supports modern + legacy paths)."""

    def __init__(self) -> None:
        self.usage = {INPUT_TOKENS_KEY: 0, OUTPUT_TOKENS_KEY: 0}

    def on_llm_end(self, response, **kwargs) -> None:
        # Newer path
        um = getattr(response, "usage_metadata", None)
        if isinstance(um, dict):
            self.usage = {
                INPUT_TOKENS_KEY: int(um.get(INPUT_TOKENS_KEY, 0)),
                OUTPUT_TOKENS_KEY: int(um.get(OUTPUT_TOKENS_KEY, 0)),
            }
            return

        # Legacy OpenAI path
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            tu = llm_output.get("token_usage", {})
            self.usage = {
                INPUT_TOKENS_KEY: int(tu.get(PROMPT_TOKENS_KEY, 0)),
                OUTPUT_TOKENS_KEY: int(tu.get(COMPLETION_TOKENS_KEY, 0)),
            }


def invoke_structured_with_usage(
    runnable, messages: list[dict[str, str]], run_name: str
) -> tuple[Any, dict[str, Any]]:
    """Invoke a structured output runnable and capture token usage.
    No JSON repair is attempted; errors are propagated.
    """
    cb = _UsageCaptureHandler()
    obj = runnable.with_config(callbacks=[cb], run_name=run_name).invoke(messages)
    return obj, cb.usage


def add_event_to_trajectory(state: DSState, node: str, **fields: Any) -> None:
    """Add an event to the trajectory log and invoke callback if present.

    Note: Modifies state.trajectory in-place.
    Converts DSStep objects to dicts for proper JSON serialization.
    If state has a trajectory_callback, invokes it with the new event.
    Uses deduplication to ensure each (node, step_idx) is only emitted once.
    """
    import logging

    logger = logging.getLogger(__name__)

    last_step = safe_get(state.steps, -1, None)
    # Convert DSStep to dict for proper JSON serialization
    if isinstance(last_step, DSStep):
        last_step = asdict(last_step)

    step_idx = max(len(state.steps) - 1, 0)
    event = {
        "time": time.time(),  # UTC timestamp
        "node": node,
        "steps_used": state.steps_used,
        "step_idx": step_idx,
        "last_step": last_step,
    }
    event.update(fields)
    state.trajectory.append(event)

    # Invoke callback if present, with deduplication
    callback = getattr(state, "trajectory_callback", None)
    if callback is not None and callable(callback):
        # Create unique key for this event
        event_key = (node, step_idx)

        # Ensure dedup state is persisted on `state`
        emitted_events = getattr(state, "_emitted_events", None)
        if emitted_events is None:
            emitted_events = set()
            state._emitted_events = emitted_events

        # Only emit if we haven't seen this (node, step_idx) before
        if event_key not in emitted_events:
            try:
                logger.debug(f"Emitting event: node={node}, step_idx={step_idx}")
                callback(event)
                emitted_events.add(event_key)
            except Exception as e:
                # Don't let callback errors break the agent
                logger.warning(f"trajectory_callback error: {e}")
        else:
            logger.debug(f"Skipping duplicate event: node={node}, step_idx={step_idx}")


def safe_get(lst: list, idx: int, default: Any = None) -> Any:
    """Safely get an item from a list by index, returning default if out of bounds."""
    try:
        return lst[idx]
    except (IndexError, TypeError):
        return default


def normalize_tool_result(result: Any) -> Any:
    """Normalize various tool return shapes into a usable Python primitive.

    - Unwrap CallToolResult-like objects with a `.content` list of TextContent
    - Accept lists of TextContent and return the first `.text` if present
    - Extract `.text` or `.content` attributes when present
    - Return DataFrames and other runtime objects unchanged
    - Fall back to str() for unknown structured objects
    """
    import numpy as np
    import pandas as pd

    # None or basic scalar types
    if result is None or isinstance(result, (str, int, float, bool, dict, tuple)):
        return result

    if callable(result):
        return result

    # Return pandas/numpy objects unchanged for code execution
    if isinstance(result, (pd.DataFrame, pd.Series, pd.Index, np.ndarray, np.generic)):
        return result

    # Return stream-like objects unchanged
    if hasattr(result, "read"):
        return result

    # Handle list/sequence results (TextContent-like or primitives)
    if isinstance(result, list):
        if not result:
            return result
        first = result[0]
        # If list contains primitives, return as-is
        if (
            isinstance(first, (str, int, float, bool, dict, list, tuple))
            or first is None
        ):
            return result
        # If list contains TextContent-like objects, extract first text
        text = getattr(first, "text", None) or getattr(first, "content", None)
        if isinstance(text, str):
            return text
        return str(first)

    # CallToolResult-like: prefer .content which may be a list or string
    content = getattr(result, "content", None)
    if content:
        if isinstance(content, list) and len(content) > 0:
            first = content[0]
            text = getattr(first, "text", None) or getattr(first, "content", None)
            if isinstance(text, str):
                return text
            return str(first)
        if isinstance(content, str):
            return content

    # Structured object with textual field
    text_attr = getattr(result, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    cont_attr = getattr(result, "content", None)
    if isinstance(cont_attr, str):
        return cont_attr

    # Fallback
    return str(result)


def build_tools_map(tools_list: list[Any]) -> dict[str, Callable[..., Any]]:
    """Wrap LangChain/LangGraph tools into simple **kwargs callables keyed by tool.name.
    """
    mapping: dict[str, Callable[..., Any]] = {}

    for tool in tools_list:
        name = getattr(tool, "name", None) or tool.__class__.__name__

        def make_call(t, _name=name):
            def _call(*args, **kwargs):
                try:
                    # Try .func first (StructuredTool has this)
                    fn = getattr(t, "func", None)
                    if callable(fn):
                        result = fn(*args, **kwargs)
                        return normalize_tool_result(result)

                    # Then try .invoke (other LangChain tools)
                    if hasattr(t, "invoke"):
                        # For invoke, if we have positional args, we need to handle them
                        if args:
                            # If there's only one positional arg and no kwargs, pass it directly
                            if len(args) == 1 and not kwargs:
                                result = t.invoke(args[0])
                            else:
                                # Otherwise, combine args and kwargs into a dict
                                # This is a fallback - ideally tools should use kwargs
                                result = t.invoke(
                                    kwargs if not args else {"args": args, **kwargs}
                                )
                        else:
                            result = t.invoke(kwargs)
                        return normalize_tool_result(result)

                    # Finally try calling directly
                    if callable(t):
                        result = t(*args, **kwargs)
                        return normalize_tool_result(result)

                    raise ToolExecutionError(
                        f"Tool '{_name}' is not callable and has no .invoke/.func"
                    )
                except Exception as e:
                    # Check if it's an event loop error from langflow MCP
                    error_msg = str(e)
                    if (
                        "Event loop is closed" in error_msg
                        or "RuntimeError: Event loop" in error_msg
                    ):
                        logger.error(
                            f"Tool '{_name}' failed due to event loop issue (likely langflow MCP): {e}"
                        )
                        raise ToolExecutionError(
                            f"Tool '{_name}' failed due to async event loop issue. "
                            f"This may be caused by langflow MCP integration. "
                            f"Try using standalone MCP integration instead. Error: {e}"
                        ) from e
                    logger.exception(f"Tool '{_name}' execution failed")
                    raise ToolExecutionError(
                        f"Tool '{_name}' execution failed: {e}"
                    ) from e

            return _call

        mapping[name] = make_call(tool)

    return mapping


def format_tools_spec(tools_list: list[Any]) -> str:
    """Build a compact JSON spec containing name, description, and a light param sketch.
    """
    spec = []
    for t in tools_list:
        # name
        try:
            name = t.name  # prefer direct access when possible
        except Exception:
            name = None
        name = name or t.__class__.__name__

        # description
        desc = ""
        for attr in ("description", "desc"):
            try:
                val = getattr(t, attr, "")
            except Exception:
                val = ""
            if val:
                desc = str(val)
                break

        # schema
        schema_dict: dict[str, Any] | None = None
        try:
            args_schema = t.args_schema
        except Exception:
            args_schema = None

        if args_schema:
            try:
                schema_dict = args_schema.model_json_schema()
            except Exception:
                try:
                    schema_dict = args_schema.schema()
                except Exception:
                    schema_dict = None

        params: dict[str, Any] = {}
        required: list[str] = []
        if isinstance(schema_dict, dict):
            params = schema_dict.get("properties") or {}
            required = schema_dict.get("required") or []

        # If you *really* want to fail on param-less tools, keep the raise.
        # Otherwise, allow empty param sets:
        entry = {"name": name, "description": desc, "params": params}
        if required:
            entry["required"] = required

        spec.append(entry)

    return json.dumps(spec, ensure_ascii=False, indent=2)


def steps_to_plan_string(steps: list[Any]) -> str:
    """Convert steps list to a formatted plan string.
    """
    msg = "\n\nPlan: "
    lines = []
    for i, s in enumerate(steps):
        plan_text = getattr(s, "plan", None)
        if plan_text:
            lines.append(f"{i}: {plan_text}")
    msg += "\n".join(lines) if lines else "<plan is empty>"
    return msg


# --- Helpers ---


def _format_tools(tools: dict[str, Callable[..., Any]]) -> str:
    """Format tool names and docstrings for inclusion in prompts."""
    if not tools:
        return "(no tools available)"
    lines = []
    for name, fn in tools.items():
        doc = (fn.__doc__ or "").strip()
        if doc:
            lines.append(f"- {name}: {doc}")
        else:
            lines.append(f"- {name}: (no description provided)")
    return "\n".join(lines)


def truncate_text(text: str | None, max_len: int, suffix: str = "...") -> str:
    """Truncate text to max_len characters, appending suffix if truncated.

    Args:
        text: Text to truncate (can be None)
        max_len: Maximum length before truncation
        suffix: String to append when truncated (default "...")

    Returns:
        Truncated text with suffix, or original text if within limit.
        Returns empty string if text is None or empty.
    """
    if not text:
        return ""

    if len(text) > max_len:
        suffix_len = len(suffix)
        return text[: (max_len - suffix_len)] + suffix

    return text


class _ImportStripper(ast.NodeTransformer):
    """Removes all import statements at any depth.
    Replaces each with `pass` so block structure remains valid.
    """

    def visit_Import(self, node: ast.Import) -> ast.AST:
        return ast.copy_location(ast.Pass(), node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        return ast.copy_location(ast.Pass(), node)


def _fallback_remove_imports(code: str) -> str:
    """Best-effort fallback for syntactically invalid Python.
    Removes lines that start with import/from and attempts to swallow
    multiline continuations using parentheses and backslashes.
    """
    lines = code.splitlines()
    out: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        is_import_start = re.match(r"^(import|from)\b", stripped) is not None
        if not is_import_start:
            out.append(line)
            i += 1
            continue

        # Swallow this import statement, including multiline continuations.
        # Handles:
        #   import a, \
        #          b
        #   from x import (
        #       a,
        #       b,
        #   )
        balance = line.count("(") - line.count(")")
        cont_backslash = line.rstrip().endswith("\\")

        i += 1
        while i < len(lines):
            if balance <= 0 and not cont_backslash:
                break
            nxt = lines[i]
            balance += nxt.count("(") - nxt.count(")")
            cont_backslash = nxt.rstrip().endswith("\\")
            i += 1

    # Keep surrounding content; trim only extra trailing newlines.
    return "\n".join(out).rstrip()


def _assert_no_import_nodes(tree: ast.AST) -> None:
    """Defensive check: ensure no import statements remain."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RuntimeError("Import stripping failed: import node still present.")


def remove_imports(code: str) -> str:
    """Remove ALL `import ...` and `from ... import ...` statements at any depth,
    including nested imports inside functions/classes/conditionals/try blocks.

    Behavior:
    - Valid Python: uses AST-based line removal to preserve comments and formatting.
    - Invalid Python: uses best-effort line-based fallback.

    Returns transformed code as a string.
    """
    src = "" if code is None else str(code)

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return _fallback_remove_imports(src)

    # Collect line numbers of all import statements (1-based)
    import_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Add all lines this import spans
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                for line_num in range(
                    node.lineno, (node.end_lineno or node.lineno) + 1
                ):
                    import_lines.add(line_num)

    # Remove import lines while preserving everything else
    lines = src.splitlines()
    result_lines = []
    for i, line in enumerate(lines, start=1):
        if i not in import_lines:
            result_lines.append(line)

    return "\n".join(result_lines).rstrip()


def assert_no_imports(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RuntimeError("Imports are not allowed in sandbox code.")


FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "breakpoint",
}
FORBIDDEN_ATTR_BASES = {"os", "sys", "subprocess", "pathlib", "builtins", "importlib", "pickle", "threading"}


class CodeValidationError(Exception):
    pass


def validate_generated_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeValidationError(f"SyntaxError: {e}") from e

    for node in ast.walk(tree):
        # No imports should remain after remove_imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise CodeValidationError("Imports are not allowed.")

        # Block explicit global/nonlocal if you want stricter behavior
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise CodeValidationError(f"{type(node).__name__} is not allowed.")

        # Block risky calls by name
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_CALLS:
                raise CodeValidationError(f"Call to '{fn.id}' is not allowed.")
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                if fn.value.id in FORBIDDEN_ATTR_BASES:
                    raise CodeValidationError(
                        f"Access to '{fn.value.id}.{fn.attr}' is not allowed."
                    )

        # Block dunder attribute tricks
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CodeValidationError(f"Dunder attribute '{node.attr}' is not allowed.")


def build_execution_environment_instructions(
    state=None,
    mode: str = "stepwise",
    include_prev_step_outputs: bool = True,
    include_tools_instructions: bool = True,
    logs_max_length: int = 2000,
) -> str:
    """Build standardized execution environment instructions for coder and debugger.

    Args:
        state: Current state (DSState or dict-like), optional for simple cases
        mode: Either "stepwise" or "full" to determine which instructions to include
        include_prev_step_outputs: Whether to include prev_step_outputs in globals list
        include_tools_instructions: Whether to include tools in the instructions
        logs_max_length: Maximum length for print output (default 2000)

    Returns:
        Formatted execution environment instruction block
    """
    from agents.ds_star.nodes.coder import get_available_packages_list

    # Extract logs_max_length from state if available
    if state is not None:
        logs_max_length = getattr(state, "logs_max_length", logs_max_length)

    # Base instructions
    base_instructions = [
        "Execution environment:",
        "- No imports allowed; any `import` or `from ... import ...` will fail.",
    ]

    # Add filesystem/network restrictions only if tools are included
    if include_tools_instructions:
        base_instructions.extend(
            [
                "- The environment has no filesystem access. Never use pd.read_csv(), pd.read_parquet(), or similar file reading functions with file paths.",
                "- No network access.",
                "- Tools are available as regular Python functions. Call them directly with their arguments (e.g., `result = search_tool(query='test')`).",
                "- To read files, use the provided tools. Check each tool's description for usage examples.",
            ]
        )
    else:
        base_instructions.append(
            "- No filesystem or network access beyond reading the specified file."
        )

    # Global variables section - differs by mode and options
    globals_section = ["- Global variables provided:"]
    if include_tools_instructions:
        globals_section.append("    * state (read-only)")

    if mode == "stepwise" and include_prev_step_outputs:
        globals_section.extend(
            [
                "    * outputs (dict for storing results of THIS step)",
                "    * prev_step_outputs (aggregated outputs of steps 0..k-1)",
            ]
        )
    else:
        output_desc = (
            "dict for final results" if mode == "full" else "dict for storing results"
        )
        globals_section.append(f"    * outputs ({output_desc})")

    # Libraries and print statement instructions
    packages_list = get_available_packages_list()
    footer_instructions = [
        f"- Preloaded libraries (available to use, no need to import!): {packages_list}.",
        "- Other libraries are NOT available.",
    ]

    # Only add print statement info if logs_max_length is meaningful
    if logs_max_length > 0:
        footer_instructions.append(
            f"- Print statements: print() output will be visible to agent for next step planning (truncated to {logs_max_length} chars)."
        )
        footer_instructions.append(
            "  Use print() for key intermediate results, sanity checks, and concise summaries."
        )

    # Error handling instructions
    error_handling_instructions = [
        "- Error handling: ",
        "  - If you detect a problem during code execution, set an informative error message in outputs['_error'].",
        "  - Also add relevant information that can help agent debug the problem and solve it.",
        "  Example:",
        "  ```python",
        "  try:",
        "      # your code here",
        "      result = some_operation()",
        "      outputs['result'] = result",
        "  except Exception as e:",
        "      outputs['_error'] = str(e)",
        "      outputs['_error'] += '\\n<additional information for debugging>'",
        "      outputs['_traceback'] = traceback.format_exc()  # Optional: include full stack trace",
        "  ```",
        "  This allows the debugger to identify and fix execution errors.",
    ]

    # Combine all sections
    all_lines = (
        base_instructions
        + globals_section
        + footer_instructions
        + error_handling_instructions
    )
    return "\n".join(all_lines) + "\n"
