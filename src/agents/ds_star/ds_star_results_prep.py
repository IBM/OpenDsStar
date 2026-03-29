from __future__ import annotations

import ast
import json
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from agents.ds_star.ds_star_state import DSState
from agents.ds_star.ds_star_utils import steps_to_plan_string


def _truncate_str(s: str, max_len: int = 1000) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _dsstate_to_dict(ds_state: Any) -> dict[str, Any]:
    """Best-effort conversion of DSState to a dict.
    Supports pydantic (model_dump/dict), dataclasses, or __dict__ fallback.
    """
    if isinstance(ds_state, dict):
        return ds_state

    if hasattr(ds_state, "model_dump") and callable(ds_state.model_dump):
        return cast("dict[str, Any]", ds_state.model_dump())

    if hasattr(ds_state, "dict") and callable(ds_state.dict):
        return cast("dict[str, Any]", ds_state.dict())

    if is_dataclass(ds_state):
        return cast("dict[str, Any]", asdict(ds_state))

    if hasattr(ds_state, "__dict__"):
        return cast("dict[str, Any]", dict(ds_state.__dict__))

    raise TypeError(f"Unsupported DSState type for serialization: {type(ds_state)!r}")


def _jsonify_and_truncate(value: Any) -> Any:
    """Convert value into JSON-serializable types and truncate long strings recursively."""
    if value is None or isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, str):
        return _truncate_str(value)

    if isinstance(value, dict):
        return {str(k): _jsonify_and_truncate(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_jsonify_and_truncate(v) for v in value]

    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonify_and_truncate(value.model_dump())

    if hasattr(value, "dict") and callable(value.dict):
        return _jsonify_and_truncate(value.dict())

    if is_dataclass(value):
        return _jsonify_and_truncate(asdict(value))

    return _truncate_str(str(value))


def _split_top_level_commas(s: str) -> list[str]:
    """Split `s` on commas that are not inside quotes/brackets/braces/parens.
    Used to parse repr strings like DSStep(a='..', b={...}, c=[...]).
    """
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = depth_brack = depth_brace = 0
    in_single = in_double = False
    escape = False

    for ch in s:
        if escape:
            buf.append(ch)
            escape = False
            continue

        if ch == "\\":
            buf.append(ch)
            escape = True
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            continue

        if not in_single and not in_double:
            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
            elif ch == "[":
                depth_brack += 1
            elif ch == "]":
                depth_brack = max(0, depth_brack - 1)
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace = max(0, depth_brace - 1)

            if ch == "," and depth_paren == depth_brack == depth_brace == 0:
                parts.append("".join(buf).strip())
                buf = []
                continue

        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_repr_step_to_dict(step_repr: str) -> dict[str, Any]:
    """Parse strings like:
      DSStep(plan='..', code='..', outputs={...}, execution_error=None, verifier_sufficient=False, ...)
    into a Python dict.

    Best-effort: if a value can't be literal-evaluated, it's stored as a raw string.
    """
    s = step_repr.strip()
    if "(" not in s or not s.endswith(")"):
        raise ValueError("Not a callable-like repr ending in ')'")

    left = s.find("(")
    inner = s[left + 1 : -1].strip()

    items = _split_top_level_commas(inner)
    out: dict[str, Any] = {}

    for item in items:
        if not item or "=" not in item:
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()

        try:
            out[key] = ast.literal_eval(val)
        except Exception:
            out[key] = val

    return out


def _normalize_trajectory_event(event: Any) -> dict[str, Any]:
    """Make a single trajectory event JSON-safe + truncate long strings.
    Note: last_step is already a dict (converted in add_event_to_trajectory).
    """
    # Make dict-ish
    if isinstance(event, dict):
        evt: dict[str, Any] = dict(event)
    elif hasattr(event, "model_dump") and callable(event.model_dump):
        evt = cast("dict[str, Any]", event.model_dump())
    elif hasattr(event, "dict") and callable(event.dict):
        evt = cast("dict[str, Any]", event.dict())
    elif is_dataclass(event):
        evt = cast("dict[str, Any]", asdict(event))
    elif hasattr(event, "__dict__"):
        evt = cast("dict[str, Any]", dict(event.__dict__))
    else:
        evt = {"value": str(event)}

    # last_step is already a dict from add_event_to_trajectory, no conversion needed

    return cast("dict[str, Any]", _jsonify_and_truncate(evt))


def prepare_result_from_graph_state_ds_star_agent(
    state: DSState | dict[str, Any],
) -> dict[str, Any]:
    """Extract and format key results from the final graph state.

    Requirements:
    1) Shorten strings longer than 1000 characters to 997 chars + "..." (recursively).
    2) last_step is now a string; return last_step as a dict (parsed from repr-like string).
    3) Return the full state as a dict (JSON) under "state".
    4) Normalize trajectory events so their "last_step" becomes a dict (not a string).
    """
    ds_state = state if isinstance(state, DSState) else DSState(**state)
    state_dict = _dsstate_to_dict(ds_state)

    token_usage = getattr(ds_state, "token_usage", None) or []
    input_tokens = sum(int(tu.get("input_tokens", 0)) for tu in token_usage)
    output_tokens = sum(int(tu.get("output_tokens", 0)) for tu in token_usage)

    steps = getattr(ds_state, "steps", None) or []
    last_step_raw = str(steps[-1]) if steps else ""

    last_step: dict[str, Any] = {}
    if last_step_raw.strip():
        try:
            last_step = _parse_repr_step_to_dict(last_step_raw)
        except Exception:
            last_step = {}

    # Best-effort: steps_to_plan_string might fail if steps are strings
    try:
        plan = str(steps_to_plan_string(steps))
    except Exception:
        plan = ""

    verifier_sufficient = bool(
        last_step.get(
            "verifier_sufficient", state_dict.get("verifier_sufficient", False)
        )
    )
    execution_error_val = last_step.get(
        "execution_error", state_dict.get("execution_error", "")
    )
    execution_error = (
        "" if execution_error_val in (None, "") else str(execution_error_val)
    )

    # Normalize trajectory events (fixes per-event last_step being a string)
    raw_traj = state_dict.get("trajectory", []) or []
    if isinstance(raw_traj, list):
        normalized_trajectory = [_normalize_trajectory_event(e) for e in raw_traj]
    else:
        normalized_trajectory = []

    # Update state snapshot to include normalized trajectory
    state_dict["trajectory"] = normalized_trajectory

    result: dict[str, Any] = {
        "answer": str(getattr(ds_state, "final_answer", "") or ""),
        "trajectory": normalized_trajectory,
        "plan": plan,
        "steps_used": int(getattr(ds_state, "steps_used", 0) or 0),
        "max_steps": int(getattr(ds_state, "max_steps", 0) or 0),
        "verifier_sufficient": verifier_sufficient,
        "fatal_error": str(getattr(ds_state, "fatal_error", "") or ""),
        "execution_error": execution_error,
        "last_step": last_step,
        "last_step_raw": last_step_raw,
        "context": state_dict.get("context", []) or [],
        "context_ids": state_dict.get("context_ids", []) or [],
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "num_llm_calls": len(token_usage),
        "state": state_dict,
    }

    # Ensure JSON-serializable + truncate long strings everywhere
    result = cast("dict[str, Any]", _jsonify_and_truncate(result))
    json.dumps(result)
    return result
