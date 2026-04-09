"""
DS-Star code execution environment.

This module executes generated Python code in a restricted sandbox with:

- user-written imports removed and rejected
- a curated builtins environment
- a pre-injected scientific/runtime environment
- tool access from generated code
- hard timeout enforcement that works in both the main thread and worker threads

Design summary
--------------
Generated code is ALWAYS executed in a dedicated child process. Timeout is
enforced by terminating that child process if it exceeds the configured limit.

Tools are NOT executed inside the child process. Instead, tools remain in the
parent process and are exposed to the child through RPC-style proxy callables
over a multiprocessing Pipe. This gives us both:

- reliable timeout enforcement, because the child can always be terminated
- tool usability, because generated code can still call tools as normal
  Python functions (e.g. `search_files(...)`)

Important guarantees
--------------------
1. Timeout works regardless of whether this module is called from:
   - the main thread
   - a worker thread
   - an environment with an already-running asyncio event loop

2. Generated code can always use tools, as long as:
   - tool arguments crossing child -> parent are serializable
   - tool results crossing parent -> child are serializable

3. Tool objects/functions themselves do NOT need to be picklable because they
   never leave the parent process.

Important caveats
-----------------
1. Timeout is enforced by killing the CHILD PROCESS, not by interrupting Python
   execution inside the same thread. This is much more reliable than
   signal.alarm(), but it means:
   - `finally` blocks in generated user code are NOT guaranteed to finish on timeout
   - in-child cleanup is NOT guaranteed on timeout

2. Tools run in the PARENT PROCESS. Therefore:
   - tool side effects happen in the parent process
   - a blocking tool can still stall the overall execution unless the tool
     itself has its own timeout/cancellation policy

   In other words, the subprocess timeout guarantees that user code cannot run
   forever, but it does NOT magically make every parent-side tool bounded.

3. Tool args/results must cross a process boundary. Therefore they must be
   serializable by `pickle` in this implementation.
   Usually fine:
   - str / int / float / bool / None
   - dict / list / tuple
   - pandas DataFrames / Series
   - numpy arrays / scalars
   Usually NOT fine:
   - open file handles
   - DB connections
   - sockets
   - locks
   - generators / iterators with live state
   - arbitrary non-picklable class instances

4. Only picklable parts of the execution scope are copied into the child.
   Non-picklable values in state or injected scope are silently omitted from
   the child environment.

Execution model note
--------------------
Code is executed with ONE shared namespace dict for both globals and locals:

    exec(code, scope, scope)

This is important because functions defined inside exec() resolve names via
their globals. Using separate globals/locals can cause nested functions to fail
to see variables created earlier in the same step.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import logging
import math
import multiprocessing
import pickle
import platform
import queue
import random
import statistics
import sys
import threading
import time
import traceback
from pathlib import PurePath
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly
import plotly.express as _px
import plotly.graph_objects as _go
import scipy as sp
from langchain.tools import BaseTool

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState
from OpenDsStar.agents.ds_star.ds_star_utils import (
    assert_no_imports,
    remove_imports,
    validate_generated_code,
)
from OpenDsStar.agents.utils.safe_imports import (
    get_safe_builtins,
    get_safe_scientific_env,
)

logger = logging.getLogger(__name__)

# Default maximum virtual memory (address space) for the child process.
# Prevents generated code from consuming unbounded memory (e.g. [0]*10**9).
# Must be large enough to accommodate the Python interpreter + preloaded
# scientific libraries (numpy, pandas, scipy, plotly ≈ 1–1.5 GB virtual).
# Set to 0 to disable the limit.
# Note: Only enforced on Linux. macOS does not support RLIMIT_AS enforcement.
# Can be overridden per-call via the max_memory_bytes parameter.
DEFAULT_MAX_CHILD_MEMORY_BYTES: int = 1 * 1024 * 1024 * 1024  # 1 GB


class TimeoutException(Exception):
    """Raised when code execution exceeds the configured timeout."""


class RemoteToolError(RuntimeError):
    """
    Raised in the child process when a parent-hosted tool fails.

    The original tool runs in the parent process. Any exception from that tool
    is captured, serialized, and re-raised in the child as a RemoteToolError
    with the original message/traceback included when possible.
    """


def _is_picklable(obj: Any) -> bool:
    """
    Return True if the object can be serialized with pickle.

    This implementation uses standard pickle because data must cross process
    boundaries over multiprocessing primitives.

    Note:
        Tool objects themselves do NOT need to be picklable because they are
        kept in the parent process. This helper is used only for data that
        actually crosses process boundaries or is copied into the child.
    """
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError, AttributeError):
        return False


def _serialize_tool_result(result: Any) -> Dict[str, Any]:
    """
    Serialize a tool result for cross-process transfer.

    For DataFrames (>100KB), uses Parquet format which is 5-10x faster
    than pickle and avoids Pipe blocking. For other objects, uses standard pickle.

    Args:
        result: The tool result to serialize

    Returns:
        Dict with 'type' and 'data' keys for deserialization

    Raises:
        TypeError: If result cannot be serialized
    """
    # Check if it's a DataFrame and use Parquet if available and beneficial
    if isinstance(result, pd.DataFrame) and PYARROW_AVAILABLE:
        # Estimate size - use Parquet for DataFrames >100KB to avoid Pipe blocking
        # (rough heuristic: memory_usage().sum() approximates in-memory size)
        try:
            estimated_size = result.memory_usage().sum()
            if estimated_size > 100 * 1024:  # 100KB threshold (lowered to avoid Pipe blocking)
                logger.debug(
                    f"Serializing DataFrame (~{estimated_size / 1024 / 1024:.1f}MB) "
                    "with Parquet for faster cross-process transfer"
                )
                # Type ignore: pa and pq are checked via PYARROW_AVAILABLE
                table = pa.Table.from_pandas(result)  # type: ignore
                sink = pa.BufferOutputStream()  # type: ignore
                pq.write_table(table, sink, compression="snappy")  # type: ignore
                return {
                    "type": "dataframe_parquet",
                    "data": sink.getvalue().to_pybytes(),  # type: ignore
                }
        except Exception as e:
            # Fall back to pickle if Parquet serialization fails
            logger.debug(f"Parquet serialization failed, falling back to pickle: {e}")

    # Default: use pickle for everything else
    try:
        return {"type": "pickle", "data": pickle.dumps(result)}
    except (pickle.PicklingError, TypeError, AttributeError) as e:
        raise TypeError(
            f"Tool result of type {type(result).__name__} cannot be serialized: {e}"
        ) from e


def _deserialize_tool_result(serialized: Dict[str, Any]) -> Any:
    """
    Deserialize a tool result from cross-process transfer.

    Args:
        serialized: Dict with 'type' and 'data' keys from _serialize_tool_result

    Returns:
        The deserialized tool result

    Raises:
        ValueError: If serialization type is unknown or data is missing
    """
    result_type = serialized.get("type")
    data = serialized.get("data")

    if data is None:
        raise ValueError("Serialized data is missing 'data' field")

    if result_type == "dataframe_parquet":
        if not PYARROW_AVAILABLE:
            raise RuntimeError(
                "Received Parquet-serialized DataFrame but pyarrow is not available"
            )
        logger.debug("Deserializing DataFrame from Parquet")
        # Type ignore: pa and pq are checked via PYARROW_AVAILABLE
        return pq.read_table(pa.BufferReader(data)).to_pandas()  # type: ignore

    if result_type == "pickle":
        return pickle.loads(data)

    raise ValueError(f"Unknown serialization type: {result_type}")


def _serialize_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize an outputs dict for cross-process transfer.

    Walks the dict and applies Parquet serialization to any DataFrame values
    so the entire dict can be pickled cheaply by Queue.put().
    """
    serialized: Dict[str, Any] = {}
    for key, value in outputs.items():
        if isinstance(value, pd.DataFrame) and PYARROW_AVAILABLE:
            try:
                serialized[key] = _serialize_tool_result(value)
                continue
            except Exception:
                pass  # fall through to raw assignment
        serialized[key] = value
    return serialized


def _deserialize_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize an outputs dict received from cross-process transfer.

    Reverses _serialize_outputs: any values that are serialized-tool-result
    dicts are deserialized back to their original types.
    """
    deserialized: Dict[str, Any] = {}
    for key, value in outputs.items():
        if isinstance(value, dict) and "type" in value and "data" in value:
            try:
                deserialized[key] = _deserialize_tool_result(value)
                continue
            except Exception:
                pass  # fall through to raw assignment
        deserialized[key] = value
    return deserialized


def _build_base_env() -> Dict[str, Any]:
    """
    Build the default runtime/scientific environment exposed to generated code.

    This environment is layered on top of the safe builtins/scientific env and
    provides familiar aliases for common data-science workflows.
    """
    env: Dict[str, Any] = {
        "math": math,
        "statistics": statistics,
        "random": random,
        "time": time,
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        "scipy": sp,
        "sp": sp,
        "Path": PurePath,
        "json": json,
        "traceback": traceback,
        "plotly": plotly,
        "plt": plotly,
        "go": _go,
        "px": _px,
        "Figure": _go.Figure,
        "FigurePlot": _go.Figure,
    }
    return env


def _run_coroutine_safely(coro: Any) -> Any:
    """
    Run an awaitable safely and return its result.

    Behavior:
    - If no event loop is running in the current thread, use asyncio.run().
    - If an event loop is already running, execute the coroutine in a dedicated
      background thread with its own event loop.

    Why this exists:
    Parent-hosted tools may be synchronous or async. Since tools execute in the
    parent process, this helper lets the parent safely await async tool results
    in both sync and async hosting contexts.

    Caveat:
    This helper waits for the coroutine to complete. It does NOT impose any
    timeout of its own.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=1)

    def runner() -> None:
        try:
            result = asyncio.run(coro)
            result_queue.put(("ok", result))
        except Exception as exc:  # pragma: no cover
            result_queue.put(("err", exc))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    status, value = result_queue.get()
    if status == "err":
        raise value
    return value


def _load_normalize_tool_result() -> Optional[Callable[[Any], Any]]:
    """
    Load the optional tool-result normalizer.

    This is intended only for agent/display-friendly values and must NOT be
    applied to runtime objects that generated code should manipulate directly.

    Examples of values that should generally NOT be normalized:
    - pandas DataFrames / Series
    - numpy arrays
    - callables
    - streams
    - arbitrary runtime objects
    """
    try:
        from OpenDsStar.agents.ds_star.ds_star_utils import (
            normalize_tool_result,  # type: ignore
        )
    except Exception:
        return None
    return normalize_tool_result


def _should_normalize_tool_result(result: Any) -> bool:
    """
    Decide whether a tool result is safe to normalize.

    Normalize only plain display-oriented values.
    Do NOT normalize runtime/code-facing objects.

    Note:
        Even if a value is picklable, that does NOT mean it should be normalized.
        Serialization and normalization are separate concerns.
    """
    if result is None:
        return True

    if isinstance(result, (str, int, float, bool)):
        return True

    if isinstance(result, (dict, list, tuple)):
        return True

    if isinstance(result, (pd.DataFrame, pd.Series, pd.Index, np.ndarray, np.generic)):
        return False

    if callable(result):
        return False

    if hasattr(result, "read"):
        return False

    return False


def _tool_to_runtime_callable(tool: Any) -> Callable[..., Any]:
    """
    Convert a tool-like object into a plain callable suitable for execution.

    Important:
    - LangChain BaseTool instances may expose formatted wrapper behavior when
      invoked normally.
    - For runtime execution we want the raw Python implementation, typically
      `._run()`.

    Returns:
        A plain callable that executes the underlying tool logic.

    Raises:
        TypeError: If the provided object cannot be converted into a callable.
    """
    if isinstance(tool, BaseTool):
        if not hasattr(tool, "_run"):
            raise TypeError(f"Tool {tool!r} does not expose _run()")
        return tool._run

    if callable(tool):
        return tool

    raise TypeError(f"Unsupported tool type for runtime execution: {type(tool)!r}")


def _wrap_tool(
    tool_func: Callable[..., Any],
    normalize_tool_result: Optional[Callable[[Any], Any]],
) -> Callable[..., Any]:
    """
    Wrap a tool callable for runtime use.

    The wrapper handles:
    - awaitable results
    - asyncio.Task results
    - optional normalization for simple display-friendly values only

    Critical behavior:
    Runtime objects like pandas DataFrames must be returned unchanged so
    generated code can use them directly.

    Note:
        This wrapper is used in the parent process when tools are actually
        executed, including in RPC-served tool mode.
    """

    def wrapper(*args, **kwargs):
        result = tool_func(*args, **kwargs)

        if inspect.isawaitable(result):
            result = _run_coroutine_safely(result)

        if isinstance(result, asyncio.Task):
            if result.done():
                result = result.result()
            else:
                raise RuntimeError("Tool returned an unfinished asyncio.Task")

        if normalize_tool_result is not None and _should_normalize_tool_result(result):
            try:
                return normalize_tool_result(result)
            except Exception as exc:
                raise RuntimeError(
                    f"Tool result normalization failed for {tool_func.__name__}: {exc}"
                ) from exc

        return result

    return wrapper


def _collect_prev_outputs(state: DSState) -> Dict[str, Any]:
    """
    Collect outputs from previous steps in STEPWISE mode.

    By convention, the current step is excluded and only prior completed steps
    contribute to `prev_step_outputs`.
    """
    prev_outputs: Dict[str, Any] = {}
    for i in range(0, max(0, len(state.steps) - 1)):
        step = state.steps[i]
        if hasattr(step, "outputs") and isinstance(step.outputs, dict):
            prev_outputs.update(step.outputs)
    return prev_outputs


def _extract_outputs_from_scope(scope: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the user-visible `outputs` dictionary from an execution scope.

    If `outputs` exists but is not a dict, it is coerced into a diagnostic
    dictionary instead of failing hard.
    """
    outputs = scope.get("outputs", {})
    if not isinstance(outputs, dict):
        return {
            "_note": "outputs was not a dict; coerced.",
            "_value": str(outputs),
        }
    return outputs


def _build_shared_execution_scope(
    state: Optional[DSState],
    tools: Optional[Dict[str, Any]] = None,
    normalize_tool_result: Optional[Callable[[Any], Any]] = None,
    initial_scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the shared namespace used as both globals and locals for exec().

    This function is used to prepare the logical execution scope. In actual
    execution, only the picklable subset of this scope is copied into the child
    process.

    Parameters:
        state:
            Current DS-Star state.
        tools:
            Optional tool mapping. If provided, tools are injected directly into
            the scope as wrapped callables. In the production timeout path we do
            NOT inject tools here because tools are hosted in the parent and
            proxied over IPC.
        normalize_tool_result:
            Optional result normalizer for injected direct-call tools.
        initial_scope:
            Optional extra symbols to merge into the scope.

    Caveat:
        Non-picklable values in the returned scope may later be omitted when the
        scope is serialized into the child process.
    """
    outputs_dict: Dict[str, Any] = {}

    scope: Dict[str, Any] = {
        "__builtins__": get_safe_builtins(),
        "__name__": "__main__",
        "__package__": None,
        **get_safe_scientific_env(),
        **_build_base_env(),
        "outputs": outputs_dict,
    }

    if initial_scope:
        scope.update(initial_scope)

    if state is not None:
        scope["state"] = dict(state)
        if state.code_mode == CodeMode.STEPWISE:
            scope["prev_step_outputs"] = _collect_prev_outputs(state)

    if tools:
        for tool_name, tool in tools.items():
            runtime_tool = _tool_to_runtime_callable(tool)
            scope[tool_name] = _wrap_tool(runtime_tool, normalize_tool_result)

    return scope


def _make_remote_tool_proxy(
    tool_name: str,
    tool_request_queue: multiprocessing.Queue,
    tool_result_queue: multiprocessing.Queue,
) -> Callable[..., Any]:
    """
    Create a child-process proxy callable for a parent-hosted tool.

    Generated code calls this proxy exactly like a normal function. The proxy:
    - serializes the call request
    - sends it to the parent via queue
    - waits for the parent's response via queue
    - returns the serialized result
    - re-raises tool failures as RemoteToolError

    Uses Queues instead of Pipes to avoid blocking with large data transfers.

    Requirement:
        Both the arguments and returned value must be picklable because they
        cross the process boundary.

    Caveat:
        This proxy is blocking. If the parent-side tool hangs, the child waits.
        The subprocess timeout can still kill the child process, but the parent
        tool may need its own timeout/cancellation policy for full robustness.
    """

    def remote_tool(*args, **kwargs):
        import sys
        if not _is_picklable(args):
            raise TypeError(
                f"Arguments for tool '{tool_name}' are not picklable; "
                "cross-process tool calls require picklable args"
            )
        if not _is_picklable(kwargs):
            raise TypeError(
                f"Keyword arguments for tool '{tool_name}' are not picklable; "
                "cross-process tool calls require picklable kwargs"
            )

        try:
            request = {
                "op": "call_tool",
                "tool_name": tool_name,
                "args": args,
                "kwargs": kwargs,
            }
            tool_request_queue.put(request, block=False)
        except Exception as exc:
            raise RemoteToolError(
                f"Failed sending tool request for '{tool_name}': {exc}"
            ) from exc

        try:
            response = tool_result_queue.get(timeout=60)
        except Exception as exc:
            raise RemoteToolError(
                f"Failed receiving tool response for '{tool_name}': {exc}"
            ) from exc

        status = response.get("status")
        if status == "ok":
            serialized_result = response.get("result")
            # Deserialize the result (handles Parquet-serialized DataFrames)
            return _deserialize_tool_result(serialized_result)

        if status == "error":
            error_msg = response.get("error", f"Tool '{tool_name}' failed")
            tb = response.get("traceback")
            if tb:
                raise RemoteToolError(f"{error_msg}\n{tb}")
            raise RemoteToolError(error_msg)

        raise RemoteToolError(f"Invalid tool response for '{tool_name}': {response!r}")

    remote_tool.__name__ = tool_name
    return remote_tool


def _serve_tools_over_connection(
    conn: Any,
    tools: Dict[str, Any],
    normalize_tool_result: Optional[Callable[[Any], Any]],
) -> None:
    """
    Serve tool calls in the parent process over a multiprocessing connection.

    This function runs in a background thread in the parent process while the
    child executes user code.

    Flow:
    - child sends {"op": "call_tool", ...}
    - parent executes the real tool
    - parent sends back {"status": "ok", "result": ...}
      or {"status": "error", ...}

    Important:
        Tools themselves do not cross the process boundary and therefore do not
        need to be picklable.

    Requirement:
        Returned results MUST be picklable so they can be sent back to the child.

    Caveat:
        This loop does not enforce per-tool timeouts. A blocking tool can still
        block the serving thread unless the tool has its own timeout policy.
    """
    wrapped_tools: Dict[str, Callable[..., Any]] = {}
    for name, tool in tools.items():
        runtime_tool = _tool_to_runtime_callable(tool)
        wrapped_tools[name] = _wrap_tool(runtime_tool, normalize_tool_result)

    try:
        while True:
            try:
                request = conn.recv()
            except EOFError:
                break

            if not isinstance(request, dict):
                try:
                    conn.send(
                        {
                            "status": "error",
                            "error": f"Invalid tool request: {type(request)!r}",
                        }
                    )
                except Exception:
                    break
                continue

            op = request.get("op")
            if op == "shutdown":
                try:
                    conn.send({"status": "ok"})
                except Exception:
                    pass
                break

            if op != "call_tool":
                try:
                    conn.send({"status": "error", "error": f"Unknown op: {op!r}"})
                except Exception:
                    break
                continue

            tool_name = request.get("tool_name")
            args = request.get("args", ())
            kwargs = request.get("kwargs", {})

            if tool_name not in wrapped_tools:
                try:
                    conn.send(
                        {"status": "error", "error": f"Tool '{tool_name}' not found"}
                    )
                except Exception:
                    break
                continue

            try:
                result = wrapped_tools[tool_name](*args, **kwargs)
                # Serialize the result (uses Parquet for large DataFrames)
                serialized_result = _serialize_tool_result(result)
                conn.send({"status": "ok", "result": serialized_result})
            except Exception as exc:
                tb = traceback.format_exc(limit=8)
                try:
                    conn.send(
                        {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": tb,
                        }
                    )
                except Exception:
                    break
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _serve_tools_over_queues(
    tool_request_queue: multiprocessing.Queue,
    tool_result_queue: multiprocessing.Queue,
    tools: Dict[str, Any],
    normalize_tool_result: Optional[Callable[[Any], Any]],
) -> None:
    """
    Serve tool calls in the parent process using queues (no Pipe blocking).

    This function runs in a background thread in the parent process while the
    child executes user code.

    Flow:
    - child puts {"op": "call_tool", ...} in tool_request_queue
    - parent executes the real tool
    - parent puts {"status": "ok", "result": ...} in tool_result_queue

    Uses Queues instead of Pipes to avoid blocking with large data transfers.

    Important:
        Tools themselves do not cross the process boundary and therefore do not
        need to be picklable.

    Requirement:
        Returned results MUST be picklable so they can be sent back to the child.

    Caveat:
        This loop does not enforce per-tool timeouts. A blocking tool can still
        block the serving thread unless the tool has its own timeout policy.
    """
    import queue
    
    wrapped_tools: Dict[str, Callable[..., Any]] = {}
    for name, tool in tools.items():
        runtime_tool = _tool_to_runtime_callable(tool)
        wrapped_tools[name] = _wrap_tool(runtime_tool, normalize_tool_result)

    try:
        while True:
            try:
                request = tool_request_queue.get(timeout=1)
                logger.debug(f"Got tool request: {request.get('tool_name', 'unknown')}")
            except queue.Empty:
                continue

            if not isinstance(request, dict):
                tool_result_queue.put({
                    "status": "error",
                    "error": f"Invalid tool request: {type(request)!r}",
                })
                continue

            op = request.get("op")
            if op == "shutdown":
                logger.debug("Tool thread shutting down")
                break

            if op != "call_tool":
                tool_result_queue.put({
                    "status": "error",
                    "error": f"Unknown op: {op!r}",
                })
                continue

            tool_name = request.get("tool_name")
            args = request.get("args", ())
            kwargs = request.get("kwargs", {})

            if tool_name not in wrapped_tools:
                tool_result_queue.put({
                    "status": "error",
                    "error": f"Tool '{tool_name}' not found",
                })
                continue

            try:
                result = wrapped_tools[tool_name](*args, **kwargs)
                # Serialize the result (uses Parquet for large DataFrames)
                serialized_result = _serialize_tool_result(result)
                tool_result_queue.put({"status": "ok", "result": serialized_result})
            except Exception as exc:
                tb = traceback.format_exc(limit=8)
                logger.debug(f"Tool '{tool_name}' failed: {exc}")
                tool_result_queue.put({
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": tb,
                })
    except Exception as e:
        logger.error(f"Tool serving thread error: {e}")



def _execute_code_in_process(
    code: str,
    env_globals_serializable: Dict[str, Any],
    env_locals: Dict[str, Any],
    result_queue: multiprocessing.Queue,
    tool_request_queue: multiprocessing.Queue,
    tool_result_queue: multiprocessing.Queue,
    tool_names: list[str],
    max_memory_bytes: int = 0,
) -> None:
    """
    Execute generated code in the child process and return results by queue.

    Queue payloads:
        Success:
            ("success", logs: str, outputs: dict)

        Error:
            ("error", logs: str, {"_error": str, "_traceback": str, ...})

    Notes:
    - The child only receives the serializable subset of the scope.
    - Tools are not executed here; instead, named proxy callables are injected.
    - stdout/stderr are captured and returned as logs.
    - Uses Queues for tool communication to avoid Pipe blocking.
    """
    # Enforce memory limit on the child process (Linux only).
    if max_memory_bytes > 0 and platform.system() == "Linux":
        try:
            import resource

            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            new_soft = (
                min(max_memory_bytes, hard)
                if hard != resource.RLIM_INFINITY
                else max_memory_bytes
            )
            resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
        except (ValueError, OSError) as exc:
            # Non-fatal: log and continue without memory limit.
            # Can fail if the requested limit is below current usage or
            # if the OS doesn't support RLIMIT_AS.
            logger.warning("Could not set child memory limit: %s", exc)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stderr_buf

    try:
        shared_scope = {
            "__builtins__": get_safe_builtins(),
            "__name__": "__main__",
            "__package__": None,
            **get_safe_scientific_env(),
            **_build_base_env(),
            **env_globals_serializable,
            **env_locals,
        }

        for tool_name in tool_names:
            shared_scope[tool_name] = _make_remote_tool_proxy(
                tool_name, tool_request_queue, tool_result_queue
            )

        if "outputs" not in shared_scope or not isinstance(
            shared_scope["outputs"], dict
        ):
            shared_scope["outputs"] = {}

        exec(code, shared_scope, shared_scope)
        outputs = _extract_outputs_from_scope(shared_scope)
        outputs = _serialize_outputs(outputs)

        logs = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        if err:
            logs += "\n[STDERR]\n" + err

        result_queue.put(("success", logs.strip(), outputs))
    except (Exception, SystemExit) as exc:
        tb = traceback.format_exc(limit=8)
        logs = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        if err:
            logs += "\n[STDERR]\n" + err

        payload: Dict[str, Any] = {
            "_error": str(exc),
            "_traceback": tb,
        }
        if isinstance(exc, SyntaxError):
            payload["_code"] = code

        result_queue.put(("error", logs.strip(), payload))
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def run_code_with_timeout(
    code: str,
    env_globals: dict,
    env_locals: dict,
    seconds: int,
    tools_dict: Optional[Dict[str, Any]] = None,
    normalize_tool_result: Optional[Callable[[Any], Any]] = None,
    max_memory_bytes: int = DEFAULT_MAX_CHILD_MEMORY_BYTES,
) -> tuple[str, Dict[str, Any]]:
    """
    Execute code in a separate process with hard timeout enforcement.

    This is the core execution primitive used by execute_user_code().

    Architecture:
    - child process runs the generated code
    - parent process hosts the real tools
    - a parent background thread serves tool calls over a Pipe
    - timeout is enforced by terminating the child process

    Why this approach:
    - works in main thread and worker threads
    - does not rely on signal.alarm()
    - preserves tool usability

    Parameters:
        code:
            Generated code to execute.
        env_globals, env_locals:
            Shared-scope dicts. Only their picklable contents are copied to the child.
        seconds:
            Hard timeout for child execution.
        tools_dict:
            Mapping of tool names to real parent-hosted tools.
        normalize_tool_result:
            Optional normalizer applied in the parent when tools return simple
            display-oriented values.
        max_memory_bytes:
            Maximum virtual memory for the child process in bytes.
            Set to 0 to disable. Only enforced on Linux.
            Defaults to DEFAULT_MAX_CHILD_MEMORY_BYTES (1 GB).

    Returns:
        (logs, outputs)

    Raises:
        TimeoutException:
            If the child process exceeds the timeout.
        RuntimeError:
            If execution fails, child exits without a result, or serialized
            error information is returned.

    Important caveat:
        Timeout kills the child process only. If a parent-hosted tool blocks,
        the serving thread can still be stuck unless the tool itself has a
        separate timeout/cancellation policy.
    """
    env_globals_serializable = {
        k: v
        for k, v in env_globals.items()
        if k not in ("__builtins__", "call_tool") and _is_picklable(v)
    }
    env_locals_serializable = {k: v for k, v in env_locals.items() if _is_picklable(v)}

    tools_dict = tools_dict or {}

    try:
        ctx = multiprocessing.get_context("spawn")
    except ValueError:
        ctx = multiprocessing.get_context("fork")

    result_queue: multiprocessing.Queue = ctx.Queue()  # type: ignore
    tool_request_queue: multiprocessing.Queue = ctx.Queue()  # type: ignore
    tool_result_queue: multiprocessing.Queue = ctx.Queue()  # type: ignore

    tool_thread = threading.Thread(
        target=_serve_tools_over_queues,
        args=(tool_request_queue, tool_result_queue, tools_dict, normalize_tool_result),
        daemon=True,
    )
    tool_thread.start()

    proc: multiprocessing.Process = ctx.Process(  # type: ignore
        target=_execute_code_in_process,
        args=(
            code,
            env_globals_serializable,
            env_locals_serializable,
            result_queue,
            tool_request_queue,
            tool_result_queue,
            list(tools_dict.keys()),
            max_memory_bytes,
        ),
        daemon=True,
    )

    proc.start()

    # Read the result queue with a timeout rather than proc.join() first.
    # With multiprocessing Queue + spawn context, the child's feeder thread
    # keeps the process alive until the parent drains the Queue. Calling
    # proc.join() before result_queue.get() would always appear to timeout.
    try:
        status, logs, payload = result_queue.get(timeout=seconds)
    except queue.Empty:
        # Timed out waiting for result — kill the child
        if proc.is_alive():
            proc.terminate()
            proc.join(1)
        tool_thread.join(timeout=1)
        raise TimeoutException(f"Code execution exceeded {seconds} seconds")

    # Result received — wait briefly for child to exit, then clean up
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join(1)

    # Signal tool thread to shutdown
    try:
        tool_request_queue.put({"op": "shutdown"}, timeout=1)
    except Exception:
        pass

    tool_thread.join(timeout=1)

    if status == "error":
        error_msg = payload.get("_error", "Execution error")
        tb = payload.get("_traceback", "N/A")
        if payload.get("_code") is not None:
            error_msg += f"\nCode:\n{payload.get('_code')}"
        if len(tb) > 3000:
            tb = tb[:3000] + "\n... (traceback truncated)"
        if len(logs) > 2000:
            logs = logs[:2000] + "\n... (logs truncated)"
        raise RuntimeError(f"{error_msg}\nLogs: {logs}\nTraceback: {tb}")

    outputs = payload
    if not isinstance(outputs, dict):
        raise RuntimeError(
            f"Child process returned non-dict outputs: {type(outputs)!r}"
        )

    outputs = _deserialize_outputs(outputs)
    return logs, outputs


def execute_user_code(
    code: str,
    state: DSState,
    tools: Dict[str, Any],
    timeout: int = 30,
    max_memory_bytes: int = DEFAULT_MAX_CHILD_MEMORY_BYTES,
) -> Tuple[str, Dict[str, Any]]:
    """
    Execute generated user code with safety checks, tool access, and timeout.

    This is the public entry point.

    Behavior:
    1. Remove/reject user-written imports.
    2. Validate the generated code.
    3. Build the logical execution scope.
    4. Execute the code in a dedicated subprocess with hard timeout.
    5. Expose tools to the child through parent-hosted RPC proxies.

    Returns:
        (logs, outputs)
        - logs: captured stdout/stderr text
        - outputs: user-produced outputs dict, or an error payload

    Timeout semantics:
        Timeout enforcement is uniform across main-thread and worker-thread use
        because it does NOT rely on Python signals. The child process is killed
        if it exceeds the configured timeout.

    Tool semantics:
        Tools run in the parent process. Generated code in the child sees normal
        callables, but they are actually RPC proxies. Tool objects therefore do
        NOT need to be picklable. However, tool call args/results must be
        picklable because they cross a process boundary.

    Scope semantics:
        Only the picklable subset of the prepared scope is copied into the child.
        If some injected values are non-picklable, they will not be available in
        the child's scope.

    Caveat:
        A parent-side blocking tool can still hang unless that tool has its own
        timeout/cancellation policy.
    """
    normalize_tool_result = _load_normalize_tool_result()

    try:
        code_without_imports = remove_imports(code)
        assert_no_imports(code_without_imports)
        validate_generated_code(code_without_imports)
        code_to_execute = code_without_imports
    except SyntaxError as exc:
        tb = traceback.format_exc(limit=6)
        logger.error("Syntax error during code validation: %s", exc)
        return "", {"_error": f"SyntaxError: {exc}", "_traceback": tb}

    shared_scope = _build_shared_execution_scope(
        state=state,
        tools=None,  # tools are hosted in parent and proxied over IPC
        normalize_tool_result=None,
    )

    try:
        logger.info("Executing code in subprocess with timeout=%s", timeout)
        logs, outputs = run_code_with_timeout(
            code=code_to_execute,
            env_globals=shared_scope,
            env_locals=shared_scope,
            seconds=timeout,
            tools_dict=tools,
            normalize_tool_result=normalize_tool_result,
            max_memory_bytes=max_memory_bytes,
        )
        logger.info("Code execution finished.")
        return logs.strip(), outputs

    except TimeoutException as exc:
        tb = traceback.format_exc(limit=6)
        logger.error("Code execution timeout: %s", exc)
        return "", {"_error": str(exc), "_traceback": tb}

    except Exception as exc:
        tb = traceback.format_exc(limit=8)
        logger.error("Code execution error: %s", exc)
        return "", {"_error": str(exc), "_traceback": tb}
