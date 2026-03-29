"""Safe imports for code execution environments.

Provides safe builtins and scientific libraries for sandboxed code execution.
Used by both DS-Star and CodeAct agents for consistency.

Key goals:
- User code cannot arbitrarily import modules (imports are stripped).
- Libraries (e.g., pandas) can perform internal imports needed for normal operation.
- No filesystem/process/network access exposed through imports.
- Plotting must be safe in headless / non-main-thread / subprocess environments.

Matplotlib note
---------------
This module forces a non-interactive matplotlib backend ("Agg") BEFORE importing
`matplotlib.pyplot`.

Why:
- Interactive backends such as `macosx`, `tkagg`, `qt*`, etc. may try to start
  GUI/event-loop behavior.
- In agent execution, worker threads, subprocesses, or headless environments,
  that can block, hang, or emit messages like:
      "Backend macosx is interactive backend. Turning interactive mode on."
- The Agg backend is safe for server/sandbox use and supports saving figures
  without opening windows.

This file intentionally exposes matplotlib/pyplot only in a headless mode.
"""

from __future__ import annotations

import builtins as _builtins
import io
import json
import math
import random
import statistics
import traceback
from collections.abc import Iterable
from typing import Any

import matplotlib

# Force a non-interactive backend before importing pyplot.
# `force=True` ensures the backend is set even if matplotlib has a default
# interactive backend configured in the environment.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import scipy as sp

# Capture the real import once (important if someone patches builtins later)
_REAL_IMPORT = _builtins.__import__

# Limits for potentially-abusive builtins
MAX_RANGE_SIZE = 10_000_000
MAX_EXPONENT = 10_000

# Modules that are allowed to be imported by safe __import__
# (Top-level packages only; submodules are allowed if top-level is allowed)
_ALLOWED_IMPORT_TOP_LEVEL = frozenset(
    {
        # scientific stack
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "plotly",
        # stdlib commonly needed by pandas/numpy internals
        "math",
        "statistics",
        "random",
        "json",
        "datetime",
        "time",
        "calendar",
        "locale",
        "re",
        "collections",
        "itertools",
        "functools",
        "operator",
        "numbers",
        "decimal",
        "fractions",
        "warnings",
        "typing",
        "dataclasses",
        "enum",
        "contextlib",
        "weakref",
        "copy",
        # "pickle" intentionally removed — deserialization is an RCE vector
        "struct",
        "abc",
        "bisect",
        "heapq",
        "array",
        "base64",
        "hashlib",
        "hmac",
        "secrets",
        "uuid",
        "platform",
        "threading",
        "queue",
        "traceback",
        "pprint",
        "textwrap",
        "string",
        "unicodedata",
        "csv",
        "gzip",
        "bz2",
        "lzma",
        "zlib",
        "zoneinfo",  # py>=3.9; harmless if present
        "dateutil",  # pandas depends on python-dateutil
        "pytz",  # sometimes used by pandas installations
    }
)

# Explicit deny list: never allow these even if mistakenly added later
# (filesystem/process/network/import system)
_DENY_IMPORT_TOP_LEVEL = frozenset(
    {
        "os",
        "sys",
        "pathlib",
        "glob",
        "fnmatch",
        "shutil",
        "tempfile",
        "subprocess",
        "multiprocessing",
        "ctypes",
        "signal",
        "resource",
        "socket",
        "ssl",
        "http",
        "urllib",
        "ftplib",
        "telnetlib",
        "webbrowser",
        "importlib",
        "pkgutil",
        "runpy",
        "inspect",
        "types",
        "builtins",
        "pickle",
    }
)


def _safe_import(
    name: str,
    globals=None,
    locals=None,
    fromlist=(),
    level: int = 0,
    *,
    extra_allowed: Iterable[str] | None = None,
):
    """Safe import that allows internal imports only for approved modules.

    Behavior:
    - Allows importing submodules if the top-level package is allowed.
    - Denies filesystem/process/network/import-system related packages.
    - Delegates to the real __import__ for allowed modules.

    Notes:
    - This is intended to support internal imports performed by already-approved
      libraries such as pandas/numpy/matplotlib.
    - User code imports should still be stripped/rejected by the higher-level
      code-generation validation layer.
    """
    if not isinstance(name, str) or not name:
        raise ImportError(f"Invalid import name: {name!r}")

    allowed: set[str] = set(_ALLOWED_IMPORT_TOP_LEVEL)
    if extra_allowed:
        allowed.update(extra_allowed)

    # For absolute import names: "pandas.core..." -> "pandas"
    top_level = name.split(".", 1)[0]

    if top_level in _DENY_IMPORT_TOP_LEVEL:
        raise ImportError(f"Import of '{name}' is denied in sandbox")

    if top_level not in allowed:
        raise ImportError(f"Import of '{name}' is not allowed in sandbox")

    return _REAL_IMPORT(name, globals, locals, fromlist, level)


def safe_range(*args: Any) -> range:
    """A bounded replacement for built-in range().

    Prevents creation of extremely large ranges that could be used to consume
    memory/CPU unexpectedly.
    """
    argc = len(args)
    if argc not in (1, 2, 3):
        raise TypeError("range expected 1-3 arguments")

    if argc == 1:
        start, stop, step = 0, int(args[0]), 1
    elif argc == 2:
        start, stop, step = int(args[0]), int(args[1]), 1
    else:
        start, stop, step = int(args[0]), int(args[1]), int(args[2])
        if step == 0:
            raise ValueError("range step cannot be 0")

    # compute produced length without floats
    if (step > 0 and start >= stop) or (step < 0 and start <= stop):
        n = 0
    else:
        span = (stop - start) if step > 0 else (start - stop)
        step_abs = step if step > 0 else -step
        n = (span + step_abs - 1) // step_abs

    if n > MAX_RANGE_SIZE:
        raise ValueError(f"range size exceeds limit of {MAX_RANGE_SIZE}")

    return range(start, stop, step)


def safe_pow(base: Any, exp: Any, mod: Any = None) -> Any:
    """A bounded replacement for built-in pow().

    Prevents extremely large exponents that could be abused for CPU/memory
    exhaustion.
    """
    exp_int = int(exp)
    if abs(exp_int) > MAX_EXPONENT:
        raise ValueError(f"exponent exceeds limit of {MAX_EXPONENT}")
    if mod is None:
        return pow(base, exp_int)
    return pow(base, exp_int, mod)


def safe_getattr(obj: Any, name: Any, default: Any = None) -> Any:
    """Safe getattr that blocks dunder attribute access.
    """
    s = name if isinstance(name, str) else str(name)
    if s.startswith("__") or s.endswith("__"):
        raise ValueError("dunder attribute access is not allowed")
    try:
        return getattr(obj, s)
    except AttributeError:
        return default


def safe_hasattr(obj: Any, name: Any) -> bool:
    """Safe hasattr that blocks dunder attribute checks.
    """
    s = name if isinstance(name, str) else str(name)
    if s.startswith("__") or s.endswith("__"):
        return False
    return hasattr(obj, s)


def safe_type(*args):
    """Safe replacement for type().

    - 1 arg: type(x) returns the type of x (normal usage).
    - 3 args: type(name, bases, dict) creates a new class dynamically.
      This is blocked to prevent metaclass/dynamic class creation exploits.
    """
    if len(args) == 1:
        return type(args[0])
    raise TypeError(
        "Dynamic class creation with type(name, bases, dict) is not allowed in sandbox"
    )


def _safe_exit(code: Any = 0) -> None:
    """Safe replacement for exit() that raises SystemExit cleanly."""
    raise SystemExit(f"exit() called with code={code}")


def _safe_quit(code: Any = 0) -> None:
    """Safe replacement for quit() that raises SystemExit cleanly."""
    raise SystemExit(f"quit() called with code={code}")


def get_safe_builtins() -> dict[str, Any]:
    """Return a dictionary of safe built-in functions and types.

    Includes a safe __import__ to support internal library imports (e.g., pandas),
    while preventing users from importing arbitrary modules.

    Intentionally omitted:
    - open
    - exec / eval / compile
    - input
    - filesystem / process helpers
    """
    return {
        # --- Critical: allow internal imports safely ---
        "__import__": _safe_import,
        # iteration / selection
        "iter": iter,
        "next": next,
        "enumerate": enumerate,
        "zip": zip,
        "range": safe_range,
        "reversed": reversed,
        # functional-ish helpers
        "map": map,
        "filter": filter,
        "any": any,
        "all": all,
        "sorted": sorted,
        # numerics
        "abs": abs,
        "round": round,
        "sum": sum,
        "min": min,
        "max": max,
        "pow": safe_pow,
        "divmod": divmod,
        # types / containers
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "str": str,
        "list": list,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "dict": dict,
        "len": len,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "type": safe_type,
        "hash": hash,
        "slice": slice,
        # bytes / buffer helpers
        "bytes": bytes,
        "bytearray": bytearray,
        "memoryview": memoryview,
        # string / debug helpers
        "repr": repr,
        "format": format,
        "ascii": ascii,
        "ord": ord,
        "chr": chr,
        "print": print,
        "exit": _safe_exit,
        "quit": _safe_quit,
        # misc common helpers
        "object": object,
        "callable": callable,
        "id": id,
        # errors / exceptions
        "Exception": Exception,
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "NameError": NameError,
        "StopIteration": StopIteration,
        "AssertionError": AssertionError,
        "ZeroDivisionError": ZeroDivisionError,
        "FileNotFoundError": FileNotFoundError,
        # safe attribute access
        "getattr": safe_getattr,
        "hasattr": safe_hasattr,
        # In-memory IO only (do NOT expose io module; do NOT expose open)
        "StringIO": io.StringIO,
        "BytesIO": io.BytesIO,
    }


def get_safe_scientific_env() -> dict[str, Any]:
    """Return pre-imported numerical / plotting libraries.

    Matplotlib is exposed in headless mode only:
    - `matplotlib` is already configured to use backend "Agg"
    - `plt` is therefore safe for non-interactive rendering/saving
    """
    return {
        "math": math,
        "statistics": statistics,
        "random": random,
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        "scipy": sp,
        "json": json,
        "traceback": traceback,
        "plt": plt,
        "matplotlib": matplotlib,
        "plotly": plotly,
        "go": go,
        "px": px,
    }


def get_authorized_imports_list() -> list[str]:
    """Return canonical top-level package names exposed in the scientific environment.

    This is derived from get_safe_scientific_env() so that documentation and
    authorization stay aligned.
    """
    env = get_safe_scientific_env()
    module_names = set()
    for value in env.values():
        if hasattr(value, "__name__"):
            module_names.add(value.__name__.split(".")[0])
    return sorted(module_names)
