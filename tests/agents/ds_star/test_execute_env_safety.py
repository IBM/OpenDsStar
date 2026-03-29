"""Safety and correctness tests for ds_star_execute_env.py."""

import multiprocessing
import threading

import numpy as np
import pandas as pd
import pytest

from agents.ds_star.ds_star_execute_env import (
    _build_base_env,
    _build_shared_execution_scope,
    _collect_prev_outputs,
    _extract_outputs_from_scope,
    _is_picklable,
    _run_coroutine_safely,
    _serve_tools_over_connection,
    _should_normalize_tool_result,
    _tool_to_runtime_callable,
    _wrap_tool,
    execute_user_code,
    run_code_with_timeout,
)
from agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from agents.ds_star.ds_star_utils import CodeValidationError, validate_generated_code


def _make_state(**overrides):
    defaults = dict(user_query="test", tools={}, steps=[], code_mode=CodeMode.STEPWISE)
    defaults.update(overrides)
    return DSState(**defaults)


def _exec(code, tools=None, state=None, timeout=30):
    tools = tools or {}
    state = state or _make_state(tools=tools)
    return execute_user_code(code, state, tools, timeout=timeout)


# ---------------------------------------------------------------------------
# Validation-time blocking (CodeValidationError)
# ---------------------------------------------------------------------------


class TestValidationBlocking:
    """Calls/attrs rejected at AST validation before execution starts."""

    @pytest.mark.parametrize(
        "code",
        [
            "open('/etc/passwd', 'r')",
            "exec('x = 1')",
            "eval('1+1')",
            "compile('1', '<>', 'eval')",
            "__import__('os')",
            "input('>')",
            "globals()",
            "locals()",
            "vars()",
            "breakpoint()",
        ],
    )
    def test_forbidden_calls(self, code):
        with pytest.raises(CodeValidationError):
            validate_generated_code(code)

    @pytest.mark.parametrize(
        "code",
        [
            "os.system('ls')",
            "sys.exit(1)",
            "subprocess.run(['ls'])",
            "pathlib.Path('.')",
            "builtins.open('x')",
            "importlib.import_module('os')",
            "pickle.loads(b'')",
            "threading.Thread(target=print)",
        ],
    )
    def test_forbidden_attr_bases(self, code):
        with pytest.raises(CodeValidationError):
            validate_generated_code(code)

    @pytest.mark.parametrize(
        "code",
        [
            "(1).__class__",
            "int.__subclasses__()",
            "(lambda:0).__globals__",
            "(lambda:0).__code__",
            "object.__dict__",
            "int.__bases__",
        ],
    )
    def test_dunder_attrs_blocked(self, code):
        with pytest.raises(CodeValidationError):
            validate_generated_code(code)

    def test_global_statement_blocked(self):
        with pytest.raises(CodeValidationError):
            validate_generated_code("def f():\n    global x\n    x=1")

    def test_nonlocal_statement_blocked(self):
        with pytest.raises(CodeValidationError):
            validate_generated_code("def o():\n x=1\n def i():\n  nonlocal x\n  x=2")


# ---------------------------------------------------------------------------
# Import stripping
# ---------------------------------------------------------------------------


class TestImportStripping:
    @pytest.mark.parametrize(
        "imp",
        [
            "import os",
            "import subprocess",
            "import shutil",
            "import socket",
            "import ctypes",
            "from os import path",
        ],
    )
    def test_imports_stripped_code_still_runs(self, imp):
        logs, out = _exec(f"{imp}\noutputs['x'] = 42")
        assert out.get("x") == 42

    def test_nested_import_stripped(self):
        code = "def h():\n    import os\n    return 42\noutputs['x'] = h()"
        _, out = _exec(code)
        assert out.get("x") == 42

    def test_multiline_from_import_stripped(self):
        code = "from os import (\n    path,\n    getcwd,\n)\noutputs['x'] = 7"
        _, out = _exec(code)
        assert out.get("x") == 7


# ---------------------------------------------------------------------------
# Filesystem / PurePath
# ---------------------------------------------------------------------------


class TestFilesystemBlocked:
    def test_purepath_no_io_methods(self):
        code = """
p = Path("foo.txt")
outputs['r'] = hasattr(p, 'read_text')
outputs['w'] = hasattr(p, 'write_text')
outputs['o'] = hasattr(p, 'open')
outputs['g'] = hasattr(p, 'glob')
outputs['e'] = hasattr(p, 'exists')
"""
        _, out = _exec(code)
        for k in "rwoge":
            assert out[k] is False

    def test_purepath_algebra(self):
        code = 'p = Path("a")/"b"/"c.txt"\noutputs["n"] = p.name'
        _, out = _exec(code)
        assert out["n"] == "c.txt"


# ---------------------------------------------------------------------------
# Safe builtins
# ---------------------------------------------------------------------------


class TestSafeBuiltins:
    def test_range_ok(self):
        _, out = _exec("outputs['r'] = list(range(5))")
        assert out["r"] == [0, 1, 2, 3, 4]

    def test_range_too_large(self):
        _, out = _exec("outputs['r'] = list(range(100_000_000))")
        assert "_error" in out

    def test_pow_ok(self):
        _, out = _exec("outputs['r'] = pow(2, 10)")
        assert out["r"] == 1024

    def test_pow_too_large(self):
        _, out = _exec("outputs['r'] = pow(2, 100_000)")
        assert "_error" in out

    def test_safe_getattr_blocks_dunder(self):
        code = "try:\n getattr(int,'__subclasses__')\nexcept ValueError:\n outputs['b']=True"
        _, out = _exec(code)
        assert out.get("b") is True

    def test_safe_hasattr_blocks_dunder(self):
        _, out = _exec("outputs['h'] = hasattr(int, '__subclasses__')")
        assert out["h"] is False

    def test_type_single_arg(self):
        _, out = _exec("outputs['t'] = str(type(42))")
        assert "int" in out["t"]

    def test_type_three_arg_blocked(self):
        _, out = _exec('type("X",(object,),{"x":1})\noutputs["c"]=True')
        assert "_error" in out

    def test_stringio(self):
        _, out = _exec("b=StringIO(); b.write('hi'); outputs['v']=b.getvalue()")
        assert out["v"] == "hi"

    def test_bytesio(self):
        _, out = _exec("b=BytesIO(); b.write(b'hi'); outputs['v']=b.getvalue()")
        assert out["v"] == b"hi"


# ---------------------------------------------------------------------------
# exit / quit
# ---------------------------------------------------------------------------


class TestExitQuit:
    def test_exit(self):
        _, out = _exec("exit(0)")
        assert "_error" in out and "exit()" in out["_error"]

    def test_quit(self):
        _, out = _exec("quit(1)")
        assert "_error" in out and "quit()" in out["_error"]


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_infinite_loop(self):
        _, out = _exec("while True: pass", timeout=2)
        assert "_error" in out

    def test_fast_code(self):
        _, out = _exec("outputs['x']=42", timeout=30)
        assert out.get("x") == 42


# ---------------------------------------------------------------------------
# Stdout/stderr
# ---------------------------------------------------------------------------


class TestOutputCapture:
    def test_print(self):
        logs, out = _exec("print('hello')\noutputs['d']=True")
        assert "hello" in logs and out.get("d") is True


# ---------------------------------------------------------------------------
# _extract_outputs_from_scope
# ---------------------------------------------------------------------------


class TestExtractOutputs:
    def test_normal(self):
        assert _extract_outputs_from_scope({"outputs": {"a": 1}}) == {"a": 1}

    def test_missing(self):
        assert _extract_outputs_from_scope({}) == {}

    def test_not_dict(self):
        r = _extract_outputs_from_scope({"outputs": "bad"})
        assert "_note" in r


# ---------------------------------------------------------------------------
# _is_picklable / _should_normalize_tool_result
# ---------------------------------------------------------------------------


class TestPicklableAndNormalize:
    def test_basic_picklable(self):
        for v in [1, "s", 3.14, None, [1], {"a": 1}]:
            assert _is_picklable(v)

    def test_lock_not_picklable(self):
        assert _is_picklable(threading.Lock()) is False

    def test_normalize_scalars(self):
        for v in [None, "s", 42, True, {"a": 1}, [1]]:
            assert _should_normalize_tool_result(v) is True

    def test_no_normalize_runtime(self):
        assert _should_normalize_tool_result(pd.DataFrame()) is False
        assert _should_normalize_tool_result(np.array([1])) is False
        assert _should_normalize_tool_result(lambda x: x) is False


# ---------------------------------------------------------------------------
# _tool_to_runtime_callable / _wrap_tool
# ---------------------------------------------------------------------------


class TestToolConversion:
    def test_plain_function(self):
        def f(x):
            return x

        assert _tool_to_runtime_callable(f) is f

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            _tool_to_runtime_callable(42)

    def test_wrap_sync(self):
        def tool_func():
            return 10

        assert _wrap_tool(tool_func, None)() == 10

    def test_wrap_normalizer_applied(self):
        def tool_func():
            return "v"

        def normalizer(v):
            return f"n:{v}"

        assert _wrap_tool(tool_func, normalizer)() == "n:v"

    def test_wrap_normalizer_skipped_df(self):
        df = pd.DataFrame({"a": [1]})

        def tool_func():
            return df

        def normalizer(v):
            return "bad"

        assert isinstance(_wrap_tool(tool_func, normalizer)(), pd.DataFrame)

    def test_wrap_normalizer_error(self):
        def bad(v):
            raise ValueError("boom")

        with pytest.raises(RuntimeError, match="normalization failed"):
            _wrap_tool(lambda: "v", bad)()


# ---------------------------------------------------------------------------
# _collect_prev_outputs
# ---------------------------------------------------------------------------


class TestCollectPrevOutputs:
    def test_no_steps(self):
        assert _collect_prev_outputs(_make_state()) == {}

    def test_single_step(self):
        assert (
            _collect_prev_outputs(
                _make_state(steps=[DSStep(plan="s0", outputs={"a": 1})])
            )
            == {}
        )

    def test_aggregates(self):
        s = _make_state(
            steps=[
                DSStep(plan="s0", outputs={"a": 1}),
                DSStep(plan="s1", outputs={"b": 2}),
                DSStep(plan="s2"),
            ]
        )
        assert _collect_prev_outputs(s) == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# _build_shared_execution_scope
# ---------------------------------------------------------------------------


class TestBuildScope:
    def test_basics(self):
        s = _build_shared_execution_scope(state=None)
        assert "__builtins__" in s and "np" in s and isinstance(s["outputs"], dict)

    def test_stepwise_prev(self):
        st = _make_state(
            code_mode=CodeMode.STEPWISE,
            steps=[DSStep(plan="s0", outputs={"d": 1}), DSStep(plan="s1")],
        )
        s = _build_shared_execution_scope(state=st)
        assert s["prev_step_outputs"]["d"] == 1

    def test_full_no_prev(self):
        st = _make_state(
            code_mode=CodeMode.FULL,
            steps=[DSStep(plan="s0", outputs={"d": 1}), DSStep(plan="s1")],
        )
        s = _build_shared_execution_scope(state=st)
        assert "prev_step_outputs" not in s


# ---------------------------------------------------------------------------
# Tool RPC through subprocess
# ---------------------------------------------------------------------------


class TestToolRPC:
    def test_simple_call(self):
        log = []

        def search(query):
            log.append(query)
            return [{"path": "/f.csv"}]

        _, out = _exec('outputs["r"]=search(query="hi")', tools={"search": search})
        assert log == ["hi"] and out["r"] == [{"path": "/f.csv"}]

    def test_tool_returning_df(self):
        _, out = _exec(
            'df=get_df(); outputs["s"]=int(df["c"].sum())',
            tools={"get_df": lambda: pd.DataFrame({"c": [1, 2, 3]})},
        )
        assert out["s"] == 6

    def test_tool_error(self):
        def bad():
            raise ValueError("boom")

        _, out = _exec("bad()", tools={"bad": bad})
        assert "_error" in out and "boom" in out["_error"]

    def test_chained_tools(self):
        _, out = _exec(
            "r=a(x=5); outputs['r']=b(x=r)",
            tools={"a": lambda x: x * 2, "b": lambda x: x + 10},
        )
        assert out["r"] == 20


# ---------------------------------------------------------------------------
# _serve_tools_over_connection edge cases
# ---------------------------------------------------------------------------


class TestServeTools:
    def _pipe(self):
        return multiprocessing.Pipe(duplex=True)

    def _run_server(self, parent_conn, tools):
        t = threading.Thread(
            target=_serve_tools_over_connection,
            args=(parent_conn, tools, None),
            daemon=True,
        )
        t.start()
        return t

    def test_unknown_tool(self):
        p, c = self._pipe()
        t = self._run_server(p, {})
        c.send({"op": "call_tool", "tool_name": "nope", "args": (), "kwargs": {}})
        assert "not found" in c.recv()["error"]
        c.send({"op": "shutdown"})
        c.recv()
        t.join(2)

    def test_invalid_request(self):
        p, c = self._pipe()
        t = self._run_server(p, {})
        c.send("not a dict")
        assert "Invalid" in c.recv()["error"]
        c.send({"op": "shutdown"})
        c.recv()
        t.join(2)

    def test_unknown_op(self):
        p, c = self._pipe()
        t = self._run_server(p, {})
        c.send({"op": "bad_op"})
        assert "Unknown op" in c.recv()["error"]
        c.send({"op": "shutdown"})
        c.recv()
        t.join(2)

    def test_eof_stops(self):
        p, c = self._pipe()
        t = self._run_server(p, {})
        c.close()
        t.join(2)
        assert not t.is_alive()


# ---------------------------------------------------------------------------
# Async tool handling
# ---------------------------------------------------------------------------


class TestAsync:
    def test_basic_coro(self):
        async def f():
            return 42

        assert _run_coroutine_safely(f()) == 42

    def test_coro_exception(self):
        async def f():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _run_coroutine_safely(f())


# ---------------------------------------------------------------------------
# Runtime errors
# ---------------------------------------------------------------------------


class TestRuntimeErrors:
    def test_division_by_zero(self):
        _, out = _exec("1/0")
        assert "_error" in out and "_traceback" in out

    def test_name_error(self):
        _, out = _exec("outputs['x']=undefined_var")
        assert "_error" in out

    def test_recursion(self):
        _, out = _exec("def r(): return r()\nr()")
        assert "_error" in out


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------


class TestSyntaxErrors:
    def test_unclosed_paren(self):
        _, out = _exec("outputs['x'] = (1 + 2")
        assert "_error" in out and "SyntaxError" in out["_error"]

    def test_missing_colon(self):
        _, out = _exec("if True\n    outputs['x'] = 1")
        assert "_error" in out and "SyntaxError" in out["_error"]


# ---------------------------------------------------------------------------
# Preloaded libraries
# ---------------------------------------------------------------------------


class TestPreloaded:
    @pytest.mark.parametrize(
        "code,key,expected",
        [
            ("outputs['r']=list(np.arange(3))", "r", [0, 1, 2]),
            ("outputs['r']=int(pd.DataFrame({'a':[1,2,3]})['a'].sum())", "r", 6),
            ("outputs['r']=round(math.pi,2)", "r", 3.14),
            ("outputs['r']=statistics.mean([1,2,3,4,5])", "r", 3),
            ("outputs['r']=json.loads(json.dumps({'a':1}))", "r", {"a": 1}),
        ],
    )
    def test_library(self, code, key, expected):
        _, out = _exec(code)
        assert out[key] == expected


# ---------------------------------------------------------------------------
# Complex code patterns
# ---------------------------------------------------------------------------


class TestCodePatterns:
    def test_list_comprehension(self):
        _, out = _exec("outputs['r']=[x**2 for x in range(5)]")
        assert out["r"] == [0, 1, 4, 9, 16]

    def test_nested_functions(self):
        _, out = _exec(
            "def o(x):\n def i(y): return x+y\n return i(10)\noutputs['r']=o(5)"
        )
        assert out["r"] == 15

    def test_try_except(self):
        _, out = _exec("try:\n 1/0\nexcept ZeroDivisionError:\n x=-1\noutputs['r']=x")
        assert out["r"] == -1

    def test_pandas_groupby(self):
        code = """
df=pd.DataFrame({'n':['A','B','A'],'s':[10,20,30]})
outputs['r']=float(df.groupby('n')['s'].mean()['A'])
"""
        _, out = _exec(code)
        assert out["r"] == 20.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_code(self):
        _, out = _exec("")
        assert "_error" not in out

    def test_only_imports(self):
        _, out = _exec("import os\nimport sys")
        assert "_error" not in out

    def test_overwrite_outputs(self):
        _, out = _exec("outputs['x']=1\noutputs['x']=2")
        assert out["x"] == 2

    def test_non_picklable_scope_doesnt_crash(self):
        scope = _build_shared_execution_scope(
            state=None, initial_scope={"lock": threading.Lock()}
        )
        logs, out = run_code_with_timeout("outputs['v']=42", scope, scope, seconds=30)
        assert out.get("v") == 42


# ---------------------------------------------------------------------------
# Memory limit
# ---------------------------------------------------------------------------


class TestMemoryLimit:
    @pytest.mark.skipif(
        __import__("platform").system() != "Linux",
        reason="RLIMIT_AS is only enforced on Linux",
    )
    def test_large_allocation_fails(self):
        """Allocating far more than MAX_CHILD_MEMORY_BYTES should raise MemoryError."""
        # Try to allocate 4 GB — well above the 1 GB limit
        code = "x = bytearray(4 * 1024 * 1024 * 1024)\noutputs['done'] = True"
        _, out = _exec(code, timeout=30)
        assert "_error" in out
        assert "done" not in out


class TestBuildBaseEnv:
    def test_path_is_purepath(self):
        from pathlib import PurePath

        assert _build_base_env()["Path"] is PurePath
