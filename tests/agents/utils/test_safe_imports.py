import types

import pytest

from agents.utils.safe_imports import (
    _ALLOWED_IMPORT_TOP_LEVEL,
    _DENY_IMPORT_TOP_LEVEL,
    _safe_exit,
    _safe_import,
    _safe_quit,
    get_authorized_imports_list,
    get_safe_builtins,
    get_safe_scientific_env,
    safe_type,
)


def _canonical_importable_modules_from_env(env: dict) -> set[str]:
    """
    Convert an env dict (like get_safe_scientific_env()) into the canonical set of
    importable module/package names:
      - include only actual modules
      - canonicalize to top-level package (e.g., "scipy.stats" -> "scipy")
    """
    out: set[str] = set()
    for v in env.values():
        if isinstance(v, types.ModuleType):
            out.add(v.__name__.split(".")[0])
    return out


def test_authorized_imports_identical_to_exposed_modules():
    # What the execution environment exposes
    builtins_env = get_safe_builtins()
    scientific_env = get_safe_scientific_env()

    # Sanity: builtins should not be modules (mostly), but ensure we don't accidentally
    # leak something module-like there in the future.
    builtin_modules = _canonical_importable_modules_from_env(builtins_env)
    assert (
        builtin_modules == set()
    ), f"Unexpected module objects in builtins: {builtin_modules}"

    exposed_modules = _canonical_importable_modules_from_env(scientific_env)

    # What CodeAct is allowed to import
    authorized = set(get_authorized_imports_list())

    assert authorized == exposed_modules, (
        "Authorized import list must match the canonical set of module/package names "
        "exposed by get_safe_scientific_env().\n"
        f"authorized={sorted(authorized)}\n"
        f"exposed={sorted(exposed_modules)}\n"
        f"missing={sorted(exposed_modules - authorized)}\n"
        f"extra={sorted(authorized - exposed_modules)}"
    )


def test_authorized_imports_are_importable():
    """
    Optional: ensure each authorized module is actually importable in the runtime.
    If your CI intentionally runs without some deps, mark this xfail/skip accordingly.
    """
    for name in get_authorized_imports_list():
        __import__(name)


# --- pickle deny tests ---


class TestPickleDenied:
    def test_pickle_not_in_allowed(self):
        assert "pickle" not in _ALLOWED_IMPORT_TOP_LEVEL

    def test_pickle_in_deny_list(self):
        assert "pickle" in _DENY_IMPORT_TOP_LEVEL

    def test_safe_import_blocks_pickle(self):
        with pytest.raises(ImportError, match="denied"):
            _safe_import("pickle")

    def test_safe_import_blocks_pickle_submodule(self):
        with pytest.raises(ImportError, match="denied"):
            _safe_import("pickle.compat")


# --- safe_type tests ---


class TestSafeType:
    def test_single_arg_returns_type(self):
        assert safe_type(42) is int
        assert safe_type("hello") is str
        assert safe_type([1, 2]) is list

    def test_three_arg_blocked(self):
        with pytest.raises(TypeError, match="Dynamic class creation"):
            safe_type("MyClass", (object,), {})

    def test_two_arg_blocked(self):
        with pytest.raises(TypeError, match="Dynamic class creation"):
            safe_type("MyClass", (object,))

    def test_builtins_use_safe_type(self):
        builtins = get_safe_builtins()
        assert builtins["type"] is safe_type


# --- safe exit/quit tests ---


class TestSafeExitQuit:
    def test_safe_exit_raises_system_exit(self):
        with pytest.raises(SystemExit, match="exit\\(\\) called"):
            _safe_exit()

    def test_safe_exit_with_code(self):
        with pytest.raises(SystemExit, match="code=1"):
            _safe_exit(1)

    def test_safe_quit_raises_system_exit(self):
        with pytest.raises(SystemExit, match="quit\\(\\) called"):
            _safe_quit()

    def test_builtins_use_safe_versions(self):
        builtins = get_safe_builtins()
        assert builtins["exit"] is _safe_exit
        assert builtins["quit"] is _safe_quit
