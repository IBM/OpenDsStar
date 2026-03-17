import types

from agents.utils.safe_imports import (
    get_authorized_imports_list,
    get_safe_builtins,
    get_safe_scientific_env,
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
