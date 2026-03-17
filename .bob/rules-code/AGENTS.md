# Code Mode Rules (Non-Obvious Only)

## No Hidden Defaults (CRITICAL PRINCIPLE)
**All configuration defaults must be explicit in main/entry points, NOT hidden in internal functions.**

This codebase follows a strict principle: configuration values flow from the outside in.
- Defaults are set in `main()` functions or experiment entry points
- Internal functions/classes receive explicit parameters
- No fallback defaults in utility functions or builders

Example:
```python
# CORRECT - defaults in main
def main():
    cache_dir = args.cache_dir or Path("./cache")  # Default here
    experiment = MyExperiment(cache_dir=cache_dir)

# WRONG - hidden default in utility
def configure_cache(cache_dir: Optional[Path] = None):
    if cache_dir is None:
        cache_dir = Path("./.cache")  # Hidden default - BAD!
```

This ensures:
- Configuration is transparent and discoverable
- No surprises from hidden fallbacks
- Easy to understand what values are actually being used


## Critical Import Pattern
- Use `from agents import OpenDsStarAgent` NOT `from src.agents import`
- Use `from tools import VectorStoreTool` NOT `from src.tools import`
- Use `from experiments import ExperimentPipeline` NOT `from src.experiments import`
- Root conftest.py adds src/ to sys.path - all imports assume this

## Recreatable Pattern (CRITICAL - Easy to Miss)
Classes inheriting from `Recreatable` MUST call `self._capture_init_args(locals())` as FIRST line in `__init__`:
```python
class MyClass(Recreatable):
    def __init__(self, arg1: int, arg2: str):
        self._capture_init_args(locals())  # MUST BE FIRST LINE
        self.arg1 = arg1
        # ... rest of init
```
Missing this breaks save/load functionality. This is NOT enforced by linters.

## Agent Type Usage
Use `AgentType` enum and `AgentFactory` from agent_factory:
```python
from experiments.implementations.agent_factory import AgentFactory, AgentType
agent_type = AgentType.DS_STAR  # Correct
# agent_type = "ds_star"  # Backwards compatible but discouraged

# AgentFactory is the recommended generic agent builder
agent_builder = AgentFactory(agent_type=AgentType.DS_STAR, model="watsonx/...")
```

## Code Execution Modes (Performance Impact)
- `code_mode="stepwise"`: Executes steps separately, reuses results (default, efficient)
- `code_mode="full"`: Re-executes entire plan each iteration (wasteful, use only for debugging)
- This is a performance-critical choice, not just a preference

## Test File Location
Tests can be anywhere under tests/ directory - pytest discovers them automatically.
No need to mirror src/ structure exactly.
Only write non-trivial tests, tests that test real functionality.


## Code Style (Enforced by Pre-commit)
- Black formatter (line length 88)
- isort with black profile
- Ruff linter with auto-fix
- Run `pre-commit run --all-files` before committing
