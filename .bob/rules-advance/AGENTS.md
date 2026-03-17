# Advance Mode Rules (Non-Obvious Only)

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
Use `AgentType` enum from `experiments.implementations.agent_factory`:
```python
from experiments.implementations.agent_factory import AgentType
agent_type = AgentType.DS_STAR  # Correct
# agent_type = "ds_star"  # Backwards compatible but discouraged
```

## Code Execution Modes (Performance Impact)
- `code_mode="stepwise"`: Executes steps separately, reuses results (default, efficient)
- `code_mode="full"`: Re-executes entire plan each iteration (wasteful, use only for debugging)
- This is a performance-critical choice, not just a preference

## MCP and Browser Access
Advance mode has access to MCP tools and browser capabilities not available in Code mode.

## Code Style (Enforced by Pre-commit)
- Black formatter (line length 88)
- isort with black profile
- Ruff linter with auto-fix
- Run `pre-commit run --all-files` before committing
