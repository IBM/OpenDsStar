# AGENTS.md

This file provides guidance to agents when working with code in this repository.

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

## Python Environment (CRITICAL)
**ALWAYS use Python from the `.venv` virtual environment, NEVER use system/global Python.**

All Python commands must use the virtual environment:
```bash
# Correct - uses .venv Python
.venv/bin/python -m pytest tests/
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main

# WRONG - uses system Python
python -m pytest tests/
python3 -m src.experiments.benchmarks.hotpotqa.hotpotqa_main
```

When executing commands, always prefix with `.venv/bin/python` or activate the venv first:
```bash
source .venv/bin/activate && python -m pytest tests/
```

## Critical Import Pattern
- Use `from agents import OpenDsStarAgent` NOT `from src.agents import`
- Use `from tools import VectorStoreTool` NOT `from src.tools import`
- Use `from experiments import ExperimentPipeline` NOT `from src.experiments import`
- Root conftest.py adds src/ to sys.path - all imports assume this

## Recreatable Pattern (Non-Obvious Requirement)
Classes inheriting from `Recreatable` MUST call `self._capture_init_args(locals())` as the FIRST line in `__init__`:
```python
class MyClass(Recreatable):
    def __init__(self, arg1: int, arg2: str):
        self._capture_init_args(locals())  # MUST BE FIRST LINE
        self.arg1 = arg1
        # ... rest of init
```
This captures constructor args for config serialization. Missing this breaks save/load functionality.

## Test Commands
```bash
# Run single test file
.venv/bin/pytest tests/agents/test_open_ds_star_agent.py -v

# Run specific test
.venv/bin/pytest tests/agents/test_open_ds_star_agent.py::TestOpenDsStarAgent::test_initialization -v

# Run by marker
.venv/bin/pytest -m unit              # Unit tests only
.venv/bin/pytest -m "not slow"        # Skip slow tests
.venv/bin/pytest -m integration       # Integration tests
.venv/bin/pytest -m e2e               # End-to-end (requires API keys)
```

## Code Execution Modes (Performance Critical)
- `code_mode="stepwise"`: Executes each step separately, reuses intermediate results (efficient for expensive operations)
- `code_mode="full"`: Re-executes entire plan from scratch each iteration (matches original DS-Star but wasteful)
- Default is "stepwise" - only use "full" for debugging or exact DS-Star replication

## Agent Type Usage
Use `AgentType` enum and `AgentFactory` from agent_factory:
```python
from experiments.implementations.agent_factory import AgentFactory, AgentType
agent_type = AgentType.DS_STAR  # Correct
# agent_type = "ds_star"  # Backwards compatible but discouraged

# AgentFactory is the recommended generic agent builder
agent_builder = AgentFactory(agent_type=AgentType.DS_STAR, model="watsonx/...")
```

## Experiment Output Files
Output and params files share the same timestamp for traceability:
- `result_<agent_type>_<experiment>[_<limit>]_<timestamp>_output.json`
- `result_<agent_type>_<experiment>[_<limit>]_<timestamp>_params.json`
Example: `result_ds_star_hotpotqa_20_20260127_161619_output.json`

## Running Experiments
```bash
# HotpotQA with DS-Star
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main --agent-type ds_star --max-steps 5

# KramaBench
.venv/bin/python -m src.experiments.benchmarks.kramabench.kramabench_main

# Rerun from saved params
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main --load-params path/to/params.json
```

## Code Style
- Black formatter (line length 88)
- isort with black profile
- Ruff linter with auto-fix
- Pre-commit hooks enforce all style rules

## Python Version
Requires Python >=3.11.8, <3.13 (specified in pyproject.toml)
