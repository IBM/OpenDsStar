# Ask Mode Rules (Non-Obvious Only)

## Project Structure Context
- `src/agents/` contains agent implementations (DS-Star, ReAct, CodeAct)
- `src/tools/` contains reusable tools (vector store, retrievers)
- `src/experiments/` contains experiment framework and benchmarks
- Root conftest.py adds src/ to sys.path - affects all imports

## Import Pattern (Non-Standard)
- Use `from agents import OpenDsStarAgent` NOT `from src.agents import`
- Use `from tools import VectorStoreTool` NOT `from src.tools import`
- Use `from experiments import ExperimentPipeline` NOT `from src.experiments import`
- This is due to sys.path modification in root conftest.py

## Experiment Output Files (Traceability)
Output and params files share the same timestamp:
- `result_<agent_type>_<experiment>[_<limit>]_<timestamp>_output.json`
- `result_<agent_type>_<experiment>[_<limit>]_<timestamp>_params.json`
Example: `result_ds_star_hotpotqa_20_20260127_161619_output.json`

## Code Execution Modes (Performance Context)
- `code_mode="stepwise"`: Executes steps separately, reuses intermediate results (efficient)
- `code_mode="full"`: Re-executes entire plan each iteration (wasteful but matches original DS-Star)
- Default is "stepwise" for performance reasons

## Agent Types (Enum vs String)
- Use `AgentType.DS_STAR` from `experiments.implementations.agent_factory`
- String values like "ds_star" work but are discouraged
- Available: DS_STAR, REACT_LANGCHAIN, CODEACT_SMOLAGENTS, REACT_SMOLAGENTS

## Recreatable Pattern (Hidden Requirement)
Classes inheriting from `Recreatable` must call `self._capture_init_args(locals())` as first line in `__init__`.
This is required for config serialization but not enforced by linters.

## Test Structure
- Tests use markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.slow`
- Run specific markers: `pytest -m unit` or `pytest -m "not slow"`
- Tests can be anywhere under tests/ - pytest discovers them automatically
