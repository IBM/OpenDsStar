# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenDsStar is an open-source, tool-centric implementation of the DS-Star agent (a Programmatic Tool Calling agent). It uses LangGraph to orchestrate a plan-code-execute-debug-verify loop. The agent decomposes tasks into tool-call sequences rather than file operations, supporting both stepwise (incremental) and full (re-run-all) execution modes.

## Python Environment

**Always use `.venv/bin/python` or `.venv/bin/pytest`** — never system Python.

```bash
# Install from source
pip install -e .

# Install with dev tools
pip install -e ".[dev]"

# Install with test dependencies (ragbench)
pip install -e ".[test]"
```

## Commands

```bash
# Run all tests
.venv/bin/pytest tests/

# Run a single test file
.venv/bin/pytest tests/agents/test_open_ds_star_agent.py -v

# Run a specific test
.venv/bin/pytest tests/agents/test_open_ds_star_agent.py::TestOpenDsStarAgent::test_initialization -v

# Run by marker
.venv/bin/pytest -m unit
.venv/bin/pytest -m "not slow"
.venv/bin/pytest -m integration
.venv/bin/pytest -m e2e               # Requires API keys

# Run experiments
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main --question-limit 5 --agent-type ds_star --model-agent gpt-4o-mini
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main --question-limit 20 --agent-type ds_star --model gpt-4o-mini
.venv/bin/python -m src.experiments.benchmarks.kramabench.kramabench_main --agent-type ds_star --model-agent gpt-4o-mini

# Rerun from saved params
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main --load-params path/to/params.json

# Linting/formatting (via pre-commit)
.venv/bin/pre-commit run --all-files
```

## Code Style

- **Black** formatter (line length 88)
- **isort** with black profile
- **Ruff** linter with auto-fix
- Pre-commit hooks enforce all rules (trailing whitespace, end-of-file, YAML check)

## Import Pattern (Critical)

Root `conftest.py` adds `src/` to `sys.path`. All imports use short-form paths:

```python
from agents import OpenDsStarAgent          # NOT from src.agents import
from tools import VectorStoreTool           # NOT from src.tools import
from experiments import ExperimentPipeline  # NOT from src.experiments import
```

Module entry points (experiments) are invoked with full dotted paths: `python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main`

## Architecture

### Three-layer structure

```
src/
├── agents/       # Agent implementations (LangGraph-based)
├── tools/        # Shared, agent-agnostic tools
├── experiments/  # Experiment framework (pipeline, interfaces, benchmarks)
├── ingestion/    # Document ingestion (Docling-based)
├── runner/       # Simple interactive QA loop
├── core/         # Shared core utilities
└── ui/           # Streamlit UI
```

Dependencies flow: **Experiments → Agents → Tools**

### DS-Star Agent (LangGraph graph)

The core agent (`src/agents/ds_star/`) is a LangGraph `StateGraph` with these nodes:

- **PlannerNode** — generates a plan as a sequence of tool calls
- **CoderNode** — translates the plan into executable Python code
- **ExecutorNode** — runs the generated code in a sandboxed environment
- **DebuggerNode** — analyzes execution errors and retries (up to `max_debug_tries`)
- **RouterNode** — decides next step: continue planning, debug, or finalize
- **VerifierNode** — validates the output before returning
- **FinalizerNode** — extracts and formats the final answer

The graph state is defined in `ds_star_state.py` (`DSState`). Execution happens in `ds_star_execute_env.py` which provides a sandboxed `exec()` environment with tool access.

### Other agents

- **ReactAgentLangchain** — LangChain ReAct wrapper
- **ReactAgentSmolagents** — Smolagents ReAct
- **CodeActAgentSmolagents** — Smolagents CodeAct

All agents extend `BaseAgent` and implement `invoke(query) -> dict` with at minimum an `"answer"` key.

### Experiments Framework

The `ExperimentPipeline` runs a 5-step workflow:
1. **Read data** (DataReader) — loads corpus and benchmark questions
2. **Create tools** (ToolBuilder) — builds tools from corpus
3. **Build agent** (AgentBuilder) — creates agent with tools
4. **Run agent** (AgentRunner) — executes agent on each benchmark question
5. **Evaluate** (Evaluator) — scores agent outputs

New experiments extend `BaseExperiment` and implement `get_data_reader()`, `get_tools_builder()`, `get_agent_builder()`, `get_evaluators()`.

Existing benchmarks: DataBench, HotpotQA, KramaBench, Demo.

## Key Patterns

### Recreatable Pattern
Classes inheriting from `Recreatable` **must** call `self._capture_init_args(locals())` as the first line of `__init__`. This enables config serialization for experiment reproducibility.

### No Hidden Defaults
Configuration defaults must be explicit in `main()` / entry points, never in internal functions or utility code.

### Code Execution Modes
- `code_mode="stepwise"` (default) — executes each step separately, reuses intermediate results
- `code_mode="full"` — re-executes entire plan from scratch each iteration

### Agent Type Usage
```python
from experiments.implementations.agent_factory import AgentFactory, AgentType
agent_builder = AgentFactory(agent_type=AgentType.DS_STAR, model="gpt-4o-mini")
```

## Experiment Output

Files are saved to each benchmark's `output/` directory:
```
result_<agent_type>_<experiment>[_<limit>]_<timestamp>_output.json
result_<agent_type>_<experiment>[_<limit>]_<timestamp>_params.json
```
Both files share the same timestamp. The params file enables exact reruns.
