# Plan Mode Rules (Non-Obvious Only)

## Architecture Overview (Non-Standard Patterns)
- Agents, tools, and experiments are strictly separated layers
- Tools are agent-agnostic and experiment-agnostic (in `src/tools/`)
- Experiments depend on agent interfaces, not implementations
- Root conftest.py modifies sys.path - all imports assume src/ is in path

## Import Pattern (Critical)
- Use `from agents import OpenDsStarAgent` NOT `from src.agents import`
- Use `from tools import VectorStoreTool` NOT `from src.tools import`
- Use `from experiments import ExperimentPipeline` NOT `from src.experiments import`

## Execution Modes (Architectural Decision)
- `code_mode="stepwise"`: Incremental execution, reuses intermediate results (default)
- `code_mode="full"`: Re-executes entire plan each iteration (original DS-Star behavior)
- Stepwise mode is a deliberate enhancement over original DS-Star for efficiency

## Recreatable Pattern (Hidden Coupling)
Classes inheriting from `Recreatable` have hidden dependency on `_capture_init_args(locals())` call.
Must be first line in `__init__` but not enforced by type system or linters.
Breaking this breaks experiment reproducibility (save/load functionality).

## Experiment Pipeline Flow
1. DataReader loads corpus and benchmarks
2. ToolBuilders create tools from corpus
3. AgentBuilder creates agent with tools
4. AgentRunner executes agent on benchmarks
5. Evaluators assess results
Each step is interface-driven for modularity.

## Agent Factory Pattern
- Uses `AgentType` enum from `experiments.implementations.agent_factory`
- String values work for backwards compatibility but discouraged
- Factory decouples experiment code from agent implementations

## Configuration Hierarchy (Non-Obvious)
- `AgentConfig`: Agent-specific (model, temperature, max_steps, code_mode)
- `ExperimentConfig`: Experiment-level (run_id, fail_fast, output_dir, cache_dir)
- `PipelineConfig`: Pipeline execution (use_cache, log_level)
- Nested structure: ExperimentConfig contains AgentConfig
