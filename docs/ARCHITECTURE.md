# OpenDsStar Architecture

## Overview

This document describes the improved architecture of the OpenDsStar project, focusing on clear separation of concerns between agents, tools, and experiments.

## Directory Structure

```
src/
├── agents/                         # Agent implementations
│   ├── base_agent.py              # Base agent class
│   ├── ds_star/                   # DS-Star agent
│   │   ├── open_ds_star_agent.py  # Main agent class
│   │   ├── ds_star_graph.py       # Graph implementation
│   │   ├── ds_star_state.py       # State definition
│   │   ├── ds_star_execute_env.py # Execution environment
│   │   ├── ds_star_utils.py       # Utilities
│   │   └── nodes/                 # Graph nodes
│   ├── analyzer/                  # Analyzer agent
│   │   ├── analyzer_graph.py      # Graph implementation
│   │   ├── analyzer_state.py      # State definition
│   │   ├── analyzer_execute_env.py # Execution environment
│   │   └── nodes/                 # Graph nodes
│   ├── react_langchain/           # ReAct agent (LangChain)
│   │   └── react_agent_langchain.py
│   ├── react_smolagents/          # ReAct agent (SmoLAgents)
│   │   └── react_agent_smolagents.py
│   ├── codeact_smolagents/        # CodeAct agent (SmoLAgents)
│   │   └── codeact_agent_smolagents.py
│   └── utils/                     # Agent-specific utilities
│
├── ingestion/                      # Document ingestion utilities
│   ├── analyzer.py                # Analyzer-based processor
│   └── docling_analyzer.py        # Docling-based processor
│
├── tools/                          # Shared, reusable tools
│   ├── __init__.py
│   ├── vector_store_tool.py        # Semantic search tool
│   └── analyzer_retriever.py       # Analyzer summary retriever
│
├── experiments/             # Experiment framework
│   ├── core/                       # Core types and configuration
│   │   ├── config.py              # Configuration classes
│   │   ├── context.py             # Pipeline context
│   │   ├── types.py               # Type definitions
│   │   └── enums.py               # Enumerations
│   ├── interfaces/                 # Abstract interfaces
│   │   ├── agent_builder.py       # Agent builder interface
│   │   ├── tool_builder.py        # Tool builder interface
│   │   ├── data_reader.py         # Data reader interface
│   │   ├── evaluator.py           # Evaluator interface
│   │   └── agent_runner.py        # Agent runner interface
│   ├── implementations/            # Concrete implementations
│   │   └── invoke_agent_runner.py # Default agent runner
│   ├── evaluators/                 # Evaluation implementations
│   │   └── unitxt_llm_judge.py    # LLM-as-judge evaluator
│   ├── utils/                      # Utility functions
│   │   ├── cache.py               # Caching utilities
│   │   ├── evaluation_cache.py    # Evaluation caching
│   │   ├── logging.py             # Logging utilities
│   │   └── validation.py          # Validation utilities
│   ├── pipeline.py                 # Main experiment pipeline
│   └── experiments/                # Specific experiments
│       ├── base_experiment.py     # Base experiment class
│       ├── demo/                   # Demo experiment
│       └── hotpotqa/              # HotpotQA experiment
│
└── runner/                         # Simple runner utilities
    └── simple_qa_loop.py          # Interactive QA loop
```

## Architecture Principles

### 1. Separation of Concerns

The architecture maintains clear boundaries between three main layers:

#### **Agents Layer** (`src/agents/`)
- **Responsibility**: Agent implementations and their internal logic
- **Contains**: Agent classes, graph definitions, nodes, and agent-specific utilities
- **Does NOT contain**: Experiment configuration, evaluation logic, or tool definitions

#### **Tools Layer** (`src/tools/`)
- **Responsibility**: Reusable tools that can be used by any agent
- **Contains**: Tool implementations (retrievers, calculators, etc.)
- **Key principle**: Tools are agent-agnostic and experiment-agnostic

#### **Experiments Layer** (`src/experiments/`)
- **Responsibility**: Orchestrating experiments, evaluation, and benchmarking
- **Contains**: Pipeline, interfaces, evaluators, and experiment configurations
- **Does NOT contain**: Agent implementation details

### 2. Dependency Flow

```
Experiments Layer
    ↓ (uses interfaces)
Agents Layer
    ↓ (uses)
Tools Layer
```

- Experiments depend on agent interfaces, not implementations
- Agents use tools but don't own them
- Tools are independent and reusable

### 3. Configuration Hierarchy

The new configuration system provides clear separation:

```python
# Agent-specific configuration
AgentConfig:
    - model
    - temperature
    - max_steps
    - code_timeout
    - code_mode
    - system_prompt
    - task_prompt

# Experiment-specific configuration
ExperimentConfig:
    - run_id
    - fail_fast
    - output_dir
    - cache_dir
    - agent_config  # Nested agent config
    - use_cache
    - log_level

# Tool-specific configuration
ToolConfig:
    - embedding_model
    - chunk_size
    - chunk_overlap
    - top_k
```

## Key Components

### Pipeline (`experiments/pipeline.py`)

The `ExperimentPipeline` orchestrates the complete experiment workflow:

1. **Read Data**: Load corpus and benchmarks
2. **Create Tools**: Build tools from corpus using ToolBuilders
3. **Build Agent**: Create agent with tools using AgentBuilder
4. **Run Agent**: Execute agent on benchmarks using AgentRunner
5. **Evaluate**: Assess results using Evaluators

### Interfaces (`experiments/interfaces/`)

All interfaces follow the dependency inversion principle:

- **AgentBuilder**: Creates agents with tools
- **ToolBuilder**: Creates tools from corpus/benchmarks
- **DataReader**: Loads data for experiments
- **Evaluator**: Evaluates agent outputs
- **AgentRunner**: Executes agents on benchmarks

### Base Experiment (`experiments/base/base_experiment.py`)

Provides a template for creating new experiments:

```python
class MyExperiment(BaseExperiment):
    def get_data_reader(self) -> DataReader:
        # Return data reader implementation

    def get_tools_builder(self) -> Sequence[ToolBuilder]:
        # Return tool builders

    def get_agent_builder(self) -> AgentBuilder:
        # Return agent builder

    def get_evaluators(self) -> Sequence[Evaluator]:
        # Return evaluators
```

## Design Patterns

### 1. Builder Pattern

Used for constructing complex objects (agents, tools):

```python
# Tool Builder
class HotpotQAToolsBuilder(ToolBuilder):
    def build_tools(self, ctx, benchmarks, corpus):
        return [VectorStoreTool(corpus=corpus, ...)]

# Agent Builder
class DemoAgentBuilder(AgentBuilder):
    def build_agent(self, ctx, tools):
        return OpenDsStarAgent(tools=tools, ...)
```

### 2. Strategy Pattern

Used for different evaluation strategies:

```python
class UnitxtLLMJudge(Evaluator):
    def evaluate_one(self, ctx, output, benchmark):
        # LLM-based evaluation logic
```

### 3. Template Method Pattern

Used in `BaseExperiment` to define experiment structure:

```python
class BaseExperiment(ABC):
    def experiment_main(self):
        # Template method defining the workflow
        data_reader = self.get_data_reader()  # Abstract
        tools = self.get_tools_builder()      # Abstract
        agent = self.get_agent_builder()      # Abstract
        evaluators = self.get_evaluators()    # Abstract
        # ... run pipeline
```

## Benefits of This Architecture

### 1. **Modularity**
- Each component has a single, well-defined responsibility
- Components can be developed and tested independently

### 2. **Reusability**
- Tools can be shared across different agents and experiments
- Evaluators can be reused for different benchmarks
- Agent implementations are decoupled from experiments

### 3. **Testability**
- Clear interfaces make mocking easy
- Each layer can be unit tested independently
- Integration tests can focus on specific interactions

### 4. **Extensibility**
- New agents can be added without modifying experiments
- New tools can be added without changing agents
- New experiments can reuse existing components

### 5. **Maintainability**
- Changes to agents don't affect experiments
- Changes to tools don't affect agents
- Clear boundaries reduce coupling

## Migration Guide

### Moving from Old Structure

#### Tools
**Before:**
```python
from src.agents.tools.retrievers import AnalyzerSummaryRetrievalTool
from src.experiments.tools import VectorStoreTool
```

**After:**
```python
from tools import AnalyzerSummaryRetrievalTool, VectorStoreTool
```

#### Configuration
**Before:**
```python
agent = OpenDsStarAgent(
    model="watsonx/mistralai/mistral-medium-2505",
    temperature=0.0,
    max_steps=5,
    ...
)
```

**After:**
```python
from experiments.core.config import AgentConfig

config = AgentConfig(
    model="watsonx/mistralai/mistral-medium-2505",
    temperature=0.0,
    max_steps=5,
)
agent = OpenDsStarAgent(**config.to_dict())
```

## Best Practices

### 1. **Keep Agents Simple**
- Agents should focus on reasoning and execution
- Don't mix tool management with agent logic
- Use configuration objects for parameters

### 2. **Make Tools Reusable**
- Tools should be agent-agnostic
- Avoid hardcoding agent-specific logic in tools
- Use clear, descriptive tool names and descriptions

### 3. **Use Interfaces**
- Always program to interfaces, not implementations
- This allows easy swapping of components
- Makes testing much easier

### 4. **Separate Configuration**
- Keep agent config separate from experiment config
- Use dataclasses for type safety
- Validate configuration early

### 5. **Document Interfaces**
- Clear docstrings for all interface methods
- Include examples in documentation
- Specify expected behavior and contracts

## Future Improvements

1. **Plugin System**: Allow dynamic loading of agents and tools
2. **Configuration Validation**: Add schema validation for configs
3. **Metrics Collection**: Standardized metrics across experiments
4. **Distributed Execution**: Support for parallel experiment runs
5. **Visualization**: Tools for visualizing agent trajectories

## Conclusion

This architecture provides a solid foundation for building, testing, and evaluating AI agents. The clear separation of concerns makes the codebase more maintainable and extensible, while the use of interfaces and configuration objects improves testability and flexibility.
