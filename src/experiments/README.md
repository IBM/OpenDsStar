# Experiments Framework

A modular, interface-driven framework for running agent experiments over benchmarks.

> **CRITICAL**: Always use `.venv/bin/python` for all Python commands, NOT system Python.

## Overview

The experiments framework provides a clean 5-step pipeline for evaluating agents:

1. **Read benchmark data** - Load dataset and corpus
2. **Create tools** - Build tools from the corpus (e.g., retrievers)
3. **Build agent** - Create agent with tools
4. **Run agent** - Execute agent over benchmark
5. **Evaluate** - Assess agent's output against ground truth

## Key Principles

### No Hidden Defaults
**All configuration defaults must be explicit in main/entry points, NOT hidden in internal functions.**

Configuration values flow from the outside in:
- Defaults are set in `main()` functions or experiment entry points
- Internal functions/classes receive explicit parameters
- No fallback defaults in utility functions or builders

### Recreatable Pattern
Classes inheriting from `Recreatable` MUST call `self._capture_init_args(locals())` as the **FIRST line** in `__init__`:

```python
class MyExperiment(BaseExperiment):
    def __init__(self, split: str, model: str):
        self._capture_init_args(locals())  # MUST BE FIRST LINE!
        super().__init__(split=split, model=model)
```

This enables complete reproducibility via params files.

## Architecture

### Core Components

```
src/experiments/
├── README.md                      # This file
├── pipeline.py                    # Main orchestrator
├── base/                          # Base experiment classes
│   └── base_experiment.py         # BaseExperiment class
├── core/                          # Core data types
│   ├── types.py                   # RawDocument, RawBenchmarkEntry, etc.
│   ├── enums.py                   # Enums (currently empty)
│   └── context.py                 # PipelineContext, PipelineConfig
├── interfaces/                    # All interfaces (ABC)
│   ├── data_reader.py             # Step 1: Read benchmarks
│   ├── tool_builder.py            # Step 2: Create tools
│   ├── agent_builder.py           # Step 3: Build agent
│   ├── agent_runner.py            # Step 4: Run agent
│   └── evaluator.py               # Step 5: Evaluate
├── implementations/               # Reusable implementations
│   ├── agent_factory.py
│   ├── ds_star_agent_builder.py
│   ├── flexible_agent_builder.py
│   └── invoke_agent_runner.py
├── evaluators/                    # Evaluator implementations
│   └── unitxt_llm_judge.py
├── utils/                         # Utilities
│   ├── logging.py                 # Logger, StageTimer
│   ├── cache.py                   # FileCache for persistent caching
│   ├── validation.py              # Validation helpers
│   └── tool_registry.py           # ToolRegistry
└── benchmarks/                    # Benchmark implementations
    ├── demo/                      # Demo benchmark
    └── hotpotqa/                  # HotpotQA benchmark
        ├── data_reader.py
        ├── tools_builder.py
        ├── hotpotqa_main.py
        ├── output/                # Experiment results
        └── cache/                 # Experiment cache
```

## Creating a New Benchmark

Each benchmark is self-contained in its own directory under `benchmarks/`. To create a new benchmark (e.g., for dataset "hotpotqa"):

### 1. Create Experiment Directory Structure

```bash
mkdir -p src/experiments/benchmarks/hotpotqa/{output,cache}
```

### 2. Implement Required Classes

Each experiment must implement 5 classes:

#### **data_reader.py** - Implements `BenchmarkReader`

Loads the dataset and corpus using `RagDataLoaderFactory`:

```python
from typing import Sequence, Tuple
from rag_unitxt_cards.data_loaders.rag_data_loader_factory import RagDataLoaderFactory
from ...interfaces.benchmark_reader import BenchmarkReader
from ...core.types import RawBenchmarkEntry
from ...core.context import PipelineContext

class HotpotQADataReader(BenchmarkReader):
    """Reads HotpotQA dataset."""

    def __init__(self, split: str = "test"):
        self.split = split
        self.corpus = None
        self.benchmark = None

    def read(self, ctx: PipelineContext) -> Sequence[RawBenchmarkEntry]:
        """Load HotpotQA data and return benchmarks."""
        data_loader = RagDataLoaderFactory.create(
            dataset_name="hotpotqa",
            split=self.split
        )
        self.corpus = data_loader.get_corpus()
        self.benchmark = data_loader.get_benchmark()

        # Convert to RawBenchmarkEntry format
        return self._convert_to_raw_benchmarks(self.benchmark)

    def get_corpus(self):
        """Return the loaded corpus for tool building."""
        return self.corpus
```

#### **tools_builder.py** - Implements `ToolBuilder`

Builds tools (e.g., retrievers) from the corpus:

```python
from typing import Sequence
from ...interfaces.tool_builder import ToolBuilder
from langchain_core.tools import BaseTool
from ...core.types import RawBenchmarkEntry
from ...core.context import PipelineContext

class HotpotQAToolsBuilder(ToolBuilder):
    """Builds retrieval tools from HotpotQA corpus."""

    def __init__(self, corpus):
        self.corpus = corpus

    @property
    def name(self) -> str:
        return "hotpotqa_tools"

    def build_tools(
        self,
        ctx: PipelineContext,
        benchmarks: Sequence[RawBenchmarkEntry],
    ) -> Sequence[BaseTool]:
        """Build retriever tools from corpus."""
        # Create retriever tool from corpus
        retriever_tool = self._create_retriever_tool(self.corpus)
        return [retriever_tool]
```

#### **agent_builder.py** - Implements `AgentBuilder`

Creates the agent (e.g., DsStarAgent) with tools:

```python
from typing import Any, List
from ...interfaces.agent_builder import AgentBuilder
from langchain_core.tools import BaseTool
from ...core.context import PipelineContext
from src.agents.open_ds_star_agent import OpenDsStarAgent

class HotpotQAAgentBuilder(AgentBuilder):
    """Builds DsStarAgent for HotpotQA."""

    def build_agent(
        self,
        ctx: PipelineContext,
        tools: List[BaseTool],
    ) -> Any:
        """Build and configure DsStarAgent."""
        return OpenDsStarAgent(
            tools=tools,
            # Add any HotpotQA-specific configuration
        )
```

#### **evaluators_builder.py** - Factory for evaluators

Configures evaluators using pre-implemented ones:

```python
from typing import List
from ...interfaces.evaluator import Evaluator
from ...implementations.evaluators import TextExactEvaluator

class HotpotQAEvaluatorsBuilder:
    """Configures evaluators for HotpotQA."""

    @staticmethod
    def build_evaluators() -> List[Evaluator]:
        """Return list of evaluators for HotpotQA."""
        return [
            TextExactEvaluator(),  # HotpotQA uses text answers
        ]
```

#### **experiment_main.py** - Orchestrates the experiment

Main entry point that ties everything together:

```python
from pathlib import Path
from ...pipeline import ExperimentPipeline
from ...core.context import PipelineContext, PipelineConfig
from ...utils.logging import StdoutLogger
from ...implementations.agent_runner import SimpleAgentRunner
from .data_reader import HotpotQADataReader
from .tools_builder import HotpotQAToolsBuilder
from .agent_builder import HotpotQAAgentBuilder
from .evaluators_builder import HotpotQAEvaluatorsBuilder

def run_hotpotqa_experiment():
    """Main entry point for HotpotQA experiment."""

    # Setup paths
    experiment_dir = Path(__file__).parent
    output_dir = experiment_dir / "output"
    cache_dir = experiment_dir / "cache"
    output_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    # Create context
    # Note: cache parameter is optional and not used by the pipeline
    # Actual caching is handled by AgentCache and EvaluationCache
    ctx = PipelineContext(
        config=PipelineConfig(
            run_id="hotpotqa_experiment",
            output_dir=output_dir,
            cache_dir=cache_dir,
        ),
        logger=StdoutLogger(),
    )

    # Step 1: Read data
    data_reader = HotpotQADataReader(split="test")

    # Create pipeline
    pipeline = ExperimentPipeline(
        ctx=ctx,
        benchmark_reader=data_reader,
        tool_builders=[HotpotQAToolsBuilder(data_reader.corpus)],
        agent_builder=HotpotQAAgentBuilder(),
        agent_runner=SimpleAgentRunner(),
        evaluators=HotpotQAEvaluatorsBuilder.build_evaluators(),
    )

    # Run experiment
    outputs, results = pipeline.run()

    # Save results
    print(f"Experiment complete! Results saved to {output_dir}")
    return outputs, results

if __name__ == "__main__":
    run_hotpotqa_experiment()
```
## Critical Implementation Details

### 1. Recreatable Pattern (CRITICAL - Easy to Miss)

Classes inheriting from `Recreatable` MUST call `self._capture_init_args(locals())` as FIRST line in `__init__`:

```python
class MyClass(Recreatable):
    def __init__(self, arg1: int, arg2: str):
        self._capture_init_args(locals())  # MUST BE FIRST LINE
        self.arg1 = arg1
        # ... rest of init
```

Missing this breaks save/load functionality. This is NOT enforced by linters.

### 2. Python Environment (CRITICAL)

**ALWAYS use Python from the `.venv` virtual environment, NEVER use system/global Python.**

```bash
# Correct - uses .venv Python
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main

# WRONG - uses system Python
python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main
```

### 3. Import Pattern

Use `from experiments import` NOT `from src.experiments import`:

```python
# Correct
from experiments.benchmarks.hotpotqa.hotpotqa_main import HotpotQAExperiment
from experiments.implementations.agent_factory import AgentFactory

# Wrong
from src.experiments.benchmarks.hotpotqa.hotpotqa_main import HotpotQAExperiment
```

Root conftest.py adds src/ to sys.path - all imports assume this.


### 3. Run Your Experiment

```bash
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main
```

## Key Features

### Self-Contained Experiments

Each experiment has its own:
- **Data loading logic** - Custom data reader for the dataset
- **Tool building** - Dataset-specific tools (retrievers, calculators, etc.)
- **Agent configuration** - Customized agent setup
- **Evaluation logic** - Appropriate evaluators for the task
- **Output directory** - Results saved in `experiments/<name>/output/`
- **Cache directory** - Intermediate results in `experiments/<name>/cache/`

### Reusable Components

The `implementations/` directory provides base implementations that experiments can use or extend:
- **Tool builders** - Common tool patterns
- **Evaluators** - Standard evaluation metrics
- **Agent runners** - Different execution strategies

### Clean Pipeline

The 5-step pipeline in `pipeline.py` orchestrates everything:
1. Reads benchmarks via `BenchmarkReader`
2. Creates tools via `ToolBuilder`
3. Builds agent via `AgentBuilder`
4. Runs agent via `AgentRunner`
5. Evaluates via `Evaluator`

## Example: HotpotQA Experiment

See `experiments/hotpotqa/` for a complete working example that:
- Loads HotpotQA dataset using `RagDataLoaderFactory`
- Builds retrieval tools from the corpus
- Creates a `DsStarAgent` with those tools
- Runs the agent over the benchmark
- Evaluates using text exact match
- Saves all results to `experiments/hotpotqa/output/`

## Output Files

The experiment runner automatically saves two files for each run:

### File Naming Convention

Both files use the same naming pattern and timestamp:

```
result_<agent_type>_<model_name>_<experiment_name>[_<question_limit>]_<timestamp>_<suffix>.json
```

**Components:**
- `agent_type`: Type of agent used (e.g., "ds_star", "react")
- `model_name`: Short name of the model (e.g., "wx_mistral_medium", "gpt_4o_mini")
- `experiment_name`: Name of the experiment (e.g., "hotpotqa")
- `question_limit`: (Optional) Number of questions if limited (e.g., "3", "10")
- `timestamp`: Shared timestamp in format `YYYYMMDD_HHMMSS`
- `suffix`: Either "output" or "params"

### Examples

**With question_limit:**
```
result_react_gpt_4o_mini_hotpotqa_3_20260127_161619_output.json
result_react_gpt_4o_mini_hotpotqa_3_20260127_161619_params.json
```

**Without question_limit:**
```
result_ds_star_wx_mistral_medium_hotpotqa_20260127_161619_output.json
result_ds_star_wx_mistral_medium_hotpotqa_20260127_161619_params.json
```

### Output File (`*_output.json`)

Contains experiment results with:
- `run_id`: Experiment identifier
- `timestamp`: ISO format timestamp
- `agent_type`: Agent type used
- `question_limit`: (Optional) Number of questions if limited
- `config`: Pipeline configuration
- `summary`: Aggregate statistics (avg_score, passed_count, etc.)
- `outputs`: Agent outputs for each question
- `results`: Evaluation results for each question
- `items`: Combined output + evaluations per question
- `failures`: Any errors encountered

### Params File (`*_params.json`)

Contains experiment parameters for reproducibility:
- `experiment_class`: Class name of the experiment
- `timestamp`: ISO format timestamp
- `run_id`: Experiment identifier
- `parameters`: All experiment parameters (model, agent_type, question_limit, etc.)

### Key Features

- **Same timestamp**: Both files share the exact same timestamp for easy matching
- **Same naming logic**: Both use identical filename generation
- **Reproducibility**: Params file can be used to re-run the exact same experiment

## Benefits

1. **Modularity** - Each experiment is independent
2. **Reusability** - Share common implementations
3. **Clarity** - Clear 5-step process
4. **Testability** - Each component can be tested independently
5. **Extensibility** - Easy to add new experiments or components
6. **Organization** - All experiment artifacts in one place
7. **Traceability** - Matching output and params files with consistent naming

## Interface Reference

See the `interfaces/` directory for detailed documentation on each interface:
- `BenchmarkReader` - How to load benchmark data
- `ToolBuilder` - How to create tools
- `AgentBuilder` - How to build agents
- `AgentRunner` - How to run agents
- `Evaluator` - How to evaluate results
