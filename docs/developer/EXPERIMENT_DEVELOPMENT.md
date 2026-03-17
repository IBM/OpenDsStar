# Experiments

This directory contains experiment implementations using the `BaseExperiment` class.

## BaseExperiment Class

All experiments should inherit from `BaseExperiment` and implement the following abstract methods:

### Required Methods

1. **`get_data_reader()`** - Returns a `DataReader` instance
   - Loads and provides both corpus data and benchmark questions
   - The DataReader must implement:
     - `read_data(ctx)`: Load all data (corpus and benchmarks)
     - `get_data()`: Return the corpus/data
     - `get_benchmark()`: Return the benchmark entries

2. **`get_tools_builder()`** - Returns a list of `ToolBuilder` instances
   - Creates tools that will be available to the agent
   - Can use data from `get_data_reader().get_data()` to build tools

3. **`get_agent_builder()`** - Returns an `AgentBuilder` instance
   - Configures and builds the agent for the experiment

4. **`get_evaluators()`** - Returns a list of `Evaluator` instances
   - Provides evaluators for different answer types

### Optional Methods

5. **`get_agent_runner()`** - Returns an `AgentRunner` instance (optional)
   - By default, uses `SimpleAgentRunner`
   - Override if you need custom agent execution logic

6. **`experiment_main()`** - Main experiment execution method (optional)
   - Default implementation handles the complete pipeline
   - Override if you need custom experiment flow (e.g., saving results, custom logging)

## Creating a New Experiment

### Basic Structure

```python
from typing import Sequence
from ..base_experiment import BaseExperiment
from ...interfaces.data_reader import DataReader
from ...interfaces.tool_builder import ToolBuilder
from ...interfaces.agent_builder import AgentBuilder
from ...interfaces.evaluator import Evaluator

class MyExperiment(BaseExperiment):
    """My custom experiment."""

    def __init__(self, **kwargs):
        """Initialize with experiment-specific parameters."""
        super().__init__(**kwargs)
        # Initialize experiment-specific attributes

    def get_data_reader(self) -> DataReader:
        """Return the data reader."""
        return MyDataReader()

    def get_tools_builder(self) -> Sequence[ToolBuilder]:
        """Return tool builders."""
        return [MyToolBuilder()]

    def get_agent_builder(self) -> AgentBuilder:
        """Return the agent builder."""
        return MyAgentBuilder()

    def get_evaluators(self) -> Sequence[Evaluator]:
        """Return evaluators."""
        return [MyEvaluator()]

# Entry point function
def run_my_experiment(**kwargs):
    """Run the experiment."""
    experiment = MyExperiment(**kwargs)
    return experiment.experiment_main()

if __name__ == "__main__":
    run_my_experiment()
```

### Directory Structure

Each experiment should have its own directory:

```
my_experiment/
├── __init__.py
├── experiment_main.py      # Main entry point with MyExperiment class
├── data_reader.py          # DataReader implementation
├── tools_builder.py        # ToolBuilder implementation(s)
├── agent_builder.py        # AgentBuilder implementation
├── evaluators_builder.py   # Evaluator factory/implementations
├── output/                 # Output directory (created automatically)
│   ├── result_<agent_type>_<model_name>_<experiment_name>[_<question_limit>]_<timestamp>_output.json
│   └── result_<agent_type>_<model_name>_<experiment_name>[_<question_limit>]_<timestamp>_params.json
└── cache/                  # Cache directory (created automatically)
```

### Output Files

Each experiment run generates two files with matching timestamps:

1. **Output file** (`*_output.json`): Contains results, evaluations, and summary
2. **Params file** (`*_params.json`): Contains all experiment parameters for reproducibility

**Naming convention:**
```
result_<agent_type>_<model_name>_<experiment_name>[_<question_limit>]_<timestamp>_<suffix>.json
```

**Examples:**
- `result_ds_star_wx_mistral_medium_hotpotqa_20260127_161619_output.json`
- `result_react_gpt_4o_mini_hotpotqa_3_20260127_161619_params.json` (with question_limit=3)

## Examples

### Demo Experiment

See `demo/experiment_main.py` for a simple example with stub implementations.

```python
class DemoExperiment(BaseExperiment):
    """Demo experiment with simple stub implementations."""

    def __init__(self):
        super().__init__()
        self.benchmarks = create_sample_benchmarks()

    def get_data_reader(self) -> DataReader:
        return SimpleBenchmarkReader(self.benchmarks)

    def get_tools_builder(self) -> Sequence[ToolBuilder]:
        return [EchoToolBuilder()]

    def get_agent_builder(self) -> AgentBuilder:
        return SimpleAgentBuilder()

    def get_evaluators(self) -> Sequence[Evaluator]:
        return [NumericExactEvaluator(), TextExactEvaluator()]
```

### HotpotQA Experiment

See `hotpotqa/experiment_main.py` for a more complex example with:
- Custom pipeline that passes corpus to tools builder
- Result saving functionality
- Command-line argument parsing

```python
from core.model_registry import ModelRegistry

class HotpotQAExperiment(BaseExperiment):
    """HotpotQA experiment with corpus-based retrieval."""

    def __init__(self, split="test", model=None, max_steps=5):
        # Use ModelRegistry default if not provided
        if model is None:
            model = ModelRegistry.WX_MISTRAL_MEDIUM
        super().__init__(split=split, model=model, max_steps=max_steps)
        self.split = split
        self.model = model
        self.max_steps = max_steps
        self.data_reader = HotpotQADataReader(split=split)
        self.tools_builder = HotpotQAToolsBuilder(corpus=None)

    def create_pipeline(self, ctx: PipelineContext) -> ExperimentPipeline:
        """Override to create custom pipeline that handles corpus passing."""
        # Custom implementation that passes corpus from data_reader to tools_builder
        ...

    def experiment_main(self, run_id=None, fail_fast=False):
        """Override to add result saving."""
        outputs, results = super().experiment_main(run_id, fail_fast)
        self.save_results(output_dir, outputs, results)
        return outputs, results
```

## Running Experiments

### From Python

```python
from src.experiments.benchmarks.demo.experiment_main import run_demo_experiment
from src.experiments.benchmarks.hotpotqa.hotpotqa_main import run_hotpotqa_experiment
from core.model_registry import ModelRegistry

# Run demo
outputs, results = run_demo_experiment()

# Run HotpotQA with specific model
outputs, results = run_hotpotqa_experiment(
    split="test",
    model=ModelRegistry.WX_MISTRAL_MEDIUM,
    max_steps=5,
    fail_fast=False
)

# Or use defaults (will use ModelRegistry.WX_MISTRAL_MEDIUM)
outputs, results = run_hotpotqa_experiment(split="test")
```

### From Command Line

```bash
# Demo experiment
python -m src.experiments.benchmarks.demo.experiment_main

# HotpotQA experiment with default model
python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main --split test

# HotpotQA experiment with specific model (use full model ID from ModelRegistry)
python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main --split test --model "gpt-4o-mini"
```

## Benefits of BaseExperiment

1. **Consistency**: All experiments follow the same structure
2. **Reusability**: Common functionality (setup, logging, summary) is shared
3. **Flexibility**: Can override methods for custom behavior
4. **Maintainability**: Changes to base functionality automatically apply to all experiments
5. **Documentation**: Clear contract for what each experiment must provide
