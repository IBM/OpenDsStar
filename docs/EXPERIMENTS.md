# Experiments Guide

This guide explains how to run, rerun, and create experiments in OpenDsStar.

## Table of Contents

- [Running Experiments](#running-experiments)
- [Rerunning Experiments](#rerunning-experiments)
- [Available Experiments](#available-experiments)
- [Creating New Experiments](#creating-new-experiments)
- [Experiment Output](#experiment-output)
- [Configuration Patterns](#configuration-patterns)
- [AI Assistant Experience](#ai-assistant-experience)

## Running Experiments

> **IMPORTANT**: Always use Python from the `.venv` virtual environment, NOT system Python.
> All commands below should use `.venv/bin/python` or activate the venv first.

### HotpotQA Experiment

Run the HotpotQA benchmark:

```bash
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main \
    --split test \
    --model watsonx/mistralai/mistral-medium-2505 \
    --agent-type ds_star \
    --max-steps 5
```

#### Command-Line Options

- `--split` - Dataset split: `train` or `test` (default: `test`)
- `--model` - Model ID (default: `watsonx/mistralai/mistral-medium-2505`)
- `--agent-type` - Agent type: `ds_star` or `react` (default: `ds_star`)
- `--max-steps` - Maximum reasoning steps (default: `5`)
- `--temperature` - Generation temperature (default: `0.0`)
- `--fail-fast` - Stop on first error (default: `false`)

### Demo Experiment

Run a simple demonstration experiment:

```bash
.venv/bin/python -m src.experiments.benchmarks.demo.experiment_main
```

### KramaBench Experiment

Run the KramaBench benchmark:

```bash
.venv/bin/python -m src.experiments.benchmarks.kramabench.kramabench_main \
    --agent-type ds_star \
    --max-steps 5
```

### DataBench Experiment

Run the DataBench benchmark:

```bash
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
    --agent-type ds_star \
    --max-steps 5
```

## Rerunning Experiments

Every experiment automatically saves its parameters to `experiment_params_<run_id>.json`. You can rerun an experiment with the exact same configuration using this file.

### Method 1: Using the load_and_run_experiment Function

```python
from src.experiments.benchmarks.hotpotqa.experiment_main import load_and_run_experiment

# Rerun with saved parameters
outputs, results = load_and_run_experiment(
    "src/experiments/experiments/hotpotqa/output/experiment_params_hotpotqa_sample_20.json"
)
```

### Method 2: Using Command-Line

```bash
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main \
    --load-params src/experiments/benchmarks/hotpotqa/output/result_ds_star_mistral-medium_hotpotqa_20_20260216_172530_params.json
```

### Method 3: Programmatically Load and Modify

```python
import json
from pathlib import Path
from src.experiments.benchmarks.hotpotqa.experiment_main import run_hotpotqa_experiment

# Load saved parameters
params_file = Path("src/experiments/experiments/hotpotqa/output/experiment_params_hotpotqa_sample_20.json")
with open(params_file) as f:
    data = json.load(f)

# Extract parameters from nested structure
params = data["parameters"]

# Optionally modify parameters
params["max_steps"] = 10
params["temperature"] = 0.5

# Run with modified parameters
outputs, results = run_hotpotqa_experiment(**params)
```

## Available Experiments

### HotpotQA

Multi-hop question answering benchmark that requires reasoning over multiple documents.

**Location:** `src/experiments/experiments/hotpotqa/`

**Features:**
- Retrieval-based QA over Wikipedia corpus
- Multi-hop reasoning required
- Text exact match evaluation
- Supports both DS-Star and React agents

**Run:**
```bash
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main
```

### KramaBench

Knowledge-intensive reasoning and multi-step analysis benchmark.

**Location:** `src/experiments/benchmarks/kramabench/`

**Features:**
- Complex reasoning tasks
- Multi-document retrieval
- Specialized evaluation metrics

**Run:**
```bash
.venv/bin/python -m src.experiments.benchmarks.kramabench.kramabench_main
```

### DataBench

Data analysis and manipulation tasks over structured datasets.

**Location:** `src/experiments/benchmarks/databench/`

**Features:**
- CSV/tabular data processing
- Code generation and execution
- Numeric and text evaluations

**Run:**
```bash
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main
```

### Demo

Simple demonstration experiment for testing the framework.

**Location:** `src/experiments/experiments/demo/`

**Features:**
- Minimal example
- Good starting point for new experiments
- Shows basic pipeline usage

**Run:**
```bash
.venv/bin/python -m src.experiments.benchmarks.demo.experiment_main
```

## Creating New Experiments

See the [Experiment Runner README](src/experiments/README.md) for detailed instructions on creating new experiments.

### Quick Overview

1. **Create experiment directory:**
   ```bash
   mkdir -p src/experiments/experiments/my_experiment/{output,cache}
   ```

2. **Implement required components:**
   - `data_reader.py` - Load dataset and corpus
   - `tools_builder.py` - Build tools from corpus
   - `agent_builder.py` - Create agent with tools
   - `evaluators_builder.py` - Configure evaluators
   - `experiment_main.py` - Orchestrate the experiment

3. **Run your experiment:**
   ```bash
   .venv/bin/python -m src.experiments.benchmarks.my_experiment.experiment_main
   ```

## Experiment Output

Each experiment saves results to its `output/` directory with a consistent naming convention.

### File Naming Convention

Both output and params files use the same naming pattern and timestamp:

```
result_<agent_type>_<experiment_name>[_<question_limit>]_<timestamp>_<suffix>.json
```

**Components:**
- `agent_type`: Type of agent used (e.g., "ds_star", "react")
- `experiment_name`: Name of the experiment (e.g., "hotpotqa")
- `question_limit`: (Optional) Number of questions if limited (e.g., "3", "10")
- `timestamp`: Shared timestamp in format `YYYYMMDD_HHMMSS`
- `suffix`: Either "output" or "params"

**Examples:**

With question_limit:
```
result_react_hotpotqa_3_20260127_161619_output.json
result_react_hotpotqa_3_20260127_161619_params.json
```

Without question_limit:
```
result_ds_star_hotpotqa_20260127_161619_output.json
result_ds_star_hotpotqa_20260127_161619_params.json
```

### Files Generated

1. **`result_<agent_type>_<experiment_name>[_<question_limit>]_<timestamp>_params.json`**
   - Complete experiment configuration
   - Use this to rerun experiments with exact same parameters
   - Both files share the same timestamp for easy matching
   - Example:
     ```json
     {
       "experiment_class": "HotpotQAExperiment",
       "timestamp": "2026-01-27T15:38:54.049522",
       "run_id": "hotpotqa",
       "parameters": {
         "split": "test",
         "model": "watsonx/mistralai/mistral-medium-2505",
         "agent_type": "ds_star",
         "max_steps": 5,
         "temperature": 0.0,
         "question_limit": 20,
         "document_factor": 10,
         "seed": 43
       }
     }
     ```

2. **`result_<agent_type>_<experiment_name>[_<question_limit>]_<timestamp>_output.json`**
   - Complete experiment results including:
     - Agent outputs for each question
     - Evaluation results for each question
     - Summary statistics
     - Configuration used
   - Example structure:
     ```json
     {
       "run_id": "hotpotqa",
       "timestamp": "2026-01-27T16:16:19.123456",
       "agent_type": "react",
       "question_limit": 3,
       "config": {
         "fail_fast": false,
         "continue_on_error": true
       },
       "summary": {
         "total_results": 3,
         "avg_score": 1.0,
         "passed_count": 3,
         "passed_rate": 1.0
       },
       "outputs": [...],
       "results": [...],
       "items": [...],
       "failures": []
     }
     ```

### Output Directory Structure

```
src/experiments/experiments/hotpotqa/
├── output/
│   ├── result_ds_star_hotpotqa_20260127_161619_output.json
│   ├── result_ds_star_hotpotqa_20260127_161619_params.json
│   ├── result_react_hotpotqa_3_20260127_162030_output.json
│   └── result_react_hotpotqa_3_20260127_162030_params.json
└── cache/
    └── (cached intermediate results)
```

### Key Features

- **Same timestamp**: Both output and params files share the exact same timestamp
- **Agent type in name**: Easy to identify which agent was used
- **Question limit in name**: When limited, the number is included in the filename
- **Reproducibility**: Params file contains all information needed to rerun the experiment
- **Traceability**: Matching filenames make it easy to find related files

## Agent Types

OpenDsStar supports two agent types:

### DS-Star Agent (`AgentType.DS_STAR`)

- **Best for:** Complex data science workflows
- **Features:** Multi-step planning, code generation, stepwise/full execution modes
- **Use when:** Tasks require explicit planning and code execution

### React Agent (`AgentType.REACT`)

- **Best for:** General tasks, quick prototyping
- **Features:** Simpler reasoning, flexible tool use
- **Use when:** Tasks are straightforward or you need faster iteration

### Specifying Agent Type

**In code:**
```python
from src.experiments.implementations.agent_factory import AgentType

experiment = HotpotQAExperiment(
    agent_type=AgentType.DS_STAR,  # or AgentType.REACT
    # ... other params
)
```

**Command-line:**
```bash
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main \
    --agent-type ds_star  # or react
```

## Tips and Best Practices

### 1. Start Small
Run experiments on a small sample first to validate your setup:
```python
experiment = HotpotQAExperiment(
    question_limit=10,  # Only 10 questions
    # ... other params
)
```

### 2. Use Caching
Enable caching to avoid recomputing expensive operations:
```python
ctx = PipelineContext(
    config=PipelineConfig(
        use_cache=True,
        cache_dir=Path("cache/"),
    )
)
```

### 3. Save Parameters
Always save experiment parameters for reproducibility:
```python
# Parameters are automatically saved to:
# experiments/<name>/output/experiment_params_<run_id>.json
```

### 4. Monitor Progress
Use the built-in logging to track experiment progress:
```python
ctx = PipelineContext(
    logger=StdoutLogger(),  # Logs to console
    # ... other config
)
```

### 5. Handle Errors Gracefully
Use `fail_fast=False` to continue on errors:
```python
outputs, results = experiment.experiment_main(
    fail_fast=False,  # Continue even if some questions fail
)
```

## Troubleshooting

### Experiment Fails to Load Parameters

**Problem:** `FileNotFoundError` when loading parameters

**Solution:** Check the file path is correct and relative to your working directory:
```python
from pathlib import Path

params_file = Path("src/experiments/experiments/hotpotqa/output/experiment_params_hotpotqa_sample_20.json")
if not params_file.exists():
    print(f"File not found: {params_file.absolute()}")
```

### Agent Type Errors

**Problem:** `ValueError: Unknown agent_type`

**Solution:** Use the `AgentType` enum:
```python
from src.experiments.implementations.agent_factory import AgentType

# Correct
agent_type = AgentType.DS_STAR

# Incorrect (old way)
# agent_type = "ds_star"
```

### Out of Memory

**Problem:** Experiment runs out of memory

**Solution:** Reduce batch size or question limit:
```python
experiment = HotpotQAExperiment(
    question_limit=10,  # Process fewer questions
    document_factor=5,  # Use fewer documents
)
```

## Configuration Patterns

### Recreatable Pattern

Experiments use the `Recreatable` pattern for configuration persistence:

```python
class MyExperiment(BaseExperiment):
    def __init__(self, split: str, model: str, max_steps: int):
        self._capture_init_args(locals())  # MUST BE FIRST LINE!
        super().__init__(split=split, model=model, max_steps=max_steps)
```

This enables:
- Automatic serialization/deserialization
- Complete reproducibility
- Type-safe enum conversion
- Backwards compatibility

See [EXPERIMENT_PARAMS.md](docs/EXPERIMENT_PARAMS.md) for details.

## AI Assistant Experience

### Using Bob for Experiments Development

The experiments framework was documented with assistance from Bob, an AI coding assistant. Here's what worked well:

**Strengths:**
- **Multi-file Analysis**: Bob efficiently read and synthesized information from multiple documentation files
- **Pattern Recognition**: Quickly identified architectural patterns like the 5-step pipeline and Recreatable pattern
- **Context Retention**: Remembered project conventions (Python venv, import patterns, Recreatable pattern)
- **Technical Accuracy**: Correctly identified critical implementation details like `_capture_init_args(locals())`
- **Error Correction**: When incorrect information was added (non-existent config methods), Bob quickly verified against actual code and corrected the documentation

**Best Practices When Using Bob:**
1. Point Bob to relevant documentation files (README.md, AGENTS.md, etc.)
2. Ask for specific patterns or mechanisms rather than general overviews
3. Request code examples to verify understanding
4. Have Bob validate against existing implementations

**Example Interaction:**
```
User: "Explain the experiments folder mechanism"
Bob: [Reads multiple files, identifies 5-step pipeline, explains Recreatable pattern]
User: "Update docs with insights"
Bob: [Updates EXPERIMENTS.md with Python venv requirement, config patterns, etc.]
```

The collaboration was highly effective for understanding complex codebases and maintaining documentation quality.

## Further Reading

- [Experiment Runner README](src/experiments/README.md) - Framework documentation
- [Experiment Parameters](docs/EXPERIMENT_PARAMS.md) - Reproducibility and configuration
- [Agent Implementations](docs/AGENT_IMPLEMENTATIONS.md) - Available agent types
- [Recreatable Config](docs/RECREATABLE_CONFIG.md) - Configuration persistence pattern
