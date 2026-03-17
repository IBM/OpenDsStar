# Experiment Parameters and Reproducibility

This document explains how experiment configurations are saved and loaded for complete reproducibility.

## Overview

All experiment configurations are automatically saved to `_params.json` files that include:
- **Experiment parameters**: All constructor arguments (split, model, max_steps, etc.)
- **Tools configurations**: Embedding models, batch sizes, chunk sizes, etc.
- **Agent configurations**: Agent type, model, temperature, max_steps, etc.
- **Data reader configurations**: Split, question limits, sampling parameters, etc.

This ensures that any experiment can be exactly reproduced by loading its params file.

## Params File Format

The params file uses a comprehensive JSON format:

```json
{
  "type": "experiments.benchmarks.hotpotqa.hotpotqa_main.HotpotQAExperiment",
  "args": {
    "split": "test",
    "model": "watsonx/mistralai/mistral-medium-2505",
    "embedding_model": "ibm/slate-125m-english-rtrvr",
    "max_steps": 5,
    "agent_type": "ds_star",
    "temperature": 0.0,
    "question_limit": 20,
    "document_factor": null,
    "seed": 43,
    "output_max_length": 500,
    "logs_max_length": 20000
  },
  "components": {
    "tools": [
      {
        "tool_name": "search_hotpotqa",
        "model": "watsonx/mistralai/mistral-medium-2505",
        "temperature": 0.0,
        "embedding_model": "ibm/slate-125m-english-rtrvr",
        "batch_size": 8,
        "chunk_size": 1000,
        "chunk_overlap": 200
      }
    ],
    "agent": {
      "agent_type": "ds_star",
      "model": "watsonx/mistralai/mistral-medium-2505",
      "temperature": 0.0,
      "max_steps": 5,
      "max_debug_tries": 5,
      "code_timeout": 30,
      "code_mode": "stepwise",
      "output_max_length": 500,
      "logs_max_length": 20000
    },
    "data_reader": {
      "split": "test",
      "question_limit": 20,
      "document_factor": null,
      "seed": 43
    }
  }
}
```

## File Naming Convention

Params files share the same timestamp as their corresponding output files:

```
result_<agent_type>_<model_name>_<experiment>[_<limit>]_<timestamp>_params.json
result_<agent_type>_<model_name>_<experiment>[_<limit>]_<timestamp>_output.json
result_<agent_type>_<model_name>_<experiment>[_<limit>]_<timestamp>_log.txt
```

Example:
```
result_ds_star_mistral-medium_hotpotqa_20_20260216_172530_params.json
result_ds_star_mistral-medium_hotpotqa_20_20260216_172530_output.json
result_ds_star_mistral-medium_hotpotqa_20_20260216_172530_log.txt
```

## Automatic Saving

Params files are automatically saved when running experiments:

```bash
# Run experiment - params are saved automatically
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main \
    --agent-type ds_star \
    --max-steps 5 \
    --question-limit 20
```

The params file will be saved in the experiment's output directory (e.g., `src/experiments/benchmarks/hotpotqa/output/`).

## Loading and Reproducing Experiments

### Command Line

Use the `--load-params` flag to reproduce an experiment:

```bash
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main \
    --load-params path/to/result_ds_star_mistral-medium_hotpotqa_20_20260216_172530_params.json
```

This will:
1. Load all configurations from the params file
2. Recreate the exact same experiment setup
3. Run the experiment with identical parameters
4. Save new output files with a new timestamp

### Programmatic

Load and run experiments programmatically using Recreatable:

```python
from pathlib import Path
from experiments.benchmarks.hotpotqa.hotpotqa_main import HotpotQAExperiment

# Load experiment from params file using Recreatable
experiment = HotpotQAExperiment.load_instance(Path("path/to/params.json"))

# Run the experiment
outputs, results = experiment.experiment_main(fail_fast=False)
```

## What Gets Captured

### Experiment Parameters
All constructor arguments to the experiment class:
- `split`: Dataset split (train/test)
- `model`: Agent model ID
- `embedding_model`: Embedding model for retrieval
- `max_steps`: Maximum reasoning steps
- `agent_type`: Type of agent (ds_star, react, etc.)
- `temperature`: Generation temperature
- `question_limit`: Number of questions to process
- `document_factor`: Document sampling factor
- `seed`: Random seed for reproducibility
- `output_max_length`: Output truncation length
- `logs_max_length`: Log truncation length

### Tools Configuration
For each tool used in the experiment:
- Tool name and type
- Model used by the tool
- Embedding model
- Batch size
- Chunk size and overlap
- Temperature
- Any tool-specific parameters

### Agent Configuration
Complete agent setup:
- Agent type (DS-Star, React, etc.)
- Model ID
- Temperature
- Max steps
- Max debug tries (DS-Star)
- Code timeout (DS-Star)
- Code execution mode (DS-Star)
- Output/log truncation lengths
- System and task prompts (if any)

### Data Reader Configuration
Data loading parameters:
- Dataset split
- Question limit
- Document sampling factor
- Random seed

## Implementation Details

### Recreatable Pattern

The system uses the **Recreatable pattern** for configuration persistence:

1. **Experiments inherit from `BaseExperiment` (which inherits from `Recreatable`)**
2. **Constructor arguments are captured** via `_capture_init_args(locals())`
3. **Configurations are serialized** to JSON with class type information
4. **Instances are recreated** using `Recreatable.load_instance()`

This provides:
- Automatic serialization/deserialization
- Type resolution and instantiation
- Enum conversion (e.g., AgentType string ↔ enum)
- Backwards compatibility

### For Experiment Developers

To ensure your experiment supports full reproducibility:

1. **Inherit from `BaseExperiment`**:
```python
from experiments.base.base_experiment import BaseExperiment

class MyExperiment(BaseExperiment):
    def __init__(self, param1, param2, ...):
        self._capture_init_args(locals())  # FIRST LINE - captures config!
        super().__init__(param1=param1, param2=param2, ...)
```

2. **Implement required methods**:
```python
def get_data_reader(self) -> DataReader:
    return MyDataReader(...)

def get_tools_builder(self) -> Sequence[ToolBuilder]:
    return [MyToolsBuilder(...)]

def get_agent_builder(self) -> AgentBuilder:
    return AgentFactory(...)

def get_evaluators(self) -> Sequence[Evaluator]:
    return [MyEvaluator()]
```


### Recreatable Pattern

The system uses the `Recreatable` base class pattern:
- Classes capture their `__init__` arguments via `_capture_init_args(locals())`
- Configurations are serialized to JSON with class type information
- Instances can be recreated from saved configurations

See `docs/RECREATABLE_CONFIG.md` for more details.

## Benefits

1. **Complete Reproducibility**: Every parameter that affects experiment behavior is saved
2. **Easy Comparison**: Compare params files to understand differences between runs
3. **Experiment Replication**: Reproduce any experiment exactly from its params file
4. **Debugging**: Understand exact configuration used in any experiment run
5. **Sharing**: Share params files to enable others to reproduce your results

## Best Practices

1. **Always save params files**: They're saved automatically, keep them with outputs
2. **Version control params**: Check in representative params files for documentation
3. **Document changes**: If you modify experiment code, note how it affects params
4. **Test reproducibility**: Periodically verify that loading params produces identical results
5. **Use meaningful names**: The automatic naming includes key parameters for easy identification

## Troubleshooting

### Params file missing components

If an older params file doesn't have the `components` section, it will still load but may not capture all configurations. Re-run the experiment to generate a comprehensive params file.

### Import errors when loading

Ensure the experiment class is importable from the path specified in the `type` field. The system uses dynamic imports to recreate the experiment class.

### Agent type conversion

Agent types are automatically converted between enum and string representations. Both formats are supported for backwards compatibility.

## Examples

### Save and Load Cycle

```bash
# 1. Run experiment (saves params automatically)
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main \
    --agent-type ds_star \
    --max-steps 5 \
    --question-limit 10

# Output: result_ds_star_mistral-medium_hotpotqa_10_20260216_172530_params.json

# 2. Reproduce experiment from params
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main \
    --load-params src/experiments/benchmarks/hotpotqa/output/result_ds_star_mistral-medium_hotpotqa_10_20260216_172530_params.json
```

### Programmatic Usage

```python
from pathlib import Path
from experiments.benchmarks.hotpotqa.hotpotqa_main import HotpotQAExperiment
from experiments.implementations.agent_factory import AgentType

# Create and run experiment
experiment = HotpotQAExperiment(
    split="test",
    model="watsonx/mistralai/mistral-medium-2505",
    embedding_model="ibm/slate-125m-english-rtrvr",
    max_steps=5,
    agent_type=AgentType.DS_STAR,
    question_limit=10,
)
outputs, results = experiment.experiment_main()

# Later: Load and reproduce from saved params
outputs2, results2 = HotpotQAExperiment.load_from_params(
    params_file=Path("path/to/params.json")
)
```

## Related Documentation

- `docs/RECREATABLE_CONFIG.md`: Details on the Recreatable pattern
- `docs/EXPERIMENTS.md`: General experiment framework documentation
- `docs/AGENTS.md`: Agent configuration and usage
