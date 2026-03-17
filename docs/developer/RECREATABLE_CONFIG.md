# Recreatable-by-Config Pattern

## Overview

The recreatable-by-config pattern allows experiments to be saved and loaded from JSON configuration files while maintaining normal Python `__init__` signatures. This enables:

- **Reproducibility**: Save experiment configurations and recreate them exactly
- **Version Control**: Track experiment configurations in git
- **Sharing**: Share experiment setups as simple JSON files
- **Auditing**: Review what parameters were used in past experiments

## Architecture

### Core Components

1. **Recreatable Base Class** (`src/experiments/utils/recreatable.py`)
   - Provides config capture and serialization
   - Handles class resolution and instantiation
   - Supports security via allowed types

2. **Experiment** (inherits from Recreatable)
   - `BaseExperiment`: Experiment configuration
   - Contains factory methods for all components:
     - `get_data_reader()` → DataReader
     - `get_agent_builder()` → AgentBuilder
     - `get_tools_builder()` → ToolBuilders
     - `get_evaluators()` → Evaluators

3. **Simple Persistence**
   - Save experiment configuration to JSON
   - Load experiment from JSON
   - Loaded experiment has all factory methods
   - Running `experiment_main()` recreates the full experiment

## Usage

### Basic Pattern

Every recreatable class follows this pattern:

```python
from experiments.utils.recreatable import Recreatable

class MyClass(Recreatable):
    def __init__(self, arg1: int, arg2: str, scale: float = 1.0):
        self._capture_init_args(locals())  # FIRST LINE - captures config
        self.arg1 = arg1
        self.arg2 = arg2
        self.scale = scale
```

**Key Rules:**
1. Call `self._capture_init_args(locals())` as the **FIRST LINE** in `__init__`
2. All init arguments must be JSON-serializable (str, int, float, bool, None, list, dict)
3. The `__init__` signature is the single source of truth for configuration

### Saving Configuration

```python
# Create experiment
experiment = DemoExperiment(
    model="watsonx/mistralai/mistral-medium-2505",
    max_steps=5,
    temperature=0.0,
)

# Save to file
experiment.save_config("my_experiment.json")
```

This creates a JSON file with structure:

```json
{
  "type": "experiments.benchmarks.demo.experiment_main.DemoExperiment",
  "args": {
    "model": "watsonx/mistralai/mistral-medium-2505",
    "max_steps": 5,
    "temperature": 0.0,
    "code_timeout": 30
  }
}
```

### Loading Configuration

```python
# Load experiment
experiment = DemoExperiment.load_instance("my_experiment.json")

# Experiment has all factory methods
data_reader = experiment.get_data_reader()
agent_builder = experiment.get_agent_builder()
tool_builders = experiment.get_tools_builder()
evaluators = experiment.get_evaluators()

# Run the experiment
outputs, results = experiment.experiment_main()
```

### Security: Allowed Types

When loading configurations from untrusted sources, use `allowed_types` to restrict what can be instantiated:

```python
# Only allow specific experiment types
experiment = DemoExperiment.load_instance(
    "config.json",
    allowed_types=[
        "experiments.benchmarks.demo.experiment_main.DemoExperiment",
        "experiments.benchmarks.hotpotqa.hotpotqa_main.HotpotQAExperiment"
    ]
)
```

If the config contains a type not in the allowed list, a `ValueError` is raised.

## Implementation Guide

### Creating a New Recreatable Component

#### 1. Inherit from Recreatable

```python
from experiments.utils.recreatable import Recreatable

class MyToolBuilder(ToolBuilder):  # ToolBuilder already inherits from Recreatable
    pass
```

#### 2. Add `_capture_init_args` Call

```python
def __init__(self, param1: str, param2: int = 10):
    self._capture_init_args(locals())  # FIRST LINE
    self.param1 = param1
    self.param2 = param2
```

#### 3. Ensure JSON-Serializable Arguments

All `__init__` parameters must be JSON-serializable:

**✓ Supported:**
- `str`, `int`, `float`, `bool`, `None`
- `list`, `dict` (with serializable contents)

**✗ Not Supported (without custom handling):**
- `Path` objects
- `Enum` instances
- NumPy arrays
- Custom objects

**Workaround for complex types:**

```python
from pathlib import Path
from enum import Enum

class MyClass(Recreatable):
    def __init__(self, path: str, mode: str):  # Use str, not Path or Enum
        self._capture_init_args(locals())
        self.path = Path(path)  # Convert after capture
        self.mode = MyEnum(mode)  # Convert after capture
```

### Updating Existing Components

For existing components, add the `_capture_init_args` call:

```python
# Before
class MyExperiment(BaseExperiment):
    def __init__(self, model: str, max_steps: int = 5):
        super().__init__(model=model, max_steps=max_steps)
        self.model = model
        self.max_steps = max_steps

# After
class MyExperiment(BaseExperiment):
    def __init__(self, model: str, max_steps: int = 5):
        self._capture_init_args(locals())  # ADD THIS LINE FIRST
        super().__init__(model=model, max_steps=max_steps)
        self.model = model
        self.max_steps = max_steps
```

## Best Practices

### 1. Single Source of Truth

The `__init__` signature is the **only** place where configuration is defined:

```python
# ✓ Good: All config in __init__
def __init__(self, model: str, temperature: float = 0.0):
    self._capture_init_args(locals())
    self.model = model
    self.temperature = temperature

# ✗ Bad: Config derived from runtime state
def __init__(self, model: str):
    self._capture_init_args(locals())
    self.model = model
    self.temperature = self._compute_temperature()  # Not reproducible!
```

### 2. Immutable Configuration

Don't modify captured init args after construction:

```python
# ✓ Good: Config stays constant
def __init__(self, max_steps: int = 5):
    self._capture_init_args(locals())
    self.max_steps = max_steps

# ✗ Bad: Modifying config after capture
def __init__(self, max_steps: int = 5):
    self._capture_init_args(locals())
    self.max_steps = max_steps
    self.max_steps += 1  # Config no longer matches reality!
```

### 3. Explicit Defaults

Always provide explicit defaults in `__init__`:

```python
# ✓ Good: Clear defaults
def __init__(
    self,
    model: str = "watsonx/mistralai/mistral-medium-2505",
    temperature: float = 0.0,
    max_steps: int = 5,
):
    self._capture_init_args(locals())
    # ...

# ✗ Bad: Implicit defaults
def __init__(self, model: str = None):
    self._capture_init_args(locals())
    self.model = model or "watsonx/mistralai/mistral-medium-2505"  # Hidden default
```

### 4. Backward Compatibility

When changing `__init__` parameters, consider backward compatibility:

```python
# Version 1
def __init__(self, model: str):
    self._capture_init_args(locals())
    self.model = model

# Version 2 - backward compatible
def __init__(self, model: str, temperature: float = 0.0):
    self._capture_init_args(locals())
    self.model = model
    self.temperature = temperature  # New parameter with default
```

Old configs will still load because the new parameter has a default.

## Examples

### Complete Example

See `examples/recreatable_config_demo.py` for a full working example that demonstrates:

1. Creating an experiment with all components
2. Saving the complete configuration
3. Loading and recreating all components
4. Verifying configurations match
5. Security with allowed types

Run it:

```bash
python examples/recreatable_config_demo.py
```

### Quick Start

```python
from experiments.benchmarks.demo.experiment_main import DemoExperiment

# Create and configure
experiment = DemoExperiment(
    model="watsonx/mistralai/mistral-medium-2505",
    max_steps=3,
    temperature=0.1,
)

# Save complete config
experiment.save_experiment_config(
    path="my_experiment.json",
    agent_builder=experiment.get_agent_builder(),
    tool_builders=experiment.get_tools_builder(),
    evaluators=experiment.get_evaluators(),
)

# Later: Load and run
loaded_exp, agent, tools, evals = DemoExperiment.load_experiment_config(
    "my_experiment.json"
)
outputs, results = loaded_exp.experiment_main()
```

## Troubleshooting

### "did not call _capture_init_args()"

**Problem:** Forgot to call `_capture_init_args` in `__init__`

**Solution:** Add as first line:
```python
def __init__(self, ...):
    self._capture_init_args(locals())  # Add this
    # rest of init
```

### "Object is not JSON serializable"

**Problem:** Using non-JSON-serializable types in `__init__`

**Solution:** Convert to JSON-serializable types:
```python
# Before
def __init__(self, path: Path):
    self._capture_init_args(locals())
    self.path = path

# After
def __init__(self, path: str):
    self._capture_init_args(locals())
    self.path = Path(path)  # Convert after capture
```

### "Type X is not in allowed types"

**Problem:** Trying to load a type not in the allowed list

**Solution:** Either:
1. Add the type to allowed list
2. Verify the config file is from a trusted source
3. Update the config to use an allowed type

## API Reference

### Recreatable Class

#### Methods

- `_capture_init_args(locals_: Dict[str, Any]) -> None`
  - Capture constructor arguments (call as first line in `__init__`)

- `get_config() -> Dict[str, Any]`
  - Get the configuration dict for this instance

- `save_config(path: Path | str) -> None`
  - Save configuration to JSON file

- `load_instance(path: Path | str, allowed_types: List[str] | None = None) -> T`
  - Load instance from JSON file (class method)

- `_type_id() -> str`
  - Get type identifier (class method)

- `_resolve_type(type_id: str) -> Type[Recreatable]`
  - Resolve type identifier to class (static method)

### BaseExperiment Methods

Inherits all methods from Recreatable:
- `save_config(path)` - Save experiment configuration
- `load_instance(path, allowed_types)` - Load experiment (class method)
- `get_config()` - Get configuration dict

## Related Documentation

- [Experiment Framework](../EXPERIMENTS.md)
- [Architecture](../ARCHITECTURE.md)
- [Agent Implementations](../AGENT_IMPLEMENTATIONS.md)
