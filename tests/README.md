# OpenDsStar Test Suite

This directory contains comprehensive unit tests for the OpenDsStar project.

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures and test configuration
├── agents/                              # Agent layer tests
│   ├── test_open_ds_star_agent.py      # Main agent class tests
│   └── graphs/
│       └── test_ds_star_state.py       # State management tests
├── experiments/                  # Experiments layer tests
│   ├── core/
│   │   └── test_types.py               # Core type tests
│   └── utils/
│       ├── test_cache.py               # Cache utilities tests
│       ├── test_agent_cache.py         # Agent cache tests
│       ├── test_evaluation_cache.py    # Evaluation cache tests
│       └── test_evaluation_cache_integration.py
└── README.md                            # This file
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/agents/test_open_ds_star_agent.py -v
```

### Run Tests with Coverage
```bash
pytest --cov=src --cov-report=html
```

### Run Tests by Marker
```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration
```

## Test Coverage

### Current Coverage (93 tests passing)

#### ✅ Agent Layer (20 tests)
- **OpenDsStarAgent** (20 tests)
  - Initialization with various configurations
  - Model validation (string and BaseChatModel)
  - Code mode validation
  - Invoke method with different scenarios
  - Tool description updates
  - Error handling

#### ✅ State Management (13 tests)
- **DSStep** (7 tests)
  - Dictionary-like interface
  - Field access and modification
  - Iteration and keys
- **DSState** (13 tests)
  - Dictionary-like interface
  - State accumulation (steps, trajectory, tokens)
- **CodeMode** (5 tests)
  - Enum values and conversion

#### ✅ Core Types (30 tests)
- **Document** (3 tests)
  - Stream factory functionality
  - Immutability
- **GroundTruth** (4 tests)
  - Default values
  - Immutability
- **BenchmarkEntry** (3 tests)
  - Creation and defaults
  - Immutability
- **ProcessedBenchmark** (3 tests)
  - Creation and defaults
  - Immutability
- **AgentOutput** (4 tests)
  - Various answer types
  - Immutability
- **EvalResult** (4 tests)
  - Score and pass/fail
  - Immutability
- **Type Equality** (3 tests)
  - Equality comparison
  - Hashability

#### ✅ Cache Utilities (30 tests)
- **NullCache** (3 tests)
  - No-op behavior
- **FileCache** (24 tests)
  - Basic operations (get, put, clear)
  - Various data types
  - Context manager
  - Persistence
  - Special characters
- **EvaluationCache** (2 tests)
  - Cache key generation
  - Get/put operations
- **AgentCache** (1 test)
  - Agent configuration hashing

## Test Fixtures

### Common Fixtures (from conftest.py)

- `temp_cache_dir` - Temporary cache directory
- `sample_document` - Sample document for testing
- `sample_documents` - Multiple sample documents
- `sample_ground_truth` - Sample ground truth data
- `sample_benchmark` - Sample benchmark entry
- `sample_benchmarks` - Multiple benchmark entries
- `sample_processed_benchmark` - Processed benchmark
- `sample_agent_output` - Sample agent output
- `sample_eval_result` - Sample evaluation result
- `mock_llm` - Mock LLM for testing
- `mock_tool` - Mock tool with proper schema
- `mock_tools` - Multiple mock tools
- `mock_evaluator` - Mock evaluator
- `agent_config` - Sample agent configuration
- `experiment_config` - Sample experiment configuration
- `pipeline_config` - Sample pipeline configuration
- `pipeline_context` - Pipeline context with logger and cache

## Writing New Tests

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<functionality>`

### Example Test Structure

```python
"""Tests for MyModule."""

import pytest
from my_module import MyClass


class TestMyClass:
    """Test MyClass functionality."""

    def test_initialization(self):
        """Test class initialization."""
        obj = MyClass()
        assert obj is not None

    def test_method_with_fixture(self, mock_tool):
        """Test method using fixture."""
        obj = MyClass(tool=mock_tool)
        result = obj.process()
        assert result is not None

    @pytest.mark.parametrize("input,expected", [
        ("a", "A"),
        ("b", "B"),
    ])
    def test_parametrized(self, input, expected):
        """Test with multiple inputs."""
        obj = MyClass()
        assert obj.transform(input) == expected
```

### Best Practices

1. **Use Fixtures**: Leverage shared fixtures from conftest.py
2. **Mock External Dependencies**: Use mocks for LLMs, APIs, databases
3. **Test Edge Cases**: Include tests for error conditions
4. **Keep Tests Fast**: Unit tests should run in milliseconds
5. **One Assertion Per Test**: Focus each test on one behavior
6. **Descriptive Names**: Test names should describe what they test
7. **Arrange-Act-Assert**: Follow AAA pattern in test structure

## Test Categories

### Unit Tests (mark with `@pytest.mark.unit`)
- Test individual functions/methods in isolation
- Use mocks for dependencies
- Fast execution (< 1 second)

### Integration Tests (mark with `@pytest.mark.integration`)
- Test component interactions
- May use real dependencies
- Slower execution (1-10 seconds)

### Slow Tests (mark with `@pytest.mark.slow`)
- Tests that take > 10 seconds
- Can be skipped with `-m "not slow"`

## Continuous Integration

Tests are automatically run on:
- Every commit
- Pull requests
- Before merges

### CI Requirements
- All tests must pass
- Coverage must be > 80%
- No new warnings

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the project root
cd /path/to/OpenDsStar

# Activate virtual environment
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

**Fixture Not Found**
- Check that conftest.py is in the tests directory
- Verify fixture name matches exactly

**Test Fails Locally But Passes in CI**
- Check for hardcoded paths
- Verify environment variables
- Look for timing-dependent code

## Future Test Coverage

### Priority 2: Interfaces & Configuration
- [ ] Interface abstract method enforcement
- [ ] Configuration validation
- [ ] Config serialization

### Priority 3: Nodes & Utilities
- [ ] All 7 node types (Planner, Coder, Executor, etc.)
- [ ] Validation utilities
- [ ] Logging utilities
- [ ] Tool registry

### Priority 4: Integration & E2E
- [ ] Pipeline integration tests
- [ ] Agent integration tests
- [ ] Complete workflow tests

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure tests pass locally
3. Add appropriate markers
4. Update this README if needed
5. Maintain > 80% coverage

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest Parametrize](https://docs.pytest.org/en/stable/parametrize.html)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
