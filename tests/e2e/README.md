# End-to-End Tests

This directory contains end-to-end (E2E) tests that validate complete workflows in the OpenDsStar project.

## Overview

E2E tests differ from unit tests in that they:
- Test complete workflows from start to finish
- Use real components (not mocks) where possible
- Require external dependencies (APIs, models, etc.)
- Take longer to run (30-60 seconds or more)
- Validate system behavior in realistic scenarios

## Directory Structure

E2E test artifacts are organized by benchmark:

```
tests/e2e/
├── cache_outputs/           # All test artifacts (persisted after runs)
│   ├── hotpotqa/           # HotpotQA benchmark artifacts
│   │   ├── cache/          # LLM cache, embeddings, etc.
│   │   └── output/         # Results, logs, trajectories
│   ├── databench/          # DataBench benchmark artifacts
│   │   ├── cache/
│   │   └── output/
│   └── kramabench/         # KramaBench benchmark artifacts
│       ├── cache/
│       └── output/
├── conftest.py             # Shared test configuration
├── test_hotpotqa_e2e.py
├── test_databench_e2e.py
├── test_kramabench_e2e.py
└── test_databench_codeact_e2e.py
```

This structure ensures:
- Each benchmark's artifacts are isolated
- Easy to inspect results for specific benchmarks
- No pollution of benchmark source directories
- Artifacts persist after test runs for debugging

## Running E2E Tests

### Run all E2E tests
```bash
pytest tests/e2e/ -v
```

### Run specific E2E test
```bash
pytest tests/e2e/test_hotpotqa_e2e.py::test_hotpotqa_sample_experiment -v
```

### Skip E2E tests (run only unit tests)
```bash
pytest tests/ -m "not e2e"
```

### Run only E2E tests
```bash
pytest tests/ -m "e2e"
```

### Skip slow tests
```bash
pytest tests/ -m "not slow"
```

### Run only E2E tests (excluding slow ones)
```bash
pytest tests/ -m "e2e and not slow"
```

## Prerequisites

E2E tests require:

1. **Environment Variables**: Create a `.env` file with required API keys:
   ```
   WATSONX_API_KEY=your_key_here
   WATSONX_PROJECT_ID=your_project_id
   # Add other required credentials
   ```

2. **Internet Connection**: Tests make real API calls to LLM services

3. **Sufficient Time**: E2E tests are marked as `slow` and may take 30-60 seconds each

## Available E2E Tests

### HotpotQA E2E Test (`test_hotpotqa_e2e.py`)

#### `test_hotpotqa_sample_experiment`
- **Purpose**: Comprehensive validation of complete HotpotQA experiment workflow
- **Sample Size**: 3 questions
- **Expected**: Average score > 0.9
- **Duration**: ~30-60 seconds
- **What it tests**:
  - Data loading from HotpotQA dataset
  - Tool building (retrieval tools with VectorStore)
  - Agent creation and execution
  - Evaluation with exact match
  - Result aggregation
  - **Output structure validation**:
    - AgentOutput instances with question_id and answer fields
    - EvalResult instances with question_id, score, and passed fields
    - Score range validation (0.0 to 1.0)
    - Type checking (boolean for passed field)
    - Alignment between outputs and results (matching question_ids)

### DataBench CodeAct E2E Tests (`test_databench_codeact_e2e.py`)

#### `test_databench_codeact_watsonx`
- **Purpose**: Validate CodeAct agent on DataBench with WatsonX Llama Maverick model
- **Sample Size**: 1 question
- **Max Steps**: 3
- **Duration**: ~60-300 seconds (5 minute timeout)
- **Model**: `wx_llama_maverick` for both agent and file descriptions
- **What it tests**:
  - CodeAct agent execution on DataBench
  - WatsonX Llama Maverick model integration
  - File description generation
  - Agent completion without fatal errors
  - Output contains "stage_end:run_agent" or "Experiment completed"

#### `test_databench_codeact_custom_api`
- **Purpose**: Validate CodeAct agent with custom API provider
- **Sample Size**: 1 question
- **Max Steps**: 3
- **Duration**: ~60-300 seconds (5 minute timeout)
- **Models**:
  - Agent: `tpm/GCP/gemini-2.5-flash` (custom API provider)
  - File descriptions: `wx_llama_maverick`
- **What it tests**:
  - CodeAct agent with custom API provider
  - Mixed model provider usage (custom API + WatsonX)
  - Agent completion without fatal errors
  - Output contains "stage_end:run_agent" or "Experiment completed"

## Test Markers

E2E tests use pytest markers:

```python
@pytest.mark.e2e      # Marks as end-to-end test
@pytest.mark.slow     # Marks as slow test (>1 second)
```

## Writing New E2E Tests

### Template

```python
"""E2E test for [component/workflow]."""

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.e2e
@pytest.mark.slow
def test_my_e2e_workflow():
    """
    E2E test: [Description of what this tests].

    This test:
    1. [Step 1]
    2. [Step 2]
    3. [Step 3]

    Note: Requires [prerequisites]
    """
    # Import real components (not mocks)
    from src.module import RealComponent

    # Setup
    component = RealComponent(config)

    # Execute complete workflow
    result = component.run_complete_workflow()

    # Validate end-to-end behavior
    assert result.success
    assert result.metric > threshold
```

### Best Practices

1. **Use Real Components**: Don't mock unless absolutely necessary
2. **Test Complete Workflows**: Cover the full user journey
3. **Set Realistic Expectations**: Account for API variability
4. **Add Debugging Output**: Print summaries for troubleshooting
5. **Handle Failures Gracefully**: Use `fail_fast=False` where appropriate
6. **Document Prerequisites**: Clearly state what's needed to run
7. **Keep Tests Focused**: One workflow per test function
8. **Use Parametrization**: Test variations efficiently

## Troubleshooting

### Test Fails with API Error
- Check `.env` file has correct credentials
- Verify internet connection
- Check API service status

### Test Times Out
- Increase timeout if needed
- Check if API is responding slowly
- Consider reducing sample size for faster tests

### Inconsistent Results
- E2E tests may have some variability due to LLM non-determinism
- Use seeds where possible for reproducibility
- Set reasonable thresholds (e.g., >0.9 instead of ==1.0)

### Import Errors
- Ensure all dependencies are installed
- Check that `src/` is in PYTHONPATH
- Verify virtual environment is activated

## CI/CD Integration

E2E tests can be configured to run:
- On pull requests (with API credentials in secrets)
- On scheduled nightly builds
- Manually triggered for release validation

Example GitHub Actions configuration:
```yaml
- name: Run E2E tests
  env:
    WATSONX_API_KEY: ${{ secrets.WATSONX_API_KEY }}
    WATSONX_PROJECT_ID: ${{ secrets.WATSONX_PROJECT_ID }}
  run: |
    pytest tests/e2e/ -v -m e2e
```

## Performance Benchmarks

Expected execution time (approximate):
- `test_hotpotqa_sample_experiment`: 30-60 seconds
- `test_databench_codeact_watsonx`: 60-300 seconds
- `test_databench_codeact_custom_api`: 60-300 seconds

Total E2E suite: ~2-7 minutes (depending on API response times)

## Future E2E Tests

Planned additions:
- Demo experiment E2E test
- Multi-agent workflow tests
- Cache integration E2E tests
- Tool builder E2E tests
- Custom dataset E2E tests
- Tests with different sample sizes (if needed for performance validation)
