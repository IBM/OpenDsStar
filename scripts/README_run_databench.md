# DataBench Launch Script

## Overview

`run_databench_3questions.py` is a launch script that runs the DataBench experiment on 3 sample questions using the DS-Star agent.

## Usage

### Basic Usage

```bash
python scripts/run_databench_3questions.py
```

Or make it executable and run directly:

```bash
chmod +x scripts/run_databench_3questions.py
./scripts/run_databench_3questions.py
```

### Prerequisites

1. **Environment Setup**: Ensure you have a `.env` file with valid API credentials:
   ```
   WATSONX_API_KEY=your_key_here
   WATSONX_PROJECT_ID=your_project_id_here
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install -e .
   ```

3. **Internet Connection**: Required for:
   - Model API access (WatsonX)
   - HuggingFace dataset downloads (first run only)

## Configuration

The script is configured with the following defaults:

- **Agent Type**: DS-Star (multi-step reasoning agent)
- **Model**: `watsonx/mistralai/mistral-medium-2505`
- **Embedding Model**: `ibm-granite/granite-embedding-english-r2`
- **Max Steps**: 10 reasoning steps
- **Questions**: 3 (from DataBench train split)
- **Code Mode**: `stepwise` (efficient execution)
- **Seed**: 43 (for reproducibility)

## Output

Results are saved to:
- **Outputs**: `experiments/databench/outputs/databench_3questions/`
- **Cache**: `experiments/databench/cache/`

The script prints:
- Configuration summary
- Progress during execution
- Final results with scores
- Pass/fail statistics

## Customization

To modify the configuration, edit the `DataBenchExperiment` parameters in the script:

```python
experiment = DataBenchExperiment(
    qa_split="train",
    model_agent="watsonx/mistralai/mistral-medium-2505",  # Change model
    max_steps=10,  # Adjust reasoning steps
    question_limit=3,  # Change number of questions
    # ... other parameters
)
```

## Expected Runtime

- **First run**: ~2-3 minutes (includes dataset download)
- **Subsequent runs**: ~1-2 minutes (uses cached data)

## Troubleshooting

### Import Errors
If you see import errors, ensure you're running from the project root:
```bash
cd /path/to/OpenDsStar
python scripts/run_databench_3questions.py
```

### API Errors
Check your `.env` file has valid credentials:
```bash
cat .env | grep WATSONX
```

### Dataset Download Issues
If HuggingFace download fails, check your internet connection and try again.

## Related Files

- **Test File**: `tests/e2e/test_databench_e2e.py` - E2E test with similar setup
- **Experiment**: `src/OpenDsStar/experiments/benchmarks/databench/databench_main.py`
- **Data Loader**: `src/OpenDsStar/experiments/demo/databench_loader.py`
