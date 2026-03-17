# Running Benchmarks

This guide shows you how to run evaluation benchmarks with OpenDsStar agents.

## Available Benchmarks

OpenDsStar includes three main benchmarks:

| Benchmark | Description | Questions | Focus Area |
|-----------|-------------|-----------|------------|
| **DataBench** | Data analysis tasks with CSV files | 70 | Data science, analysis |
| **HotpotQA** | Multi-hop question answering | 7,405 | Information retrieval |
| **KramaBench** | Complex data analysis | 31 | Advanced reasoning |

## Quick Start

### DataBench (Recommended for Testing)

Run a quick test with 5 questions:

```bash
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --question-limit 5 \
  --agent-type ds_star \
  --model-agent gpt-4o-mini
```

### HotpotQA

```bash
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main \
  --question-limit 20 \
  --agent-type ds_star \
  --model gpt-4o-mini
```

### KramaBench

```bash
.venv/bin/python -m src.experiments.benchmarks.kramabench.kramabench_main \
  --agent-type ds_star \
  --model-agent gpt-4o-mini
```

## Common Options

### Agent Selection

Choose which agent to use:

```bash
--agent-type ds_star              # DS-Star agent (recommended)
--agent-type react_langchain      # React agent
--agent-type codeact_smolagents   # CodeAct agent
```

### Model Selection

Specify the LLM model:

```bash
--model-agent gpt-4o-mini         # OpenAI GPT-4o mini
--model-agent gpt-4o              # OpenAI GPT-4o
--model-agent wx_llama_maverick   # WatsonX Llama
--model-agent custom_prefix/...   # Custom API provider (configure via env vars)
```

See [Model Providers](MODEL_PROVIDERS.md) for all available models.

### Question Limits

Control how many questions to run:

```bash
--question-limit 5     # Run only 5 questions (for testing)
--question-limit 20    # Run 20 questions
# Omit to run all questions
```

### Execution Control

```bash
--max-steps 10         # Maximum reasoning steps (default: 5)
--code-mode stepwise   # Stepwise execution (default, efficient)
--code-mode full       # Full re-execution (original DS-Star)
--parallel-workers 4   # Run 4 questions in parallel
```

## Understanding Results

### Output Files

Each run creates three files in the benchmark's `output/` directory:

```
result_<agent>_<model>_<benchmark>_<limit>_<timestamp>_output.json
result_<agent>_<model>_<benchmark>_<limit>_<timestamp>_params.json
result_<agent>_<model>_<benchmark>_<limit>_<timestamp>_log.txt
```

Example:
```
result_ds_star_gpt-4o-mini_databench_5_20260312_143022_output.json
result_ds_star_gpt-4o-mini_databench_5_20260312_143022_params.json
result_ds_star_gpt-4o-mini_databench_5_20260312_143022_log.txt
```

### Output File Structure

The `*_output.json` file contains:

```json
{
  "run_id": "databench_20260312_143022",
  "agent_type": "ds_star",
  "summary": {
    "total_questions": 5,
    "passed": 4,
    "failed": 1,
    "avg_score": 0.85,
    "total_tokens": 12500,
    "total_llm_calls": 45
  },
  "items": [
    {
      "question_id": "001_Forbes_q1",
      "output": {
        "answer": "The average revenue is $2.5B",
        "metadata": {
          "input_tokens": 2500,
          "output_tokens": 150,
          "num_llm_calls": 8
        }
      },
      "evaluations": [
        {
          "score": 1.0,
          "passed": true
        }
      ]
    }
  ]
}
```

### Params File

The `*_params.json` file contains all configuration used, enabling exact reproduction:

```json
{
  "type": "experiments.benchmarks.databench.databench_main.DataBenchExperiment",
  "args": {
    "agent_type": "ds_star",
    "model": "gpt-4o-mini",
    "max_steps": 5,
    "question_limit": 5
  }
}
```

## Analyzing Results

### Using the Analysis Script

Analyze results across multiple runs:

```bash
.venv/bin/python scripts/analyze_experiment_results.py \
  src/experiments/benchmarks/databench/output/
```

Output:
```
┌─────────────────────┬────────┬───────────┬──────────────┬──────────────┬──────────────┐
│ Subdirectory        │ Files  │ Questions │ Avg Score    │ Total Tokens │ LLM Calls    │
├─────────────────────┼────────┼───────────┼──────────────┼──────────────┼──────────────┤
│ ds_star_gpt4o       │ 1      │ 5         │ 0.85         │ 12,500       │ 45           │
│ react_gpt4o_mini    │ 1      │ 5         │ 0.72         │ 8,200        │ 32           │
└─────────────────────┴────────┴───────────┴──────────────┴──────────────┴──────────────┘
```

### Key Metrics

- **Score**: 0.0 to 1.0, where 1.0 is perfect
- **Passed**: Questions that met the success threshold
- **Total Tokens**: Input + output tokens (for cost estimation)
- **LLM Calls**: Number of API calls made

## Reproducing Results

To reproduce an experiment exactly:

```bash
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --load-params path/to/result_*_params.json
```

This will use the exact same configuration as the original run.

## Comparing Agents

Run the same benchmark with different agents:

```bash
# DS-Star
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --question-limit 10 --agent-type ds_star --model-agent gpt-4o-mini

# React
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --question-limit 10 --agent-type react_langchain --model-agent gpt-4o-mini

# CodeAct
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --question-limit 10 --agent-type codeact_smolagents --model-agent gpt-4o-mini
```

Then compare results using the analysis script.

## Benchmark-Specific Details

### DataBench

- **70 questions** across 10 datasets
- Each question requires analyzing CSV data
- Tests data manipulation, aggregation, and analysis
- Recommended for testing data science capabilities

**Example question**: "What is the average revenue of companies in the Forbes dataset?"

### HotpotQA

- **7,405 questions** in test set
- Multi-hop reasoning required
- Tests information retrieval and reasoning
- Includes supporting documents for each question

**Example question**: "What is the capital of the country where the Eiffel Tower is located?"

### KramaBench

- **31 complex questions**
- Advanced data analysis tasks
- Tests sophisticated reasoning and code generation
- Smaller but more challenging than DataBench

**Example question**: "Calculate the correlation between temperature and sales, then predict next month's sales."

## Performance Tips

### For Faster Testing

```bash
# Use smaller model
--model-agent gpt-4o-mini

# Limit questions
--question-limit 5

# Use parallel execution
--parallel-workers 4
```

### For Best Results

```bash
# Use more capable model
--model-agent gpt-4o

# Allow more reasoning steps
--max-steps 10

# Use stepwise execution
--code-mode stepwise
```

### For Cost Optimization

```bash
# Use efficient model
--model-agent gpt-4o-mini

# Limit steps
--max-steps 5

# Monitor token usage in results
```

## Troubleshooting

### "Out of memory"
- Reduce `--parallel-workers`
- Process fewer questions at once
- Use a smaller model

### "API rate limit exceeded"
- Reduce `--parallel-workers` to 1
- Add delays between questions
- Use a different model provider

### "Questions timing out"
- Increase `--max-steps`
- Check if model is responding
- Review logs in `*_log.txt` file

## Next Steps

- Review [Experiments Guide](EXPERIMENTS.md) for advanced configuration
- See [Model Providers](MODEL_PROVIDERS.md) for using different models
- Check [EXPERIMENT_PARAMS.md](EXPERIMENT_PARAMS.md) for reproducibility details

## Example Workflow

```bash
# 1. Quick test with 5 questions
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --question-limit 5 --agent-type ds_star --model-agent gpt-4o-mini

# 2. Review results
cat src/experiments/benchmarks/databench/output/result_*_output.json | jq '.summary'

# 3. If good, run full benchmark
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --agent-type ds_star --model-agent gpt-4o-mini

# 4. Analyze results
.venv/bin/python scripts/analyze_experiment_results.py \
  src/experiments/benchmarks/databench/output/
```

Happy benchmarking! 📊
