# Token Tracking in OpenDsStar

This document describes how token usage is tracked across different agent implementations.

## Overview

All agent implementations in OpenDsStar track token usage (input tokens, output tokens, and number of LLM calls) and include this information in their output metadata. This enables:

1. Cost analysis and optimization
2. Performance monitoring
3. Experiment result comparison
4. Resource usage reporting

## Agent Implementations

### DS-Star Agent

The DS-Star agent uses LangChain's callback system to capture token usage from each LLM call:

- **Implementation**: `src/agents/ds_star/ds_star_utils.py`
- **Method**: `_UsageCaptureHandler` callback class
- **Storage**: Token usage is accumulated in `state.token_usage` list
- **Extraction**: Summed in `ds_star_results_prep.py`

Token usage is captured for:
- Planner node
- Coder node
- Debugger node
- Verifier node
- Router node
- Finalizer node

### CodeAct Agent (Smolagents)

The CodeAct agent uses smolagents' built-in monitor for token tracking:

- **Implementation**: `src/agents/codeact_smolagents/codeact_agent_smolagents.py`
- **Method**: `agent.monitor.get_total_token_counts()`
- **Source**: Smolagents Monitor class automatically tracks all LLM calls

The monitor provides a `TokenUsage` object with:
- `input_tokens`: Total input/prompt tokens
- `output_tokens`: Total output/completion tokens
- `total_tokens`: Sum of input and output tokens

### React Agent (Smolagents)

The React agent (smolagents) uses the same monitor-based approach:

- **Implementation**: `src/agents/react_smolagents/react_agent_smolagents.py`
- **Method**: `agent.monitor.get_total_token_counts()`
- **Source**: Smolagents Monitor class

Uses the same token tracking mechanism as CodeAct agent.

### React Agent (LangChain)

The React agent (LangChain) extracts token usage from LangChain messages:

- **Implementation**: `src/agents/react_langchain/react_agent_langchain.py`
- **Method**: `_extract_token_usage()` method
- **Source**: Extracts from message metadata

Supports multiple metadata formats:
- `usage_metadata` (newer LangChain)
- `response_metadata.token_usage` (older LangChain/LiteLLM)

## Output Format

All agents return token usage in their invoke() response:

```python
{
    "answer": "...",
    "trajectory": [...],
    "input_tokens": 1234,      # Total input tokens
    "output_tokens": 567,      # Total output tokens
    "num_llm_calls": 5,        # Number of LLM API calls
    # ... other fields
}
```

## Experiment Output Files

Token usage is stored in experiment output files (`*_output.json`):

```json
{
  "items": [
    {
      "question_id": "q1",
      "output": {
        "metadata": {
          "input_tokens": 1234,
          "output_tokens": 567,
          "num_llm_calls": 5
        }
      }
    }
  ]
}
```

## Analysis Scripts

The `scripts/analyze_experiment_results.py` script aggregates token usage across experiments:

```bash
.venv/bin/python scripts/analyze_experiment_results.py /path/to/results/
```

Output includes:
- Total input tokens per subdirectory
- Total output tokens per subdirectory
- Total tokens (input + output)
- Total LLM calls
- Summary statistics across all experiments

## Implementation Notes

### Token Extraction Strategies

Different agent frameworks provide token usage in different formats:

1. **LangChain Callbacks**: DS-Star uses callbacks to intercept LLM responses
2. **Smolagents Monitor**: Smolagents agents use the built-in Monitor class
3. **Message Metadata**: LangChain React agent inspects message metadata

### Fallback Behavior

If token usage information is not available:
- Agents return 0 for input_tokens and output_tokens
- num_llm_calls is estimated from trajectory/steps length
- No errors are raised (graceful degradation)

### Testing

Token tracking is tested in `tests/agents/test_token_tracking.py`:
- Unit tests for each extraction method
- Tests for different metadata formats
- Tests for missing/empty usage data

## Best Practices

1. **Always check token usage**: Review token usage in experiment results to identify inefficiencies
2. **Compare across agents**: Use token metrics to compare agent efficiency
3. **Monitor costs**: Track token usage to estimate API costs
4. **Optimize prompts**: Use token metrics to guide prompt optimization

## Troubleshooting

### Tokens showing as 0

If tokens are showing as 0 in results:

1. **Check LLM provider**: Ensure your LLM provider returns token usage in responses
2. **Check LiteLLM version**: Some older versions may not include usage metadata
3. **Enable verbose logging**: Set log level to DEBUG to see token extraction attempts
4. **Verify trajectory format**: Inspect the trajectory structure to ensure it contains usage data

### Inconsistent token counts

If token counts seem inconsistent:

1. **Check caching**: Cached responses may not include token usage
2. **Verify extraction logic**: Different providers use different field names
3. **Review trajectory**: Inspect the raw trajectory to see what usage data is available

## Future Improvements

Potential enhancements to token tracking:

1. **Real-time monitoring**: Stream token usage during execution
2. **Cost estimation**: Automatically calculate costs based on model pricing
3. **Token budgets**: Implement token limits per question/experiment
4. **Detailed breakdowns**: Track tokens per node/step for fine-grained analysis
