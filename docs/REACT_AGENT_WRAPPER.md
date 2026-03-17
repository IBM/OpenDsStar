# ReactAgentLangchain Documentation

## Overview

The `ReactAgentLangchain` is a wrapper around the [LangChain React Agent](https://github.com/langchain-ai/react-agent) that integrates it seamlessly with the OpenDsStar framework. This wrapper provides a consistent interface while leveraging the React Agent's reasoning and action capabilities.

## What is ReAct?

ReAct (Reasoning and Acting) is an agent paradigm that combines reasoning traces and task-specific actions in an interleaved manner. The agent:

1. **Reasons** about the user's query
2. **Acts** by calling tools to gather information
3. **Observes** the results
4. Repeats steps 1-3 until it can provide a final answer

This approach allows the agent to dynamically decide which tools to use and when, making it highly flexible and adaptable to various tasks.

## Installation

The ReactAgent is included in OpenDsStar. No additional dependencies are required beyond the standard OpenDsStar installation:

```bash
pip install opendsstar
```

## Quick Start

```python
from dotenv import load_dotenv
from agents import ReactAgentLangchain

# Load environment variables (for API keys)
load_dotenv()

# Create the agent
agent = ReactAgentLangchain()

# Run a query
result = agent.invoke("What is 15 * 23 + 42?")
print(result['answer'])
```

## API Reference

### ReactAgentLangchain

The main wrapper class for the LangChain React Agent.

#### Parameters

- `model` (str or BaseChatModel): Either a model ID string (e.g., "watsonx/mistralai/mistral-medium-2505") or a LangChain chat model instance
- `temperature` (float): Temperature for generation, 0.0 = deterministic (default: 0.0)
- `tools` (List, optional): List of LangChain tools the agent can use
- `system_prompt` (str, optional): Custom system prompt for the agent
- `max_iterations` (int): Maximum number of iterations the agent can take (default: 25)

#### Methods

**`invoke(query, config=None, return_state=False)`**

Execute the agent with a query.

- `query` (str): The question or task to solve
- `config` (dict, optional): LangGraph configuration
- `return_state` (bool): Return full state if True (default: False)

Returns a dictionary with:
- `answer`: Final answer to the query
- `messages`: List of all messages in the conversation
- `trajectory`: List of events showing agent's reasoning process
- `num_iterations`: Number of iterations used

## Examples

### Basic Usage

```python
from agents import ReactAgentLangchain

agent = ReactAgentLangchain()
result = agent.invoke("What is the capital of France?")
print(result['answer'])
```

### Using a Specific Model

```python
agent = ReactAgentLangchain(
    model="gpt-4o",  # or "anthropic/claude-3-sonnet-20240229"
    temperature=0.0,
)
result = agent.invoke("Explain quantum computing in simple terms")
```

### Using Custom Tools

```python
from langchain_core.tools import tool
from agents import ReactAgentLangchain

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

agent = ReactAgentLangchain(
    model="watsonx/mistralai/mistral-medium-2505",
    tools=[multiply, add],
    system_prompt="You are a helpful math assistant.",
)

result = agent.invoke("What is (5 * 3) + 7?")
print(result['answer'])
```

### Accessing the Trajectory

The trajectory shows the agent's step-by-step reasoning and actions:

```python
agent = ReactAgentLangchain()
result = agent.invoke("What is the square root of 144?")

print(f"Answer: {result['answer']}")
print(f"\nTrajectory:")
for i, step in enumerate(result['trajectory'], 1):
    print(f"Step {i}: {step['type']}")
    if step['type'] == 'tool_call':
        print(f"  Tool calls: {step['tool_calls']}")
```

### Custom System Prompt

```python
agent = ReactAgentLangchain(
    system_prompt="""You are a data science expert assistant.
    You help users analyze data and solve data science problems.
    Always explain your reasoning clearly."""
)

result = agent.invoke("How would I analyze customer churn data?")
```

## Comparison with OpenDsStarAgent

| Feature | ReactAgentLangchain | OpenDsStarAgent |
|---------|---------------------|-----------------|
| **Origin** | LangChain React Agent | DS-Star paper implementation |
| **Approach** | Simple ReAct loop | Multi-stage planning & execution |
| **Planning** | Implicit (per-step) | Explicit multi-step plans |
| **Code Generation** | Via tools | Built-in coder node |
| **Execution Modes** | Single mode | Stepwise & Full modes |
| **Complexity** | Simpler, more flexible | More structured, data science focused |
| **Best For** | General tasks, quick prototyping | Complex data science workflows |

## When to Use ReactAgentLangchain

Use `ReactAgentLangchain` when:

- You need a simple, flexible agent for general tasks
- You want to quickly prototype with custom tools
- Your tasks don't require complex multi-step planning
- You prefer the standard ReAct pattern
- You want to leverage existing LangChain React Agent patterns

Use `OpenDsStarAgent` when:

- You need explicit multi-step planning
- You're working on complex data science workflows
- You need stepwise execution with state management
- You want built-in code generation and debugging capabilities

## Configuration

### Environment Variables

Set your API keys in a `.env` file:

```bash
# For OpenAI models
OPENAI_API_KEY=your-openai-key

# For Anthropic models
ANTHROPIC_API_KEY=your-anthropic-key

# For other providers via LiteLLM
# See: https://docs.litellm.ai/docs/providers
```

### LangGraph Configuration

You can pass custom LangGraph configuration:

```python
config = {
    "configurable": {"thread_id": "my_thread"},
    "recursion_limit": 50,
}

result = agent.invoke("Your query here", config=config)
```

## Advanced Usage

### Combining with OpenDsStar Tools

You can use OpenDsStar's tools with the ReactAgentLangchain:

```python
from agents import ReactAgentLangchain
from tools import VectorStoreTool

# Create a vector store tool
vector_tool = VectorStoreTool(
    corpus=your_documents,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)

# Use it with the React Agent
agent = ReactAgentLangchain(
    tools=[vector_tool],
    system_prompt="You are a helpful assistant with access to a document database.",
)

result = agent.invoke("What does the documentation say about X?")
```

### Error Handling

```python
try:
    result = agent.invoke("Your query")
    print(result['answer'])
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"Error during execution: {e}")
```

## Limitations

- The agent will stop after `max_iterations` even if it hasn't found an answer
- Tool selection is done by the LLM, which may not always be optimal
- No built-in code execution environment (use tools for that)
- Simpler than OpenDsStarAgent for complex multi-step workflows

## Contributing

To extend the ReactAgent:

1. Add new tools in your application code
2. Customize the system prompt for your use case
3. Adjust `max_iterations` based on task complexity
4. Use LangGraph configuration for advanced control

## References

- [LangChain React Agent Repository](https://github.com/langchain-ai/react-agent)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenDsStar Documentation](../README.md)

## License

Apache License 2.0 (same as OpenDsStar)
