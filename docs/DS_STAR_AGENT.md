# DS_Star Agent Documentation

## Overview

The **OpenDsStarAgent** is the core implementation of the DS-Star agent architecture in OpenDsStar. It provides a sophisticated multi-step reasoning system that plans, codes, executes, debugs, verifies, and routes through complex tasks using a tool-centric approach.

## Architecture

The DS_Star agent is built on a **LangGraph state machine** with the following nodes:

```
┌─────────┐
│ Planner │ ──> Plans next step
└────┬────┘
     │
     ▼
┌─────────┐
│  Coder  │ ──> Generates code for the step
└────┬────┘
     │
     ▼
┌──────────┐
│ Executor │ ──> Executes the code
└────┬─────┘
     │
     ├──> (on error) ──┐
     │                 ▼
     │            ┌──────────┐
     │            │ Debugger │ ──> Fixes code and retries
     │            └────┬─────┘
     │                 │
     │                 └──> (back to Executor)
     │
     └──> (on success)
          │
          ▼
     ┌──────────┐
     │ Verifier │ ──> Checks if answer is sufficient
     └────┬─────┘
          │
          ├──> (sufficient) ──> Finalizer ──> END
          │
          └──> (insufficient)
               │
               ▼
          ┌────────┐
          │ Router │ ──> Decides next action (add_next_step or fix_step)
          └────┬───┘
               │
               └──> (back to Planner)
```

## Key Features

### 1. **Tool-Centric Planning**
Plans are expressed as sequences of tool invocations rather than direct file manipulation, making the agent general-purpose and extensible.

### 2. **Execution Modes**

#### Stepwise Mode (Default)
- Executes one step at a time
- Reuses outputs from previous steps
- More efficient for expensive operations
- Recommended for most use cases

#### Full Mode
- Executes the entire plan end-to-end
- Mirrors the original DS-Star behavior
- Useful for workflows where steps are interdependent

### 3. **Automatic Error Recovery**
- Failed steps are automatically debugged
- Up to 3 debug attempts per step
- Debugger analyzes errors and generates fixes

### 4. **Result Verification**
- Each step's output is verified for sufficiency
- Agent continues planning if more information is needed
- Ensures high-quality final answers

## Installation

```bash
pip install opendsstar
```

For local development:

```bash
git clone https://github.com/IBM/OpenDsStar.git
cd OpenDsStar
pip install -e .
```

## Quick Start

### Basic Usage

```python
from dotenv import load_dotenv
from agents import OpenDsStarAgent

load_dotenv()

# Create agent with default settings
agent = OpenDsStarAgent(
    model="gpt-4o-mini",  # or any LiteLLM-supported model
)

# Run a query
result = agent.invoke("What is 15 * 23 + 42?")
print(result["answer"])
```

### Generating Plots

The agent can build and render Plotly figures without accessing the
filesystem.  In a Python step simply create the figure object (using the
preloaded `px`/`go` modules) and store it in the outputs dict; e.g.:```
fig = px.line(df, x="time", y="value")
outputs["figure"] = fig
```
When the final answer is produced the system will convert the figure to a
PNG data‑URI and embed it as an image block in the chat.

No special tools are required for this – just save the figure and the
finalizer handles the rest.

### With Custom Tools

```python
from langchain_core.tools import tool
from agents import OpenDsStarAgent

@tool
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width

agent = OpenDsStarAgent(
    model="gpt-4o-mini",
    tools=[calculate_area],
)

result = agent.invoke("What is the area of a rectangle with length 5 and width 3?")
print(result["answer"])
```

## Initialization

### Constructor Parameters

```python
OpenDsStarAgent(
    model: str | BaseChatModel,
    temperature: float = 0.0,
    tools: list[any] | None = None,
    system_prompt: str | None = None,
    task_prompt: str | None = None,
    max_steps: int = 5,
    code_timeout: int = 30,
    code_mode: str = "stepwise",
)
```

#### Parameters

- **`model`** (str or BaseChatModel, required)
  - Model identifier string (e.g., `"gpt-4o-mini"`, `"claude-3-sonnet-20240229"`, `"watsonx/mistralai/mistral-medium-2505"`)
  - Or a LangChain `BaseChatModel` instance for custom models
  - Supports any model available through [LiteLLM](https://docs.litellm.ai/docs/providers)

- **`temperature`** (float, default: 0.0)
  - Controls randomness in generation
  - Range: 0.0 (deterministic) to 1.0 (creative)
  - Lower values recommended for reasoning tasks

- **`tools`** (list, optional)
  - List of LangChain tools available to the agent
  - Can be custom tools decorated with `@tool` or built-in LangChain tools
  - Defaults to empty list (agent can still reason without tools)

- **`system_prompt`** (str, optional)
  - Custom system prompt to guide agent behavior
  - Defaults to: `"You are a helpful data science assistant. You can break down complex queries into steps, write code, and provide answers."`
  - Use to customize agent personality or domain expertise

- **`task_prompt`** (str, optional)
  - Task-specific prompt for additional context
  - Useful for domain-specific instructions

- **`max_steps`** (int, default: 5)
  - Maximum number of planning steps allowed
  - Prevents infinite loops
  - Increase for complex multi-step tasks

- **`code_timeout`** (int, default: 30)
  - Timeout in seconds for code execution
  - Prevents hanging on long-running operations
  - Adjust based on expected computation time

- **`code_mode`** (str, default: "stepwise")
  - Execution strategy: `"stepwise"` or `"full"`
  - **stepwise**: Execute one step at a time, reuse outputs (recommended)
  - **full**: Execute entire plan end-to-end (original DS-Star behavior)

### Example Configurations

#### For Data Science Tasks

```python
agent = OpenDsStarAgent(
    model="gpt-4o",
    temperature=0.0,
    max_steps=10,
    code_timeout=60,
    system_prompt="You are an expert data scientist specializing in statistical analysis and machine learning.",
)
```

#### For Quick Calculations

```python
agent = OpenDsStarAgent(
    model="gpt-4o-mini",
    temperature=0.0,
    max_steps=3,
    code_timeout=10,
    code_mode="full",
)
```

#### With Custom Model

```python
from langchain_openai import ChatOpenAI

custom_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key="your-api-key",
)

agent = OpenDsStarAgent(
    model=custom_model,
    tools=[...],
)
```

## Methods

### `invoke(query, config=None, return_state=False)`

Execute the agent with a given query.

#### Parameters

- **`query`** (str, required)
  - The user's question or task to solve
  - Must be a non-empty string

- **`config`** (dict, optional)
  - LangGraph configuration dictionary
  - Default: `{"configurable": {"thread_id": "default"}, "recursion_limit": 1000}`
  - Use for custom thread management or recursion limits

- **`return_state`** (bool, default: False)
  - If `True`, returns the full `DSState` object with all internal details
  - If `False`, returns a cleaned result dictionary (recommended for most use cases)

#### Returns

Dictionary containing:

```python
{
    "answer": str,              # Final answer to the query
    "trajectory": list[dict],   # List of events showing reasoning process
    "plan": str,                # String representation of the execution plan
    "steps_used": int,          # Number of steps actually used
    "max_steps": int,           # Maximum steps allowed
    "verifier_sufficient": bool, # Whether answer was deemed sufficient
    "fatal_error": str,         # Any fatal errors (empty if none)
    "execution_error": str,     # Execution errors in last step (empty if none)
    "input_tokens": int,        # Total input tokens used
    "output_tokens": int,       # Total output tokens used
    "num_llm_calls": int,       # Number of LLM API calls made
}
```

#### Examples

**Basic usage:**

```python
result = agent.invoke("What is the square root of 144?")
print(result["answer"])  # "12"
print(f"Used {result['steps_used']} steps")
print(f"Made {result['num_llm_calls']} LLM calls")
```

**With custom config:**

```python
result = agent.invoke(
    query="Analyze this dataset...",
    config={
        "configurable": {"thread_id": "analysis_123"},
        "recursion_limit": 2000,
    }
)
```

**Getting full state:**

```python
state = agent.invoke(
    query="Complex task...",
    return_state=True,
)
# Access internal state
print(state.steps)
print(state.trajectory)
```

**Error handling:**

```python
try:
    result = agent.invoke("What is the capital of France?")
    if result["fatal_error"]:
        print(f"Fatal error: {result['fatal_error']}")
    elif result["execution_error"]:
        print(f"Execution error: {result['execution_error']}")
    else:
        print(f"Answer: {result['answer']}")
except Exception as e:
    print(f"Invocation failed: {e}")
```

### `model_id` (property)

Get the model identifier string.

```python
agent = OpenDsStarAgent(model="gpt-4o-mini")
print(agent.model_id)  # "gpt-4o-mini"
```

## State Management

### DSState

The internal state object that flows through the graph:

```python
@dataclass
class DSState:
    user_query: str                          # Original user query
    tools: Dict[str, Callable]               # Available tools
    steps: List[DSStep]                      # List of execution steps
    final_answer: Optional[str]              # Final answer
    fatal_error: Optional[str]               # Fatal errors
    steps_used: int                          # Steps used so far
    max_steps: int                           # Maximum allowed steps
    trajectory: List[Dict[str, Any]]         # Event log
    token_usage: List[Dict[str, Any]]        # Token usage per LLM call
    code_mode: CodeMode                      # Execution mode
```

### DSStep

Each step in the plan:

```python
@dataclass
class DSStep:
    plan: str                                # Step description
    code: Optional[str]                      # Generated code
    logs: Optional[str]                      # Execution logs
    outputs: Optional[Dict[str, Any]]        # Step outputs
    execution_error: Optional[str]           # Execution errors
    verifier_sufficient: Optional[bool]      # Verification result
    verifier_explanation: Optional[str]      # Verification reasoning
    router_action: Optional[str]             # Router decision
    router_fix_index: Optional[int]          # Step to fix (if any)
    router_explanation: Optional[str]        # Router reasoning
    debug_tries: int                         # Number of debug attempts
```

## Advanced Usage

### Custom System Prompts

```python
agent = OpenDsStarAgent(
    model="gpt-4o",
    system_prompt="""You are a financial analyst AI assistant.
    You specialize in analyzing market data, calculating financial metrics,
    and providing investment insights. Always show your calculations step by step.""",
)
```

### Working with Multiple Tools

```python
from langchain_core.tools import tool

@tool
def fetch_stock_price(symbol: str) -> float:
    """Fetch current stock price for a given symbol."""
    # Implementation here
    pass

@tool
def calculate_roi(initial: float, final: float) -> float:
    """Calculate return on investment."""
    return ((final - initial) / initial) * 100

agent = OpenDsStarAgent(
    model="gpt-4o",
    tools=[fetch_stock_price, calculate_roi],
    max_steps=10,
)

result = agent.invoke("What's the ROI if I bought AAPL at $150 and it's now $180?")
```

### Analyzing the Trajectory

```python
result = agent.invoke("Complex multi-step task...")

# Examine the reasoning process
for event in result["trajectory"]:
    print(f"{event['node']}: {event.get('note', '')}")

# Check token usage
print(f"Total tokens: {result['input_tokens'] + result['output_tokens']}")
print(f"Cost estimate: ${(result['input_tokens'] * 0.01 + result['output_tokens'] * 0.03) / 1000}")
```

### Handling Long-Running Tasks

```python
agent = OpenDsStarAgent(
    model="gpt-4o",
    max_steps=20,           # Allow more steps
    code_timeout=120,       # 2 minute timeout
    code_mode="stepwise",   # Reuse intermediate results
)

result = agent.invoke("Perform comprehensive data analysis on large dataset...")
```

## Best Practices

### 1. **Choose the Right Execution Mode**

- Use **stepwise** (default) for:
  - Tasks with expensive computations
  - Long-running operations
  - When intermediate results should be cached

- Use **full** for:
  - Simple calculations
  - When steps are tightly coupled
  - Debugging (easier to see full execution)

### 2. **Set Appropriate Limits**

```python
# For simple tasks
agent = OpenDsStarAgent(model="gpt-4o-mini", max_steps=3, code_timeout=10)

# For complex tasks
agent = OpenDsStarAgent(model="gpt-4o", max_steps=15, code_timeout=60)
```

### 3. **Provide Clear Tool Descriptions**

```python
@tool
def analyze_sentiment(text: str) -> str:
    """
    Analyze the sentiment of the given text.

    Args:
        text: The text to analyze (must be non-empty)

    Returns:
        One of: "positive", "negative", or "neutral"
    """
    # Implementation
    pass
```

### 4. **Handle Errors Gracefully**

```python
result = agent.invoke(query)

if result["fatal_error"]:
    # Max steps reached or critical failure
    print(f"Task failed: {result['fatal_error']}")
elif result["execution_error"]:
    # Last step had an error but agent continued
    print(f"Warning: {result['execution_error']}")
else:
    # Success
    print(f"Answer: {result['answer']}")
```

## Troubleshooting

### Agent Reaches Max Steps

**Problem**: Agent hits the step limit before completing the task.

**Solutions**:
- Increase `max_steps` parameter
- Simplify the query
- Provide more specific instructions
- Add relevant tools to reduce reasoning steps

### Code Execution Timeouts

**Problem**: Code execution exceeds the timeout limit.

**Solutions**:
- Increase `code_timeout` parameter
- Optimize tool implementations
- Break down complex operations into smaller tools

### Poor Answer Quality

**Problem**: Agent provides incomplete or incorrect answers.

**Solutions**:
- Use a more capable model (e.g., `gpt-4o` instead of `gpt-4o-mini`)
- Provide a more detailed `system_prompt`
- Add domain-specific tools
- Increase `max_steps` to allow more reasoning


## Integration Examples

### With LangChain Tools

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

agent = OpenDsStarAgent(
    model="gpt-4o",
    tools=[wikipedia],
)

result = agent.invoke("What is the population of Tokyo?")
```

### With Custom Retrievers

```python
from tools import VectorStoreTool

retriever = VectorStoreTool(
    corpus=documents,
    embedding_model="text-embedding-3-small",
)

agent = OpenDsStarAgent(
    model="gpt-4o",
    tools=[retriever],
)

result = agent.invoke("What does the document say about climate change?")
```

## Related Documentation

- [Main README](../README.md) - Project overview and quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture details
- [EXPERIMENTS.md](EXPERIMENTS.md) - Running experiments and benchmarks
- [REACT_AGENT_WRAPPER.md](REACT_AGENT_WRAPPER.md) - Alternative agent implementation

## Contributing

When contributing to the DS_Star agent:

1. Maintain backward compatibility with the existing API
2. Add tests for new features in `tests/agents/ds_star/`
3. Update this documentation for any API changes
4. Follow the existing code style and patterns

## License

Apache License 2.0
