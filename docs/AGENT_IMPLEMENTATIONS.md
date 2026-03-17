# Agent Implementations

This document describes the different agent implementations available in OpenDsStar.

## Available Agents

### 1. OpenDsStarAgent

The main DS-Star agent with multi-step planning, code generation, and verification.

**Usage:**
```python
from agents import OpenDsStarAgent

agent = OpenDsStarAgent(
    model="gpt-4o-mini",
    temperature=0.0,
    tools=[],  # List of LangChain tools
    max_steps=5,
    code_mode="stepwise"  # or "full"
)

result = agent.invoke("What is the capital of France?")
print(result["answer"])
```

**Features:**
- Multi-step planning with planner, coder, executor, verifier, and router nodes
- Stepwise or full code execution modes
- Automatic error recovery with debugger
- Result verification

See [DS_STAR_AGENT.md](DS_STAR_AGENT.md) for detailed documentation.

---

### 2. ReactAgentLangchain

A ReAct-style agent using LangChain's agent framework.

**Usage:**
```python
from agents import ReactAgentLangchain

agent = ReactAgentLangchain(
    model="gpt-4o-mini",
    temperature=0.0,
    tools=[],  # List of LangChain tools
    max_steps=5
)

result = agent.invoke("What is the capital of France?")
print(result["answer"])
```

**Features:**
- Uses LangChain's built-in ReAct agent
- Supports all LangChain tools
- Follows ReAct reasoning pattern (Reason + Act)

---

### 3. CodeActAgentSmolagents

A code-based agent using smolagents' `CodeAgent`.

**Usage:**
```python
from agents import CodeActAgentSmolagents

agent = CodeActAgentSmolagents(
    model="gpt-4o-mini",
    temperature=0.0,
    tools=[],  # List of tools (LangChain or smolagents format)
    max_steps=5
)

result = agent.invoke("Calculate the sum of numbers from 1 to 100")
print(result["answer"])
```

**Features:**
- Generates and executes Python code to solve tasks
- Uses smolagents' CodeAgent implementation
- Good for computational and data analysis tasks
- Can import numpy, pandas, matplotlib by default

---

### 4. ReactAgentSmolagents

A ReAct-style agent using smolagents' `ToolCallingAgent`.

**Usage:**
```python
from agents import ReactAgentSmolagents

agent = ReactAgentSmolagents(
    model="gpt-4o-mini",
    temperature=0.0,
    tools=[],  # List of tools (LangChain or smolagents format)
    max_steps=5
)

result = agent.invoke("Search for information about Python")
print(result["answer"])
```

**Features:**
- Uses smolagents' ToolCallingAgent
- Follows ReAct reasoning pattern
- Alternative to ReactAgentLangchain with smolagents backend

---

## Common Interface

All agents implement the same `BaseAgent` interface:

```python
class BaseAgent(ABC):
    def __init__(
        self,
        model: str | BaseChatModel = "gpt-4o-mini",
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        task_prompt: str | None = None,
        max_steps: int = 5,
        code_timeout: int = 30,
        code_mode: str = "stepwise",
    ) -> None:
        ...

    def invoke(
        self,
        query: str,
        config: dict[str, Any] | None = None,
        return_state: bool = False,
    ) -> dict[str, Any]:
        ...

    @property
    def model_id(self) -> str:
        ...
```

### Return Format

All agents return a dictionary with the following keys:

```python
{
    "answer": str,              # Final answer to the query
    "trajectory": list,         # List of reasoning steps
    "plan": str,                # Execution plan (if applicable)
    "steps_used": int,          # Number of steps taken
    "max_steps": int,           # Maximum steps allowed
    "verifier_sufficient": bool,# Whether answer is sufficient
    "fatal_error": str,         # Fatal errors (empty if none)
    "execution_error": str,     # Execution errors (empty if none)
    "input_tokens": int,        # Input tokens used
    "output_tokens": int,       # Output tokens used
    "num_llm_calls": int,       # Number of LLM calls made
}
```

---

## Choosing an Agent

- **OpenDsStarAgent**: Best for complex data science workflows requiring multi-step planning, code generation, and verification
- **ReactAgentLangchain**: Best for general-purpose tasks with LangChain tools
- **CodeActAgentSmolagents**: Best for computational tasks requiring code execution
- **ReactAgentSmolagents**: Alternative ReAct implementation using smolagents

---

## Installation

Make sure to install the required dependencies:

```bash
pip install -e .
```

This will install both LangChain and smolagents dependencies.

---

## Example: Comparing Agents

```python
from agents import ReactAgentLangchain, CodeActAgentSmolagents, ReactAgentSmolagents

query = "What is 15 factorial?"

# Try with LangChain ReAct
agent1 = ReactAgentLangchain(model="gpt-4o-mini")
result1 = agent1.invoke(query)
print(f"ReactAgentLangchain: {result1['answer']}")

# Try with smolagents CodeAgent
agent2 = CodeActAgentSmolagents(model="gpt-4o-mini")
result2 = agent2.invoke(query)
print(f"CodeActAgentSmolagents: {result2['answer']}")

# Try with smolagents ToolCallingAgent
agent3 = ReactAgentSmolagents(model="gpt-4o-mini")
result3 = agent3.invoke(query)
print(f"ReactAgentSmolagents: {result3['answer']}")
