# Getting Started with OpenDsStar

Welcome! This guide will get you up and running with OpenDsStar in 5 minutes.

## What is OpenDsStar?

OpenDsStar is an open-source implementation of the DS-Star agent - a sophisticated AI agent that can break down complex tasks into steps, write code, execute it, and verify results. Think of it as an AI assistant that can plan, code, and reason through problems.

## Quick Install

### 1. Prerequisites
- Python 3.11 or 3.12
- An API key for your preferred LLM provider (OpenAI, Anthropic, etc.)

### 2. Install
```bash
# Clone the repository
git clone https://github.com/IBM/OpenDsStar.git
cd OpenDsStar

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### 3. Configure API Keys
Create a `.env` file in the project root:
```bash
# For OpenAI (recommended for getting started)
OPENAI_API_KEY=your_key_here

# Or for other providers - see docs/INSTALLATION.md
```

## Your First Agent

### Simple Question Answering

```python
from dotenv import load_dotenv
from agents import OpenDsStarAgent

load_dotenv()

# Create the agent
agent = OpenDsStarAgent(model="gpt-4o-mini")

# Ask a question
result = agent.invoke("What is 15 * 23 + 42?")
print(result["answer"])
# Output: 387
```

### With Custom Tools

```python
from langchain_core.tools import tool
from agents import OpenDsStarAgent

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    # Your weather API logic here
    return f"The weather in {city} is sunny"

agent = OpenDsStarAgent(
    model="gpt-4o-mini",
    tools=[get_weather]
)

result = agent.invoke("What's the weather in Paris?")
print(result["answer"])
```

## Understanding the Output

Every agent returns a dictionary with useful information:

```python
result = agent.invoke("Calculate 2 + 2")

print(result["answer"])           # "4"
print(result["steps_used"])       # Number of reasoning steps
print(result["input_tokens"])     # Tokens used (for cost tracking)
print(result["trajectory"])       # Step-by-step reasoning process
```

## Choosing an Agent

OpenDsStar includes multiple agent types:

| Agent | Best For | Complexity |
|-------|----------|------------|
| **OpenDsStarAgent** | Complex multi-step tasks, data science | High |
| **ReactAgentLangchain** | General tasks, quick prototyping | Low |
| **CodeActAgentSmolagents** | Computational tasks requiring code | Medium |

**Start with OpenDsStarAgent** - it's the most capable and handles a wide range of tasks.

## Next Steps

### Learn More About Agents
- [DS-Star Agent Guide](DS_STAR_AGENT.md) - Detailed documentation
- [React Agent Guide](REACT_AGENT_WRAPPER.md) - Simpler alternative
- [Agent Comparison](AGENT_IMPLEMENTATIONS.md) - Choose the right agent

### Run Experiments
- [Installation Guide](INSTALLATION.md) - Complete setup instructions
- [Benchmarks Guide](BENCHMARKS.md) - Run evaluation benchmarks
- [Experiments Guide](EXPERIMENTS.md) - Advanced experiment configuration

### Advanced Features
- [Model Providers](MODEL_PROVIDERS.md) - Use different LLM providers
- [MCP Integration](DS_STAR_MCP_INTEGRATION.md) - Connect to MCP tools

## Common Issues

### "API key not found"
Make sure your `.env` file is in the project root and contains the correct key for your model provider.

### "Module not found"
Activate your virtual environment: `source .venv/bin/activate`

### "Model not supported"
Check [Model Providers](MODEL_PROVIDERS.md) for supported models and configuration.

## Getting Help

- Check the [FAQ](FAQ.md) for common questions
- Review the [Installation Guide](INSTALLATION.md) for detailed setup
- See example code in the `examples/` directory

## What's Next?

Now that you have OpenDsStar running, try:
1. **Experiment with different models** - Compare GPT-4, Claude, or local models
2. **Add custom tools** - Extend the agent with your own functions
3. **Run benchmarks** - See how agents perform on standard datasets
4. **Build applications** - Integrate OpenDsStar into your projects

Happy coding! 🚀
