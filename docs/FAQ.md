# Frequently Asked Questions

## General Questions

### What is OpenDsStar?

OpenDsStar is an open-source implementation of the DS-Star agent, a sophisticated AI agent that can plan, code, execute, debug, and verify solutions to complex tasks. It's designed for data science workflows but works for general-purpose tasks too.

### How is it different from other AI agents?

OpenDsStar uses a **tool-centric approach** with explicit multi-step planning:
- Plans tasks as sequences of tool calls
- Executes code incrementally (stepwise mode)
- Automatically debugs and retries failed steps
- Verifies results before returning answers

Most other agents (like ReAct) use simpler reasoning loops without explicit planning or verification.

### Which agent should I use?

- **OpenDsStarAgent**: Complex tasks, data science, multi-step reasoning
- **ReactAgentLangchain**: Simple tasks, quick prototyping, general Q&A
- **CodeActAgentSmolagents**: Computational tasks requiring code execution

Start with **OpenDsStarAgent** - it's the most capable.

## Installation & Setup

### What Python version do I need?

Python 3.11 or 3.12. Check with:
```bash
python --version
```

### How do I install dependencies?

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

### Which API keys do I need?

Only for the LLM providers you plan to use:
- **OpenAI**: `OPENAI_API_KEY` (recommended for getting started)
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Google**: `GEMINI_API_KEY`
- **WatsonX**: `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID`

Add them to a `.env` file in the project root.

### Can I use local models?

Yes! Use Ollama for local models:
```python
agent = OpenDsStarAgent(model="ollama/phi4:latest")
```

No API key needed. Install Ollama from https://ollama.ai

## Usage Questions

### How do I run a simple query?

```python
from dotenv import load_dotenv
from agents import OpenDsStarAgent

load_dotenv()
agent = OpenDsStarAgent(model="gpt-4o-mini")
result = agent.invoke("What is 15 * 23?")
print(result["answer"])
```

### How do I add custom tools?

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Description of what the tool does."""
    return f"Result: {param}"

agent = OpenDsStarAgent(
    model="gpt-4o-mini",
    tools=[my_tool]
)
```

### How do I track token usage?

Token usage is automatically included in results:
```python
result = agent.invoke("Your question")
print(f"Input tokens: {result['input_tokens']}")
print(f"Output tokens: {result['output_tokens']}")
print(f"Total calls: {result['num_llm_calls']}")
```

### How do I see the agent's reasoning process?

Check the trajectory:
```python
result = agent.invoke("Your question")
for step in result["trajectory"]:
    print(f"{step['node']}: {step.get('note', '')}")
```

## Configuration Questions

### What's the difference between stepwise and full mode?

- **Stepwise** (default): Executes one step at a time, reuses results. More efficient.
- **Full**: Re-executes entire plan each iteration. Matches original DS-Star behavior.

```python
agent = OpenDsStarAgent(
    model="gpt-4o-mini",
    code_mode="stepwise"  # or "full"
)
```

### How do I increase the number of reasoning steps?

```python
agent = OpenDsStarAgent(
    model="gpt-4o-mini",
    max_steps=10  # default is 5
)
```

### How do I change the temperature?

```python
agent = OpenDsStarAgent(
    model="gpt-4o-mini",
    temperature=0.7  # default is 0.0 (deterministic)
)
```

### How do I set a custom system prompt?

```python
agent = OpenDsStarAgent(
    model="gpt-4o-mini",
    system_prompt="You are an expert financial analyst..."
)
```

## Benchmark Questions

### How do I run a benchmark?

```bash
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --question-limit 5 \
  --agent-type ds_star \
  --model-agent gpt-4o-mini
```

See [Benchmarks Guide](BENCHMARKS.md) for details.

### Where are results saved?

In the benchmark's `output/` directory:
```
src/experiments/benchmarks/databench/output/
src/experiments/benchmarks/hotpotqa/output/
src/experiments/benchmarks/kramabench/output/
```

### How do I reproduce an experiment?

Use the params file:
```bash
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --load-params path/to/result_*_params.json
```

### How do I compare different agents?

Run the same benchmark with different `--agent-type` values, then use the analysis script:
```bash
.venv/bin/python scripts/analyze_experiment_results.py \
  src/experiments/benchmarks/databench/output/
```

## Troubleshooting

### "API key not found" error

1. Check your `.env` file exists in project root
2. Verify the key name matches your provider (e.g., `OPENAI_API_KEY`)
3. Make sure you called `load_dotenv()` in your code
4. Restart your Python session after adding keys

### "Module not found" error

Activate your virtual environment:
```bash
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### Agent reaches max steps without completing

Increase the step limit:
```python
agent = OpenDsStarAgent(
    model="gpt-4o-mini",
    max_steps=15  # increase from default 5
)
```

Or simplify your query.

### Code execution timeout

Increase the timeout:
```python
agent = OpenDsStarAgent(
    model="gpt-4o-mini",
    code_timeout=60  # default is 30 seconds
)
```

### "Out of memory" during benchmarks

Reduce parallel workers:
```bash
--parallel-workers 1  # or 2, default is sequential
```

### Poor answer quality

Try:
1. Use a more capable model (e.g., `gpt-4o` instead of `gpt-4o-mini`)
2. Increase `max_steps` to allow more reasoning
3. Provide a more detailed system prompt
4. Add relevant tools for your domain

### Results show 0 tokens

Some LLM providers don't return token usage. This is normal for:
- Some local models via Ollama
- Certain API configurations

The agent still works correctly; token tracking just isn't available.

## Model Questions

### Which models are supported?

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers):
- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- Google: `gemini-2.5-flash`, `gemini-2.5-pro`
- Local: `ollama/phi4`, `ollama/llama3.2`
- Many more...

See [Model Providers](MODEL_PROVIDERS.md) for configuration details.

### How do I use a different model?

Just change the model parameter:
```python
agent = OpenDsStarAgent(model="claude-3-5-sonnet-20241022")
```

### Can I use multiple models?

Yes, create multiple agents:
```python
agent1 = OpenDsStarAgent(model="gpt-4o-mini")
agent2 = OpenDsStarAgent(model="claude-3-5-sonnet-20241022")

result1 = agent1.invoke("Your question")
result2 = agent2.invoke("Your question")
```

### How do I estimate costs?

Check token usage in results:
```python
result = agent.invoke("Your question")
input_tokens = result["input_tokens"]
output_tokens = result["output_tokens"]

# Example pricing (check your provider's actual rates)
cost = (input_tokens * 0.01 + output_tokens * 0.03) / 1000
print(f"Estimated cost: ${cost:.4f}")
```

## Advanced Questions

### Can I use MCP tools?

Yes! See [MCP Integration Guide](DS_STAR_MCP_INTEGRATION.md):
```python
from tools.mcp_integration_standalone import create_langchain_tools_from_mcp

mcp_tools = create_langchain_tools_from_mcp(mcp_servers)
agent = OpenDsStarAgent(model="gpt-4o-mini", tools=mcp_tools)
```

### How do I create custom evaluators?

See the evaluators documentation in `src/experiments/evaluators/README.md`.

### Can I extend the framework?

Yes! The framework is modular:
- Add custom tools by implementing LangChain tools
- Add custom agents by inheriting from `BaseAgent`
- Add custom experiments by inheriting from `BaseExperiment`

### How do I contribute?

Check the developer documentation in `docs/developer/` for:
- Architecture details
- Coding standards
- Testing requirements
- Contribution guidelines

## Performance Questions

### How can I make it faster?

1. Use a faster model (e.g., `gpt-4o-mini` instead of `gpt-4o`)
2. Reduce `max_steps`
3. Use `code_mode="full"` for simple tasks
4. Run benchmarks with `--parallel-workers`

### How can I reduce costs?

1. Use cheaper models (e.g., `gpt-4o-mini`)
2. Limit `max_steps`
3. Use local models via Ollama (free)
4. Monitor token usage and optimize prompts

### Why is stepwise mode slower sometimes?

Stepwise mode makes more LLM calls (one per step) but reuses computation. It's faster for expensive operations but may be slower for simple tasks. Use `code_mode="full"` for simple calculations.

## Still Have Questions?

- Check the [Getting Started Guide](GETTING_STARTED.md)
- Review the [Installation Guide](INSTALLATION.md)
- Read the [DS-Star Agent Documentation](DS_STAR_AGENT.md)
- Look at examples in the `examples/` directory
- Check the source code - it's well-documented!
