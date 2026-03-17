# OpenDsStar

**OpenDsStar** is an open-source implementation of the **[DS-Star agent](https://arxiv.org/abs/2509.21825)**, with several deliberate design enhancements that improve modularity, extensibility, and execution efficiency.

The original DS-Star agent is primarily built around **file-based artifacts**: reasoning, planning, and execution revolve around reading, writing, and modifying files that represent intermediate and final results. OpenDsStar preserves the core planning-and-coding philosophy of DS-Star, but redefines the execution model around a **tool-centric abstraction**.

DS-Star is a **Programmatic Tool Calling (PTC)** agent. Rather than reasoning directly over files, OpenDsStar plans and executes workflows by composing **explicit tool invocations**, drawing inspiration from the **ReAct** and **CodeAct** paradigms. Tools can encapsulate file access, database queries, API calls, external services, computation engines, or arbitrary custom functions. This decouples the agent’s reasoning logic from the underlying execution environment and storage format.

This design generalizes the DS-Star approach beyond data-science-specific workflows. Any task that can be expressed as a sequence of tool calls—whether it involves data processing, information retrieval, programmatic reasoning, or system interaction—can be handled without changing the agent’s core structure.

OpenDsStar also introduces more flexible execution control than the original DS-Star.

In the original design, the agent typically **re-executes the entire planned workflow from the beginning** whenever the plan is revised, even if earlier steps have already completed successfully. This can be wasteful when early steps involve expensive computation, slow external calls, or large-scale data processing.

OpenDsStar explicitly separates planning from execution and supports **incremental, stepwise execution**. In this mode, completed steps produce persistent intermediate results and are **not re-run**. When the plan is extended or refined, only the **newly introduced step** is executed, while previous outputs are reused. This significantly reduces redundant computation and makes the agent more practical for workflows in which individual steps are costly, long-running, or stateful.

## Summary of Design Enhancements

| Aspect | Original DS-Star | OpenDsStar |
|--------|------------------|------------|
| Core abstraction | Files | Tools |
| Planning representation | File/code actions | Tool-call sequences |
| Scope | Data-science focused | General purpose |
| Execution strategy | Re-run full plan | Incremental execution |
| Intermediate results | Recomputed | Persisted and reused |
| Planning vs. execution | Coupled | Explicitly separated |
| Extensibility | File-centric | Tool-based |

## Features

- **Programmatic Tool Calling (PTC)**: Plans are represented as sequences of tool invocations rather than direct file manipulation
- **Explicit multi-step reasoning**: Complex tasks are decomposed into structured, inspectable plans
- **Code generation and execution**: Generates and runs code when needed
- **Stepwise execution mode**: Executes plans incrementally while reusing intermediate outputs
- **Full execution mode**: Runs the entire plan end-to-end, mirroring the original DS-Star behavior
- **Error handling and recovery**: Failed steps are debugged and retried automatically
- **Result verification**: Outputs are validated before returning final answers
- **LLM-agnostic**: Works with OpenAI, Anthropic, Azure, WatsonX, Ollama, and more through LiteLLM

## Execution Modes

OpenDsStar supports two execution modes:

- **Full mode**
  Plans and executes the entire workflow end-to-end, closely matching the original DS-Star execution model.

- **Stepwise mode**
  Produces plans incrementally and executes only the newest step while reusing outputs from previous steps. This mode is more efficient when steps are expensive or computationally heavy.

## Installation

### From PyPI (Recommended)

```bash
pip install opendsstar
```

### From Source

```bash
git clone https://github.com/IBM/OpenDsStar.git
cd OpenDsStar
python -m venv .venv
source .venv/bin/activate  # macOS/Linux or .venv\Scripts\activate on Windows
pip install -e .
```

### Configuration

Create a `.env` file with your API keys (only include keys for providers you'll use):
```bash
OPENAI_API_KEY=your_key_here
# Add other provider keys as needed
```

See [Installation Guide](docs/INSTALLATION.md) for detailed setup instructions, environment variables, and troubleshooting.

## Quick Start

### OpenDsStarAgent (DS-Star Implementation)

```python
from dotenv import load_dotenv
from agents import OpenDsStarAgent

load_dotenv()

agent = OpenDsStarAgent(model="gpt-4o-mini")

result = agent.invoke("What is 15 * 23 + 42?")
print(result["answer"])
```

For detailed usage, see [DS-Star Agent Documentation](docs/DS_STAR_AGENT.md).

### ReactAgent (LangChain ReAct Agent)

```python
from dotenv import load_dotenv
from agents import ReactAgent

load_dotenv()

agent = ReactAgent()

result = agent.invoke("What is the capital of France?")
print(result["answer"])
```

See [ReactAgent Documentation](docs/REACT_AGENT_WRAPPER.md) for more details.

## Running Experiments

### Quick Start - DataBench with 5 Questions

```bash
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --question-limit 5 \
  --agent-type ds_star \
  --model-agent gpt-4o-mini
```

### Other Benchmarks

```bash
# HotpotQA
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main \
  --question-limit 20 --agent-type ds_star --model gpt-4o-mini

# KramaBench
.venv/bin/python -m src.experiments.benchmarks.kramabench.kramabench_main \
  --agent-type ds_star --model-agent gpt-4o-mini
```

See [Installation Guide](docs/INSTALLATION.md) for detailed command options, model aliases, and parameter explanations.

See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for comprehensive experiment documentation.

## Agent Implementations

OpenDsStar includes multiple agent implementations for comparison and benchmarking:

- **OpenDsStarAgent**: Main DS-Star implementation — a Programmatic Tool Calling (PTC) agent with planning, coding, execution, debugging, and verification ([Documentation](docs/DS_STAR_AGENT.md))
- **ReactAgentLangchain**: Lightweight wrapper around the LangChain ReAct agent ([Documentation](docs/REACT_AGENT_WRAPPER.md))
- **ReactAgentSmolagents**: Smolagents-based ReAct implementation
- **CodeActAgentSmolagents**: Smolagents-based CodeAct implementation

All agents share a common interface, making it easy to compare different agent paradigms on the same tasks.

## Experiments Framework

OpenDsStar includes a comprehensive **experiments framework** for reproducible benchmarking and evaluation. It provides modular experiment design, automatic caching, multi-agent support, and built-in evaluation.

The framework includes:

- **Modular experiment design**: Each experiment is self-contained, with its own data reader, tools builder, agent configuration, and evaluators
- **Easy extensibility**: New experiments can be added by implementing a small set of simple interfaces, without modifying core framework code
- **Automatic caching**: Intermediate results are cached to avoid redundant computation
- **Reproducibility**: Experiment parameters are automatically saved, enabling exact reruns
- **Multiple agent support**: The same experiment can be run with different agents (e.g., DS-Star, ReAct, CodeAct) for direct comparison
- **Built-in evaluation**: Integrated evaluation metrics and result tracking

See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for more details.

## Benchmark Results

### Kramabench Dataset Evaluation

Comparison of DS-Star and CodeAct agents on the Kramabench dataset (31 questions) across multiple LLM providers:

| Agent | Model | Total Tokens | LLM Calls | LLM Judge Score |
|-------|-------|--------------|-----------|-----------------|
| **CodeAct** | Llama Maverick | 3.3M | 271 | 0.224 |
| **DS-Star** | Llama Maverick | 2.5M | 548 | **0.248** |
| **CodeAct** | Gemini 2.5 Flash | 4.0M | 275 | 0.297 |
| **DS-Star** | Gemini 2.5 Flash | 6.1M | 664 | **0.303** |
| **CodeAct** | Gemini 2.5 Pro | 1.6M | 235 | 0.312 |
| **DS-Star** | Gemini 2.5 Pro | 1.1M | 701 | **0.387** |

**Experimental setup**

- Both agents use identical tools and data access methods
- Both use the same data-ingestion pipeline and file descriptions
- File descriptions were generated using WatsonX Llama Maverick for all configurations
- Observed performance differences are therefore attributable primarily to the agent architecture and reasoning strategy

**Key findings**

- DS-Star consistently outperforms CodeAct across all tested models in answer quality
- DS-Star achieves better results with fewer total tokens on Llama Maverick and Gemini 2.5 Pro
- The planning, debugging, and verification cycle improves answer accuracy, even when it requires more LLM calls
- Best overall result: **DS-Star with Gemini 2.5 Pro** (0.387 judge score, with the lowest token usage among the Gemini Pro runs)

## Project Structure

```text
src/
├── agents/
├── tools/
└── experiments/
```

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [EXPERIMENTS.md](docs/EXPERIMENTS.md)

## License

Apache License 2.0
