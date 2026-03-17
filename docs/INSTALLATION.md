# Installation Guide

This guide provides detailed instructions for setting up OpenDsStar.

## Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Activate Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

## Environment Variables

Create a `.env` file in the project root with your API keys. Only include keys for providers you plan to use.

### Complete .env Template

```bash
# OpenAI (for GPT models)
OPENAI_API_KEY=your_openai_key_here

# Anthropic (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here

# WatsonX (IBM-specific)
WATSONX_APIKEY=your_watsonx_key_here
WATSONX_URL=your_watsonx_url_here
WATSONX_PROJECT_ID=your_project_id_here

# Google (for Gemini models via LiteLLM)
GEMINI_API_KEY=your_gemini_key_here

# Ollama (local models - no key needed)
OLLAMA_BASE_URL=http://localhost:11434
```

**Note**: API keys are only required when using models from the corresponding providers. The framework will only access the keys needed for your chosen models.

## Running Experiments

### DataBench Benchmark

Run DataBench experiment with a question limit (useful for testing):

```bash
# Run with 5 questions using DS-Star agent
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --question-limit 5 \
  --agent-type ds_star \
  --model-agent gpt-4o-mini

# Run with WatsonX models (IBM-specific)
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --question-limit 5 \
  --agent-type ds_star \
  --model-agent wx_llama_maverick \
  --model-file-descriptions wx_llama_maverick

# Run with multiple models for comparison
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --question-limit 5 \
  --agent-type ds_star \
  --model-agent gpt-4o-mini gpt-4o

# Run all questions (no limit)
.venv/bin/python -m src.experiments.benchmarks.databench.databench_main \
  --agent-type ds_star \
  --model-agent gpt-4o-mini
```

#### Common DataBench Options

- `--question-limit N`: Limit to N questions (omit to run all)
- `--agent-type`: Agent type (`ds_star`, `react_langchain`, `codeact_smolagents`)
- `--model-agent`: Model(s) for the agent - the LLM that powers the agent's reasoning, planning, coding, and decision-making (can specify multiple for comparison)
- `--model-file-descriptions`: Model for generating file descriptions during data ingestion - analyzes CSV/data files to create natural language descriptions that help the agent understand the data structure
- `--embedding-model`: Embedding model for vector store (default: `granite_embedding`)
- `--max-steps`: Maximum reasoning steps (default: 10)
- `--code-mode`: Execution mode (`stepwise` or `full`, default: `full`)
- `--parallel-workers N`: Run N questions in parallel (default: sequential)

#### Available Model Aliases

You can use short aliases or full model identifiers for `--model-agent` and `--model-file-descriptions`:
- OpenAI: `gpt_4o`, `gpt_4o_mini`, or full IDs like `gpt-4o`
- WatsonX: `wx_mistral_medium`, `wx_mistral_small`, `wx_llama_maverick`
- Custom API models: Configure via `CUSTOM_API_*` environment variables (see [Model Providers](MODEL_PROVIDERS.md))
- Or any LiteLLM-compatible model string (e.g., `anthropic/claude-3-5-sonnet-20241022`)

**Note**: Model aliases are defined in `src/core/model_registry.py` (class `ModelRegistry`). You can add new aliases by adding constants to this class.

#### Model Parameter Roles

- `--model-agent`: The primary LLM that runs the agent - handles all reasoning, planning, code generation, debugging, and verification
- `--model-file-descriptions`: Used during data ingestion to analyze files and generate descriptions - runs once per dataset, results are cached
- `--embedding-model`: Converts text to vectors for semantic search in the vector store

### Other Benchmarks

**HotpotQA**:
```bash
.venv/bin/python -m src.experiments.benchmarks.hotpotqa.hotpotqa_main \
  --question-limit 20 \
  --agent-type ds_star \
  --model gpt-4o-mini
```

**KramaBench**:
```bash
.venv/bin/python -m src.experiments.benchmarks.kramabench.kramabench_main \
  --agent-type ds_star \
  --model-agent gpt-4o-mini
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed experiment documentation.

## Installation from Source

For development or to install from source:

```bash
git clone https://github.com/IBM/OpenDsStar.git
cd OpenDsStar
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv sync
```

## Troubleshooting

### Python Version
Requires Python >=3.10, <3.14 (specified in pyproject.toml)

### Virtual Environment
Always use `.venv/bin/python` to ensure you're using the virtual environment Python, not the system Python.

### API Keys
If you get authentication errors, verify that:
1. Your `.env` file is in the project root
2. The correct API key is set for your chosen model provider
3. The key has the necessary permissions
