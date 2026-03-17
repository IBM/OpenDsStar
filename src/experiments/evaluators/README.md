# Evaluators

This directory contains evaluator implementations for the experiment runner.

## Available Evaluators

### UnitxtLLMJudgeEvaluator

An evaluator that uses Unitxt's LLM-as-a-Judge metric to evaluate agent outputs by comparing them against gold answers using an LLM.

#### Features

- Uses IBM's Unitxt library for LLM-based evaluation
- Configurable LLM model (use ModelRegistry for available models)
- Automatic score normalization to [0, 1] range
- Detailed error handling and logging

#### Usage

```python
from experiments.evaluators import UnitxtLLMJudgeEvaluator
from core.model_registry import ModelRegistry

# Create the evaluator - must specify a model
evaluator = UnitxtLLMJudgeEvaluator(model=ModelRegistry.WX_MISTRAL_MEDIUM)

# Or customize the model and parameters
evaluator = UnitxtLLMJudgeEvaluator(
    model=ModelRegistry.WX_MISTRAL_MEDIUM,
    max_tokens=2048,
    temperature=0.0,
    task_description="Evaluate if the answer is correct and complete"
)

# Use in an evaluators builder
class MyEvaluatorsBuilder:
    @staticmethod
    def build_evaluators():
        return [
            UnitxtLLMJudgeEvaluator(),
        ]
```

#### Integration Example

To use the UnitxtLLMJudgeEvaluator in your experiment:

```python
# In your evaluators_builder.py
from experiments.evaluators import UnitxtLLMJudgeEvaluator
from core.model_registry import ModelRegistry

class HotpotQAEvaluatorsBuilder:
    @staticmethod
    def build_evaluators():
        return [
            UnitxtLLMJudgeEvaluator(
                model=ModelRegistry.WX_MISTRAL_MEDIUM,
                temperature=0.0,
            ),
        ]
```

#### Parameters

- `model` (str): The LLM model to use for judging. Use ModelRegistry constants (e.g., ModelRegistry.WX_MISTRAL_MEDIUM)
- `max_tokens` (int): Maximum tokens for LLM response. Default: `2048`
- `temperature` (float): Temperature for LLM sampling. Default: `0.0`
- `task_description` (str | None): Description of the task for the judge. If None, uses a default description.

#### Output

The evaluator returns an `EvalResult` with:
- `score`: Float in [0, 1] range indicating the quality of the answer
- `passed`: Boolean indicating if score >= 0.5
- `details`: Dictionary containing:
  - `predicted`: The agent's answer
  - `ground_truth`: The reference answers
  - `model`: The model used for evaluation
  - `raw_score`: The raw score from the LLM judge

#### Requirements

Make sure you have the required dependencies installed:

```bash
pip install unitxt
```

And ensure you have the necessary environment variables set for the LLM model (e.g., API keys for watsonx).

## Creating Custom Evaluators

To create a custom evaluator, inherit from the `Evaluator` interface:

```python
from experiments.interfaces.evaluator import Evaluator
from experiments.core.types import AgentOutput, BenchmarkEntry, EvalResult
from experiments.core.context import PipelineContext

class MyCustomEvaluator(Evaluator):
    def evaluate_one(
        self,
        ctx: PipelineContext,
        output: AgentOutput,
        benchmark: BenchmarkEntry,
    ) -> EvalResult:
        # Your evaluation logic here
        score = 1.0  # Calculate your score
        passed = score >= 0.5

        return EvalResult(
            question_id=output.question_id,
            answer_type=self.answer_type,
            score=score,
            passed=passed,
            details={
                "predicted": output.answer,
                "ground_truth": benchmark.ground_truth.answers,
            },
        )
