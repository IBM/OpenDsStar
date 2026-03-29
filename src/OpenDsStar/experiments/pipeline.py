"""Main experiment pipeline orchestrator."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.tools import BaseTool

from .core.context import PipelineContext
from .core.types import AgentOutput, BenchmarkEntry, EvalResult, ProcessedBenchmark
from .interfaces.agent_builder import AgentBuilder
from .interfaces.agent_runner import AgentRunner
from .interfaces.data_reader import DataReader
from .interfaces.evaluator import Evaluator
from .interfaces.tool_builder import ToolBuilder
from .utils.evaluation_cache import EvaluationCache
from .utils.logging import StageTimer
from .utils.tool_registry import ToolRegistry
from .utils.validation import ensure_unique_question_ids, index_by_question_id

logger = logging.getLogger(__name__)


class ExperimentPipeline:
    """
    Main pipeline for running experiments.

    Steps:
    1. Read data (corpus and benchmarks)
    2. Create tools
    3. Build agent with tools
    4. Run agent over benchmark
    5. Evaluate agent's output over benchmark (all evaluators for every question)
    """

    def __init__(
        self,
        ctx: PipelineContext,
        data_reader: DataReader,
        tool_builders: Sequence[ToolBuilder],
        agent_builder: AgentBuilder,
        agent_runner: AgentRunner,
        evaluators: Sequence[Evaluator],
        experiment_params: Dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> None:
        self._ctx = ctx
        self._data_reader = data_reader
        self._tool_builders = list(tool_builders)
        self._agent_builder = agent_builder
        self._agent_runner = agent_runner
        self._evaluators = list(evaluators)
        self._experiment_params = experiment_params or {}
        self._timestamp = timestamp

        # Generate base filename once and cache it
        self._base_filename = self._generate_base_filename()

    def run(self) -> Tuple[Sequence[AgentOutput], Sequence[EvalResult]]:
        """
        Run the complete experiment pipeline.

        Returns:
            Tuple of (agent outputs (with evaluation attached in metadata), evaluation results)
        """
        print("\nRunning 5-Step Pipeline:")
        print("1. Read benchmark data")
        print("2. Create tools")
        print("3. Build agent with tools")
        print("4. Run agent over benchmark")
        print("5. Evaluate agent's output")

        failures: List[Dict[str, Any]] = []

        # Step 1: Read data (corpus and benchmarks)
        benchmarks, corpus = self._read_data()
        processed_benchmarks = self._process_benchmarks(benchmarks)

        # Step 2: Create tools (pass corpus to tool builders)
        tools = self._create_tools(benchmarks, corpus, failures)

        # Step 3: Build agent with tools
        agent = self._build_agent(tools)

        # Step 4: Run agent over benchmark
        outputs = self._run_agent(agent, processed_benchmarks, failures)

        # Step 5: Evaluate agent's output
        outputs_by_qid = self._validate_outputs(outputs, processed_benchmarks, failures)
        results = self._evaluate_outputs(
            outputs_by_qid, benchmarks, processed_benchmarks
        )

        # Attach evaluation to outputs (metadata["evaluations"])
        results_by_qid = self._group_results_by_question(results)
        self._attach_evaluations_to_outputs(outputs_by_qid, results_by_qid)

        # Log summary
        self._log_summary(results, failures)

        # Return outputs in benchmark order (only those produced)
        outputs_ordered = [
            outputs_by_qid[b.question_id]
            for b in processed_benchmarks
            if b.question_id in outputs_by_qid
        ]

        # Save results to disk if output_dir is configured
        if self._ctx.config.output_dir:
            self._save_results(outputs_ordered, results, failures)

        return outputs_ordered, results

    def _should_fail_fast(self) -> bool:
        """Check if pipeline should fail fast on errors."""
        return self._ctx.config.fail_fast and not self._ctx.config.continue_on_error

    def _read_data(self) -> Tuple[Sequence[BenchmarkEntry], Any]:
        """Step 1: Read data (corpus and benchmarks)."""
        with StageTimer(logger, "read_data"):
            self._data_reader.read_data()
            corpus = self._data_reader.get_data()
            benchmarks = self._data_reader.get_benchmark()
            return benchmarks, corpus

    def _process_benchmarks(
        self, benchmarks: Sequence[BenchmarkEntry]
    ) -> Sequence[ProcessedBenchmark]:
        """Process raw benchmarks into processed format."""
        processed = [
            ProcessedBenchmark(
                question_id=b.question_id,
                question=b.question,
                ground_truth=b.ground_truth,
                metadata=dict(b.additional_information),
            )
            for b in benchmarks
        ]
        ensure_unique_question_ids(processed)
        return processed

    def _create_tools(
        self,
        benchmarks: Sequence[BenchmarkEntry],
        corpus: Any,
        failures: List[Dict[str, Any]],
    ) -> List[BaseTool]:
        """Step 2: Create tools from tool builders."""
        registry = ToolRegistry()

        with StageTimer(logger, "create_tools"):
            for builder in self._tool_builders:
                tools = builder.build_tools(self._ctx, benchmarks, corpus)
                registry.add_all(list(tools), namespace=builder.name)

        tools_final = registry.list()
        logger.info(
            f"tools_ready count={len(tools_final)} names={[t.name for t in tools_final]}"
        )
        return tools_final

    def _build_agent(self, tools: List[BaseTool]) -> Any:
        """Step 3: Build agent with tools."""
        with StageTimer(logger, "build_agent"):
            return self._agent_builder.build_agent(self._ctx, tools)

    def _run_agent(
        self,
        agent: Any,
        benchmarks: Sequence[ProcessedBenchmark],
        failures: List[Dict[str, Any]],
    ) -> List[AgentOutput]:
        """Step 4: Run agent over benchmarks."""
        with StageTimer(logger, "run_agent"):
            try:
                return list(self._agent_runner.run_batch(self._ctx, agent, benchmarks))
            except Exception as e:
                logger.error(f"agent_failed error={str(e)}", exc_info=True)
                if self._should_fail_fast():
                    raise
                failures.append({"stage": "run_agent", "error": str(e)})
                return []

    def _validate_outputs(
        self,
        outputs: List[AgentOutput],
        processed_benchmarks: Sequence[ProcessedBenchmark],
        failures: List[Dict[str, Any]],
    ) -> Dict[str, AgentOutput]:
        """Validate and index outputs by question_id."""
        outputs_by_qid: Dict[str, AgentOutput] = {}

        for o in outputs:
            if o.question_id in outputs_by_qid:
                msg = f"Duplicate agent output for question_id={o.question_id}"
                logger.error(f"agent_output_duplicate question_id={o.question_id}")
                if self._should_fail_fast():
                    raise ValueError(msg)
                failures.append({"stage": "run_agent", "error": msg})
                continue
            outputs_by_qid[o.question_id] = o

        # Check for missing outputs
        missing = [
            b.question_id
            for b in processed_benchmarks
            if b.question_id not in outputs_by_qid
        ]
        if missing:
            logger.warning(
                f"agent_outputs_missing count={len(missing)} sample={missing[:10]}"
            )

        return outputs_by_qid

    def _evaluate_outputs(
        self,
        outputs_by_qid: Dict[str, AgentOutput],
        raw_benchmarks: Sequence[BenchmarkEntry],
        processed_benchmarks: Sequence[ProcessedBenchmark],
    ) -> List[EvalResult]:
        """Step 5: Evaluate agent outputs against ground truth (run ALL evaluators for EVERY question)."""
        raw_by_qid = index_by_question_id(raw_benchmarks, "question_id")

        # Initialize evaluation cache once for all evaluations
        cache_dir = self._ctx.config.cache_dir if self._ctx.config.cache_dir else None
        evaluation_cache = EvaluationCache(
            cache_base_dir=cache_dir, enabled=cache_dir is not None
        )
        total_cache_hits = 0

        results: List[EvalResult] = []
        with StageTimer(logger, "evaluate"):
            # Keep stable benchmark order
            for proc in processed_benchmarks:
                qid = proc.question_id
                raw = raw_by_qid.get(qid)
                out = outputs_by_qid.get(qid)

                if raw is None:
                    # mismatch between processed/raw
                    if self._evaluators:
                        for ev in self._evaluators:
                            results.append(
                                EvalResult(
                                    question_id=qid,
                                    score=0.0,
                                    passed=False,
                                    details={"reason": "missing_raw_benchmark_entry"},
                                )
                            )
                    else:
                        results.append(
                            EvalResult(
                                question_id=qid,
                                score=0.0,
                                passed=False,
                                details={"reason": "missing_raw_benchmark_entry"},
                            )
                        )
                    continue

                eval_results, cache_hits = self._evaluate_single_all(
                    qid=qid,
                    proc=proc,
                    raw=raw,
                    out=out,
                    evaluation_cache=evaluation_cache,
                )
                results.extend(eval_results)
                total_cache_hits += cache_hits

            # Log overall cache statistics
            if total_cache_hits > 0:
                total_evaluations = len(results)
                cache_hit_rate = (
                    total_cache_hits / total_evaluations if total_evaluations > 0 else 0
                )
                logger.info(
                    f"evaluation_cache_summary total_cache_hits={total_cache_hits} "
                    f"total_evaluations={total_evaluations} cache_hit_rate={cache_hit_rate}"
                )

        return results

    def _evaluate_single_all(
        self,
        qid: str,
        proc: ProcessedBenchmark,
        raw: BenchmarkEntry,
        out: AgentOutput | None,
        evaluation_cache: EvaluationCache,
    ) -> tuple[List[EvalResult], int]:
        """
        Evaluate one question with all evaluators.

        Returns:
            Tuple of (evaluation results, cache hits count)
        """
        if not self._evaluators:
            return [
                EvalResult(
                    question_id=qid,
                    score=0.0,
                    passed=False,
                    details={"reason": "no_evaluators_configured"},
                )
            ], 0

        # Missing output -> one result per evaluator
        if out is None:
            return [
                EvalResult(
                    question_id=qid,
                    score=0.0,
                    passed=False,
                    details={"reason": "missing_agent_output"},
                )
                for ev in self._evaluators
            ], 0

        per_eval: List[EvalResult] = []
        cache_hits = 0

        for ev in self._evaluators:
            try:
                # Prepare ground truth answers for cache key
                ground_truth_answers = [str(a) for a in raw.ground_truth.answers]

                # Check cache first
                cached_result = evaluation_cache.get(
                    ev, qid, str(out.answer), ground_truth_answers
                )

                if cached_result is not None:
                    # Reconstruct EvalResult from cached dict
                    result = EvalResult(
                        question_id=cached_result["question_id"],
                        score=cached_result["score"],
                        passed=cached_result["passed"],
                        details=cached_result["details"],
                    )
                    cache_hits += 1
                    # Cache hit - no need to log for each hit, summary is logged at the end
                else:
                    # Evaluate and cache result
                    result = ev.evaluate_one(self._ctx, output=out, benchmark=raw)

                    # Add evaluator class name to details if not already present
                    if "evaluator_class" not in result.details:
                        # Create a new result with evaluator class name in details
                        updated_details = dict(result.details)
                        updated_details["evaluator_class"] = ev.__class__.__name__
                        result = EvalResult(
                            question_id=result.question_id,
                            score=result.score,
                            passed=result.passed,
                            details=updated_details,
                        )

                    # Store result in cache as dict
                    result_dict = {
                        "question_id": result.question_id,
                        "score": result.score,
                        "passed": result.passed,
                        "details": result.details,
                    }
                    evaluation_cache.put(
                        ev, qid, str(out.answer), ground_truth_answers, result_dict
                    )

                per_eval.append(result)

            except Exception as e:
                logger.error(
                    f"evaluator_failed evaluator={ev.__class__.__name__} "
                    f"question_id={qid} error={str(e)}",
                    exc_info=True,
                )
                per_eval.append(
                    EvalResult(
                        question_id=qid,
                        score=0.0,
                        passed=False,
                        details={
                            "reason": "evaluator_exception",
                            "error": str(e),
                            "evaluator": ev.__class__.__name__,
                        },
                    )
                )

        return per_eval, cache_hits

    def _group_results_by_question(
        self, results: Sequence[EvalResult]
    ) -> Dict[str, List[EvalResult]]:
        by_qid: Dict[str, List[EvalResult]] = {}
        for r in results:
            by_qid.setdefault(r.question_id, []).append(r)
        return by_qid

    def _attach_evaluations_to_outputs(
        self,
        outputs_by_qid: Dict[str, AgentOutput],
        results_by_qid: Dict[str, List[EvalResult]],
    ) -> None:
        """
        Add evaluations into each output.metadata["evaluations"].

        If AgentOutput or metadata is immutable, we log a warning. Saving still includes
        evaluation in the merged 'items' section in the JSON.
        """
        for qid, out in outputs_by_qid.items():
            evals = results_by_qid.get(qid, [])
            try:
                # If metadata is None or not a dict, normalize as best we can.
                if out.metadata is None:
                    out.metadata = {}  # type: ignore[attr-defined]
                out.metadata["evaluations"] = [self._serialize_result(r) for r in evals]
            except Exception as e:
                logger.warning(
                    f"attach_evaluations_failed question_id={qid} error={str(e)}"
                )

    def _log_summary(
        self, results: List[EvalResult], failures: List[Dict[str, Any]]
    ) -> None:
        """Log pipeline summary."""
        if results:
            avg = sum(r.score for r in results) / len(results)
            passed = sum(1 for r in results if r.passed)
            logger.info(f"summary n={len(results)} avg_score={avg} passed={passed}")
        if failures:
            logger.warning(
                f"pipeline_failures count={len(failures)} sample={failures[:5]}"
            )

    def _make_serializable(self, obj: Any) -> Any:
        """
        Recursively convert an object to a JSON-serializable form.

        Improvements:
        - Parses JSON strings to objects when possible
        - Shortens values longer than 1000 characters to 997 chars + "..."
        """
        # Handle primitives
        if isinstance(obj, (int, float, bool, type(None))):
            return obj

        # Handle strings - try to parse as JSON, then shorten if needed
        if isinstance(obj, str):
            # Try to parse as JSON first
            if obj.strip().startswith(("{", "[")):
                try:
                    parsed = json.loads(obj)
                    # Recursively process the parsed object
                    return self._make_serializable(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass  # Not valid JSON, treat as regular string

            # Shorten if too long
            if len(obj) > 1000:
                return obj[:997] + "..."
            return obj

        # Handle dicts
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]

        # Convert other types to string and shorten if needed
        str_obj = str(obj)
        if len(str_obj) > 1000:
            return str_obj[:997] + "..."
        return str_obj

    def _serialize_output(self, output: AgentOutput) -> Dict[str, Any]:
        """Serialize an AgentOutput to a JSON-compatible dict."""
        return {
            "question_id": str(output.question_id),
            "answer": str(output.answer),
            "metadata": self._make_serializable(dict(output.metadata)),
        }

    def _serialize_result(self, result: EvalResult) -> Dict[str, Any]:
        """Serialize an EvalResult to a JSON-compatible dict."""
        return {
            "question_id": str(result.question_id),
            "score": float(result.score),
            "passed": bool(result.passed),
            "details": self._make_serializable(dict(result.details)),
        }

    def _serialize_item(
        self,
        question_id: str,
        output: AgentOutput | None,
        evals: Sequence[EvalResult],
    ) -> Dict[str, Any]:
        """Serialize a merged per-question record (output + evaluations)."""
        return {
            "question_id": question_id,
            "output": self._serialize_output(output) if output is not None else None,
            "evaluations": [self._serialize_result(r) for r in evals],
        }

    def _compute_summary(self, results: Sequence[EvalResult]) -> Dict[str, Any]:
        """Compute summary statistics from results."""
        if not results:
            return {
                "total_questions": 0,
                "avg_score": 0.0,
                "passed_count": 0,
                "passed_rate": 0.0,
            }

        total_results = len(results)
        avg_score = sum(r.score for r in results) / total_results
        passed_count = sum(1 for r in results if r.passed)
        passed_rate = passed_count / total_results

        # Group by evaluator class name
        by_evaluator: Dict[str, List[float]] = {}
        for r in results:
            evaluator_name = r.details.get("evaluator_class", "unknown")
            by_evaluator.setdefault(str(evaluator_name), []).append(r.score)

        type_stats = {
            type_str: {
                "count": len(scores),
                "avg_score": sum(scores) / len(scores),
            }
            for type_str, scores in by_evaluator.items()
        }

        return {
            "total_results": total_results,
            "avg_score": avg_score,
            "passed_count": passed_count,
            "passed_rate": passed_rate,
            "by_evaluator": type_stats,
        }

    def _generate_base_filename(self) -> str:
        """
        Generate the base filename (without suffix) using experiment parameters.
        This is called once during initialization and cached.

        Naming convention: result_<agent_type>_<model_name>_<experiment_name>[_<question_limit>]_<timestamp>

        Returns:
            Generated base filename (without suffix and extension)
        """
        from OpenDsStar.core.model_registry import ModelRegistry

        # Use the timestamp passed to the pipeline, or generate a new one
        timestamp = (
            self._timestamp
            if self._timestamp
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        # Get agent_type from experiment params
        agent_type = self._experiment_params.get("agent_type", "unknown")
        if hasattr(agent_type, "value"):
            agent_type = str(agent_type.value)
        else:
            agent_type = str(agent_type)

        # Get model name from parameters
        model = self._experiment_params.get("model", "unknown")
        model_name = ModelRegistry.get_model_name(model)

        # Get experiment name from run_id
        run_id = self._ctx.config.run_id
        experiment_name = run_id.split("_")[0] if run_id else "experiment"

        # Get question_limit from experiment params
        question_limit = self._experiment_params.get("question_limit")

        # Build base filename
        if question_limit is not None:
            return f"result_{agent_type}_{model_name}_{experiment_name}_{question_limit}_{timestamp}"
        else:
            return f"result_{agent_type}_{model_name}_{experiment_name}_{timestamp}"

    def _generate_filename(self, suffix: str) -> str:
        """
        Generate a complete filename by adding suffix to the cached base filename.

        Args:
            suffix: File suffix (e.g., "output" or "params")

        Returns:
            Complete filename with suffix and extension
        """
        return f"{self._base_filename}_{suffix}.json"

    def _save_results(
        self,
        outputs: Sequence[AgentOutput],
        results: Sequence[EvalResult],
        failures: List[Dict[str, Any]],
    ) -> None:
        """
        Save experiment results to disk.

        Saves results as JSON with timestamp in the output directory.
        Includes a merged per-question `items` list containing output + evaluations.

        Naming convention: result_<agent_type>_<model_name>_<experiment_name>[_<question_limit>]_<timestamp>_output.json
        """
        assert self._ctx.config.output_dir is not None
        output_dir = Path(self._ctx.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename using shared method
        filename = self._generate_filename("output")
        output_file = output_dir / filename

        # Get agent_type and question_limit for JSON data
        agent_type = self._experiment_params.get("agent_type", "unknown")
        if hasattr(agent_type, "value"):
            agent_type = str(agent_type.value)
        else:
            agent_type = str(agent_type)

        question_limit = self._experiment_params.get("question_limit")

        # Group results by question_id
        results_by_qid: Dict[str, List[EvalResult]] = {}
        for r in results:
            results_by_qid.setdefault(r.question_id, []).append(r)

        outputs_by_qid: Dict[str, AgentOutput] = {o.question_id: o for o in outputs}

        # Build merged items with deterministic ordering:
        # first outputs order, then any qids that only appear in results.
        output_qids_in_order = [o.question_id for o in outputs]
        all_qids = set(outputs_by_qid.keys()) | set(results_by_qid.keys())
        ordered_qids = output_qids_in_order + sorted(
            all_qids - set(output_qids_in_order)
        )

        items: List[Dict[str, Any]] = [
            self._serialize_item(
                question_id=qid,
                output=outputs_by_qid.get(qid),
                evals=results_by_qid.get(qid, []),
            )
            for qid in ordered_qids
        ]

        data = {
            "run_id": self._ctx.config.run_id,
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "config": {
                "fail_fast": self._ctx.config.fail_fast,
                "continue_on_error": self._ctx.config.continue_on_error,
            },
            "summary": self._compute_summary(results),
            # Backwards compatibility
            "outputs": [self._serialize_output(o) for o in outputs],
            "results": [self._serialize_result(r) for r in results],
            # Preferred combined representation
            "items": items,
            "failures": failures,
        }

        # Add question_limit to data if available
        if question_limit is not None:
            data["question_limit"] = question_limit

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"results_saved output_file={str(output_file)} "
            f"total_outputs={len(outputs)} total_results={len(results)}"
        )
