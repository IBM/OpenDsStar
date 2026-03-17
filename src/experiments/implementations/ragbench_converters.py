"""Utility functions for converting between ragbench data models and our types."""

from __future__ import annotations

from typing import Any, Sequence

from ..core.types import BenchmarkEntry, Document, GroundTruth


def convert_ragbench_corpus_to_documents(corpus: Any) -> Sequence[Document]:
    """
    Convert ragbench RagCorpus to sequence of Document objects.

    Args:
        corpus: RagCorpus object from ragbench package

    Returns:
        Sequence of Document objects
    """
    documents = []

    for doc_obj in corpus.documents:
        # Create a stream factory that returns a fresh stream each time
        # Use default argument to capture the current stream value (avoids closure bug)
        def make_stream_factory(doc_stream=doc_obj.stream):
            def stream_factory():
                # Reset stream position and return it
                doc_stream.seek(0)
                return doc_stream

            return stream_factory

        document = Document(
            document_id=doc_obj.name,
            path=doc_obj.name,  # Use name as path since we don't have file paths
            mime_type=doc_obj.mime_type,
            extra_metadata=doc_obj.metadata,
            stream_factory=make_stream_factory(),
        )
        documents.append(document)

    return documents


def convert_ragbench_benchmark_to_entries(
    benchmark_data: Any,
) -> Sequence[BenchmarkEntry]:
    """
    Convert ragbench RagBenchmark to sequence of BenchmarkEntry objects.

    Args:
        benchmark_data: RagBenchmark object from ragbench package

    Returns:
        Sequence of BenchmarkEntry objects
    """
    benchmark_entries = []

    for item in benchmark_data.benchmark_entries:
        question_id = item.question_id
        question = item.question

        # Safely handle ground_truth_answers - ensure it's a list
        if item.ground_truth_answers is None:
            answers = []
        elif isinstance(item.ground_truth_answers, list):
            answers = item.ground_truth_answers
        else:
            # If it's a single value, wrap it in a list
            answers = [item.ground_truth_answers]

        # Convert GroundTruthContextId objects to simple document IDs
        # Safely handle ground_truth_context_ids
        if item.ground_truth_context_ids is None:
            context_ids = []
        elif isinstance(item.ground_truth_context_ids, list):
            context_ids = [ctx.document_id for ctx in item.ground_truth_context_ids]
        else:
            # If it's a single value, wrap it in a list
            context_ids = [item.ground_truth_context_ids.document_id]

        additional_info = item.additional_information or {}

        ground_truth = GroundTruth(
            answers=answers,
            context_ids=context_ids,
            extra=additional_info,
        )

        # Create benchmark entry
        benchmark_entry = BenchmarkEntry(
            question_id=str(question_id),
            question=str(question),
            ground_truth=ground_truth,
            additional_information=additional_info,
        )

        benchmark_entries.append(benchmark_entry)

    return benchmark_entries
