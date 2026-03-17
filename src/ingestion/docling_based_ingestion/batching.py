"""Batching utilities for processing items in chunks."""

from typing import Iterator, List, Sequence, TypeVar

T = TypeVar("T")


def iter_batches(items: Sequence[T], batch_size: int) -> Iterator[List[T]]:
    """
    Yield successive batches from a sequence.

    Args:
        items: Sequence to batch
        batch_size: Size of each batch

    Yields:
        Lists of items, each of size batch_size (except possibly the last)
    """
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])
