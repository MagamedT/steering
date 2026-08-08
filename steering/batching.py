from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence, TypeVar


T = TypeVar("T")


def chunked(values: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield lists containing at most ``size`` values."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    batch: list[T] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def chunked_with_bounds(
    values: Sequence[Any], size: int
) -> Iterator[tuple[int, int, list[Any]]]:
    """Yield ``(start, stop, values)`` batches."""
    if size <= 0:
        size = len(values) if values else 1
    for start in range(0, len(values), size):
        stop = min(len(values), start + size)
        yield start, stop, list(values[start:stop])
