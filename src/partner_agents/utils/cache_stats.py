"""Cache hit and miss counters for one run (HLD section 6.3, 6.5, T-34).

``track_cache()`` opens a counter for the work done inside the block (tasks started inside it
share it, like ``llm.track_usage``). The search cache and the embedding cache call
``record_cache`` and the result is what ``RunRecord.cache`` stores, for example
``{"search_hits": 2, "search_misses": 0, "embed_hits": 31, "embed_misses": 0}``. Outside a
block, recording does nothing.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_stats: ContextVar[dict[str, int] | None] = ContextVar("cache_stats", default=None)


@contextmanager
def track_cache() -> Iterator[dict[str, int]]:
    stats: dict[str, int] = {}
    token = _stats.set(stats)
    try:
        yield stats
    finally:
        _stats.reset(token)


def record_cache(name: str, *, hits: int = 0, misses: int = 0) -> None:
    """Count ``hits`` and ``misses`` of the cache called ``name`` in the current run."""
    stats = _stats.get()
    if stats is None:
        return
    for suffix, count in (("hits", hits), ("misses", misses)):
        if count:
            stats[f"{name}_{suffix}"] = stats.get(f"{name}_{suffix}", 0) + count
