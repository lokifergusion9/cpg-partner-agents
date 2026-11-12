"""The discovery rate limiter (HLD §7, §12): a token bucket per key, own implementation.

A request names one or more keys (``user:<id>`` and ``ip:<address>``). It is allowed only if
every key has a token; then one token is taken from each. If any key is empty nothing is taken
and the longest wait is returned, so a refused request costs nothing. Keys are held in memory
(one worker, ADR-010); idle full buckets are dropped when the table grows.
"""

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from partner_agents.outreach.policy import TokenBucket

HOUR_S = 3600.0
MAX_KEYS = 10_000


class RateLimiter:
    def __init__(
        self,
        capacity: int,
        period_s: float = HOUR_S,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.capacity = capacity
        self._period_s = period_s
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}

    def take(self, keys: Sequence[str]) -> int | None:
        """Take one token from every key. Returns ``None`` if allowed, else ``retry_after_s``."""
        now = self._clock()
        fresh = {
            key: (
                self._buckets.get(key) or TokenBucket.full(self.capacity, now, self._period_s)
            ).refilled(now)
            for key in dict.fromkeys(keys)
        }
        wait = max((bucket.wait_s(now) for bucket in fresh.values()), default=0.0)
        if wait > 0:
            self._buckets.update(fresh)  # keep the refill, spend nothing
            return math.ceil(wait) if math.isfinite(wait) else int(self._period_s)
        for key, bucket in fresh.items():
            self._buckets[key] = bucket.take(now)[1]
        self._prune(now)
        return None

    def _prune(self, now: datetime) -> None:
        if len(self._buckets) <= MAX_KEYS:
            return
        self._buckets = {
            key: bucket
            for key, bucket in self._buckets.items()
            if bucket.refilled(now).tokens < self.capacity  # still recovering
        }
