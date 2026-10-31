"""Async retry with exponential backoff, full jitter and ``Retry-After`` (HLD §11).

Errors are classified by duck typing so this module needs no HTTP or SDK imports:
anything exposing ``status_code`` (or ``response.status_code``) in ``RETRYABLE_STATUS``
is retried, as are timeouts and connection errors.
"""

import asyncio
import functools
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Indirection so tests can mock "now" for HTTP-date Retry-After values.
_now: Callable[[], datetime] = lambda: datetime.now(UTC)  # noqa: E731


@dataclass(frozen=True)
class RetryPolicy:
    """Defaults match HLD §11: 4 attempts, exponential backoff with jitter."""

    attempts: int = 4
    base_delay_s: float = 0.5
    multiplier: float = 2.0
    max_delay_s: float = 20.0
    # A server asking us to wait longer than this is treated as "give up now".
    max_retry_after_s: float = 60.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.base_delay_s < 0 or self.max_delay_s < 0 or self.max_retry_after_s < 0:
            raise ValueError("delays must be >= 0")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")

    def backoff_ceiling(self, attempt: int) -> float:
        """Upper bound of the jittered delay after failed attempt ``attempt`` (1-based)."""
        return float(min(self.max_delay_s, self.base_delay_s * self.multiplier ** (attempt - 1)))


DEFAULT_POLICY = RetryPolicy()


def status_code_of(exc: BaseException) -> int | None:
    for candidate in (exc, getattr(exc, "response", None)):
        code = getattr(candidate, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def is_retryable(exc: BaseException) -> bool:
    """Default predicate: transient network failures and retryable HTTP statuses."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    # httpx and SDK transport errors that are not subclasses of the builtins above.
    if type(exc).__name__ in {
        "TimeoutException",
        "ConnectError",
        "ReadError",
        "RemoteProtocolError",
    }:
        return True
    code = status_code_of(exc)
    return code is not None and code in RETRYABLE_STATUS


def parse_retry_after(value: str | float | int | None) -> float | None:
    """Seconds to wait from a ``Retry-After`` value (delta-seconds or HTTP-date)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = value.strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - _now()).total_seconds())


def retry_after_of(exc: BaseException) -> float | None:
    """``Retry-After`` carried by an exception: a ``retry_after`` attribute or header."""
    explicit = getattr(exc, "retry_after", None)
    if isinstance(explicit, (int, float, str)):
        return parse_retry_after(explicit)
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None and hasattr(headers, "get"):
        return parse_retry_after(headers.get("Retry-After") or headers.get("retry-after"))
    return None


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy = DEFAULT_POLICY,
    retryable: Callable[[BaseException], bool] = is_retryable,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Call ``fn`` until it succeeds, a non-retryable error occurs, or attempts run out.

    The delay after failed attempt ``n`` is ``Retry-After`` when the error carries one,
    otherwise full jitter: ``rand() * min(max_delay, base * multiplier**(n-1))``.
    The last error is re-raised unchanged.
    """
    for attempt in range(1, policy.attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            if attempt == policy.attempts or not retryable(exc):
                raise
            wait = retry_after_of(exc)
            if wait is not None and wait > policy.max_retry_after_s:
                raise
            delay = wait if wait is not None else rand() * policy.backoff_ceiling(attempt)
            logger.warning(
                "retrying after error",
                extra={
                    "attempt": attempt,
                    "max_attempts": policy.attempts,
                    "delay_s": round(delay, 3),
                    "error_type": type(exc).__name__,
                    "status_code": status_code_of(exc),
                },
            )
            await sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def with_retry[**P, T](
    policy: RetryPolicy = DEFAULT_POLICY,
    retryable: Callable[[BaseException], bool] = is_retryable,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator form of ``retry_async`` for async functions."""

    def decorate(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await retry_async(
                lambda: fn(*args, **kwargs), policy=policy, retryable=retryable
            )

        return wrapper

    return decorate
