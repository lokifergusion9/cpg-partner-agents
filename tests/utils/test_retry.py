import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from partner_agents.utils import retry as r
from partner_agents.utils.retry import RetryPolicy, retry_async


class FakeResponse:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}


class HTTPError(Exception):
    """Shaped like httpx.HTTPStatusError: the status lives on ``.response``."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = FakeResponse(status_code, headers)


class SDKError(Exception):
    """Shaped like LLM SDK errors: ``status_code`` on the exception itself."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class Flaky:
    """Raises the given errors in order, then returns ``result``."""

    def __init__(self, errors: list[Exception], result: str = "ok") -> None:
        self.errors = list(errors)
        self.result = result
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


class FakeSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def run(
    fn: Callable[[], Awaitable[str]],
    sleep: FakeSleep,
    policy: RetryPolicy = r.DEFAULT_POLICY,
    rand: Callable[[], float] = lambda: 1.0,
    **kwargs: Any,
) -> str:
    return asyncio.run(retry_async(fn, policy=policy, sleep=sleep, rand=rand, **kwargs))


# --- success after N failures -------------------------------------------------------


@pytest.mark.parametrize("failures", [0, 1, 2, 3])
def test_succeeds_after_n_transient_failures(failures: int) -> None:
    fn = Flaky([TimeoutError()] * failures)
    sleep = FakeSleep()
    assert run(fn, sleep) == "ok"
    assert fn.calls == failures + 1
    assert len(sleep.delays) == failures


def test_backoff_is_exponential_and_capped() -> None:
    policy = RetryPolicy(attempts=6, base_delay_s=1, multiplier=2, max_delay_s=5)
    fn = Flaky([ConnectionError()] * 5)
    sleep = FakeSleep()
    run(fn, sleep, policy=policy, rand=lambda: 1.0)
    assert sleep.delays == [1, 2, 4, 5, 5]


def test_full_jitter_scales_the_ceiling() -> None:
    policy = RetryPolicy(attempts=3, base_delay_s=2, multiplier=2)
    rolls = iter([0.25, 0.5])
    fn = Flaky([TimeoutError(), TimeoutError()])
    sleep = FakeSleep()
    run(fn, sleep, policy=policy, rand=lambda: next(rolls))
    assert sleep.delays == [0.5, 2.0]


# --- non-retryable errors -----------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [ValueError("bad input"), HTTPError(400), HTTPError(401), HTTPError(404), SDKError(422)],
)
def test_non_retryable_error_raises_immediately(error: Exception) -> None:
    fn = Flaky([error])
    sleep = FakeSleep()
    with pytest.raises(type(error)) as caught:
        run(fn, sleep)
    assert caught.value is error
    assert fn.calls == 1
    assert sleep.delays == []


def test_custom_predicate_overrides_default() -> None:
    fn = Flaky([ValueError("flaky parse")])
    sleep = FakeSleep()
    assert run(fn, sleep, retryable=lambda e: isinstance(e, ValueError)) == "ok"
    assert fn.calls == 2


# --- max attempts -------------------------------------------------------------------


def test_gives_up_after_max_attempts_and_reraises_last_error() -> None:
    errors: list[Exception] = [SDKError(503), SDKError(502), SDKError(500), SDKError(429)]
    fn = Flaky(list(errors))
    sleep = FakeSleep()
    with pytest.raises(SDKError) as caught:
        run(fn, sleep, policy=RetryPolicy(attempts=4))
    assert caught.value is errors[-1]
    assert fn.calls == 4
    assert len(sleep.delays) == 3


def test_single_attempt_policy_never_sleeps() -> None:
    fn = Flaky([TimeoutError()])
    sleep = FakeSleep()
    with pytest.raises(TimeoutError):
        run(fn, sleep, policy=RetryPolicy(attempts=1))
    assert sleep.delays == []


# --- Retry-After --------------------------------------------------------------------


def test_retry_after_seconds_header_is_honoured() -> None:
    fn = Flaky([HTTPError(429, {"Retry-After": "7"})])
    sleep = FakeSleep()
    run(fn, sleep, rand=lambda: 0.0)
    assert sleep.delays == [7.0]


def test_retry_after_http_date_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r, "_now", lambda: datetime(2026, 10, 31, 12, 0, 0, tzinfo=UTC))
    fn = Flaky([HTTPError(503, {"Retry-After": "Sat, 31 Oct 2026 12:00:12 GMT"})])
    sleep = FakeSleep()
    run(fn, sleep)
    assert sleep.delays == [12.0]


def test_retry_after_attribute_is_honoured() -> None:
    error = SDKError(429)
    error.retry_after = 3  # type: ignore[attr-defined]
    fn = Flaky([error])
    sleep = FakeSleep()
    run(fn, sleep)
    assert sleep.delays == [3.0]


def test_retry_after_beyond_limit_gives_up_without_waiting() -> None:
    error = HTTPError(429, {"Retry-After": "3600"})
    fn = Flaky([error])
    sleep = FakeSleep()
    with pytest.raises(HTTPError):
        run(fn, sleep, policy=RetryPolicy(max_retry_after_s=60))
    assert sleep.delays == []
    assert fn.calls == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5", 5.0),
        (" 2.5 ", 2.5),
        ("-3", 0.0),
        (4, 4.0),
        ("", None),
        (None, None),
        ("not a date", None),
        ("Sat, 31 Oct 2026 11:59:00 GMT", 0.0),  # in the past
    ],
)
def test_parse_retry_after(
    monkeypatch: pytest.MonkeyPatch, value: str | int | None, expected: float | None
) -> None:
    monkeypatch.setattr(r, "_now", lambda: datetime(2026, 10, 31, 12, 0, 0, tzinfo=UTC))
    assert r.parse_retry_after(value) == expected


# --- classification -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError(), True),
        (TimeoutError(), True),
        (ConnectionResetError(), True),
        (HTTPError(429), True),
        (HTTPError(503), True),
        (SDKError(500), True),
        (HTTPError(400), False),
        (SDKError(403), False),
        (ValueError(), False),
        (type("TimeoutException", (Exception,), {})(), True),  # httpx-shaped
        (type("ReadTimeout", (type("TimeoutException", (Exception,), {}),), {})(), True),
        (type("ConnectTimeout", (type("TimeoutException", (Exception,), {}),), {})(), True),
        (type("ProtocolError", (Exception,), {})(), False),
    ],
)
def test_is_retryable(error: Exception, expected: bool) -> None:
    assert r.is_retryable(error) is expected


@pytest.mark.parametrize(
    "kwargs",
    [{"attempts": 0}, {"base_delay_s": -1}, {"multiplier": 0.5}, {"max_retry_after_s": -1}],
)
def test_invalid_policy_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


# --- decorator ----------------------------------------------------------------------


def test_with_retry_decorator() -> None:
    attempts: list[int] = []

    @r.with_retry(RetryPolicy(attempts=3, base_delay_s=0.01))
    async def fetch(url: str) -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise HTTPError(502)
        return f"body of {url}"

    assert asyncio.run(fetch("https://acme.example")) == "body of https://acme.example"
    assert len(attempts) == 3
    assert fetch.__name__ == "fetch"
