"""The discovery rate limit: the limiter with mocked time, and 429 through the API (HLD §7)."""

from datetime import UTC, datetime, timedelta
from typing import Any

import respx
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from partner_agents.api.main import create_app
from partner_agents.api.ratelimit import MAX_KEYS, RateLimiter
from tests.api.test_api import error_of
from tests.graph.test_build import (  # noqa: F401  (fixtures are used by name)
    DISCOVERY,
    client,
    decision,
    instant_retries,
    make_deps,
    serve_sites,
)

T0 = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


# --- the limiter --------------------------------------------------------------------


def test_the_sixth_run_in_an_hour_is_refused_with_the_wait() -> None:
    clock = Clock()
    limiter = RateLimiter(5, clock=clock)
    assert [limiter.take(["user:a"]) for _ in range(5)] == [None] * 5
    assert limiter.take(["user:a"]) == 720  # 3600 s / 5 runs = one token per 12 minutes
    clock.advance(minutes=6)
    assert limiter.take(["user:a"]) == 360  # the wait shrinks as time passes


def test_the_limit_recovers_after_the_refill() -> None:
    clock = Clock()
    limiter = RateLimiter(5, clock=clock)
    for _ in range(5):
        limiter.take(["user:a"])
    assert limiter.take(["user:a"]) is not None
    clock.advance(seconds=719)
    assert limiter.take(["user:a"]) == 1
    clock.advance(seconds=1)
    assert limiter.take(["user:a"]) is None  # one token is back
    assert limiter.take(["user:a"]) == 720  # and only one
    clock.advance(hours=2)
    assert [limiter.take(["user:a"]) for _ in range(5)] == [None] * 5  # full again, not more
    assert limiter.take(["user:a"]) is not None


def test_keys_are_independent() -> None:
    limiter = RateLimiter(1, clock=Clock())
    assert limiter.take(["user:a"]) is None
    assert limiter.take(["user:a"]) == 3600
    assert limiter.take(["user:b"]) is None


def test_a_refused_request_spends_nothing_on_its_other_keys() -> None:
    limiter = RateLimiter(2, clock=Clock())
    limiter.take(["user:a"])
    limiter.take(["user:a"])
    assert limiter.take(["user:a", "ip:1"]) is not None  # user:a is empty
    # ip:1 kept both of its tokens
    assert limiter.take(["ip:1"]) is None and limiter.take(["ip:1"]) is None
    assert limiter.take(["ip:1"]) is not None


def test_one_ip_is_limited_across_many_users() -> None:
    limiter = RateLimiter(2, clock=Clock())
    assert limiter.take(["user:a", "ip:9"]) is None
    assert limiter.take(["user:b", "ip:9"]) is None
    assert limiter.take(["user:c", "ip:9"]) == 1800  # a fresh user does not escape the IP


def test_a_repeated_key_in_one_request_is_counted_once() -> None:
    limiter = RateLimiter(1, clock=Clock())
    assert limiter.take(["user:a", "user:a"]) is None


def test_zero_capacity_always_refuses() -> None:
    assert RateLimiter(0, clock=Clock()).take(["user:a"]) == 3600


def test_idle_full_buckets_are_dropped_when_the_table_grows() -> None:
    clock = Clock()
    limiter = RateLimiter(5, clock=clock)
    for n in range(MAX_KEYS + 1):
        limiter.take([f"ip:{n}"])
    clock.advance(hours=2)  # every bucket has refilled
    limiter.take(["user:new"])
    assert len(limiter._buckets) < 10


# --- through the API ----------------------------------------------------------------


def discovery_app(qdrant: AsyncQdrantClient, clock: Clock, runs: int, *extra: Any) -> TestClient:
    deps = make_deps(qdrant, *[DISCOVERY] * runs, *extra)
    deps.rate_limiter = RateLimiter(5, clock=clock)
    return TestClient(create_app(deps))


def ask(web: TestClient, text: str = "find cookie brokers", user: str = "u_1") -> Any:
    return web.post("/api/chat", json={"user_id": user, "message": text})


@respx.mock
def test_the_sixth_discovery_in_an_hour_is_429_then_recovers(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    serve_sites()
    clock = Clock()
    with discovery_app(client, clock, 7) as web:
        assert [ask(web).status_code for _ in range(5)] == [200] * 5
        blocked = ask(web)
        assert blocked.status_code == 429
        error = error_of(blocked)
        assert error["code"] == "rate_limited" and error["retry_after_s"] == 720
        assert blocked.headers["Retry-After"] == "720"
        assert "12 minutes" in error["message"]
        clock.advance(minutes=12)
        assert ask(web).status_code == 200  # one token is back


@respx.mock
def test_a_refusal_does_not_stick_to_the_thread(client: AsyncQdrantClient) -> None:  # noqa: F811
    serve_sites()
    clock = Clock()
    with discovery_app(client, clock, 6, decision("cpg_qa")) as web:
        for _ in range(5):
            ask(web)
        refused = ask(web)
        assert refused.status_code == 429
        # A question is not a discovery run, so the limit does not apply to it.
        answered = ask(web, "what is a slotting fee?")
        assert answered.status_code == 200 and answered.json()["route"]["intent"] == "cpg_qa"


@respx.mock
def test_the_limit_counts_the_ip_as_well_as_the_user(client: AsyncQdrantClient) -> None:  # noqa: F811
    serve_sites()
    clock = Clock()
    with discovery_app(client, clock, 6) as web:
        for n in range(5):
            assert ask(web, user=f"u_{n}").status_code == 200  # five users, one client IP
        other = ask(web, user="u_new")
        assert other.status_code == 429 and error_of(other)["code"] == "rate_limited"
