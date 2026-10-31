import asyncio
import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from partner_agents import observability as obs
from partner_agents.config import Settings


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class Capture:
    def __init__(self, logger: logging.Logger, stream: io.StringIO) -> None:
        self.logger = logger
        self._stream = stream

    def records(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self._stream.getvalue().splitlines()]

    def last(self) -> dict[str, Any]:
        return self.records()[-1]


@pytest.fixture
def capture() -> Iterator[Capture]:
    """The production handler on the package logger, writing JSON lines to a buffer.

    Module loggers (``partner_agents.*``) propagate to it, as they do in production.
    """
    stream = io.StringIO()
    settings = make_settings(
        gemini_api_key="AIzaFAKE-but-long-enough-key-123", app_api_key="hunter2"
    )
    package = logging.getLogger("partner_agents")
    package.handlers = [obs.build_handler(settings, stream)]
    package.setLevel(logging.DEBUG)
    package.propagate = False
    yield Capture(logging.getLogger("partner_agents.tests"), stream)
    package.handlers = []
    package.setLevel(logging.NOTSET)
    package.propagate = True


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Each call to the clock pops the next value."""
    ticks: list[float] = []
    monkeypatch.setattr(obs, "_clock", lambda: ticks.pop(0))
    return ticks


# --- run context --------------------------------------------------------------------


def test_run_id_appears_in_log_record(capture: Capture) -> None:
    with obs.run_context("r_51ab"):
        capture.logger.info("hello")
    record = capture.last()
    assert record["run_id"] == "r_51ab"
    assert record["msg"] == "hello"
    assert record["level"] == "INFO"
    assert record["logger"] == "partner_agents.tests"


def test_run_id_is_reset_after_context(capture: Capture) -> None:
    with obs.run_context("r_1"):
        pass
    capture.logger.info("outside")
    assert capture.last()["run_id"] is None
    assert obs.current_run_id() is None


def test_run_id_propagates_into_tasks(capture: Capture) -> None:
    async def child() -> None:
        capture.logger.info("from task")

    async def main() -> None:
        with obs.run_context("r_async"):
            await asyncio.gather(child(), child())

    asyncio.run(main())
    assert [r["run_id"] for r in capture.records()] == ["r_async", "r_async"]


def test_extra_fields_are_included(capture: Capture) -> None:
    capture.logger.info("fetched", extra={"url": "https://example.com", "chars": 120})
    record = capture.last()
    assert record["url"] == "https://example.com"
    assert record["chars"] == 120


# --- timing -------------------------------------------------------------------------


def test_stage_timer_records_timing_and_logs_it(capture: Capture, fake_clock: list[float]) -> None:
    fake_clock.extend([10.0, 10.25])
    timings: dict[str, int] = {}
    with obs.run_context("r_t"), obs.stage_timer("scout_fetch", timings):
        pass
    assert timings == {"scout_fetch": 250}
    record = capture.last()
    assert record["msg"] == "stage finished"
    assert record["node"] == "scout_fetch"
    assert record["duration_ms"] == 250
    assert record["run_id"] == "r_t"


def test_stage_timer_records_timing_when_stage_raises(fake_clock: list[float]) -> None:
    fake_clock.extend([1.0, 1.5])
    timings: dict[str, int] = {}
    with pytest.raises(RuntimeError), obs.stage_timer("scout_search", timings):
        raise RuntimeError("boom")
    assert timings == {"scout_search": 500}
    assert obs.current_node() is None


def test_timed_node_adds_timing_to_state_update(fake_clock: list[float]) -> None:
    fake_clock.extend([0.0, 0.042])

    @obs.timed_node("charlie_route")
    async def charlie_route(state: dict[str, Any]) -> dict[str, Any]:
        assert obs.current_node() == "charlie_route"
        return {"route": "discovery"}

    update = asyncio.run(charlie_route({}))
    assert update == {"route": "discovery", "timings_ms": {"charlie_route": 42}}
    assert charlie_route.__name__ == "charlie_route"


def test_timed_node_accepts_none_update(fake_clock: list[float]) -> None:
    fake_clock.extend([0.0, 0.001])

    @obs.timed_node("refuse")
    async def refuse(state: dict[str, Any]) -> None:
        return None

    assert asyncio.run(refuse({})) == {"timings_ms": {"refuse": 1}}


# --- redaction ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("contact sales@acme-foods.com today", "contact [email] today"),
        ("obfuscated info [at] acme [dot] com", "obfuscated [email]"),
        ("call +1 (310) 555-0100 now", "call [phone] now"),
        ("call 310-555-0100", "call [phone]"),
        ("call (310) 555 0100", "call [phone]"),
        ("mobile +91 98765 43210", "mobile [phone]"),
        ("mobile 9876543210", "mobile [phone]"),
    ],
)
def test_contacts_are_redacted_at_info(capture: Capture, message: str, expected: str) -> None:
    capture.logger.info(message)
    assert capture.last()["msg"] == expected


@pytest.mark.parametrize(
    "message",
    [
        "run started 2026-10-31T06:42:00",
        "fetched 12 pages in 3400 ms",
        "run_id r_51ab took 12345 ms",
        "status 200 for https://acme.example/contact",
    ],
)
def test_non_contacts_are_not_redacted(capture: Capture, message: str) -> None:
    capture.logger.info(message)
    assert capture.last()["msg"] == message


def test_email_in_format_args_and_extras_is_redacted(capture: Capture) -> None:
    capture.logger.warning(
        "verified %s", "buyer@retailer.example", extra={"recipient": "ops@x.example"}
    )
    record = capture.last()
    assert record["msg"] == "verified [email]"
    assert record["recipient"] == "[email]"


def test_contacts_are_kept_at_debug(capture: Capture) -> None:
    capture.logger.debug("candidate sales@acme-foods.com")
    assert capture.last()["msg"] == "candidate sales@acme-foods.com"


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.ERROR])
def test_configured_secrets_are_redacted_at_every_level(capture: Capture, level: int) -> None:
    capture.logger.log(level, "key=AIzaFAKE-but-long-enough-key-123 api=hunter2")
    assert capture.last()["msg"] == "key=[secret] api=[secret]"


def test_unconfigured_key_shapes_are_redacted(capture: Capture) -> None:
    capture.logger.debug("groq gsk_abcdefghijklmnopqrstuvwx tavily tvly-abcdefghijklmnop12")
    assert capture.last()["msg"] == "groq [secret] tavily [secret]"


def test_exception_text_is_redacted(capture: Capture) -> None:
    try:
        raise ValueError("bad address sales@acme-foods.com with key hunter2")
    except ValueError:
        capture.logger.exception("extraction failed")
    exc = capture.last()["exc"]
    assert "ValueError" in exc
    assert "sales@acme-foods.com" not in exc
    assert "hunter2" not in exc


# --- configure_logging --------------------------------------------------------------


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    try:
        stream = io.StringIO()
        obs.configure_logging(make_settings(log_level="warning"), stream)
        obs.configure_logging(make_settings(log_level="warning"), stream)
        ours = [h for h in root.handlers if h.get_name() == "partner_agents.json"]
        assert len(ours) == 1
        assert root.level == logging.WARNING
        logging.getLogger("any.module").warning("hi sales@acme-foods.com")
        assert json.loads(stream.getvalue().splitlines()[-1])["msg"] == "hi [email]"
    finally:
        root.handlers = before
        root.setLevel(level)
