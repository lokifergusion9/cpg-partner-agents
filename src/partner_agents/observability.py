"""Logging, run context and stage timing (HLD §11, §13).

- ``configure_logging()`` installs one JSON handler on the root logger.
- ``run_context()`` sets the ``run_id`` that every log record carries automatically.
- ``stage_timer()`` / ``timed_node()`` record ``timings_ms`` per stage or graph node;
  ``timed_node()`` also turns an unexpected exception into a ``StageError``.
- ``RedactionFilter`` removes secrets at every level, and emails and phone numbers at
  INFO and above (CLAUDE.md hard rule 5).
"""

import functools
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from partner_agents.config import Settings, get_settings
from partner_agents.schemas import StageError

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_node: ContextVar[str | None] = ContextVar("node", default=None)

# Indirection so tests can mock the clock.
_clock: Callable[[], float] = time.perf_counter

logger = logging.getLogger(__name__)


# --- run context --------------------------------------------------------------------


def current_run_id() -> str | None:
    return _run_id.get()


def current_node() -> str | None:
    return _node.get()


@contextmanager
def run_context(run_id: str) -> Iterator[None]:
    """Attach ``run_id`` to every log record emitted inside the block (and its tasks)."""
    token = _run_id.set(run_id)
    try:
        yield
    finally:
        _run_id.reset(token)


# --- timing -------------------------------------------------------------------------


@contextmanager
def stage_timer(stage: str, timings: MutableMapping[str, int]) -> Iterator[None]:
    """Time a stage into ``timings[stage]`` (ms), even if the stage raises."""
    token = _node.set(stage)
    start = _clock()
    try:
        yield
    finally:
        duration_ms = round((_clock() - start) * 1000)
        timings[stage] = duration_ms
        logger.info("stage finished", extra={"duration_ms": duration_ms})
        _node.reset(token)


NodeUpdate = dict[str, Any]


def _stage_error(stage: str, exc: Exception) -> StageError:
    # The message goes into graph state and the API, so contact values are removed here too.
    return StageError(stage=stage, code=type(exc).__name__, message=redact_contacts(str(exc)))


def timed_node[**P](
    name: str,
) -> Callable[[Callable[P, Awaitable[NodeUpdate | None]]], Callable[P, Awaitable[NodeUpdate]]]:
    """Decorate an async graph node (HLD §11).

    The state update carries ``timings_ms[name]``. An unexpected exception becomes a
    ``StageError`` in the update's ``errors`` so a partial failure never crashes the run.
    With ``STRICT_NODES=true`` it is re-raised instead.
    """

    def decorate(
        fn: Callable[P, Awaitable[NodeUpdate | None]],
    ) -> Callable[P, Awaitable[NodeUpdate]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> NodeUpdate:
            timings: dict[str, int] = {}
            try:
                with stage_timer(name, timings):
                    update = await fn(*args, **kwargs)
            except Exception as exc:
                if get_settings().strict_nodes:
                    raise
                logger.exception("node failed", extra={"error_type": type(exc).__name__})
                update = {"errors": [_stage_error(name, exc)]}
            out = dict(update or {})
            out["timings_ms"] = {**out.get("timings_ms", {}), **timings}
            return out

        return wrapper

    return decorate


# --- redaction ----------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+\s*(?:@|\[at\]|\(at\))\s*[A-Za-z0-9.-]+\s*(?:\.|\[dot\]|\(dot\))\s*[A-Za-z]{2,}",
    re.IGNORECASE,
)
# Phone shapes: "+<country> ..." international, or 10-digit NANP / Indian mobile forms.
# Dates (2026-10-31), durations and ids do not match.
_PHONE_RE = re.compile(
    r"(?<![\w+])(?:"
    r"\+\d{1,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?){2,4}\d{2,4}"
    r"|\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
    r")(?!\w)"
)
# Well-known key shapes, in case a key reaches a log without being in Settings.
_KEY_RE = re.compile(
    r"\b(?:AIza[0-9A-Za-z_-]{20,}|gsk_[0-9A-Za-z]{20,}|tvly-[0-9A-Za-z-]{16,}"
    r"|sk-lf-[0-9A-Za-z-]{8,}|pk-lf-[0-9A-Za-z-]{8,}|sk-[0-9A-Za-z_-]{20,})"
)

# Attributes every LogRecord has; anything else came from ``extra=``.
_STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def redact_contacts(text: str) -> str:
    return _PHONE_RE.sub("[phone]", _EMAIL_RE.sub("[email]", text))


class RedactionFilter(logging.Filter):
    """Redact secrets always, and contact values from INFO up."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        # Longest first, so a secret containing another is replaced whole.
        self._secrets = sorted({s for s in secrets if len(s) >= 4}, key=len, reverse=True)

    def redact(self, text: str, *, contacts: bool) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[secret]")
        text = _KEY_RE.sub("[secret]", text)
        return redact_contacts(text) if contacts else text

    def filter(self, record: logging.LogRecord) -> bool:
        contacts = record.levelno >= logging.INFO
        record.msg = self.redact(record.getMessage(), contacts=contacts)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self.redact(record.exc_text, contacts=contacts)
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and isinstance(value, str):
                setattr(record, key, self.redact(value, contacts=contacts))
        return True


# --- formatting ---------------------------------------------------------------------


class RunContextFilter(logging.Filter):
    """Copy ``run_id`` and ``node`` from the context onto the record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "run_id", None) is None:
            record.run_id = current_run_id()
        if getattr(record, "node", None) is None:
            record.node = current_node()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg, run_id, node, extras, exc."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
            "node": getattr(record, "node", None),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and key not in payload:
                payload[key] = value
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            payload["exc"] = record.exc_text
        return json.dumps(payload, default=str, ensure_ascii=False)


def secret_values(settings: Settings) -> list[str]:
    """Every secret value configured in ``settings``."""
    values = []
    for name in type(settings).model_fields:
        value = getattr(settings, name)
        if hasattr(value, "get_secret_value"):
            values.append(value.get_secret_value())
    return values


_HANDLER_NAME = "partner_agents.json"


def build_handler(settings: Settings, stream: Any = None) -> logging.Handler:
    handler = logging.StreamHandler(stream)
    handler.set_name(_HANDLER_NAME)
    handler.addFilter(RunContextFilter())
    handler.addFilter(RedactionFilter(secret_values(settings)))
    handler.setFormatter(JsonFormatter())
    return handler


def configure_logging(settings: Settings, stream: Any = None) -> None:
    """Install the JSON handler on the root logger. Safe to call more than once."""
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == _HANDLER_NAME:
            root.removeHandler(existing)
    root.addHandler(build_handler(settings, stream))
    root.setLevel(settings.log_level)
