"""Langfuse tracing behind a small protocol (HLD section 13, ADR-029).

``build_tracing`` returns ``None`` unless both ``LANGFUSE_*`` keys are set, and the ``langfuse``
package (an optional extra: ``uv sync --extra tracing``) is imported only then. The graph code
talks to ``Tracing`` and ``RunTrace``, so tests use ``FakeTracing`` and a tracing failure can
never fail a run: ``start_trace`` and ``finish_trace`` swallow every error and return ``None``.

A trace holds what the models saw, which includes scraped page text and, for calls, the
partner's words (HLD section 12). We add only ids, the run name and the prompt versions to its
metadata, never a contact value or an email body.
"""

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, Protocol

from partner_agents.config import Settings

logger = logging.getLogger(__name__)

TAG = "cpg-partner-agents"


class TracingUnavailable(RuntimeError):
    """Tracing is switched on (keys are set) but the ``langfuse`` package is not installed."""


class RunTrace(Protocol):
    """One traced run: what to put in the LangChain ``config``, and how to close it."""

    @property
    def callbacks(self) -> list[Any]: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    def finish(self) -> str | None:
        """The trace URL, or ``None`` if it cannot be resolved. May block briefly (network)."""
        ...


class Tracing(Protocol):
    def start(
        self,
        *,
        name: str,
        session_id: str,
        user_id: str,
        metadata: Mapping[str, Any],
    ) -> RunTrace: ...

    def shutdown(self) -> None:
        """Send what is still queued. Called once when the app closes."""
        ...


def trace_metadata(
    name: str, session_id: str, user_id: str, extra: Mapping[str, Any]
) -> dict[str, Any]:
    """The keys the Langfuse handler reads (session, user, name, tags), plus ``extra``."""
    return {
        "langfuse_session_id": session_id,
        "langfuse_user_id": user_id,
        "langfuse_trace_name": name,
        "langfuse_tags": [TAG],
        **extra,
    }


class LangfuseRunTrace:
    def __init__(self, client: Any, handler: Any, metadata: dict[str, Any]) -> None:
        self._client, self._handler, self._metadata = client, handler, metadata

    @property
    def callbacks(self) -> list[Any]:
        return [self._handler]

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def finish(self) -> str | None:
        trace_id = getattr(self._handler, "last_trace_id", None)
        if not trace_id:
            return None
        url = self._client.get_trace_url(trace_id=trace_id)
        return str(url) if url else None


class LangfuseTracing:
    """The real thing. Building it needs the ``tracing`` extra."""

    def __init__(self, settings: Settings) -> None:
        public, secret = settings.langfuse_public_key, settings.langfuse_secret_key
        if public is None or secret is None:
            raise TracingUnavailable("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are not set.")
        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler
        except ImportError as exc:
            raise TracingUnavailable(
                "Install the tracing extra first: uv sync --extra tracing"
            ) from exc
        options: dict[str, Any] = {}
        if settings.langfuse_host:
            options["base_url"] = settings.langfuse_host
        self._client = Langfuse(
            public_key=public.get_secret_value(), secret_key=secret.get_secret_value(), **options
        )
        self._handler_type = CallbackHandler

    def start(
        self, *, name: str, session_id: str, user_id: str, metadata: Mapping[str, Any]
    ) -> RunTrace:
        return LangfuseRunTrace(
            self._client,
            self._handler_type(),  # one handler per run: it keeps that run's last trace id
            trace_metadata(name, session_id, user_id, metadata),
        )

    def shutdown(self) -> None:
        self._client.shutdown()


def build_tracing(settings: Settings) -> Tracing | None:
    """The tracer when both Langfuse keys are set, else ``None``. Raises ``TracingUnavailable``."""
    return LangfuseTracing(settings) if settings.tracing_enabled else None


def start_trace(
    tracing: Tracing | None,
    *,
    name: str,
    session_id: str,
    user_id: str,
    metadata: Mapping[str, Any],
) -> RunTrace | None:
    """Start a trace, or ``None`` when tracing is off or fails. Never raises."""
    if tracing is None:
        return None
    try:
        return tracing.start(name=name, session_id=session_id, user_id=user_id, metadata=metadata)
    except Exception as exc:
        logger.warning("tracing could not start", extra={"error_type": type(exc).__name__})
        return None


async def finish_trace(trace: RunTrace | None) -> str | None:
    """The trace URL, or ``None``. Never raises, and never blocks the event loop."""
    if trace is None:
        return None
    try:
        return await asyncio.to_thread(trace.finish)
    except Exception as exc:
        logger.warning("tracing could not finish", extra={"error_type": type(exc).__name__})
        return None


def traced_config[T: Mapping[str, Any]](config: T, trace: RunTrace | None) -> dict[str, Any]:
    """``config`` plus the trace's callbacks and metadata (unchanged when there is no trace)."""
    merged: dict[str, Any] = dict(config)
    if trace is not None:
        merged["callbacks"] = [*merged.get("callbacks", []), *trace.callbacks]
        merged["metadata"] = {**merged.get("metadata", {}), **trace.metadata}
    return merged
