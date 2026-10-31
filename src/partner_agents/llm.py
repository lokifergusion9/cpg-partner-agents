"""LLM layer (HLD §9). The only module that imports provider SDKs.

``get_chat_model(role)`` returns a LangChain chat model for the configured provider.
``llm_slot()`` bounds how many model calls run at once, so fan-out stays under free-tier
request limits. ``invoke_structured()`` returns a validated Pydantic object, retrying once
with the validation error, and records token usage per role. Per-role providers,
fallbacks and the response cache arrive in T-53.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, Literal
from weakref import WeakKeyDictionary

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from partner_agents.config import LLMProvider, Settings, get_settings
from partner_agents.fakes import ScriptedChatModel
from partner_agents.utils.retry import DEFAULT_POLICY, retry_async

logger = logging.getLogger(__name__)

# Transport retries for model calls (429 / 5xx / timeouts); tests swap in a zero-delay policy.
RETRY_POLICY = DEFAULT_POLICY

Role = Literal["router", "extractor", "writer"]

# HLD §9. The writer is 0.3 for knowledge-base answers; ``chase_draft`` asks for 0.7.
ROLE_TEMPERATURE: dict[Role, float] = {"router": 0.0, "extractor": 0.0, "writer": 0.3}

# Stands in for a model reply when ``LLM_PROVIDER=fake`` and nothing scripted one.
FAKE_REPLY = "fake model reply"


class LLMConfigError(ValueError):
    """The configured provider cannot be built (missing key or model id)."""


def _model_id(role: Role, settings: Settings) -> str:
    model = getattr(settings, f"llm_model_{role}")
    if not model:
        raise LLMConfigError(
            f"LLM_MODEL_{role.upper()} is not set. Model ids are config, not code: "
            "check the provider's console for a current free-tier model."
        )
    return str(model)


def _build_gemini(model: str, temperature: float, settings: Settings) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    if settings.gemini_api_key is None:
        raise LLMConfigError("LLM_PROVIDER=gemini needs GEMINI_API_KEY")
    # SDK-level retries are off: ``utils/retry.py`` owns the retry budget (HLD §9).
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=settings.gemini_api_key,
        timeout=settings.llm_timeout_s,
        retries=0,
    )


def _build_groq(model: str, temperature: float, settings: Settings) -> BaseChatModel:
    from langchain_groq import ChatGroq

    if settings.groq_api_key is None:
        raise LLMConfigError("LLM_PROVIDER=groq needs GROQ_API_KEY")
    # Field names rather than aliases (model, api_key, timeout): mypy only knows the former.
    return ChatGroq(
        model_name=model,
        temperature=temperature,
        groq_api_key=settings.groq_api_key,
        request_timeout=settings.llm_timeout_s,
        max_retries=0,
    )


def _build_ollama(model: str, temperature: float, settings: Settings) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.llm_timeout_s},
    )


def _build_fake() -> BaseChatModel:
    return ScriptedChatModel.repeating(FAKE_REPLY)


def get_chat_model(
    role: Role,
    *,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """The chat model for ``role`` on ``LLM_PROVIDER``.

    ``temperature`` overrides the role's default (for example the writer at 0.7 when
    drafting emails). Raises ``LLMConfigError`` when a key or model id is missing.
    """
    settings = settings or get_settings()
    temp = ROLE_TEMPERATURE[role] if temperature is None else temperature
    provider: LLMProvider = settings.llm_provider
    if provider == "fake":
        return _build_fake()
    model = _model_id(role, settings)
    if provider == "gemini":
        return _build_gemini(model, temp, settings)
    if provider == "groq":
        return _build_groq(model, temp, settings)
    return _build_ollama(model, temp, settings)


# --- concurrency --------------------------------------------------------------------

# One semaphore per running event loop: a semaphore must not be shared across loops.
_semaphores: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()


@asynccontextmanager
async def llm_slot(settings: Settings | None = None) -> AsyncIterator[None]:
    """Hold one of ``LLM_MAX_CONCURRENCY`` slots for the duration of a model call."""
    loop = asyncio.get_running_loop()
    semaphore = _semaphores.get(loop)
    if semaphore is None:
        limit = (settings or get_settings()).llm_max_concurrency
        semaphore = _semaphores[loop] = asyncio.Semaphore(limit)
    async with semaphore:
        yield


# --- token usage --------------------------------------------------------------------

_usage: ContextVar[dict[str, int] | None] = ContextVar("llm_usage", default=None)


@contextmanager
def track_usage() -> Iterator[dict[str, int]]:
    """Collect total tokens per role for model calls made inside the block.

    Tasks started inside the block share the same dict, so a fan-out is counted too. The
    result is what ``RunRecord.tokens`` stores.
    """
    usage: dict[str, int] = {}
    token = _usage.set(usage)
    try:
        yield usage
    finally:
        _usage.reset(token)


def _record_usage(role: Role, message: AIMessage) -> None:
    usage = _usage.get()
    meta = message.usage_metadata
    if usage is None or not meta:
        return
    total = meta.get("total_tokens") or meta.get("input_tokens", 0) + meta.get("output_tokens", 0)
    usage[role] = usage.get(role, 0) + total


# --- structured output --------------------------------------------------------------


class StructuredOutputError(Exception):
    """The model's reply did not match the schema, even after one retry with the error."""

    def __init__(self, schema: str, detail: str) -> None:
        super().__init__(f"{schema}: {detail}")
        self.schema = schema
        self.detail = detail


def _describe(error: BaseException | None) -> str:
    """A short, input-free description of why a reply was invalid."""
    if isinstance(error, ValidationError):
        problems = [
            f"{'.'.join(str(part) for part in e['loc']) or 'reply'}: {e['msg']}"
            for e in error.errors()
        ]
        return "; ".join(problems)[:600]
    if error is not None:
        return str(error)[:300]
    return "the reply held no structured output"


def _retry_messages(raw: AIMessage | None, detail: str) -> list[BaseMessage]:
    note = HumanMessage(
        content=f"Your previous reply was invalid: {detail}. "
        "Reply again with output that matches the required schema exactly."
    )
    return [raw, note] if raw is not None else [note]


async def _call(runnable: Any, conversation: Sequence[BaseMessage]) -> Any:
    """One model call: bounded by the concurrency slot, retried on 429 / 5xx / timeouts."""

    async def once() -> Any:
        async with llm_slot():
            return await runnable.ainvoke(list(conversation))

    return await retry_async(once, policy=RETRY_POLICY)


async def invoke_structured[T: BaseModel](
    model: BaseChatModel,
    schema: type[T],
    messages: Sequence[BaseMessage],
    *,
    role: Role,
) -> T:
    """Ask ``model`` for a ``schema`` object.

    An invalid reply gets one retry with the validation error appended to the
    conversation. A second invalid reply raises ``StructuredOutputError``. Transport
    errors (429, 5xx, timeouts) are retried separately and never trigger the validation
    retry. Tokens of every attempt are recorded under ``role`` (see ``track_usage``).
    """
    runnable = model.with_structured_output(schema, include_raw=True)
    conversation = list(messages)
    for attempt in (1, 2):
        raw: AIMessage | None = None
        error: BaseException | None = None
        parsed: T | None = None
        try:
            result = await _call(runnable, conversation)
        except (ValidationError, OutputParserException) as exc:
            error = exc
        else:
            if isinstance(result, dict):
                raw_value = result.get("raw")
                raw = raw_value if isinstance(raw_value, AIMessage) else None
                error = result.get("parsing_error")
                value = result.get("parsed")
            else:
                value = result
            if error is None and value is not None:
                try:
                    parsed = value if isinstance(value, schema) else schema.model_validate(value)
                except ValidationError as exc:
                    error = exc
        if raw is not None:
            _record_usage(role, raw)
        if parsed is not None:
            return parsed
        detail = _describe(error)
        logger.warning(
            "structured output invalid",
            extra={"role": role, "schema": schema.__name__, "attempt": attempt},
        )
        if attempt == 2:
            raise StructuredOutputError(schema.__name__, detail) from error
        conversation = [*conversation, *_retry_messages(raw, detail)]
    raise AssertionError("unreachable")  # pragma: no cover
