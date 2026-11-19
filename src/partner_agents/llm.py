"""LLM layer (HLD §9). The only module that imports provider SDKs.

``get_chat_model(role)`` returns a ``ModelChain``: the role's model plus its fallbacks.
``invoke_structured()`` returns a validated Pydantic object. For each call it

1. serves a cached reply when ``LLM_CACHE_DIR`` is set and the same question was answered;
2. otherwise asks the first model that works: a model is retried on 429 / 5xx / timeouts
   (``utils/retry.py``) and only when that budget is spent on a 429 / quota / connection
   error does the next fallback get a turn (ADR-018);
3. retries once, with the validation error appended, when the reply does not match.

Token usage and the provider that really answered are recorded per role
(``track_usage`` / ``track_providers``).
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, ValidationError

from partner_agents.config import ModelSpec, Role, Settings, get_settings
from partner_agents.fakes import ScriptedChatModel
from partner_agents.llm_cache import ResponseCache, cache_key
from partner_agents.utils.cache_stats import record_cache
from partner_agents.utils.retry import DEFAULT_POLICY, retry_async, status_code_of

logger = logging.getLogger(__name__)

# Transport retries for model calls (429 / 5xx / timeouts); tests swap in a zero-delay policy.
RETRY_POLICY = DEFAULT_POLICY

# HLD §9. The writer is 0.3 for knowledge-base answers; ``chase_draft`` asks for 0.7.
ROLE_TEMPERATURE: dict[Role, float] = {
    "router": 0.0,
    "extractor": 0.0,
    "writer": 0.3,
    "classifier": 0.0,
    "caller": 0.4,
    "simulator": 0.8,
    "judge": 0.0,
}

# A role without its own LLM_MODEL_<ROLE> uses another role's model id (HLD §9).
_MODEL_FALLBACK_ROLE: dict[Role, Role] = {
    "classifier": "router",
    "caller": "writer",
    "simulator": "writer",
    "judge": "extractor",
}

# Stands in for a model reply when ``LLM_PROVIDER=fake`` and nothing scripted one.
FAKE_REPLY = "fake model reply"


class LLMConfigError(ValueError):
    """The configured provider cannot be built (missing key or model id)."""


# --- building models ----------------------------------------------------------------


def _model_id(role: Role, settings: Settings) -> str:
    for candidate in (role, _MODEL_FALLBACK_ROLE.get(role)):
        model = getattr(settings, f"llm_model_{candidate}", None) if candidate else None
        if model:
            return str(model)
    raise LLMConfigError(
        f"LLM_MODEL_{role.upper()} is not set. Model ids are config, not code: "
        "check the provider's console for a current free-tier model."
    )


def _build_gemini(model: str, temperature: float, settings: Settings) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    if settings.gemini_api_key is None:
        raise LLMConfigError("a gemini model needs GEMINI_API_KEY")
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
        raise LLMConfigError("a groq model needs GROQ_API_KEY")
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
        client_kwargs={"timeout": settings.ollama_timeout_s},
    )


@dataclass(frozen=True)
class Candidate:
    """One model in a chain. ``label`` is ``provider:model`` and names who answered."""

    label: str
    model: BaseChatModel


def _build_candidate(spec: ModelSpec, temperature: float, settings: Settings) -> Candidate:
    if spec.provider == "fake":
        return Candidate("fake", ScriptedChatModel.repeating(FAKE_REPLY))
    if spec.provider == "gemini":
        return Candidate(spec.label, _build_gemini(spec.model, temperature, settings))
    if spec.provider == "groq":
        return Candidate(spec.label, _build_groq(spec.model, temperature, settings))
    return Candidate(spec.label, _build_ollama(spec.model, temperature, settings))


def _primary_spec(role: Role, settings: Settings) -> ModelSpec:
    """``LLM_ROLES`` wins over ``LLM_PROVIDER`` plus the role's model id."""
    override = settings.llm_roles.get(role)
    if override is not None:
        return override
    if settings.llm_provider == "fake":
        return ModelSpec(provider="fake")
    return ModelSpec(provider=settings.llm_provider, model=_model_id(role, settings))


class ModelChain(BaseChatModel):
    """The models one role may use, in order, plus the reply cache directory.

    ``invoke_structured`` understands the whole chain. Plain ``.invoke()`` talks to the
    first model only, with no retry, fallback or cache.
    """

    role: str
    candidates: list[Candidate]
    cache_dir: Path | None = None

    @property
    def primary(self) -> BaseChatModel:
        return self.candidates[0].model

    @property
    def label(self) -> str:
        return self.candidates[0].label

    @property
    def _llm_type(self) -> str:
        return "model-chain"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = self.primary.invoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self.primary.with_structured_output(schema, **kwargs)


def get_chat_model(
    role: Role,
    *,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> ModelChain:
    """The model chain for ``role``.

    ``temperature`` overrides the role's default (for example the writer at 0.7 when
    drafting emails). Raises ``LLMConfigError`` when a key or model id is missing, for the
    primary model or for any ``LLM_FALLBACKS`` entry.
    """
    settings = settings or get_settings()
    temp = ROLE_TEMPERATURE[role] if temperature is None else temperature
    specs = [_primary_spec(role, settings), *settings.llm_fallbacks]
    return ModelChain(
        role=role,
        candidates=[_build_candidate(spec, temp, settings) for spec in specs],
        cache_dir=settings.llm_cache_dir,
    )


# --- concurrency --------------------------------------------------------------------

# One semaphore and one pacer per running event loop: neither may be shared across loops.
_semaphores: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()


class _Pacer:
    """Spaces the start of model calls ``60 / rpm`` seconds apart (``LLM_RPM_LIMIT``)."""

    def __init__(self, rpm: int, loop: asyncio.AbstractEventLoop) -> None:
        self._interval = 60.0 / rpm
        self._loop = loop
        self._next = 0.0

    async def wait(self) -> None:
        now = self._loop.time()
        start = max(now, self._next)
        self._next = start + self._interval  # reserved before sleeping, so tasks queue in order
        if start > now:
            await asyncio.sleep(start - now)


_pacers: WeakKeyDictionary[asyncio.AbstractEventLoop, _Pacer | None] = WeakKeyDictionary()


@asynccontextmanager
async def llm_slot(settings: Settings | None = None) -> AsyncIterator[None]:
    """Hold one of ``LLM_MAX_CONCURRENCY`` slots for the duration of a model call.

    With ``LLM_RPM_LIMIT`` set, each call (a retry counts as one) also waits its turn so that
    starts are spaced evenly, which keeps a free-tier requests-per-minute limit from answering 429.
    """
    loop = asyncio.get_running_loop()
    semaphore = _semaphores.get(loop)
    if semaphore is None:
        active = settings or get_settings()
        semaphore = _semaphores[loop] = asyncio.Semaphore(active.llm_max_concurrency)
        rpm = active.llm_rpm_limit
        _pacers[loop] = _Pacer(rpm, loop) if rpm > 0 else None
    async with semaphore:
        pacer = _pacers.get(loop)
        if pacer is not None:
            await pacer.wait()
        yield


# --- token usage and providers ------------------------------------------------------

_usage: ContextVar[dict[str, int] | None] = ContextVar("llm_usage", default=None)
_providers: ContextVar[dict[str, str] | None] = ContextVar("llm_providers", default=None)


@contextmanager
def track_usage() -> Iterator[dict[str, int]]:
    """Collect total tokens per role for model calls made inside the block.

    Tasks started inside the block share the same dict, so a fan-out is counted too. The
    result is what ``RunRecord.tokens`` stores. A reply served from the cache costs 0.
    """
    usage: dict[str, int] = {}
    token = _usage.set(usage)
    try:
        yield usage
    finally:
        _usage.reset(token)


@contextmanager
def track_providers() -> Iterator[dict[str, str]]:
    """Collect, per role, the ``provider:model`` that really answered.

    When a fallback answered some calls, the labels are joined with ``+``. The result is
    what ``RunRecord.providers`` stores.
    """
    providers: dict[str, str] = {}
    token = _providers.set(providers)
    try:
        yield providers
    finally:
        _providers.reset(token)


def _record_usage(role: Role, message: AIMessage) -> None:
    usage = _usage.get()
    meta = message.usage_metadata
    if usage is None or not meta:
        return
    total = meta.get("total_tokens") or meta.get("input_tokens", 0) + meta.get("output_tokens", 0)
    usage[role] = usage.get(role, 0) + total


def _record_provider(role: Role, label: str) -> None:
    providers = _providers.get()
    if providers is None:
        return
    seen = providers.get(role, "").split("+") if role in providers else []
    if label not in seen:
        providers[role] = "+".join([*seen, label])


# --- fallback and structured output -------------------------------------------------

_CONNECTION_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutException",
    }
)


def is_fallback_worthy(exc: BaseException) -> bool:
    """429, quota exhaustion, or a connection failure: the model cannot answer right now.

    A validation failure never qualifies, and neither does a 4xx/5xx that another model
    would hit just the same.
    """
    if isinstance(exc, (ValidationError, OutputParserException)):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if type(exc).__name__ in _CONNECTION_ERROR_NAMES:
        return True
    code = status_code_of(exc)
    if code is None and isinstance(getattr(exc, "code", None), int):
        code = exc.code  # type: ignore[attr-defined]  # google-genai errors carry .code
    if code == 429:
        return True
    text = str(exc).lower()
    return "quota" in text or "resource_exhausted" in text


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


async def _call_chain(
    candidates: Sequence[Candidate],
    make_runnable: Callable[[BaseChatModel], Any],
    conversation: Sequence[BaseMessage],
    role: Role,
) -> tuple[Any, str]:
    """The first reply any candidate gives, and the label of the candidate that gave it."""
    for index, candidate in enumerate(candidates):
        try:
            return await _call(make_runnable(candidate.model), conversation), candidate.label
        except Exception as exc:
            if index + 1 == len(candidates) or not is_fallback_worthy(exc):
                raise
            logger.warning(
                "model unavailable, trying the next one",
                extra={
                    "role": role,
                    "failed": candidate.label,
                    "next": candidates[index + 1].label,
                    "error_type": type(exc).__name__,
                },
            )
    raise AssertionError("unreachable")  # pragma: no cover


async def _cached_reply[T: BaseModel](
    cache: ResponseCache | None, key: str, schema: type[T], role: Role
) -> T | None:
    if cache is None:
        return None
    entry = await cache.get(key)
    if entry is None:
        record_cache("llm", misses=1)
        return None
    try:
        parsed = schema.model_validate_json(entry["parsed"])
    except (KeyError, TypeError, ValidationError):
        record_cache("llm", misses=1)
        return None  # a stale or damaged entry is a miss
    record_cache("llm", hits=1)
    _record_provider(role, str(entry.get("label", "cache")))
    logger.info("llm cache hit", extra={"role": role, "schema": schema.__name__})
    return parsed


async def invoke_structured[T: BaseModel](
    model: BaseChatModel,
    schema: type[T],
    messages: Sequence[BaseMessage],
    *,
    role: Role,
    prompt_version: str = "",
) -> T:
    """Ask ``model`` for a ``schema`` object.

    An invalid reply gets one retry with the validation error appended to the
    conversation. A second invalid reply raises ``StructuredOutputError``. Transport
    errors (429, 5xx, timeouts) are retried separately and never trigger the validation
    retry. Tokens of every attempt are recorded under ``role`` (see ``track_usage``).

    With a ``ModelChain`` the call also uses the fallbacks and the reply cache. The cache
    key covers the primary model, temperature, ``prompt_version``, the schema and the
    messages; only valid replies are cached.
    """
    chain = model if isinstance(model, ModelChain) else None
    candidates = chain.candidates if chain else [Candidate(type(model).__name__, model)]
    cache = ResponseCache(chain.cache_dir) if chain and chain.cache_dir else None
    key = cache_key(
        candidates[0].label,
        getattr(candidates[0].model, "temperature", None),
        prompt_version,
        schema.__name__,
        schema.model_json_schema(),
        [(m.type, m.content) for m in messages],
    )
    if (hit := await _cached_reply(cache, key, schema, role)) is not None:
        return hit

    def make_runnable(candidate_model: BaseChatModel) -> Any:
        return candidate_model.with_structured_output(schema, include_raw=True)

    conversation = list(messages)
    for attempt in (1, 2):
        raw: AIMessage | None = None
        error: BaseException | None = None
        parsed: T | None = None
        label = candidates[0].label
        try:
            result, label = await _call_chain(candidates, make_runnable, conversation, role)
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
            _record_provider(role, label)
            if cache is not None:
                await cache.put(key, {"label": label, "parsed": parsed.model_dump_json()})
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
