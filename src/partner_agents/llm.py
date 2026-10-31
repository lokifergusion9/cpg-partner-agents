"""LLM layer (HLD §9). The only module that imports provider SDKs.

``get_chat_model(role)`` returns a LangChain chat model for the configured provider.
``llm_slot()`` bounds how many model calls run at once, so fan-out stays under free-tier
request limits. ``invoke_structured()`` is added in T-08; per-role providers, fallbacks
and the response cache in T-53.
"""

import asyncio
import itertools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal
from weakref import WeakKeyDictionary

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from partner_agents.config import LLMProvider, Settings, get_settings

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
    return GenericFakeChatModel(messages=itertools.repeat(AIMessage(content=FAKE_REPLY)))


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
