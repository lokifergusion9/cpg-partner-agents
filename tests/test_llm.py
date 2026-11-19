import asyncio
from itertools import pairwise
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from partner_agents import llm
from partner_agents.config import Role, Settings


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


MODELS = {
    "llm_model_router": "router-model",
    "llm_model_extractor": "extractor-model",
    "llm_model_writer": "writer-model",
}


# --- fake provider ------------------------------------------------------------------


@pytest.mark.parametrize("role", ["router", "extractor", "writer"])
def test_fake_provider_needs_no_keys_or_model_ids(role: Role) -> None:
    model = llm.get_chat_model(role, settings=make_settings(llm_provider="fake"))
    reply = model.invoke("hello")
    assert isinstance(reply, AIMessage)
    assert reply.content == llm.FAKE_REPLY


def test_fake_provider_replies_more_than_once() -> None:
    model = llm.get_chat_model("router", settings=make_settings(llm_provider="fake"))
    assert [model.invoke("a").content, model.invoke("b").content] == [llm.FAKE_REPLY] * 2


def test_get_settings_is_used_when_none_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert llm.get_chat_model("writer").invoke("hi").content == llm.FAKE_REPLY


# --- real providers (constructed, never called) -------------------------------------


@pytest.mark.parametrize(
    ("role", "expected"), [("router", 0.0), ("extractor", 0.0), ("writer", 0.3)]
)
def test_gemini_uses_the_role_model_and_temperature(role: Role, expected: float) -> None:
    settings = make_settings(llm_provider="gemini", gemini_api_key="g-key-123", **MODELS)
    model = llm.get_chat_model(role, settings=settings).primary
    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert model.model == f"{role}-model"  # type: ignore[attr-defined]
    assert model.temperature == expected  # type: ignore[attr-defined]


def test_temperature_override_beats_the_role_default() -> None:
    settings = make_settings(llm_provider="groq", groq_api_key="q-key-123", **MODELS)
    model = llm.get_chat_model("writer", temperature=0.7, settings=settings).primary
    assert type(model).__name__ == "ChatGroq"
    assert model.temperature == 0.7  # type: ignore[attr-defined]
    assert model.model_name == "writer-model"  # type: ignore[attr-defined]


def test_ollama_points_at_the_configured_host_with_a_timeout() -> None:
    settings = make_settings(
        llm_provider="ollama",
        ollama_base_url="http://100.64.0.7:11434",
        ollama_timeout_s=12,
        **MODELS,
    )
    model = llm.get_chat_model("extractor", settings=settings).primary
    assert type(model).__name__ == "ChatOllama"
    assert model.base_url == "http://100.64.0.7:11434"  # type: ignore[attr-defined]
    assert model.client_kwargs == {"timeout": 12}  # type: ignore[attr-defined]


def test_sdk_retries_are_off_so_the_retry_utility_owns_the_budget() -> None:
    settings = make_settings(llm_provider="groq", groq_api_key="q-key-123", **MODELS)
    model = llm.get_chat_model("router", settings=settings).primary
    assert model.max_retries == 0  # type: ignore[attr-defined]


# --- misconfiguration ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "message"),
    [("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY")],
)
def test_missing_api_key_is_a_config_error(provider: str, message: str) -> None:
    settings = make_settings(llm_provider=provider, **MODELS)
    with pytest.raises(llm.LLMConfigError, match=message):
        llm.get_chat_model("router", settings=settings)


@pytest.mark.parametrize("provider", ["gemini", "groq", "ollama"])
def test_missing_model_id_is_a_config_error(provider: str) -> None:
    settings = make_settings(
        llm_provider=provider, gemini_api_key="g-key-123", groq_api_key="q-key-123"
    )
    with pytest.raises(llm.LLMConfigError, match="LLM_MODEL_ROUTER"):
        llm.get_chat_model("router", settings=settings)


def test_a_blank_model_id_counts_as_missing() -> None:
    settings = make_settings(llm_provider="ollama", llm_model_writer="  ")
    with pytest.raises(llm.LLMConfigError, match="LLM_MODEL_WRITER"):
        llm.get_chat_model("writer", settings=settings)


# --- concurrency --------------------------------------------------------------------


def test_llm_slot_bounds_concurrent_calls() -> None:
    settings = make_settings(llm_max_concurrency=2)
    running = peak = 0

    async def call() -> None:
        nonlocal running, peak
        async with llm.llm_slot(settings):
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

    async def main() -> None:
        await asyncio.gather(*(call() for _ in range(8)))

    asyncio.run(main())
    assert peak == 2


def test_llm_slot_spaces_call_starts_when_an_rpm_limit_is_set() -> None:
    """T-40: 600 calls a minute is one start every 0.1 s, however many are waiting."""
    settings = make_settings(llm_max_concurrency=8, llm_rpm_limit=600)
    starts: list[float] = []

    async def call() -> None:
        async with llm.llm_slot(settings):
            starts.append(asyncio.get_running_loop().time())

    async def main() -> None:
        await asyncio.gather(*(call() for _ in range(5)))

    asyncio.run(main())
    gaps = [b - a for a, b in pairwise(starts)]
    assert len(starts) == 5 and all(gap >= 0.09 for gap in gaps)


def test_llm_slot_does_not_pace_without_an_rpm_limit() -> None:
    settings = make_settings(llm_max_concurrency=8)
    assert settings.llm_rpm_limit == 0
    starts: list[float] = []

    async def call() -> None:
        async with llm.llm_slot(settings):
            starts.append(asyncio.get_running_loop().time())

    async def main() -> None:
        await asyncio.gather(*(call() for _ in range(5)))

    asyncio.run(main())
    assert max(starts) - min(starts) < 0.05


def test_llm_slot_is_released_when_the_call_raises() -> None:
    settings = make_settings(llm_max_concurrency=1)

    async def main() -> None:
        with pytest.raises(RuntimeError):
            async with llm.llm_slot(settings):
                raise RuntimeError("boom")
        async with asyncio.timeout(1), llm.llm_slot(settings):
            pass

    asyncio.run(main())


def test_llm_slot_works_across_separate_event_loops() -> None:
    settings = make_settings(llm_max_concurrency=1)

    async def once() -> None:
        async with llm.llm_slot(settings):
            await asyncio.sleep(0)

    asyncio.run(once())
    asyncio.run(once())
