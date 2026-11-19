"""T-53: per-role providers, the fallback chain and the response cache (ADR-018)."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from partner_agents import llm
from partner_agents.config import ModelSpec, Role, Settings, parse_model_spec
from partner_agents.fakes import ScriptedChatModel
from partner_agents.llm_cache import ResponseCache, cache_key
from partner_agents.schemas import DiscoveryIntent
from partner_agents.utils.cache_stats import track_cache
from partner_agents.utils.retry import RetryPolicy
from tests.helpers import FaultyModel, RateLimited, ServerError

VALID = DiscoveryIntent(
    partner_type="broker", product_category="cookies", raw_query="cookie brokers in CA"
)
PROMPT = [HumanMessage(content="Extract the intent")]


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def chain(*models: ScriptedChatModel, cache_dir: Path | None = None) -> llm.ModelChain:
    candidates = [llm.Candidate(name, model) for name, model in zip("abc", models, strict=False)]
    return llm.ModelChain(role="router", candidates=candidates, cache_dir=cache_dir)


def ask(model: llm.ModelChain, messages: Any = PROMPT, **kwargs: Any) -> DiscoveryIntent:
    return asyncio.run(
        llm.invoke_structured(model, DiscoveryIntent, messages, role="router", **kwargs)
    )


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "RETRY_POLICY", RetryPolicy(attempts=2, base_delay_s=0))


# --- LLM_ROLES / LLM_FALLBACKS parsing -----------------------------------------------


def test_model_spec_splits_on_the_first_colon_only() -> None:
    spec = parse_model_spec("ollama:qwen2.5:7b")
    assert (spec.provider, spec.model) == ("ollama", "qwen2.5:7b")
    assert spec.label == "ollama:qwen2.5:7b"


def test_llm_roles_parse_the_documented_example() -> None:
    settings = make_settings(llm_roles="simulator=ollama:qwen2.5:7b, judge=groq:llama-x")
    assert settings.llm_roles == {
        "simulator": ModelSpec(provider="ollama", model="qwen2.5:7b"),
        "judge": ModelSpec(provider="groq", model="llama-x"),
    }


def test_llm_roles_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ROLES", "simulator=ollama:qwen2.5:7b")
    assert Settings(_env_file=None).llm_roles["simulator"].model == "qwen2.5:7b"  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "value",
    [
        "simulator",  # no '='
        "mystery=ollama:m",  # unknown role
        "simulator=skynet:m",  # unknown provider
        "simulator=ollama",  # provider without a model
        "simulator=ollama:  ",
    ],
)
def test_bad_llm_roles_are_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(llm_roles=value)


def test_llm_fallbacks_parse_in_order() -> None:
    settings = make_settings(llm_fallbacks="groq:m1, gemini:m2")
    assert [spec.label for spec in settings.llm_fallbacks] == ["groq:m1", "gemini:m2"]


def test_blank_values_mean_off() -> None:
    settings = make_settings(llm_roles="", llm_fallbacks="", llm_cache_dir="")
    assert (settings.llm_roles, settings.llm_fallbacks, settings.llm_cache_dir) == ({}, [], None)


def test_a_fallback_without_a_model_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(llm_fallbacks="groq")


# --- per-role providers ---------------------------------------------------------------

MODELS = {
    "llm_model_router": "router-model",
    "llm_model_extractor": "extractor-model",
    "llm_model_writer": "writer-model",
}


def test_simulator_override_builds_ollama_with_the_override_model() -> None:
    settings = make_settings(
        llm_provider="gemini",
        gemini_api_key="g-key-123",
        llm_roles="simulator=ollama:qwen2.5:7b",
        ollama_base_url="http://100.64.0.7:11434",
        ollama_timeout_s=33,
        **MODELS,
    )
    simulator = llm.get_chat_model("simulator", settings=settings)
    assert simulator.label == "ollama:qwen2.5:7b"
    assert type(simulator.primary).__name__ == "ChatOllama"
    assert simulator.primary.model == "qwen2.5:7b"  # type: ignore[attr-defined]
    assert simulator.primary.temperature == 0.8  # type: ignore[attr-defined]
    assert simulator.primary.client_kwargs == {"timeout": 33}  # type: ignore[attr-defined]


def test_other_roles_keep_the_default_provider() -> None:
    settings = make_settings(
        llm_provider="gemini",
        gemini_api_key="g-key-123",
        llm_roles="simulator=ollama:qwen2.5:7b",
        **MODELS,
    )
    router = llm.get_chat_model("router", settings=settings)
    assert router.label == "gemini:router-model"
    assert type(router.primary).__name__ == "ChatGoogleGenerativeAI"


@pytest.mark.parametrize(
    ("role", "expected_model", "expected_temp"),
    [
        ("classifier", "router-model", 0.0),
        ("caller", "writer-model", 0.4),
        ("simulator", "writer-model", 0.8),
        ("judge", "extractor-model", 0.0),
    ],
)
def test_new_roles_default_to_another_roles_model(
    role: Role, expected_model: str, expected_temp: float
) -> None:
    settings = make_settings(llm_provider="groq", groq_api_key="q-key-123", **MODELS)
    built = llm.get_chat_model(role, settings=settings)
    assert built.label == f"groq:{expected_model}"
    assert built.primary.temperature == pytest.approx(expected_temp, abs=1e-6)  # type: ignore[attr-defined]


def test_a_role_specific_model_id_beats_the_default() -> None:
    settings = make_settings(
        llm_provider="groq", groq_api_key="q-key-123", llm_model_judge="judge-model", **MODELS
    )
    assert llm.get_chat_model("judge", settings=settings).label == "groq:judge-model"


def test_fake_provider_labels_itself_fake() -> None:
    assert llm.get_chat_model("judge", settings=make_settings(llm_provider="fake")).label == "fake"


def test_fallback_models_are_built_with_the_role_temperature() -> None:
    settings = make_settings(
        llm_provider="groq",
        groq_api_key="q-key-123",
        llm_fallbacks="ollama:backup-model",
        **MODELS,
    )
    built = llm.get_chat_model("caller", settings=settings)
    assert [c.label for c in built.candidates] == ["groq:writer-model", "ollama:backup-model"]
    assert built.candidates[1].model.temperature == 0.4  # type: ignore[attr-defined]


def test_a_fallback_with_a_missing_key_fails_at_startup() -> None:
    settings = make_settings(llm_provider="fake", llm_fallbacks="groq:backup")
    with pytest.raises(llm.LLMConfigError, match="GROQ_API_KEY"):
        llm.get_chat_model("router", settings=settings)


def test_the_cache_dir_travels_with_the_chain(tmp_path: Path) -> None:
    settings = make_settings(llm_provider="fake", llm_cache_dir=str(tmp_path))
    assert llm.get_chat_model("router", settings=settings).cache_dir == tmp_path


# --- fallback chain -------------------------------------------------------------------


def test_a_429_that_outlasts_the_retries_is_served_by_the_fallback() -> None:
    primary = FaultyModel.scripted(VALID)
    primary.faults.extend([RateLimited(), RateLimited()])  # 2 attempts in the policy
    backup = FaultyModel.scripted(VALID)
    with llm.track_providers() as providers:
        assert ask(chain(primary, backup)) == VALID
    assert primary.transport_calls == 2  # the retry budget was spent first
    assert backup.transport_calls == 1
    assert providers == {"router": "b"}


def test_a_429_that_clears_within_the_retries_never_reaches_the_fallback() -> None:
    primary = FaultyModel.scripted(VALID)
    primary.faults.append(RateLimited())
    backup = FaultyModel.scripted(VALID)
    with llm.track_providers() as providers:
        ask(chain(primary, backup))
    assert backup.transport_calls == 0
    assert providers == {"router": "a"}


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("refused"),
        TimeoutError(),
        RuntimeError("You exceeded your current quota, please check your plan"),
        RuntimeError("RESOURCE_EXHAUSTED"),
    ],
    ids=["connection", "timeout", "quota-text", "resource-exhausted"],
)
def test_quota_and_connection_errors_fall_back(error: Exception) -> None:
    primary = FaultyModel.scripted(VALID)
    primary.faults.extend([error, error])  # outlasts the 2-attempt retry budget
    backup = FaultyModel.scripted(VALID)
    assert ask(chain(primary, backup)) == VALID
    assert backup.transport_calls == 1


def test_a_google_style_error_with_code_429_falls_back() -> None:
    class ClientError(Exception):
        code = 429

    primary = FaultyModel.scripted(VALID)
    primary.faults.append(ClientError("slow down"))
    backup = FaultyModel.scripted(VALID)
    assert ask(chain(primary, backup)) == VALID


@pytest.mark.parametrize("error", [KeyError("bug"), ValueError("bad request")], ids=repr)
def test_other_errors_do_not_fall_back(error: Exception) -> None:
    primary = FaultyModel.scripted(VALID)
    primary.faults.append(error)
    backup = FaultyModel.scripted(VALID)
    with pytest.raises(type(error)):
        ask(chain(primary, backup))
    assert backup.transport_calls == 0


def test_a_persistent_500_is_not_a_fallback_case() -> None:
    primary = FaultyModel.scripted(VALID)
    primary.faults.extend([ServerError(), ServerError()])
    backup = FaultyModel.scripted(VALID)
    with pytest.raises(ServerError):
        ask(chain(primary, backup))
    assert backup.transport_calls == 0


def test_a_validation_error_does_not_trigger_the_fallback() -> None:
    primary = FaultyModel.scripted("not json", "still not json")
    backup = FaultyModel.scripted(VALID)
    with pytest.raises(llm.StructuredOutputError):
        ask(chain(primary, backup))
    assert backup.transport_calls == 0
    assert primary.transport_calls == 2  # the first reply and the one validation retry


def test_the_last_models_failure_is_raised() -> None:
    primary = FaultyModel.scripted(VALID)
    primary.faults.extend([RateLimited(), RateLimited()])
    backup = FaultyModel.scripted(VALID)
    backup.faults.extend([RateLimited(), RateLimited()])
    with pytest.raises(RateLimited):
        ask(chain(primary, backup))


def test_the_chain_stays_with_the_fallback_across_the_validation_retry() -> None:
    primary = FaultyModel.scripted(VALID)
    primary.faults.extend([RateLimited(), RateLimited()])
    backup = FaultyModel.scripted("not json", VALID)
    assert ask(chain(primary, backup)) == VALID


def test_providers_list_every_model_that_answered_a_role() -> None:
    primary = FaultyModel.scripted(VALID, VALID)
    backup = FaultyModel.scripted(VALID)
    ordered = chain(primary, backup)
    with llm.track_providers() as providers:
        ask(ordered)  # the primary answers
        primary.faults.extend([RateLimited(), RateLimited()])
        ask(ordered, [HumanMessage(content="another question")])  # the fallback answers
    assert providers == {"router": "a+b"}


def test_a_bare_model_is_a_chain_of_one() -> None:
    with llm.track_providers() as providers:
        asyncio.run(
            llm.invoke_structured(
                ScriptedChatModel.scripted(VALID), DiscoveryIntent, PROMPT, role="router"
            )
        )
    assert providers == {"router": "ScriptedChatModel"}


# --- response cache -------------------------------------------------------------------


def reply_with_tokens(total: int) -> AIMessage:
    return AIMessage(
        content=VALID.model_dump_json(),
        usage_metadata={"input_tokens": total - 1, "output_tokens": 1, "total_tokens": total},
    )


def test_a_second_identical_call_is_served_from_the_cache_with_no_model_call(
    tmp_path: Path,
) -> None:
    model = ScriptedChatModel.scripted(VALID)  # a second call would raise: nothing left to say
    cached = chain(model, cache_dir=tmp_path)
    assert ask(cached) == VALID
    assert ask(cached) == VALID
    assert len(model.received) == 1


def test_the_cache_survives_a_new_chain_and_a_new_process_of_the_same_directory(
    tmp_path: Path,
) -> None:
    ask(chain(ScriptedChatModel.scripted(VALID), cache_dir=tmp_path))
    untouched = ScriptedChatModel.scripted()
    assert ask(chain(untouched, cache_dir=tmp_path)) == VALID
    assert untouched.received == []


def test_a_cache_hit_costs_no_tokens_and_reports_who_answered(tmp_path: Path) -> None:
    ask(chain(ScriptedChatModel.scripted(reply_with_tokens(50)), cache_dir=tmp_path))
    with llm.track_usage() as usage, llm.track_providers() as providers:
        ask(chain(ScriptedChatModel.scripted(), cache_dir=tmp_path))
    assert usage == {}
    assert providers == {"router": "a"}


def test_the_response_cache_counts_hits_and_misses_for_the_run(tmp_path: Path) -> None:
    """T-39: the eval report quotes the cache hit rate (``RunRecord.cache`` ``llm_*``)."""
    cached = chain(ScriptedChatModel.scripted(VALID), cache_dir=tmp_path)
    with track_cache() as stats:
        ask(cached)  # a miss: the model answers and the reply is stored
        ask(cached)  # a hit
        ask(cached)  # a hit
    assert stats == {"llm_misses": 1, "llm_hits": 2}
    with track_cache() as without_cache:  # no cache directory: nothing to count
        ask(chain(ScriptedChatModel.scripted(VALID)))
    assert without_cache == {}


@pytest.mark.parametrize(
    "change",
    [
        {"messages": [HumanMessage(content="A different question")]},
        {"prompt_version": "v2"},
    ],
    ids=["messages", "prompt-version"],
)
def test_a_changed_input_is_a_miss(tmp_path: Path, change: dict[str, Any]) -> None:
    model = ScriptedChatModel.scripted(VALID, VALID)
    cached = chain(model, cache_dir=tmp_path)
    ask(cached, prompt_version="v1")
    ask(cached, **{"prompt_version": "v1", **change})
    assert len(model.received) == 2


def test_a_different_schema_is_a_miss(tmp_path: Path) -> None:
    from partner_agents.schemas import RouteDecision

    route = RouteDecision(intent="cpg_qa", confidence=0.9, reason="r")
    model = ScriptedChatModel.scripted(VALID, route)
    cached = chain(model, cache_dir=tmp_path)
    ask(cached)
    asyncio.run(llm.invoke_structured(cached, RouteDecision, PROMPT, role="router"))
    assert len(model.received) == 2


def test_the_fallbacks_answer_is_cached_under_the_primary(tmp_path: Path) -> None:
    primary = FaultyModel.scripted(VALID)
    primary.faults.extend([RateLimited(), RateLimited()])
    backup = FaultyModel.scripted(VALID)
    cached = chain(primary, backup, cache_dir=tmp_path)
    ask(cached)
    with llm.track_providers() as providers:
        ask(cached)
    assert primary.transport_calls == 2  # not asked again
    assert backup.transport_calls == 1
    assert providers == {"router": "b"}  # the label of the model that originally answered


def test_invalid_replies_are_not_cached(tmp_path: Path) -> None:
    bad = ScriptedChatModel.scripted("nope", "still nope")
    with pytest.raises(llm.StructuredOutputError):
        ask(chain(bad, cache_dir=tmp_path))
    assert list(tmp_path.rglob("*.json")) == []


def test_a_damaged_cache_entry_is_a_miss_not_an_error(tmp_path: Path) -> None:
    ask(chain(ScriptedChatModel.scripted(VALID), cache_dir=tmp_path))
    [entry] = list(tmp_path.rglob("*.json"))
    entry.write_text("{ not json", encoding="utf-8")
    model = ScriptedChatModel.scripted(VALID)
    assert ask(chain(model, cache_dir=tmp_path)) == VALID
    assert len(model.received) == 1


def test_without_a_cache_dir_every_call_reaches_the_model() -> None:
    model = ScriptedChatModel.scripted(VALID, VALID)
    uncached = chain(model)
    ask(uncached)
    ask(uncached)
    assert len(model.received) == 2


# --- the cache itself -----------------------------------------------------------------


def test_cache_key_is_stable_and_order_insensitive_for_dicts() -> None:
    assert cache_key("a", {"x": 1, "y": 2}) == cache_key("a", {"y": 2, "x": 1})
    assert cache_key("a", 1) != cache_key("a", 2)
    assert len(cache_key("a")) == 64


def test_response_cache_round_trip_and_miss(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    key = cache_key("q")
    assert asyncio.run(cache.get(key)) is None
    asyncio.run(cache.put(key, {"label": "a", "parsed": "{}"}))
    assert asyncio.run(cache.get(key)) == {"label": "a", "parsed": "{}"}
    assert list(tmp_path.rglob("*.tmp")) == []  # the atomic write left nothing behind


def test_response_cache_creates_its_directory(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "deep" / "dir")
    asyncio.run(cache.put(cache_key("q"), {"x": 1}))
    assert asyncio.run(cache.get(cache_key("q"))) == {"x": 1}
