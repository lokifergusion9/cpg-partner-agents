from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from partner_agents.config import BUILTIN_SEARCH_DOMAIN_BLOCKLIST, Settings, get_settings


def make_settings(**overrides: Any) -> Settings:
    """Settings from env + overrides only, never from a local .env file."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_safe_defaults() -> None:
    s = make_settings()
    assert s.outreach_mode == "outbox"
    assert s.voice_provider == "mock"
    assert s.demo_mode is False
    assert s.verify_mx is False
    assert s.outreach_allowlist == []
    assert s.voice_allowlist == []


def test_defaults_match_hld_table() -> None:
    s = make_settings()
    assert s.llm_provider == "gemini"
    assert s.llm_max_concurrency == 4
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert (s.search_provider, s.search_fallback) == ("tavily", "ddg")
    assert (s.chunk_tokens, s.chunk_overlap) == (1000, 150)
    assert (s.evidence_k, s.evidence_max) == (4, 6)
    assert s.router_min_confidence == 0.6
    assert (s.fetch_timeout_s, s.fetch_max_bytes, s.fetch_concurrency) == (10, 2_000_000, 8)
    assert s.default_phone_region == "US"
    assert (s.outreach_daily_cap, s.outreach_dedupe_days) == (20, 30)
    assert s.rate_limit_runs_per_hour == 5
    assert s.data_ttl_days == 7


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("env_value", [None, "", "  "])
def test_gmail_files_default_outside_repo(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    if env_value is not None:
        monkeypatch.setenv("GMAIL_CREDENTIALS_PATH", env_value)
        monkeypatch.setenv("GMAIL_TOKEN_PATH", env_value)
    s = make_settings()
    for path in (s.gmail_credentials_path, s.gmail_token_path):
        assert not path.resolve().is_relative_to(REPO_ROOT)


def test_env_example_loads_with_safe_defaults() -> None:
    """`cp .env.example .env` must give a valid, safe configuration."""
    s = Settings(_env_file=REPO_ROOT / ".env.example")  # type: ignore[call-arg]
    assert s == make_settings()
    assert s.outreach_mode == "outbox"
    assert s.voice_provider == "mock"
    assert s.qdrant_is_local
    assert s.mongo_is_in_memory
    assert s.gemini_api_key is None


def test_env_example_lists_every_setting() -> None:
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    declared = {line.split("=", 1)[0] for line in text.splitlines() if "=" in line[:40]}
    declared = {name for name in declared if not name.startswith("#")}
    assert declared == {name.upper() for name in Settings.model_fields}


@pytest.mark.parametrize(
    ("env", "qdrant_local", "mongo_in_memory"),
    [
        ({}, True, True),
        ({"QDRANT_URL": "", "MONGODB_URI": ""}, True, True),
        ({"QDRANT_URL": "   ", "MONGODB_URI": "  "}, True, True),
        ({"QDRANT_URL": "https://x.qdrant.io"}, False, True),
        ({"MONGODB_URI": "mongodb+srv://u:p@cluster.example/db"}, True, False),
    ],
)
def test_empty_store_urls_select_local_backends(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    qdrant_local: bool,
    mongo_in_memory: bool,
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    s = make_settings()
    assert s.qdrant_is_local is qdrant_local
    assert s.mongo_is_in_memory is mongo_in_memory


@pytest.mark.parametrize(
    "env",
    [
        {"OUTREACH_MODE": "gmail_draft"},
        {"OUTREACH_MODE": "gmail_send", "OUTREACH_ALLOWLIST": "me@example.com"},
        {"VOICE_PROVIDER": "vapi"},
        {"OUTREACH_MODE": "gmail_send", "OUTREACH_ALLOWLIST": "a@x.com", "VOICE_PROVIDER": "vapi"},
    ],
)
def test_demo_mode_overrides_outreach_and_voice(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DEMO_MODE", "true")
    s = make_settings()
    assert s.outreach_mode == "outbox"
    assert s.voice_provider == "mock"


def test_non_demo_mode_keeps_configured_outreach() -> None:
    s = make_settings(outreach_mode="gmail_draft", voice_provider="vapi")
    assert s.outreach_mode == "gmail_draft"
    assert s.voice_provider == "vapi"


def test_gmail_send_requires_allowlist() -> None:
    with pytest.raises(ValidationError, match="OUTREACH_ALLOWLIST"):
        make_settings(outreach_mode="gmail_send")


def test_csv_lists_are_split_trimmed_and_lowercased(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTREACH_ALLOWLIST", " Me@Example.com, ,ops@example.com ")
    monkeypatch.setenv("VOICE_ALLOWLIST", "+13105550100,+919800000000")
    monkeypatch.setenv("SEARCH_DOMAIN_BLOCKLIST", "Spam-Directory.com")
    s = make_settings()
    assert s.outreach_allowlist == ["me@example.com", "ops@example.com"]
    assert s.voice_allowlist == ["+13105550100", "+919800000000"]
    assert "spam-directory.com" in s.effective_search_blocklist
    assert s.effective_search_blocklist >= BUILTIN_SEARCH_DOMAIN_BLOCKLIST


def test_blank_secret_is_none_and_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret-value")
    s = make_settings()
    assert s.gemini_api_key is None
    assert s.tavily_api_key is not None
    assert s.tavily_api_key.get_secret_value() == "tvly-secret-value"
    assert "tvly-secret-value" not in repr(s)
    assert "tvly-secret-value" not in str(s.model_dump())


def test_tracing_enabled_only_with_both_keys() -> None:
    assert make_settings().tracing_enabled is False
    assert make_settings(langfuse_public_key="pk").tracing_enabled is False
    assert make_settings(langfuse_public_key="pk", langfuse_secret_key="sk").tracing_enabled


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_provider": "openai"},
        {"outreach_mode": "send_everything"},
        {"voice_provider": "twilio"},
        {"router_min_confidence": 1.5},
        {"llm_max_concurrency": 0},
        {"chunk_tokens": 500, "chunk_overlap": 500},
        {"evidence_k": 7, "evidence_max": 6},
    ],
)
def test_invalid_values_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_env_names_are_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("llm_provider", "fake")
    monkeypatch.setenv("DEFAULT_PHONE_REGION", "in")
    s = make_settings()
    assert s.llm_provider == "fake"
    assert s.default_phone_region == "IN"


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    first = get_settings()
    assert get_settings() is first
    assert first.outreach_mode == "outbox"
