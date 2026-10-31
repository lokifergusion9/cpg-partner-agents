"""Application settings (HLD §10).

This is the only module that reads environment variables. Everything else receives a
``Settings`` instance, usually through ``get_settings()``.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_core import PydanticUseDefault
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LLMProvider = Literal["gemini", "groq", "ollama", "fake"]
SearchProviderName = Literal["tavily", "ddg"]
OutreachMode = Literal["outbox", "gmail_draft", "gmail_send"]
VoiceProvider = Literal["mock", "vapi"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

# Domains that are never partners themselves: social networks, marketplaces, large
# directories and news sites (HLD §6.3). SEARCH_DOMAIN_BLOCKLIST adds to this list.
BUILTIN_SEARCH_DOMAIN_BLOCKLIST: frozenset[str] = frozenset(
    {
        # social networks
        "linkedin.com",
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "x.com",
        "tiktok.com",
        "pinterest.com",
        "youtube.com",
        "reddit.com",
        # marketplaces
        "amazon.com",
        "ebay.com",
        "alibaba.com",
        "indiamart.com",
        "etsy.com",
        "walmart.com",
        # directories and review sites
        "yelp.com",
        "yellowpages.com",
        "bbb.org",
        "zoominfo.com",
        "crunchbase.com",
        "dnb.com",
        "manta.com",
        "thomasnet.com",
        "wikipedia.org",
        # news
        "forbes.com",
        "bloomberg.com",
        "reuters.com",
        "nytimes.com",
        "foodnavigator-usa.com",
    }
)

_GMAIL_DIR = Path.home() / ".partner_agents"


class Settings(BaseSettings):
    """Typed configuration. Defaults are the safe ones (CLAUDE.md hard rule 1)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    llm_provider: LLMProvider = "gemini"
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    llm_model_router: str | None = None
    llm_model_extractor: str | None = None
    llm_model_writer: str | None = None
    llm_max_concurrency: int = Field(4, ge=1)
    llm_timeout_s: float = Field(60, gt=0)

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Search
    search_provider: SearchProviderName = "tavily"
    search_fallback: SearchProviderName | None = "ddg"
    tavily_api_key: SecretStr | None = None
    search_domain_blocklist: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Stores: an empty URL/URI selects the local / in-memory backend
    qdrant_url: str = ""
    qdrant_api_key: SecretStr | None = None
    qdrant_local_path: str = "./.qdrant"
    mongodb_uri: SecretStr | None = None

    # Retrieval
    chunk_tokens: int = Field(1000, ge=50)
    chunk_overlap: int = Field(150, ge=0)
    evidence_k: int = Field(4, ge=1)
    evidence_max: int = Field(6, ge=1)

    # Charlie thresholds. KB_MIN_SCORE is a starting point, tuned on the kb dev split (EVAL §4.6).
    router_min_confidence: float = Field(0.6, ge=0, le=1)
    kb_min_score: float = Field(0.5, ge=0, le=1)

    # Fetching
    fetch_timeout_s: float = Field(10, gt=0)
    fetch_max_bytes: int = Field(2_000_000, gt=0)
    fetch_concurrency: int = Field(8, ge=1)

    # Verification
    verify_mx: bool = False
    default_phone_region: str = Field("US", min_length=2, max_length=2)

    # Outreach
    outreach_mode: OutreachMode = "outbox"
    outreach_allowlist: Annotated[list[str], NoDecode] = Field(default_factory=list)
    outreach_daily_cap: int = Field(20, ge=0)
    outreach_dedupe_days: int = Field(30, ge=0)
    gmail_credentials_path: Path = _GMAIL_DIR / "gmail_credentials.json"
    gmail_token_path: Path = _GMAIL_DIR / "gmail_token.json"

    # Voice (stretch)
    voice_provider: VoiceProvider = "mock"
    vapi_api_key: SecretStr | None = None
    vapi_phone_number_id: str | None = None
    voice_allowlist: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Tracing: enabled only when both keys are set
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None

    # Logging
    log_level: LogLevel = "INFO"

    # Graph nodes: true re-raises unexpected exceptions instead of recording a StageError
    strict_nodes: bool = False

    # API and demo
    app_api_key: SecretStr | None = None
    demo_mode: bool = False
    rate_limit_runs_per_hour: int = Field(5, ge=1)
    data_ttl_days: int = Field(7, ge=1)

    @field_validator(
        "gemini_api_key",
        "groq_api_key",
        "tavily_api_key",
        "qdrant_api_key",
        "mongodb_uri",
        "vapi_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        "app_api_key",
        "llm_model_router",
        "llm_model_extractor",
        "llm_model_writer",
        "vapi_phone_number_id",
        "langfuse_host",
        "search_fallback",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        """`KEY=` in a .env file means "not set", not an empty secret."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("gmail_credentials_path", "gmail_token_path", mode="before")
    @classmethod
    def _blank_path_is_default(cls, value: Any) -> Any:
        """A blank path would mean the working directory, i.e. inside the repo."""
        if isinstance(value, str) and not value.strip():
            raise PydanticUseDefault()
        return value

    @field_validator(
        "search_domain_blocklist", "outreach_allowlist", "voice_allowlist", mode="before"
    )
    @classmethod
    def _split_csv(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("search_domain_blocklist", "outreach_allowlist")
    @classmethod
    def _lowercase(cls, value: list[str]) -> list[str]:
        return [item.lower() for item in value]

    @field_validator("default_phone_region", "log_level", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _apply_safety_rules(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_TOKENS")
        if self.evidence_k > self.evidence_max:
            raise ValueError("EVIDENCE_K must not exceed EVIDENCE_MAX")
        if self.demo_mode:
            # DEMO_MODE wins over any other outreach or voice setting (HLD §6.9).
            self.outreach_mode = "outbox"
            self.voice_provider = "mock"
        if self.outreach_mode == "gmail_send" and not self.outreach_allowlist:
            raise ValueError("OUTREACH_MODE=gmail_send requires a non-empty OUTREACH_ALLOWLIST")
        return self

    @property
    def qdrant_is_local(self) -> bool:
        return not self.qdrant_url.strip()

    @property
    def mongo_is_in_memory(self) -> bool:
        return self.mongodb_uri is None

    @property
    def tracing_enabled(self) -> bool:
        return self.langfuse_public_key is not None and self.langfuse_secret_key is not None

    @property
    def effective_search_blocklist(self) -> frozenset[str]:
        return BUILTIN_SEARCH_DOMAIN_BLOCKLIST | frozenset(self.search_domain_blocklist)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once. Tests build ``Settings(...)`` directly instead."""
    return Settings()
