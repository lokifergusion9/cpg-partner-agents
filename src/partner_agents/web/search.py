"""Scout search (HLD §6.3): providers, query templates, post-processing and the cache.

``SearchService.discover(intent)`` turns a ``DiscoveryIntent`` into ranked, de-duplicated
``SearchHit``s: up to two queries, each answered by the first provider that returns
results (Tavily, then DuckDuckGo), merged, filtered through the domain blocklist, one hit
per registrable domain.
"""

import asyncio
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import tldextract
from pydantic import HttpUrl, TypeAdapter, ValidationError

from partner_agents.config import SearchProviderName, Settings
from partner_agents.schemas import DiscoveryIntent, PartnerType, SearchHit, StageError
from partner_agents.storage.repo import Repository
from partner_agents.utils.cache_stats import record_cache
from partner_agents.utils.retry import DEFAULT_POLICY, retry_async

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
SEARCH_TIMEOUT_S = 15.0
# Every query asks for the same number, so a cached answer serves any ``max_results``.
RESULTS_PER_QUERY = 20
SNIPPET_CHARS = 300
STAGE = "scout_search"

# Transport retries for the Tavily call (429 / 5xx / timeouts); tests swap in a zero-delay policy.
RETRY_POLICY = DEFAULT_POLICY

# Up to two templates per partner type (HLD §6.3). ``{loc}`` is dropped when there is none.
QUERY_TEMPLATES: dict[PartnerType, tuple[str, str]] = {
    "broker": ("{cat} food broker {loc} contact", "{cat} brokers {loc}"),
    "distributor": ("{cat} distributor {loc} contact", "{cat} wholesale distributors {loc}"),
    "retailer": ("{cat} retailer {loc} buyer contact", "independent stores selling {cat} {loc}"),
    "manufacturer": ("{cat} manufacturer {loc} contact", "{cat} co-packers {loc}"),
    "other": ("{cat} companies {loc} contact", "{cat} partners {loc}"),
}

_TRACKING_PARAMS = re.compile(
    r"^(utm_.*|gclid|fbclid|msclkid|mc_cid|mc_eid|igshid)$", re.IGNORECASE
)
_URL = TypeAdapter(HttpUrl)
# The bundled public-suffix snapshot only: no network at import or first use.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


class SearchError(Exception):
    """A provider could not answer (HTTP error, timeout, rate limit, malformed reply)."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> list[SearchHit]: ...


# --- providers ----------------------------------------------------------------------


class TavilyProvider:
    """Tavily's REST API, called with httpx (no SDK)."""

    name = "tavily"

    def __init__(self, api_key: str, *, timeout_s: float = SEARCH_TIMEOUT_S) -> None:
        self._api_key = api_key
        self._timeout_s = timeout_s

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        async def once() -> httpx.Response:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(
                    TAVILY_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                        "include_answer": False,
                    },
                )
            response.raise_for_status()
            return response

        try:
            response = await retry_async(once, policy=RETRY_POLICY)
            rows = [
                (r.get("url"), r.get("title"), r.get("content")) for r in response.json()["results"]
            ]
        except httpx.HTTPStatusError as exc:
            raise SearchError(self.name, f"HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise SearchError(self.name, type(exc).__name__) from exc
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise SearchError(self.name, "unexpected response shape") from exc
        return _to_hits(self.name, rows)


TextSearch = Callable[[str, int], list[dict[str, Any]]]


def _ddgs_text(query: str, max_results: int) -> list[dict[str, Any]]:
    from ddgs import DDGS

    return list(DDGS(timeout=int(SEARCH_TIMEOUT_S)).text(query, max_results=max_results))


class DDGProvider:
    """DuckDuckGo through the unofficial ``ddgs`` library (ADR-006), run in a thread."""

    name = "ddg"

    def __init__(self, text_search: TextSearch = _ddgs_text) -> None:
        self._text_search = text_search

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(self._text_search, query, max_results),
                timeout=SEARCH_TIMEOUT_S + 5,
            )
        except Exception as exc:  # ddgs raises its own hierarchy, including "no results"
            raise SearchError(self.name, type(exc).__name__) from exc
        return _to_hits(
            self.name, ((r.get("href"), r.get("title"), r.get("body")) for r in results)
        )


def _to_hits(provider: str, rows: Any) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for url, title, snippet in rows:
        try:
            parsed = _URL.validate_python(url)
        except ValidationError:
            continue
        hits.append(
            SearchHit(
                url=parsed,
                title=str(title or "").strip(),
                snippet=str(snippet or "").strip()[:SNIPPET_CHARS],
                rank=len(hits) + 1,
                provider=provider,
            )
        )
    return hits


def build_providers(settings: Settings) -> list[SearchProvider]:
    """The primary provider, then the fallback. Tavily without a key is skipped."""
    providers: list[SearchProvider] = []
    wanted: list[SearchProviderName] = [settings.search_provider]
    if settings.search_fallback and settings.search_fallback not in wanted:
        wanted.append(settings.search_fallback)
    for name in wanted:
        if name == "tavily":
            if settings.tavily_api_key is None:
                logger.warning("TAVILY_API_KEY is not set, skipping the Tavily provider")
                continue
            providers.append(TavilyProvider(settings.tavily_api_key.get_secret_value()))
        else:
            providers.append(DDGProvider())
    return providers


# --- queries and post-processing (pure) ----------------------------------------------


def build_queries(intent: DiscoveryIntent) -> list[str]:
    """Up to two search queries for the intent."""
    category = intent.product_category.strip()
    location = (intent.location or "").strip()
    queries: list[str] = []
    for template in QUERY_TEMPLATES[intent.partner_type]:
        query = " ".join(template.format(cat=category, loc=location).split())
        if query and query not in queries:
            queries.append(query)
    return queries


def cache_key(query: str) -> str:
    return " ".join(query.lower().split())


def normalise_url(url: str) -> str | None:
    """Lowercase host, no fragment, default port or tracking parameters. ``None`` if not http(s)."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").rstrip(".").lower()
    if parts.scheme.lower() not in {"http", "https"} or not host:
        return None
    port = parts.port
    default_port = 443 if parts.scheme.lower() == "https" else 80
    netloc = host if port in (None, default_port) else f"{host}:{port}"
    query = urlencode(
        [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not _TRACKING_PARAMS.match(k)
        ]
    )
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", query, ""))


def registrable_domain(url_or_host: str) -> str:
    """``shop.acme.co.uk`` -> ``acme.co.uk``. IP addresses and bare hosts map to themselves."""
    host = urlsplit(url_or_host).hostname or url_or_host if "//" in url_or_host else url_or_host
    host = host.lower().rstrip(".")
    return _EXTRACT(host).top_domain_under_public_suffix or host


def is_blocked(url: str, blocklist: frozenset[str]) -> bool:
    """True when the host is, or is a subdomain of, a blocklisted domain."""
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return any(host == entry or host.endswith(f".{entry}") for entry in blocklist)


def merge_hits(
    per_query: Sequence[Sequence[SearchHit]], blocklist: frozenset[str], max_results: int
) -> list[SearchHit]:
    """Normalise, drop blocklisted domains, keep the best rank per registrable domain.

    The result is re-ranked 1..N in merged order.
    """
    candidates: list[tuple[int, int, int, SearchHit, str]] = []
    for query_index, hits in enumerate(per_query):
        for position, hit in enumerate(hits):
            normal = normalise_url(str(hit.url))
            if normal is None or is_blocked(normal, blocklist):
                continue
            candidates.append((hit.rank, query_index, position, hit, normal))
    candidates.sort(key=lambda c: c[:3])
    seen: set[str] = set()
    merged: list[SearchHit] = []
    for _rank, _qi, _pos, hit, normal in candidates:
        domain = registrable_domain(normal)
        if domain in seen:
            continue
        seen.add(domain)
        merged.append(
            hit.model_copy(update={"url": _URL.validate_python(normal), "rank": len(merged) + 1})
        )
        if len(merged) == max_results:
            break
    return merged


# --- service ------------------------------------------------------------------------


class SearchOutcome:
    def __init__(self, hits: list[SearchHit], errors: list[StageError], cache_hits: int) -> None:
        self.hits = hits
        self.errors = errors
        self.cache_hits = cache_hits


class SearchService:
    def __init__(
        self,
        providers: Sequence[SearchProvider],
        repo: Repository,
        blocklist: frozenset[str],
    ) -> None:
        self._providers = list(providers)
        self._repo = repo
        self._blocklist = blocklist

    async def discover(self, intent: DiscoveryIntent) -> SearchOutcome:
        queries = build_queries(intent)
        answers = await asyncio.gather(*(self._answer(query) for query in queries))
        errors = [error for _hits, _from_cache, error in answers if error is not None]
        hits = merge_hits([hits for hits, _c, _e in answers], self._blocklist, intent.max_results)
        if not hits and not errors:
            errors.append(
                StageError(stage=STAGE, code="no_results", message="Every result was filtered out.")
            )
        return SearchOutcome(hits, errors, cache_hits=sum(c for _h, c, _e in answers))

    async def _answer(self, query: str) -> tuple[list[SearchHit], int, StageError | None]:
        """Hits for one query, whether they came from the cache, and an error if none came."""
        key = cache_key(query)
        cached = await self._repo.get_cached_search(key)
        if cached is not None:
            record_cache("search", hits=1)
            return cached, 1, None
        record_cache("search", misses=1)  # the provider is called now
        failures: list[str] = []
        for provider in self._providers:
            try:
                hits = await provider.search(query, RESULTS_PER_QUERY)
            except SearchError as exc:
                logger.warning("search provider failed", extra={"provider": provider.name})
                failures.append(str(exc))
                continue
            if hits:
                await self._repo.put_cached_search(key, hits)
                return hits, 0, None
            failures.append(f"{provider.name}: no results")
        if not self._providers:
            return (
                [],
                0,
                StageError(
                    stage=STAGE, code="no_provider", message="No search provider is configured."
                ),
            )
        code = "search_failed" if any("no results" not in f for f in failures) else "no_results"
        return [], 0, StageError(stage=STAGE, code=code, message="; ".join(failures))
