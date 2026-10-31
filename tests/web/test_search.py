import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respx

from partner_agents.config import Settings
from partner_agents.schemas import DiscoveryIntent, SearchHit
from partner_agents.storage.repo import InMemoryRepository
from partner_agents.utils.retry import RetryPolicy
from partner_agents.web import search
from partner_agents.web.search import (
    DDGProvider,
    SearchError,
    SearchService,
    TavilyProvider,
    build_providers,
    build_queries,
    is_blocked,
    merge_hits,
    normalise_url,
    registrable_domain,
)


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def run[T](coro: Any) -> T:
    return asyncio.run(coro)  # type: ignore[no-any-return]


def intent(**overrides: Any) -> DiscoveryIntent:
    fields: dict[str, Any] = {
        "partner_type": "broker",
        "product_category": "cookies",
        "location": "California",
        "raw_query": "cookie brokers in California",
    }
    return DiscoveryIntent(**{**fields, **overrides})


def hit(url: str, rank: int = 1, provider: str = "tavily") -> SearchHit:
    return SearchHit(url=url, title=url, snippet="s", rank=rank, provider=provider)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, "RETRY_POLICY", RetryPolicy(attempts=3, base_delay_s=0))


# --- Tavily (respx) -----------------------------------------------------------------

TAVILY_BODY = {
    "query": "q",
    "results": [
        {"title": "Acme Brokers", "url": "https://acme.example/contact", "content": "x" * 500},
        {"title": "Beta", "url": "https://beta.example/", "content": "Beta brokers", "score": 0.4},
        {"title": "Bad", "url": "not a url", "content": "skipped"},
    ],
}


@respx.mock
def test_tavily_sends_the_documented_request_and_parses_results() -> None:
    route = respx.post(search.TAVILY_URL).respond(200, json=TAVILY_BODY)
    hits = run(TavilyProvider("tvly-secret").search("cookie brokers", 20))
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer tvly-secret"
    assert json.loads(request.content) == {
        "query": "cookie brokers",
        "max_results": 20,
        "search_depth": "basic",
        "include_answer": False,
    }
    assert [(h.rank, str(h.url), h.provider) for h in hits] == [
        (1, "https://acme.example/contact", "tavily"),
        (2, "https://beta.example/", "tavily"),
    ]
    assert len(hits[0].snippet) == search.SNIPPET_CHARS
    assert hits[1].title == "Beta"


@respx.mock
def test_tavily_empty_results_is_an_empty_list() -> None:
    respx.post(search.TAVILY_URL).respond(200, json={"results": []})
    assert run(TavilyProvider("k").search("q", 5)) == []


@respx.mock
def test_tavily_retries_a_429_then_succeeds() -> None:
    route = respx.post(search.TAVILY_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=TAVILY_BODY)]
    )
    assert len(run(TavilyProvider("k").search("q", 5))) == 2
    assert route.call_count == 2


@respx.mock
def test_tavily_gives_up_after_the_retry_budget() -> None:
    route = respx.post(search.TAVILY_URL).respond(503)
    with pytest.raises(SearchError, match="HTTP 503"):
        run(TavilyProvider("k").search("q", 5))
    assert route.call_count == 3


@respx.mock
def test_tavily_does_not_retry_a_bad_key() -> None:
    route = respx.post(search.TAVILY_URL).respond(401)
    with pytest.raises(SearchError, match="HTTP 401"):
        run(TavilyProvider("bad").search("q", 5))
    assert route.call_count == 1


@respx.mock
def test_tavily_timeout_becomes_a_search_error() -> None:
    respx.post(search.TAVILY_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(SearchError, match="ReadTimeout"):
        run(TavilyProvider("k").search("q", 5))


@pytest.mark.parametrize("body", [{"oops": 1}, {"results": "nope"}, {"results": [1]}])
@respx.mock
def test_tavily_malformed_reply_is_a_search_error(body: dict[str, Any]) -> None:
    respx.post(search.TAVILY_URL).respond(200, json=body)
    with pytest.raises(SearchError, match="unexpected response"):
        run(TavilyProvider("k").search("q", 5))


@respx.mock
def test_tavily_errors_never_contain_the_key() -> None:
    respx.post(search.TAVILY_URL).respond(500)
    with pytest.raises(SearchError) as excinfo:
        run(TavilyProvider("tvly-very-secret").search("q", 5))
    assert "tvly-very-secret" not in str(excinfo.value)


# --- DuckDuckGo (injected search function: ddgs does not use httpx) -------------------


def test_ddg_maps_ddgs_rows_to_hits() -> None:
    rows = [
        {"title": "Acme", "href": "https://acme.example/", "body": "Brokers in CA"},
        {"title": "Nope", "href": "ftp://old.example/", "body": ""},
        {"title": "Beta", "href": "https://beta.example/x", "body": "More"},
    ]
    calls: list[tuple[str, int]] = []

    def fake(query: str, max_results: int) -> list[dict[str, Any]]:
        calls.append((query, max_results))
        return rows

    hits = run(DDGProvider(fake).search("cookie brokers", 20))
    assert calls == [("cookie brokers", 20)]
    assert [(h.rank, str(h.url), h.provider) for h in hits] == [
        (1, "https://acme.example/", "ddg"),
        (2, "https://beta.example/x", "ddg"),
    ]


def test_ddg_failure_becomes_a_search_error() -> None:
    def broken(query: str, max_results: int) -> list[dict[str, Any]]:
        raise RuntimeError("rate limited")

    with pytest.raises(SearchError, match="RuntimeError"):
        run(DDGProvider(broken).search("q", 5))


# --- queries ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("partner_type", "expected"),
    [
        ("broker", ["cookies food broker California contact", "cookies brokers California"]),
        (
            "distributor",
            ["cookies distributor California contact", "cookies wholesale distributors California"],
        ),
        (
            "retailer",
            [
                "cookies retailer California buyer contact",
                "independent stores selling cookies California",
            ],
        ),
        (
            "manufacturer",
            ["cookies manufacturer California contact", "cookies co-packers California"],
        ),
        ("other", ["cookies companies California contact", "cookies partners California"]),
    ],
)
def test_two_queries_per_partner_type(partner_type: str, expected: list[str]) -> None:
    assert build_queries(intent(partner_type=partner_type)) == expected


def test_queries_without_a_location_have_no_stray_spaces() -> None:
    assert build_queries(intent(location=None)) == [
        "cookies food broker contact",
        "cookies brokers",
    ]


# --- URL helpers --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPS://WWW.Acme.COM/Path#frag", "https://www.acme.com/Path"),
        ("https://acme.com", "https://acme.com/"),
        ("https://acme.com:443/a", "https://acme.com/a"),
        ("http://acme.com:80/a", "http://acme.com/a"),
        ("http://acme.com:8080/a", "http://acme.com:8080/a"),
        ("https://acme.com/a?utm_source=x&id=7&gclid=1", "https://acme.com/a?id=7"),
        ("https://acme.com./x", "https://acme.com/x"),
        ("ftp://acme.com/", None),
        ("mailto:sales@acme.com", None),
        ("/relative/path", None),
        ("", None),
    ],
)
def test_normalise_url(raw: str, expected: str | None) -> None:
    assert normalise_url(raw) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.acme.com/x", "acme.com"),
        ("https://shop.acme.co.uk/", "acme.co.uk"),
        ("acme.com", "acme.com"),
        ("https://acme.com.au/", "acme.com.au"),
        ("http://192.168.1.5/", "192.168.1.5"),
        ("http://localhost:8000/", "localhost"),
    ],
)
def test_registrable_domain(url: str, expected: str) -> None:
    assert registrable_domain(url) == expected


@pytest.mark.parametrize(
    ("url", "blocked"),
    [
        ("https://linkedin.com/company/x", True),
        ("https://www.linkedin.com/company/x", True),
        ("https://in.linkedin.com/x", True),
        ("https://notlinkedin.com/", False),
        ("https://linkedin.com.evil.example/", False),
        ("https://acme.example/", False),
    ],
)
def test_is_blocked_matches_the_domain_and_its_subdomains(url: str, blocked: bool) -> None:
    assert is_blocked(url, frozenset({"linkedin.com"})) is blocked


# --- merging ------------------------------------------------------------------------


def test_merge_keeps_the_best_rank_per_registrable_domain() -> None:
    first = [hit("https://www.acme.com/contact", 2), hit("https://beta.example/", 3)]
    second = [hit("https://shop.acme.com/", 1), hit("https://gamma.example/", 2)]
    merged = merge_hits([first, second], frozenset(), 10)
    assert [str(h.url) for h in merged] == [
        "https://shop.acme.com/",
        "https://www.acme.com/contact" if False else "https://gamma.example/",
        "https://beta.example/",
    ]
    assert [h.rank for h in merged] == [1, 2, 3]


def test_merge_treats_co_uk_subdomains_as_one_company() -> None:
    merged = merge_hits(
        [[hit("https://acme.co.uk/", 1), hit("https://shop.acme.co.uk/", 2)]], frozenset(), 10
    )
    assert [str(h.url) for h in merged] == ["https://acme.co.uk/"]


def test_merge_drops_blocklisted_domains_and_caps_the_count() -> None:
    hits = [
        hit("https://linkedin.com/company/a", 1),
        hit("https://a.example/", 2),
        hit("https://b.example/", 3),
        hit("https://c.example/", 4),
    ]
    merged = merge_hits([hits], frozenset({"linkedin.com"}), 2)
    assert [str(h.url) for h in merged] == ["https://a.example/", "https://b.example/"]


def test_merge_normalises_urls() -> None:
    merged = merge_hits([[hit("https://A.example/p?utm_source=x#top")]], frozenset(), 5)
    assert str(merged[0].url) == "https://a.example/p"


def test_merge_of_nothing_is_empty() -> None:
    assert merge_hits([[], []], frozenset(), 5) == []


# --- providers from settings --------------------------------------------------------


def names(settings: Settings) -> list[str]:
    return [p.name for p in build_providers(settings)]


def test_default_providers_are_tavily_then_ddg() -> None:
    assert names(make_settings(tavily_api_key="k")) == ["tavily", "ddg"]


def test_tavily_without_a_key_is_skipped() -> None:
    assert names(make_settings()) == ["ddg"]


def test_no_fallback_and_same_provider_twice() -> None:
    assert names(make_settings(tavily_api_key="k", search_fallback="")) == ["tavily"]
    assert names(make_settings(tavily_api_key="k", search_fallback="tavily")) == ["tavily"]
    assert names(make_settings(search_provider="ddg", search_fallback="tavily")) == ["ddg"]


# --- service ------------------------------------------------------------------------


class FakeProvider:
    def __init__(self, name: str, *replies: list[SearchHit] | Exception) -> None:
        self.name = name
        self.queries: list[str] = []
        self._replies = list(replies)

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.queries.append(query)
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def service(*providers: FakeProvider, blocklist: frozenset[str] = frozenset()) -> SearchService:
    return SearchService(providers, InMemoryRepository(), blocklist)


def discover(svc: SearchService, **overrides: Any) -> search.SearchOutcome:
    return run(svc.discover(intent(**overrides)))


PRIMARY = [hit("https://acme.example/", 1), hit("https://beta.example/", 2)]
BACKUP = [hit("https://gamma.example/", 1, "ddg")]


def test_both_queries_run_and_their_results_merge() -> None:
    primary = FakeProvider("tavily", PRIMARY)
    outcome = discover(service(primary))
    assert sorted(primary.queries) == sorted(build_queries(intent()))
    assert [str(h.url) for h in outcome.hits] == ["https://acme.example/", "https://beta.example/"]
    assert outcome.errors == []


def test_falls_back_when_the_primary_errors() -> None:
    primary = FakeProvider("tavily", SearchError("tavily", "HTTP 500"))
    backup = FakeProvider("ddg", BACKUP)
    outcome = discover(service(primary, backup))
    assert [str(h.url) for h in outcome.hits] == ["https://gamma.example/"]
    assert outcome.errors == []
    assert len(backup.queries) == 2


def test_falls_back_when_the_primary_returns_nothing() -> None:
    primary = FakeProvider("tavily", [])
    backup = FakeProvider("ddg", BACKUP)
    outcome = discover(service(primary, backup))
    assert [h.provider for h in outcome.hits] == ["ddg"]


def test_the_backup_is_not_called_when_the_primary_answers() -> None:
    backup = FakeProvider("ddg", BACKUP)
    discover(service(FakeProvider("tavily", PRIMARY), backup))
    assert backup.queries == []


def test_all_providers_failing_gives_a_stage_error_and_no_hits() -> None:
    svc = service(
        FakeProvider("tavily", SearchError("tavily", "HTTP 500")),
        FakeProvider("ddg", SearchError("ddg", "RuntimeError")),
    )
    outcome = discover(svc)
    assert outcome.hits == []
    assert {e.code for e in outcome.errors} == {"search_failed"}
    assert outcome.errors[0].stage == "scout_search"
    assert "tavily: HTTP 500" in outcome.errors[0].message


def test_all_providers_empty_is_no_results_not_a_failure() -> None:
    outcome = discover(service(FakeProvider("tavily", []), FakeProvider("ddg", [])))
    assert outcome.hits == []
    assert {e.code for e in outcome.errors} == {"no_results"}


def test_no_configured_provider_is_reported() -> None:
    outcome = discover(service())
    assert outcome.hits == []
    assert {e.code for e in outcome.errors} == {"no_provider"}


def test_everything_blocklisted_is_reported_not_silent() -> None:
    blocked = discover(
        service(
            FakeProvider("tavily", [hit("https://linkedin.com/a", 1)]),
            blocklist=frozenset({"linkedin.com"}),
        )
    )
    assert blocked.hits == []
    assert {e.code for e in blocked.errors} == {"no_results"}


def test_max_results_caps_the_merged_list() -> None:
    many = [hit(f"https://site{i}.example/", i) for i in range(1, 8)]
    outcome = discover(service(FakeProvider("tavily", many)), max_results=3)
    assert len(outcome.hits) == 3


def test_a_second_identical_query_is_served_from_the_cache() -> None:
    primary = FakeProvider("tavily", PRIMARY)
    svc = service(primary)
    discover(svc)
    calls_after_first = len(primary.queries)
    outcome = discover(svc)
    assert len(primary.queries) == calls_after_first  # 0 provider calls the second time
    assert outcome.cache_hits == 2
    assert [str(h.url) for h in outcome.hits] == ["https://acme.example/", "https://beta.example/"]


def test_the_cache_ignores_case_and_spacing_in_the_query() -> None:
    repo = InMemoryRepository()
    primary = FakeProvider("tavily", PRIMARY)
    svc = SearchService([primary], repo, frozenset())
    discover(svc, product_category="Cookies", location="California")
    first = len(primary.queries)
    discover(svc, product_category="  cookies ", location="CALIFORNIA")
    assert len(primary.queries) == first


def test_failures_and_empty_answers_are_not_cached() -> None:
    primary = FakeProvider("tavily", [], [], PRIMARY)  # both queries empty, then results
    svc = service(primary)
    assert discover(svc).hits == []
    assert len(discover(svc).hits) == 2  # the empty answer was not remembered


def test_the_cache_is_shared_across_max_results_values() -> None:
    primary = FakeProvider("tavily", [hit(f"https://s{i}.example/", i) for i in range(1, 6)])
    svc = service(primary)
    discover(svc, max_results=5)
    seen = len(primary.queries)
    assert len(discover(svc, max_results=2).hits) == 2
    assert len(primary.queries) == seen


ProviderFactory = Callable[[], FakeProvider]
