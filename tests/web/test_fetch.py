import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx

from partner_agents.config import Settings
from partner_agents.storage.repo import InMemoryRepository
from partner_agents.utils.retry import RetryPolicy
from partner_agents.web import fetch
from partner_agents.web.fetch import (
    Fetcher,
    FetchFailure,
    FetchOutcome,
    assert_public,
    is_public_ip,
)

PUBLIC_IP = "93.184.216.34"
HOME = "https://acme.example/"


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def resolver(table: dict[str, list[str]] | None = None) -> fetch.Resolver:
    async def resolve(host: str) -> list[str]:
        return (table or {}).get(host, [PUBLIC_IP])

    return resolve


def html_page(body: str, title: str = "Acme Brokers") -> str:
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


MAIN = (
    "<article><h1>Acme</h1><p>We broker cookies across California for many retailers.</p></article>"
)
FOOTER = '<footer>Email <a href="mailto:sales@acme.example">sales@acme.example</a></footer>'


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "RETRY_POLICY", RetryPolicy(attempts=2, base_delay_s=0))


@pytest.fixture
def repo() -> InMemoryRepository:
    return InMemoryRepository()


def get(
    url: str,
    repo: InMemoryRepository,
    *,
    table: dict[str, list[str]] | None = None,
    **settings: Any,
) -> FetchOutcome:
    async def go() -> FetchOutcome:
        async with Fetcher(make_settings(**settings), repo, resolver=resolver(table)) as fetcher:
            return await fetcher.fetch(url, "r_1")

    return asyncio.run(go())


def robots(body: str | int = 404, host: str = "acme.example") -> respx.Route:
    route = respx.get(f"https://{host}/robots.txt")
    return route.respond(body) if isinstance(body, int) else route.respond(200, text=body)


def page_route(path: str = "/", body: str = "", host: str = "acme.example") -> respx.Route:
    return respx.get(f"https://{host}{path}").respond(200, html=html_page(body))


def stored(repo: InMemoryRepository, url: str = HOME) -> Any:
    return asyncio.run(repo.get_page("r_1", url))


# --- a good page --------------------------------------------------------------------


@respx.mock
def test_ok_page_is_stored_with_main_and_contact_text(repo: InMemoryRepository) -> None:
    robots()
    page_route("/", MAIN + FOOTER)
    outcome = get(HOME, repo)
    assert outcome.error is None
    assert outcome.page.status == "ok"
    assert outcome.page.title == "Acme Brokers"
    record = stored(repo)
    assert "broker cookies across California" in record.text
    assert "mailto: sales@acme.example" in record.contact_text  # a footer mailto: is captured
    assert outcome.page.chars == len(record.text) + len(record.contact_text)
    assert record.run_id == "r_1"


@respx.mock
def test_requests_identify_the_crawler(repo: InMemoryRepository) -> None:
    robots()
    route = page_route("/", MAIN)
    get(HOME, repo)
    assert route.calls.last.request.headers["user-agent"].startswith("CPGPartnerAgents/")


# --- failure statuses ---------------------------------------------------------------


@respx.mock
def test_robots_disallow_blocks_the_page(repo: InMemoryRepository) -> None:
    robots("User-agent: *\nDisallow: /private\n")
    private = page_route("/private/team", MAIN)
    outcome = get("https://acme.example/private/team", repo)
    assert outcome.page.status == "blocked_robots"
    assert outcome.error is not None
    assert (outcome.error.stage, outcome.error.code) == ("scout_fetch", "blocked_robots")
    assert not private.called
    assert stored(repo, "https://acme.example/private/team") is None


@respx.mock
def test_robots_allows_other_paths_and_our_own_agent_rules_win(repo: InMemoryRepository) -> None:
    robots("User-agent: *\nDisallow: /\n\nUser-agent: CPGPartnerAgents\nAllow: /\n")
    page_route("/", MAIN)
    assert get(HOME, repo).page.status == "ok"


@respx.mock
def test_a_site_that_disallows_everything_for_all_agents(repo: InMemoryRepository) -> None:
    robots("User-agent: *\nDisallow: /\n")
    page_route("/", MAIN)
    assert get(HOME, repo).page.status == "blocked_robots"


@respx.mock
def test_missing_robots_txt_allows_the_site(repo: InMemoryRepository) -> None:
    robots(404)
    page_route("/", MAIN)
    assert get(HOME, repo).page.status == "ok"


@respx.mock
def test_unreachable_robots_txt_is_treated_as_disallowed(repo: InMemoryRepository) -> None:
    robots(503)
    page = page_route("/", MAIN)
    assert get(HOME, repo).page.status == "blocked_robots"
    assert not page.called


@respx.mock
def test_timeout(repo: InMemoryRepository) -> None:
    robots()
    respx.get(HOME).mock(side_effect=httpx.ReadTimeout("slow"))
    outcome = get(HOME, repo)
    assert outcome.page.status == "timeout"
    assert outcome.error is not None
    assert outcome.error.url == HOME


@respx.mock
def test_http_error_for_a_404(repo: InMemoryRepository) -> None:
    robots()
    route = respx.get(HOME).respond(404)
    outcome = get(HOME, repo)
    assert outcome.page.status == "http_error"
    assert outcome.error is not None
    assert "404" in outcome.error.message
    assert route.call_count == 1  # a 404 is not retried


@respx.mock
def test_a_503_is_retried_then_reported(repo: InMemoryRepository) -> None:
    robots()
    route = respx.get(HOME).respond(503)
    assert get(HOME, repo).page.status == "http_error"
    assert route.call_count == 2


@respx.mock
def test_a_503_that_clears_on_retry_succeeds(repo: InMemoryRepository) -> None:
    robots()
    route = respx.get(HOME).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, html=html_page(MAIN))]
    )
    assert get(HOME, repo).page.status == "ok"
    assert route.call_count == 2


@respx.mock
def test_connection_error_is_an_http_error(repo: InMemoryRepository) -> None:
    robots()
    respx.get(HOME).mock(side_effect=httpx.ConnectError("refused"))
    outcome = get(HOME, repo)
    assert outcome.page.status == "http_error"


@respx.mock
def test_too_large_by_content_length(repo: InMemoryRepository) -> None:
    robots()
    respx.get(HOME).respond(200, html="x" * 300)
    assert get(HOME, repo, fetch_max_bytes=100).page.status == "too_large"


@respx.mock
def test_too_large_while_streaming_without_a_content_length(repo: InMemoryRepository) -> None:
    async def body() -> AsyncIterator[bytes]:
        for _ in range(10):
            yield b"<p>" + b"x" * 50 + b"</p>"

    robots()
    respx.get(HOME).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/html"}, content=body())
    )
    assert get(HOME, repo, fetch_max_bytes=200).page.status == "too_large"


@pytest.mark.parametrize("content_type", ["application/pdf", "image/png", "application/json", ""])
@respx.mock
def test_non_html(repo: InMemoryRepository, content_type: str) -> None:
    robots()
    respx.get(HOME).respond(200, content=b"%PDF", headers={"content-type": content_type})
    assert get(HOME, repo).page.status == "non_html"


@respx.mock
def test_xhtml_counts_as_html(repo: InMemoryRepository) -> None:
    robots()
    respx.get(HOME).respond(
        200, content=html_page(MAIN).encode(), headers={"content-type": "application/xhtml+xml"}
    )
    assert get(HOME, repo).page.status == "ok"


# --- SSRF ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.5/admin",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0/",
        "http://100.64.0.1/",
    ],
)
@respx.mock
def test_private_ip_literals_are_blocked(repo: InMemoryRepository, url: str) -> None:
    route = respx.route().respond(200, html="secret")
    outcome = get(url, repo)
    assert outcome.page.status == "ssrf_blocked"
    assert not route.called


@respx.mock
def test_a_hostname_resolving_to_a_private_address_is_blocked(repo: InMemoryRepository) -> None:
    route = respx.route().respond(200, html="secret")
    outcome = get("https://sneaky.example/", repo, table={"sneaky.example": ["10.1.2.3"]})
    assert outcome.page.status == "ssrf_blocked"
    assert not route.called


@respx.mock
def test_one_private_address_among_public_ones_blocks_the_host(repo: InMemoryRepository) -> None:
    route = respx.route().respond(200, html="secret")
    table = {"mixed.example": [PUBLIC_IP, "127.0.0.1"]}
    assert get("https://mixed.example/", repo, table=table).page.status == "ssrf_blocked"
    assert not route.called


@respx.mock
def test_a_redirect_to_a_private_ip_is_blocked(repo: InMemoryRepository) -> None:
    robots()
    respx.get(HOME).respond(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
    internal = respx.get("http://169.254.169.254/latest/meta-data/").respond(200, html="creds")
    outcome = get(HOME, repo)
    assert outcome.page.status == "ssrf_blocked"
    assert not internal.called


@respx.mock
def test_a_redirect_to_a_hostname_that_resolves_privately_is_blocked(
    repo: InMemoryRepository,
) -> None:
    robots()
    respx.get(HOME).respond(301, headers={"location": "https://intranet.example/x"})
    internal = respx.get("https://intranet.example/x").respond(200, html="secret")
    outcome = get(HOME, repo, table={"intranet.example": ["192.168.0.9"]})
    assert outcome.page.status == "ssrf_blocked"
    assert not internal.called


@respx.mock
def test_a_redirect_to_a_non_http_scheme_is_blocked(repo: InMemoryRepository) -> None:
    robots()
    respx.get(HOME).respond(302, headers={"location": "file:///etc/passwd"})
    assert get(HOME, repo).page.status == "ssrf_blocked"


@respx.mock
def test_redirects_are_followed_up_to_three_hops(repo: InMemoryRepository) -> None:
    robots()
    respx.get("https://acme.example/a").respond(302, headers={"location": "/b"})
    respx.get("https://acme.example/b").respond(302, headers={"location": "/c"})
    respx.get("https://acme.example/c").respond(302, headers={"location": "/d"})
    page_route("/d", MAIN)
    assert get("https://acme.example/a", repo).page.status == "ok"


@respx.mock
def test_a_fourth_redirect_is_an_error(repo: InMemoryRepository) -> None:
    robots()
    for here, there in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]:
        respx.get(f"https://acme.example/{here}").respond(302, headers={"location": f"/{there}"})
    page_route("/e", MAIN)
    outcome = get("https://acme.example/a", repo)
    assert outcome.page.status == "http_error"
    assert outcome.error is not None
    assert "redirects" in outcome.error.message


@respx.mock
def test_a_redirect_loop_ends(repo: InMemoryRepository) -> None:
    robots()
    respx.get(HOME).respond(302, headers={"location": HOME})
    assert get(HOME, repo).page.status == "http_error"


@respx.mock
def test_a_redirect_without_location_is_an_error(repo: InMemoryRepository) -> None:
    robots()
    respx.get(HOME).respond(302)
    assert get(HOME, repo).page.status == "http_error"


@respx.mock
def test_a_redirect_to_another_host_is_checked_against_its_robots(repo: InMemoryRepository) -> None:
    robots()
    robots("User-agent: *\nDisallow: /\n", host="other.example")
    respx.get(HOME).respond(302, headers={"location": "https://other.example/"})
    target = respx.get("https://other.example/").respond(200, html=html_page(MAIN))
    assert get(HOME, repo).page.status == "blocked_robots"
    assert not target.called


# --- is_public_ip / assert_public ---------------------------------------------------


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("93.184.216.34", True),
        ("8.8.8.8", True),
        ("2606:4700:4700::1111", True),
        ("127.0.0.1", False),
        ("10.0.0.1", False),
        ("172.16.5.5", False),
        ("192.168.0.1", False),
        ("169.254.1.1", False),
        ("100.64.0.1", False),
        ("0.0.0.0", False),
        ("224.0.0.1", False),
        ("240.0.0.1", False),
        ("::1", False),
        ("fe80::1", False),
        ("fc00::1", False),
        ("::ffff:10.0.0.1", False),
        ("::ffff:8.8.8.8", True),
        ("fe80::1%eth0", False),
        ("not an ip", False),
    ],
)
def test_is_public_ip(address: str, public: bool) -> None:
    assert is_public_ip(address) is public


@pytest.mark.parametrize("url", ["ftp://acme.example/", "file:///etc/passwd", "gopher://x/", "//x"])
def test_assert_public_rejects_other_schemes(url: str) -> None:
    with pytest.raises(FetchFailure) as caught:
        asyncio.run(assert_public(url, resolver()))
    assert caught.value.status == "ssrf_blocked"


def test_a_dns_failure_is_an_http_error_not_an_ssrf_block() -> None:
    async def failing(host: str) -> list[str]:
        raise OSError("no such host")

    with pytest.raises(FetchFailure) as caught:
        asyncio.run(assert_public("https://nope.example/", failing))
    assert caught.value.status == "http_error"


def test_an_empty_dns_answer_is_an_http_error() -> None:
    with pytest.raises(FetchFailure) as caught:
        asyncio.run(assert_public("https://nope.example/", resolver({"nope.example": []})))
    assert caught.value.status == "http_error"


def test_a_numeric_host_that_the_resolver_maps_to_loopback_is_blocked() -> None:
    """``http://2130706433/`` is 127.0.0.1 to the OS resolver."""
    with pytest.raises(FetchFailure) as caught:
        asyncio.run(assert_public("http://2130706433/", resolver({"2130706433": ["127.0.0.1"]})))
    assert caught.value.status == "ssrf_blocked"


# --- robots.txt caching -------------------------------------------------------------


@respx.mock
def test_robots_txt_is_fetched_once_per_host(repo: InMemoryRepository) -> None:
    route = robots()
    page_route("/a", MAIN)
    page_route("/b", MAIN)

    async def go() -> list[FetchOutcome]:
        async with Fetcher(make_settings(), repo, resolver=resolver()) as fetcher:
            return await fetcher.fetch_many(
                ["https://acme.example/a", "https://acme.example/b"], "r_1"
            )

    outcomes = asyncio.run(go())
    assert [o.page.status for o in outcomes] == ["ok", "ok"]
    assert route.call_count == 1


@respx.mock
def test_each_host_has_its_own_robots_txt(repo: InMemoryRepository) -> None:
    robots("User-agent: *\nDisallow: /\n", host="closed.example")
    robots(host="open.example")
    page_route("/", MAIN, host="closed.example")
    page_route("/", MAIN, host="open.example")

    async def go() -> list[FetchOutcome]:
        async with Fetcher(make_settings(), repo, resolver=resolver()) as fetcher:
            return await fetcher.fetch_many(
                ["https://closed.example/", "https://open.example/"], "r_1"
            )

    closed, opened = asyncio.run(go())
    assert (closed.page.status, opened.page.status) == ("blocked_robots", "ok")


# --- contact-page discovery ---------------------------------------------------------

NO_CONTACT = MAIN + '<nav><a href="/contact-us">Contact us</a> <a href="/careers">Jobs</a></nav>'
CONTACT_PAGE = "<p>Reach our team.</p><address>Acme, 1 Main St. Call 310-555-0100</address>"


@respx.mock
def test_contact_page_is_followed_when_the_first_page_has_no_contact(
    repo: InMemoryRepository,
) -> None:
    robots()
    page_route("/", NO_CONTACT)
    contact = page_route("/contact-us", CONTACT_PAGE)
    outcome = get(HOME, repo)
    assert contact.call_count == 1
    assert str(outcome.page.contact_page_url) == "https://acme.example/contact-us"
    record = stored(repo)
    assert "Reach our team" in record.text
    assert "310-555-0100" in record.contact_text


@respx.mock
def test_contact_page_is_not_followed_when_the_page_already_has_a_contact(
    repo: InMemoryRepository,
) -> None:
    robots()
    page_route("/", NO_CONTACT + FOOTER)
    contact = page_route("/contact-us", CONTACT_PAGE)
    outcome = get(HOME, repo)
    assert not contact.called
    assert outcome.page.contact_page_url is None


@respx.mock
def test_contact_page_is_not_followed_when_the_main_text_has_a_phone(
    repo: InMemoryRepository,
) -> None:
    robots()
    page_route("/", NO_CONTACT + "<p>Phone 310-555-0100 for orders.</p>")
    contact = page_route("/contact-us", CONTACT_PAGE)
    get(HOME, repo)
    assert not contact.called


@respx.mock
def test_only_one_contact_page_is_followed(repo: InMemoryRepository) -> None:
    robots()
    page_route("/", NO_CONTACT)
    page_route("/contact-us", '<p>No details.</p><a href="/about-us">About</a>')
    about = page_route("/about-us", "<p>About</p>")
    get(HOME, repo)
    assert not about.called


@respx.mock
def test_a_contact_page_blocked_by_robots_is_skipped_quietly(repo: InMemoryRepository) -> None:
    robots("User-agent: *\nDisallow: /contact-us\n")
    page_route("/", NO_CONTACT)
    contact = page_route("/contact-us", CONTACT_PAGE)
    outcome = get(HOME, repo)
    assert outcome.page.status == "ok"
    assert outcome.page.contact_page_url is None
    assert not contact.called


@respx.mock
def test_a_failing_contact_page_does_not_fail_the_page(repo: InMemoryRepository) -> None:
    robots()
    page_route("/", NO_CONTACT)
    respx.get("https://acme.example/contact-us").respond(500)
    outcome = get(HOME, repo)
    assert outcome.page.status == "ok"
    assert outcome.error is None


@respx.mock
def test_a_contact_link_to_another_company_is_not_followed(repo: InMemoryRepository) -> None:
    robots()
    page_route("/", MAIN + '<a href="https://other.example/contact">Contact</a>')
    other = page_route("/contact", CONTACT_PAGE, host="other.example")
    get(HOME, repo)
    assert not other.called


# --- concurrency --------------------------------------------------------------------


@respx.mock
def test_at_most_two_requests_per_host_in_flight(repo: InMemoryRepository) -> None:
    peaks: dict[str, int] = {}
    now = 0

    async def serve(request: httpx.Request) -> httpx.Response:
        nonlocal now
        now += 1
        peaks["acme.example"] = max(peaks.get("acme.example", 0), now)
        await asyncio.sleep(0.02)
        now -= 1
        return httpx.Response(200, html=html_page(MAIN))

    robots()
    paths = [f"/p{i}" for i in range(8)]
    for path in paths:
        respx.get(f"https://acme.example{path}").mock(side_effect=serve)

    async def go() -> list[FetchOutcome]:
        async with Fetcher(make_settings(fetch_concurrency=8), repo, resolver=resolver()) as f:
            return await f.fetch_many([f"https://acme.example{p}" for p in paths], "r_1")

    outcomes = asyncio.run(go())
    assert all(o.page.status == "ok" for o in outcomes)
    assert peaks["acme.example"] == 2


@respx.mock
def test_fetch_concurrency_bounds_pages_across_hosts(repo: InMemoryRepository) -> None:
    peak = now = 0

    async def serve(request: httpx.Request) -> httpx.Response:
        nonlocal now, peak
        now += 1
        peak = max(peak, now)
        await asyncio.sleep(0.02)
        now -= 1
        return httpx.Response(200, html=html_page(MAIN))

    hosts = [f"site{i}.example" for i in range(9)]
    for host in hosts:
        robots(host=host)
        respx.get(f"https://{host}/").mock(side_effect=serve)

    async def go() -> list[FetchOutcome]:
        async with Fetcher(make_settings(fetch_concurrency=3), repo, resolver=resolver()) as f:
            return await f.fetch_many([f"https://{h}/" for h in hosts], "r_1")

    assert all(o.page.status == "ok" for o in asyncio.run(go()))
    assert peak == 3


@respx.mock
def test_one_failing_page_does_not_stop_the_others(repo: InMemoryRepository) -> None:
    robots()
    page_route("/good", MAIN)
    respx.get("https://acme.example/bad").respond(404)

    async def go() -> list[FetchOutcome]:
        async with Fetcher(make_settings(), repo, resolver=resolver()) as f:
            return await f.fetch_many(
                ["https://acme.example/bad", "https://acme.example/good"], "r_1"
            )

    bad, good = asyncio.run(go())
    assert (bad.page.status, good.page.status) == ("http_error", "ok")
    assert good.error is None
