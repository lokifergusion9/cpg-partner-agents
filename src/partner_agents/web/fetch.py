"""Safe page fetcher (HLD §6.4).

Every request, including each redirect hop and every ``robots.txt``, goes through the same
guards: http(s) only, the host must resolve only to public addresses (SSRF), the origin's
``robots.txt`` must allow the path, the response must be HTML and smaller than
``FETCH_MAX_BYTES``. At most two requests per host are in flight.

A page that fails is reported as a ``PageRef`` with a status and a ``StageError``; the run
carries on with the other pages.

A page whose text scores at or above ``INJECTION_THRESHOLD`` (``web/safety.injection_score``) is
reported as ``injection_flagged`` and is neither stored nor indexed (T-32).

Known limit: the SSRF check resolves the host and httpx resolves it again to connect, so a
DNS-rebinding server could answer differently the second time. Pinning the connection to
the checked address is a security-review item (HLD §12).
"""

import asyncio
import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from pydantic import HttpUrl, TypeAdapter

from partner_agents.config import Settings
from partner_agents.schemas import PageRecord, PageRef, StageError
from partner_agents.storage.repo import Repository
from partner_agents.utils.contacts import has_contact_pattern
from partner_agents.utils.retry import RetryPolicy, retry_async
from partner_agents.web.html import decode_html, extract_page
from partner_agents.web.safety import injection_score

logger = logging.getLogger(__name__)

STAGE = "scout_fetch"
USER_AGENT = "CPGPartnerAgents/0.1 (+https://github.com/lokifergusion9/cpg-partner-agents)"
ROBOTS_TOKEN = "CPGPartnerAgents"
MAX_REDIRECTS = 3
PER_HOST_LIMIT = 2
ROBOTS_MAX_BYTES = 512_000
REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
HTML_TYPES = ("text/html", "application/xhtml+xml")

# Transport retries for one GET (429 / 5xx / timeouts); tests swap in a zero-delay policy.
RETRY_POLICY = RetryPolicy(attempts=2, base_delay_s=0.5)

FailureStatus = Literal[
    "blocked_robots",
    "timeout",
    "http_error",
    "too_large",
    "non_html",
    "ssrf_blocked",
    "injection_flagged",
]
Resolver = Callable[[str], Awaitable[list[str]]]

_URL = TypeAdapter(HttpUrl)


class FetchFailure(Exception):
    def __init__(
        self,
        status: FailureStatus,
        message: str,
        *,
        http_status: int | None = None,
        injection_score: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.http_status = http_status
        self.injection_score = injection_score


class _Transient(Exception):
    """A 429 or 5xx answer, shaped for ``utils/retry`` (``status_code``, ``retry_after``)."""

    def __init__(self, status_code: int, retry_after: str | None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass(frozen=True)
class _Raw:
    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class FetchOutcome:
    page: PageRef
    error: StageError | None = None


# --- SSRF guard ---------------------------------------------------------------------


async def system_resolver(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


def is_public_ip(text: str) -> bool:
    """True only for globally routable unicast addresses."""
    try:
        ip = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def assert_public(url: str, resolver: Resolver) -> None:
    """Raise ``FetchFailure`` unless ``url`` is http(s) and its host resolves only publicly."""
    parts = urlsplit(url)
    host = parts.hostname
    if parts.scheme not in {"http", "https"} or not host:
        raise FetchFailure("ssrf_blocked", f"scheme {parts.scheme or '?'!r} is not allowed")
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        addresses = [host]
    except ValueError:
        try:
            addresses = await resolver(host)
        except OSError as exc:
            raise FetchFailure("http_error", "DNS lookup failed") from exc
    if not addresses:
        raise FetchFailure("http_error", "DNS lookup returned no address")
    if not all(is_public_ip(address) for address in addresses):
        raise FetchFailure("ssrf_blocked", "host resolves to a non-public address")


# --- fetcher ------------------------------------------------------------------------


def _is_html(content_type: str) -> bool:
    return content_type.split(";", 1)[0].strip().lower() in HTML_TYPES


def _allow_all(_url: str) -> bool:
    return True


def _deny_all(_url: str) -> bool:
    return False


class Fetcher:
    def __init__(
        self,
        settings: Settings,
        repo: Repository,
        *,
        client: httpx.AsyncClient | None = None,
        resolver: Resolver = system_resolver,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._resolver = resolver
        self._clock = clock
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.fetch_timeout_s), follow_redirects=False
        )
        self._slots = asyncio.Semaphore(settings.fetch_concurrency)
        self._host_slots: dict[str, asyncio.Semaphore] = {}
        self._robots: dict[str, Callable[[str], bool]] = {}
        self._robots_locks: dict[str, asyncio.Lock] = {}

    async def __aenter__(self) -> "Fetcher":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # public API

    async def fetch_many(self, urls: Sequence[str], run_id: str) -> list[FetchOutcome]:
        return list(await asyncio.gather(*(self.fetch(url, run_id) for url in urls)))

    async def fetch(self, url: str, run_id: str) -> FetchOutcome:
        key = str(_URL.validate_python(url))
        async with self._slots:
            try:
                page = await self._fetch_page(key, run_id)
            except FetchFailure as failure:
                logger.info("page not fetched", extra={"url": key, "status": failure.status})
                return FetchOutcome(
                    PageRef(
                        url=_URL.validate_python(key),
                        title=None,
                        status=failure.status,
                        injection_score=failure.injection_score,
                    ),
                    StageError(stage=STAGE, code=failure.status, message=failure.message, url=key),
                )
        logger.info("page fetched", extra={"url": key, "chars": page.chars})
        return FetchOutcome(page)

    async def fetch_html(self, url: str) -> tuple[str, str]:
        """The final URL and decoded HTML of ``url``, through every guard (SSRF, robots, size).

        Nothing is stored or scored. The eval snapshot tool uses it (``eval/snapshot.py``).
        Raises ``FetchFailure``.
        """
        key = str(_URL.validate_python(url))
        async with self._slots:
            final_url, raw = await self._get(
                key, max_bytes=self._settings.fetch_max_bytes, check_robots=True, require_html=True
            )
        return final_url, decode_html(raw.body, raw.headers.get("content-type", ""))

    # one page

    async def _fetch_page(self, url: str, run_id: str) -> PageRef:
        final_url, raw = await self._get(
            url, max_bytes=self._settings.fetch_max_bytes, check_robots=True, require_html=True
        )
        html = decode_html(raw.body, raw.headers.get("content-type", ""))
        extracted = await asyncio.to_thread(extract_page, html, final_url)
        text, contact_text = extracted.main_text, extracted.contact_text
        hidden_text = extracted.hidden_text
        contact_page_url: str | None = None
        if extracted.contact_link and not has_contact_pattern(f"{text}\n{contact_text}"):
            extra = await self._contact_page(extracted.contact_link)
            if extra is not None:
                text = f"{text}\n\n{extra[0]}" if text else extra[0]
                contact_text = f"{contact_text}\n{extra[1]}" if contact_text else extra[1]
                hidden_text = f"{hidden_text}\n{extra[2]}" if hidden_text else extra[2]
                contact_page_url = extracted.contact_link
        score = injection_score(f"{text}\n{contact_text}", hidden_text)
        if score >= self._settings.injection_threshold:
            # The page is excluded: it is neither stored nor indexed, so no model reads it.
            raise FetchFailure(
                "injection_flagged",
                f"the page looks like it tries to instruct the model (score {score})",
                injection_score=score,
            )
        await self._repo.save_page(
            PageRecord(
                run_id=run_id,
                url=url,
                text=text,
                contact_text=contact_text,
                fetched_at=self._clock(),
            )
        )
        return PageRef(
            url=_URL.validate_python(url),
            title=extracted.title,
            status="ok",
            chars=len(text) + len(contact_text),
            injection_score=score,
            contact_page_url=_URL.validate_python(contact_page_url) if contact_page_url else None,
        )

    async def _contact_page(self, url: str) -> tuple[str, str, str] | None:
        """Text, contact text and hidden text of the contact page, or ``None`` if unreadable."""
        try:
            final_url, raw = await self._get(
                url, max_bytes=self._settings.fetch_max_bytes, check_robots=True, require_html=True
            )
        except FetchFailure as failure:
            logger.info("contact page skipped", extra={"url": url, "status": failure.status})
            return None
        html = decode_html(raw.body, raw.headers.get("content-type", ""))
        extracted = await asyncio.to_thread(extract_page, html, final_url)
        return extracted.main_text, extracted.contact_text, extracted.hidden_text

    # requests

    async def _get(
        self, url: str, *, max_bytes: int, check_robots: bool, require_html: bool
    ) -> tuple[str, _Raw]:
        """GET ``url``, following at most ``MAX_REDIRECTS`` redirects, re-checking every hop."""
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            await assert_public(current, self._resolver)
            if check_robots and not await self._robots_allow(current):
                raise FetchFailure("blocked_robots", "disallowed by robots.txt")
            raw = await self._send(current, max_bytes=max_bytes, require_html=require_html)
            if raw.status not in REDIRECT_STATUS:
                return current, raw
            location = raw.headers.get("location")
            if not location:
                raise FetchFailure("http_error", f"HTTP {raw.status} without a Location header")
            current = urljoin(current, location)
        raise FetchFailure("http_error", f"more than {MAX_REDIRECTS} redirects")

    async def _send(self, url: str, *, max_bytes: int, require_html: bool) -> _Raw:
        host = urlsplit(url).hostname or ""
        slot = self._host_slots.setdefault(host, asyncio.Semaphore(PER_HOST_LIMIT))

        async def once() -> _Raw:
            async with (
                slot,
                self._client.stream(
                    "GET", url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.1"}
                ) as response,
            ):
                status = response.status_code
                if status in REDIRECT_STATUS:
                    return _Raw(status, dict(response.headers), b"")
                if status == 429 or status >= 500:
                    raise _Transient(status, response.headers.get("retry-after"))
                if status >= 400:
                    raise FetchFailure("http_error", f"HTTP {status}", http_status=status)
                content_type = response.headers.get("content-type", "")
                if require_html and not _is_html(content_type):
                    raise FetchFailure("non_html", f"content type {content_type or 'unknown'!r}")
                declared = response.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > max_bytes:
                    raise FetchFailure("too_large", f"{declared} bytes exceeds {max_bytes}")
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise FetchFailure("too_large", f"more than {max_bytes} bytes")
                    chunks.append(chunk)
                return _Raw(status, dict(response.headers), b"".join(chunks))

        try:
            return await retry_async(once, policy=RETRY_POLICY)
        except FetchFailure:
            raise
        except _Transient as exc:
            raise FetchFailure(
                "http_error", f"HTTP {exc.status_code}", http_status=exc.status_code
            ) from exc
        except httpx.TimeoutException as exc:
            raise FetchFailure(
                "timeout", f"no answer within {self._settings.fetch_timeout_s:g} s"
            ) from exc
        except httpx.HTTPError as exc:
            raise FetchFailure("http_error", type(exc).__name__) from exc

    # robots.txt

    async def _robots_allow(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        async with self._robots_locks.setdefault(origin, asyncio.Lock()):
            if origin not in self._robots:
                self._robots[origin] = await self._load_robots(origin)
        return self._robots[origin](url)

    async def _load_robots(self, origin: str) -> Callable[[str], bool]:
        """RFC 9309: a missing file (4xx) allows everything; an unreachable one allows nothing."""
        try:
            _, raw = await self._get(
                f"{origin}/robots.txt",
                max_bytes=ROBOTS_MAX_BYTES,
                check_robots=False,
                require_html=False,
            )
        except FetchFailure as failure:
            if failure.http_status is not None and 400 <= failure.http_status < 500:
                return _allow_all
            logger.info(
                "robots.txt unreachable, treating the site as disallowed", extra={"origin": origin}
            )
            return _deny_all
        parser = RobotFileParser()
        parser.parse(decode_html(raw.body, raw.headers.get("content-type", "")).splitlines())
        parser.modified()  # without this, can_fetch() assumes the file was never read
        return lambda target: parser.can_fetch(ROBOTS_TOKEN, target)
