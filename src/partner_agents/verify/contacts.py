"""Deterministic contact verification (HLD §6.8, ADR-009, CLAUDE.md hard rule 3).

A contact is ``verified`` only if it appears in the text of the page it came from. No model is
involved: the extractor proposes, this module checks. The checks are plain functions, so the
rules are table-tested.
"""

import logging
import re
import unicodedata
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import email_validator
import phonenumbers
from pydantic import HttpUrl, TypeAdapter, ValidationError

from partner_agents.config import Settings
from partner_agents.schemas import (
    ExtractedLead,
    Lead,
    StageError,
    VerificationStatus,
)
from partner_agents.storage.repo import Repository

logger = logging.getLogger(__name__)

STAGE = "scout_verify"
MX_TIMEOUT_S = 3.0

# Higher is better; used to pick between duplicate leads.
STATUS_RANK: dict[VerificationStatus, int] = {
    "verified": 4,
    "mx_failed": 3,
    "not_in_source": 2,
    "invalid_format": 1,
    "unchecked": 0,
}

MxLookup = Callable[[str], Awaitable[bool]]

_URL = TypeAdapter(HttpUrl)
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_AT = re.compile(r"\s*[\[\(\{]\s*at\s*[\]\)\}]\s*")
_DOT = re.compile(r"\s*[\[\(\{]\s*dot\s*[\]\)\}]\s*")
# "info at acme dot com": the word "at" only counts between an email-looking local part and a
# domain that ends in a real-looking suffix, so ordinary prose is left alone.
_WORD_AT = re.compile(
    r"(?<![\w.+-])([\w.+-]+)\s+at\s+((?:[\w-]+(?:\.|\s+dot\s+))+[a-z]{2,})(?![\w-])"
)
_SPACED_AT = re.compile(r"(?<=[\w.+-])\s*@\s*(?=[\w-])")
_PHONE_RUN = re.compile(r"[\d()+\-.\s]{7,}")
_LEGAL_SUFFIX = frozenset(
    {"inc", "llc", "ltd", "corp", "co", "company", "corporation", "limited", "plc"}
)

# Words in a location that name a phone region. US states are enough to cover "Los Angeles, CA"
# style locations; anything else falls back to ``DEFAULT_PHONE_REGION``.
_COUNTRY_REGIONS = {
    "united states": "US", "usa": "US", "u.s.": "US", "america": "US",
    "india": "IN", "united kingdom": "GB", "uk": "GB", "england": "GB", "scotland": "GB",
    "wales": "GB", "canada": "CA", "australia": "AU", "germany": "DE", "france": "FR",
    "mexico": "MX", "ireland": "IE", "italy": "IT", "spain": "ES", "netherlands": "NL",
}  # fmt: skip
_US_STATES = frozenset(
    [
        "alabama",
        "alaska",
        "arizona",
        "arkansas",
        "california",
        "colorado",
        "connecticut",
        "delaware",
        "florida",
        "georgia",
        "hawaii",
        "idaho",
        "illinois",
        "indiana",
        "iowa",
        "kansas",
        "kentucky",
        "louisiana",
        "maine",
        "maryland",
        "massachusetts",
        "michigan",
        "minnesota",
        "mississippi",
        "missouri",
        "montana",
        "nebraska",
        "nevada",
        "ohio",
        "oklahoma",
        "oregon",
        "pennsylvania",
        "tennessee",
        "texas",
        "utah",
        "vermont",
        "virginia",
        "washington",
        "wisconsin",
        "wyoming",
    ]
)


# --- normalising the source ---------------------------------------------------------


def normalise_source(text: str) -> str:
    """Lowercase, collapse whitespace and undo ``[at]`` / ``(dot)`` style obfuscation."""
    out = unicodedata.normalize("NFKC", text)
    out = _INVISIBLE.sub("", out).lower()
    out = _AT.sub("@", out)
    out = _DOT.sub(".", out)
    out = _WORD_AT.sub(lambda m: f"{m[1]}@{re.sub(r'\s+dot\s+', '.', m[2])}", out)
    out = _SPACED_AT.sub("@", out)
    return " ".join(out.split())


def phone_digit_runs(text: str) -> list[str]:
    """The digits of each phone-looking stretch of ``text``, kept apart so that digits from
    neighbouring numbers cannot join into a number the page does not contain."""
    runs = (re.sub(r"\D", "", match) for match in _PHONE_RUN.findall(text))
    return [run for run in runs if len(run) >= 7]


# --- email --------------------------------------------------------------------------


def clean_email(raw: str) -> str:
    value = raw.strip().lower()
    return value[7:] if value.startswith("mailto:") else value


def is_valid_email_syntax(email: str) -> bool:
    try:
        email_validator.validate_email(email, check_deliverability=False)
    except email_validator.EmailNotValidError:
        return False
    return True


def email_in_source(email: str, normalised_source: str) -> bool:
    """The whole address, not a piece of a longer one (``xsales@a.com``, ``sales@a.com.au``)."""
    pattern = rf"(?<![\w.+-]){re.escape(email)}(?![\w-]|\.\w)"
    return re.search(pattern, normalised_source) is not None


def verify_email(raw: str, normalised_source: str) -> tuple[str, VerificationStatus]:
    """The cleaned address and its status: verified, not_in_source or invalid_format."""
    email = clean_email(raw)
    if not is_valid_email_syntax(email):
        return email, "invalid_format"
    if not email_in_source(email, normalised_source):
        return email, "not_in_source"
    return email, "verified"


async def dns_mx_lookup(domain: str) -> bool:
    """True if ``domain`` has an MX record. A timeout or resolver failure counts as "has one":
    only a definite "no such record" fails the check."""
    import dns.asyncresolver
    import dns.exception
    import dns.resolver

    try:
        await dns.asyncresolver.resolve(domain, "MX", lifetime=MX_TIMEOUT_S)
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
        return False
    except dns.exception.DNSException:
        return True
    return True


# --- phone --------------------------------------------------------------------------


def region_for(location: str | None, default: str) -> str:
    """The phone region a location points at, else ``default``."""
    text = (location or "").lower()
    for name, region in _COUNTRY_REGIONS.items():
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text):
            return region
    if any(re.search(rf"(?<!\w){state}(?!\w)", text) for state in _US_STATES):
        return "US"
    return default


def verify_phone(
    raw: str, region: str, digit_runs: Sequence[str]
) -> tuple[str | None, VerificationStatus]:
    """E.164 form (when parseable) and status. The national number must sit in one run."""
    try:
        parsed = phonenumbers.parse(raw, region)
    except phonenumbers.NumberParseException:
        return None, "invalid_format"
    if not phonenumbers.is_possible_number(parsed):
        return None, "invalid_format"
    e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    national = str(parsed.national_number)
    # Leading zeros of a national number (Italian numbers) are kept by ``italian_leading_zero``.
    if parsed.italian_leading_zero:
        national = "0" * (parsed.number_of_leading_zeros or 1) + national
    found = any(national in run for run in digit_runs)
    return e164, "verified" if found else "not_in_source"


# --- leads --------------------------------------------------------------------------


def company_key(name: str) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    return " ".join(w for w in words if w not in _LEGAL_SUFFIX)


def lead_rank(lead: Lead) -> tuple[int, int]:
    """Better verified leads sort higher: best status first, then the other channel."""
    email, phone = STATUS_RANK[lead.email_status], STATUS_RANK[lead.phone_status]
    return max(email, phone), min(email, phone)


def dedupe_leads(leads: Sequence[Lead]) -> list[Lead]:
    """One lead per ``(company, email or phone)``: the better-verified one, first seen on ties."""
    best: dict[tuple[str, str], Lead] = {}
    for lead in leads:
        key = (company_key(lead.company_name), (lead.email or lead.phone or "").lower())
        current = best.get(key)
        if current is None or lead_rank(lead) > lead_rank(current):
            best[key] = lead
    return list(best.values())


def _website(value: str | None) -> HttpUrl | None:
    if not value or not value.strip():
        return None
    text = value.strip()
    for candidate in (text, f"https://{text}"):
        try:
            url = _URL.validate_python(candidate)
        except ValidationError:
            continue
        if url.host and "." in url.host:  # "http://" parses as the host "http"
            return url
    return None


@dataclass(frozen=True)
class VerificationResult:
    leads: list[Lead]  # verified and rejected, for audit and eval
    errors: list[StageError]


async def verify_candidates(
    candidates: Mapping[str, Sequence[ExtractedLead]],
    repo: Repository,
    *,
    run_id: str,
    user_id: str,
    settings: Settings,
    intent_location: str | None = None,
    mx_lookup: MxLookup = dns_mx_lookup,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    new_id: Callable[[], str] = lambda: f"l_{uuid.uuid4().hex[:10]}",
) -> VerificationResult:
    """Check every candidate against the stored body of the page it came from."""
    leads: list[Lead] = []
    errors: list[StageError] = []
    for url, extracted in candidates.items():
        page = await repo.get_page(run_id, url)
        if page is None:
            errors.append(
                StageError(
                    stage=STAGE,
                    code="page_missing",
                    message="page text is no longer stored",
                    url=url,
                )
            )
        source = normalise_source(f"{page.text}\n{page.contact_text}") if page else ""
        runs = phone_digit_runs(f"{page.text}\n{page.contact_text}") if page else []
        for item in extracted:
            lead = await _verify_one(
                item, url, source, runs, page is not None, run_id, user_id, settings,
                intent_location, mx_lookup, now(), new_id(),
            )  # fmt: skip
            if lead is not None:
                leads.append(lead)
    return VerificationResult(dedupe_leads(leads), errors)


async def _verify_one(
    item: ExtractedLead,
    url: str,
    source: str,
    digit_runs: Sequence[str],
    have_page: bool,
    run_id: str,
    user_id: str,
    settings: Settings,
    intent_location: str | None,
    mx_lookup: MxLookup,
    created_at: datetime,
    lead_id: str,
) -> Lead | None:
    email: str | None = None
    email_status: VerificationStatus = "unchecked"
    if item.email and item.email.strip():
        email, email_status = (
            verify_email(item.email, source)
            if have_page
            else (clean_email(item.email), "unchecked")
        )
        if email_status == "verified" and settings.verify_mx:
            domain = email.rsplit("@", 1)[-1]
            if not await mx_lookup(domain):
                email_status = "mx_failed"
    phone: str | None = None
    phone_status: VerificationStatus = "unchecked"
    if item.phone and item.phone.strip():
        if have_page:
            region = region_for(item.location or intent_location, settings.default_phone_region)
            e164, phone_status = verify_phone(item.phone, region, digit_runs)
            phone = e164 or item.phone.strip()
        else:
            phone = item.phone.strip()
    try:
        source_url = _URL.validate_python(url)
    except ValidationError:
        logger.warning("candidate has an invalid source url", extra={"url": url})
        return None
    return Lead(
        id=lead_id,
        run_id=run_id,
        user_id=user_id,
        company_name=item.company_name.strip(),
        partner_type=item.partner_type,
        location=item.location,
        website=_website(item.website),
        email=email,
        email_status=email_status,
        phone=phone,
        phone_status=phone_status,
        source_url=source_url,
        evidence_quote=item.evidence_quote,
        created_at=created_at,
    )
