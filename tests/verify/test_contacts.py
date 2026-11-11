import asyncio
import itertools
from datetime import UTC, datetime
from typing import Any

import pytest

from partner_agents.config import Settings
from partner_agents.schemas import ExtractedLead, Lead, PageRecord
from partner_agents.storage.repo import InMemoryRepository
from partner_agents.verify.contacts import (
    VerificationResult,
    clean_email,
    company_key,
    dedupe_leads,
    email_in_source,
    normalise_source,
    phone_digit_runs,
    region_for,
    verify_candidates,
    verify_email,
    verify_phone,
)

NOW = datetime(2026, 11, 4, 9, 0, tzinfo=UTC)
URL = "https://acme.example/contact"
OTHER = "https://beta.example/about"


def run[T](coro: Any) -> T:
    return asyncio.run(coro)  # type: ignore[no-any-return]


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def candidate(**overrides: Any) -> ExtractedLead:
    fields: dict[str, Any] = {"company_name": "Acme Foods", "evidence_quote": "quote"}
    return ExtractedLead(**{**fields, **overrides})


def verify(
    text: str,
    *candidates: ExtractedLead,
    contact_text: str = "",
    url: str = URL,
    **kwargs: Any,
) -> VerificationResult:
    # Pages are dated NOW, so keep the page TTL off the wall clock.
    repo = InMemoryRepository(clock=lambda: NOW)
    run(
        repo.save_page(
            PageRecord(run_id="r_1", url=url, text=text, contact_text=contact_text, fetched_at=NOW)
        )
    )
    ids = (f"l_{n}" for n in itertools.count())
    return run(
        verify_candidates(
            {url: list(candidates)},
            repo,
            run_id="r_1",
            user_id="u_1",
            settings=kwargs.pop("settings", settings()),
            now=lambda: NOW,
            new_id=lambda: next(ids),
            **kwargs,
        )
    )


def only(result: VerificationResult) -> Lead:
    [lead] = result.leads
    return lead


# --- source normalisation -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Email SALES@Acme.COM now", "email sales@acme.com now"),
        ("info [at] acme [dot] com", "info@acme.com"),
        ("info (at) acme (dot) com", "info@acme.com"),
        ("info {AT} acme {DOT} com", "info@acme.com"),
        ("info[at]acme.com", "info@acme.com"),
        ("write info at acme dot com today", "write info@acme.com today"),
        ("write sales at acme.co.uk today", "write sales@acme.co.uk today"),
        ("info @ acme.com", "info@acme.com"),
        ("a" + chr(0x200B) + "cme" + chr(0xA0) + " foods", "acme foods"),
        ("  lots \n\n of\t space  ", "lots of space"),
        ("we meet at noon on friday", "we meet at noon on friday"),
        ("look at example", "look at example"),
    ],
)
def test_normalise_source(raw: str, expected: str) -> None:
    assert normalise_source(raw) == expected


# --- email --------------------------------------------------------------------------


def test_a_hallucinated_email_is_not_in_source() -> None:
    lead = only(verify("Contact us at hello@acme.example", candidate(email="sales@acme.example")))
    assert lead.email_status == "not_in_source"
    assert lead.outreach_eligible is False


@pytest.mark.parametrize(
    "page",
    [
        "Email: sales@acme.example",
        "EMAIL: SALES@ACME.EXAMPLE.",
        "Write to <sales@acme.example>, thanks",
        "sales [at] acme [dot] example",
        "sales (at) acme (dot) example",
        "sales at acme dot example",
        "sales @ acme.example",
    ],
)
def test_an_email_on_the_page_is_verified_even_when_obfuscated(page: str) -> None:
    lead = only(verify(page, candidate(email="sales@acme.example")))
    assert lead.email_status == "verified"
    assert lead.email == "sales@acme.example"
    assert lead.outreach_eligible is True


def test_a_mailto_target_in_the_contact_text_verifies() -> None:
    lead = only(
        verify(
            "About us",
            candidate(email="sales@acme.example"),
            contact_text="mailto: sales@acme.example",
        )
    )
    assert lead.email_status == "verified"


@pytest.mark.parametrize(
    "bad", ["not-an-email", "sales@@acme.example", "a b@acme.example", "@acme.example", "sales@"]
)
def test_a_malformed_email_is_invalid_format(bad: str) -> None:
    lead = only(verify(f"page mentions {bad}", candidate(email=bad)))
    assert lead.email_status == "invalid_format"


@pytest.mark.parametrize(
    ("page", "email"),
    [
        ("contact xsales@acme.example", "sales@acme.example"),
        ("contact sales@acme.example.au", "sales@acme.example"),
        ("contact sales@acme.example-shop.com", "sales@acme.example"),
        ("contact my.sales@acme.example", "sales@acme.example"),
        ("contact bigsales@acme.example", "sales@acme.example"),
    ],
)
def test_an_email_that_is_only_part_of_a_longer_address_is_not_in_source(
    page: str, email: str
) -> None:
    assert verify_email(email, normalise_source(page))[1] == "not_in_source"


@pytest.mark.parametrize(
    "page",
    [
        "sales@acme.example",
        "email sales@acme.example.",
        "(sales@acme.example)",
        "sales@acme.example, or",
        "a:sales@acme.example",
    ],
)
def test_punctuation_around_an_address_does_not_hide_it(page: str) -> None:
    assert email_in_source("sales@acme.example", normalise_source(page))


def test_the_email_is_cleaned_before_it_is_checked() -> None:
    lead = only(verify("sales@acme.example", candidate(email="  mailto:Sales@Acme.Example ")))
    assert (lead.email, lead.email_status) == ("sales@acme.example", "verified")
    assert clean_email(" MAILTO:A@B.CO ") == "a@b.co"


def test_a_reserved_domain_is_invalid_format() -> None:
    assert verify_email("a@b.test", "a@b.test")[1] == "invalid_format"


# --- MX (behind VERIFY_MX) ----------------------------------------------------------


def mx(answers: dict[str, bool], calls: list[str]) -> Any:
    async def lookup(domain: str) -> bool:
        calls.append(domain)
        return answers.get(domain, True)

    return lookup


def test_mx_is_not_checked_unless_the_flag_is_on() -> None:
    calls: list[str] = []
    lead = only(
        verify(
            "sales@acme.example",
            candidate(email="sales@acme.example"),
            mx_lookup=mx({"acme.example": False}, calls),
        )
    )
    assert lead.email_status == "verified"
    assert calls == []


def test_a_domain_without_mx_is_mx_failed() -> None:
    calls: list[str] = []
    lead = only(
        verify(
            "sales@acme.example",
            candidate(email="sales@acme.example"),
            settings=settings(verify_mx=True),
            mx_lookup=mx({"acme.example": False}, calls),
        )
    )
    assert lead.email_status == "mx_failed"
    assert lead.outreach_eligible is False
    assert calls == ["acme.example"]


def test_mx_is_only_looked_up_for_emails_that_passed_the_other_checks() -> None:
    calls: list[str] = []
    result = verify(
        "hello@acme.example",
        candidate(email="ghost@acme.example"),
        settings=settings(verify_mx=True),
        mx_lookup=mx({}, calls),
    )
    assert only(result).email_status == "not_in_source"
    assert calls == []


def test_a_domain_with_mx_stays_verified() -> None:
    lead = only(
        verify(
            "sales@acme.example",
            candidate(email="sales@acme.example"),
            settings=settings(verify_mx=True),
            mx_lookup=mx({}, []),
        )
    )
    assert lead.email_status == "verified"


# --- phone --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("page", "phone"),
    [
        ("Call (310) 555-0100 today", "+1 310-555-0100"),
        ("Call (310) 555-0100 today", "310.555.0100"),
        ("Call (310) 555-0100 today", "3105550100"),
        ("Call (310) 555-0100 today", "(310) 555 0100"),
        ("Call +1 310 555 0100 today", "310-555-0100"),
        ("tel: +13105550100", "(310) 555-0100"),
        ("Call 310.555.0100 or fax 310.555.0101", "310-555-0101"),
    ],
)
def test_phone_format_variants_verify_and_become_e164(page: str, phone: str) -> None:
    lead = only(verify(page, candidate(phone=phone)))
    assert lead.phone_status == "verified"
    assert lead.phone in {"+13105550100", "+13105550101"}


def test_a_phone_not_on_the_page_is_not_in_source() -> None:
    lead = only(verify("Call (310) 555-0100 today", candidate(phone="(310) 555-0199")))
    assert lead.phone_status == "not_in_source"
    assert lead.phone == "+13105550199"


def test_digits_of_neighbouring_numbers_do_not_join_into_a_phone() -> None:
    page = "Call 310-555 or 0100-77 for sales"
    assert phone_digit_runs(page) == []  # runs shorter than 7 digits are ignored
    lead = only(verify("Orders 310 555\n\nFax 0100 777 9999", candidate(phone="310-555-0100")))
    assert lead.phone_status == "not_in_source"


def test_a_number_split_across_two_phone_runs_is_not_found() -> None:
    page = "Sales 213-555-0123 and support 310-555-0188"
    lead = only(verify(page, candidate(phone="(555) 012-3310")))
    assert lead.phone_status == "not_in_source"


@pytest.mark.parametrize("bad", ["12345", "abc", "+999 1", ""])
def test_an_impossible_phone_is_invalid_format(bad: str) -> None:
    status = verify_phone(bad, "US", ["12345678901"])[1] if bad else "unchecked"
    assert status in {"invalid_format", "unchecked"}


def test_a_uk_number_uses_the_gb_region_from_the_location() -> None:
    lead = only(
        verify(
            "Call +44 20 7946 0958", candidate(phone="020 7946 0958", location="London, England")
        )
    )
    assert (lead.phone, lead.phone_status) == ("+442079460958", "verified")


def test_the_default_region_applies_without_a_location() -> None:
    lead = only(
        verify(
            "Call 98765 43210",
            candidate(phone="098765 43210"),
            settings=settings(default_phone_region="IN"),
        )
    )
    assert (lead.phone, lead.phone_status) == ("+919876543210", "verified")


def test_the_intent_location_is_used_when_the_lead_has_none() -> None:
    lead = only(
        verify(
            "Call 020 7946 0958", candidate(phone="020 7946 0958"), intent_location="Manchester, UK"
        )
    )
    assert lead.phone == "+442079460958"
    assert lead.phone_status == "verified"


@pytest.mark.parametrize(
    ("location", "region"),
    [
        ("Los Angeles, CA", "DE-DEFAULT"),  # a bare "CA" is ambiguous, so it is not guessed
        ("California, US", "US"),
        ("Texas", "US"),
        ("United States", "US"),
        ("London, UK", "GB"),
        ("Scotland", "GB"),
        ("Mumbai, India", "IN"),
        ("Toronto, Canada", "CA"),
        ("West Virginia", "US"),
        ("Berlin", "DE-DEFAULT"),
        (None, "DE-DEFAULT"),
        ("", "DE-DEFAULT"),
        ("Pune, MH", "DE-DEFAULT"),
    ],
)
def test_region_for(location: str | None, region: str) -> None:
    assert region_for(location, "DE-DEFAULT") == region


def test_the_phone_status_is_unchecked_when_no_phone_was_extracted() -> None:
    lead = only(verify("anything", candidate()))
    assert (lead.phone, lead.phone_status) == (None, "unchecked")
    assert (lead.email, lead.email_status) == (None, "unchecked")


# --- leads, dedupe ------------------------------------------------------------------


def test_rejected_leads_are_kept_for_audit() -> None:
    result = verify(
        "sales@acme.example",
        candidate(company_name="Real Co", email="sales@acme.example"),
        candidate(company_name="Ghost Co", email="ghost@nowhere.example"),
    )
    assert {lead.company_name: lead.email_status for lead in result.leads} == {
        "Real Co": "verified",
        "Ghost Co": "not_in_source",
    }
    assert [lead.outreach_eligible for lead in result.leads] == [True, False]


def test_the_lead_carries_its_source_and_run() -> None:
    lead = only(
        verify(
            "sales@acme.example",
            candidate(
                email="sales@acme.example",
                website="acme.example",
                partner_type="broker",
                location="LA",
            ),
        )
    )
    assert (lead.run_id, lead.user_id, lead.id, lead.created_at) == ("r_1", "u_1", "l_0", NOW)
    assert str(lead.source_url) == URL
    assert str(lead.website) == "https://acme.example/"
    assert (lead.partner_type, lead.location, lead.evidence_quote) == ("broker", "LA", "quote")


@pytest.mark.parametrize("website", [None, "", "  ", "not a url at all", "http://"])
def test_a_bad_website_becomes_none(website: str | None) -> None:
    assert only(verify("x", candidate(website=website))).website is None


def test_the_better_verified_duplicate_wins() -> None:
    weak = candidate(company_name="Acme Foods, Inc.", email="sales@acme.example")
    strong = candidate(company_name="ACME FOODS", email="sales@acme.example")
    repo = InMemoryRepository(clock=lambda: NOW)
    run(repo.save_page(PageRecord(run_id="r_1", url=URL, text="no email here", fetched_at=NOW)))
    run(
        repo.save_page(
            PageRecord(run_id="r_1", url=OTHER, text="sales@acme.example", fetched_at=NOW)
        )
    )
    result = run(
        verify_candidates(
            {URL: [weak], OTHER: [strong]}, repo, run_id="r_1", user_id="u_1", settings=settings()
        )
    )
    assert len(result.leads) == 1
    assert result.leads[0].email_status == "verified"
    assert str(result.leads[0].source_url) == OTHER


def test_first_seen_wins_a_tie() -> None:
    result = verify(
        "sales@acme.example",
        candidate(company_name="Acme", email="sales@acme.example", location="first"),
        candidate(company_name="Acme", email="SALES@acme.example", location="second"),
    )
    assert [lead.location for lead in result.leads] == ["first"]


def test_different_companies_or_contacts_are_not_merged() -> None:
    result = verify(
        "a@acme.example b@acme.example c@beta.example",
        candidate(company_name="Acme", email="a@acme.example"),
        candidate(company_name="Acme", email="b@acme.example"),
        candidate(company_name="Beta", email="a@acme.example"),
    )
    assert len(result.leads) == 3


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Acme Foods, Inc.", "acme foods"),
        ("ACME   FOODS LLC", "acme foods"),
        ("Acme-Foods Co.", "acme foods"),
        ("Müller & Söhne GmbH", "m ller s hne gmbh"),
        ("Inc", ""),
    ],
)
def test_company_key(name: str, key: str) -> None:
    assert company_key(name) == key


def test_dedupe_of_nothing_is_empty() -> None:
    assert dedupe_leads([]) == []


# --- missing page -------------------------------------------------------------------


def test_a_candidate_whose_page_is_gone_stays_unverified_and_is_reported() -> None:
    result = run(
        verify_candidates(
            {URL: [candidate(email="sales@acme.example", phone="310-555-0100")]},
            InMemoryRepository(),
            run_id="r_1",
            user_id="u_1",
            settings=settings(),
        )
    )
    lead = only(result)
    assert (lead.email_status, lead.phone_status) == ("unchecked", "unchecked")
    assert lead.outreach_eligible is False
    [error] = result.errors
    assert (error.stage, error.code, error.url) == ("scout_verify", "page_missing", URL)


def test_no_candidates_gives_no_leads() -> None:
    result = run(
        verify_candidates({}, InMemoryRepository(), run_id="r", user_id="u", settings=settings())
    )
    assert (result.leads, result.errors) == ([], [])
