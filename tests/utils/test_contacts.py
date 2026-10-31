import pytest

from partner_agents.utils.contacts import has_contact_pattern, has_email, has_phone


@pytest.mark.parametrize(
    "text",
    [
        "write to sales@acme-foods.com today",
        "SALES@ACME.CO.UK",
        "first.last+tag@sub.domain.example",
        "info [at] acme [dot] com",
        "info(at)acme.com",
        "info {at} acme {dot} com",
        "info [AT] acme [DOT] co [dot] uk",
        "info at acme dot com",
    ],
)
def test_emails_are_detected(text: str) -> None:
    assert has_email(text)
    assert has_contact_pattern(text)


@pytest.mark.parametrize(
    "text",
    [
        "no contact here",
        "look at example.com for details",
        "meet us at 5 pm at the booth",
        "arrives at noon",
        "user@localhost",
        "@handle on social media",
        "at the end of the day",
    ],
)
def test_ordinary_prose_is_not_an_email(text: str) -> None:
    assert not has_email(text)


@pytest.mark.parametrize(
    "text",
    [
        "call +1 (310) 555-0100",
        "call 310-555-0100",
        "(310) 555 0100",
        "310.555.0100",
        "mobile +91 98765 43210",
        "9876543210",
        "+44 20 7946 0958",
    ],
)
def test_phones_are_detected(text: str) -> None:
    assert has_phone(text)
    assert has_contact_pattern(text)


@pytest.mark.parametrize(
    "text",
    [
        "published 2026-10-31",
        "order 12345",
        "version 1.2.3",
        "founded in 1999",
        "SKU 1234567",
        "ref 3105550100x",
    ],
)
def test_dates_and_ids_are_not_phones(text: str) -> None:
    assert not has_phone(text)
    assert not has_contact_pattern(text)
