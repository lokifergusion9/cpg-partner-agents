"""Outreach policy: every skip status, the token bucket and opt-out enforcement (HLD §6.9)."""

from datetime import UTC, datetime, timedelta

import pytest

from partner_agents.config import OutreachMode
from partner_agents.outreach.policy import (
    DAY_S,
    MAX_BODY_WORDS,
    OPT_OUT_LINE,
    TokenBucket,
    check_lead,
    enforce_body,
    has_opt_out,
    is_allowlisted,
    is_duplicate,
)
from partner_agents.schemas import Lead, OutreachDraft

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
EMAIL = "buyer@acme.example"


def lead(email: str | None = EMAIL, status: str = "verified") -> Lead:
    return Lead(
        id="l1",
        run_id="r",
        user_id="u",
        company_name="Acme Foods",
        partner_type="broker",
        location=None,
        website=None,
        email=email,
        email_status=status,  # type: ignore[arg-type]
        phone=None,
        phone_status="unchecked",
        source_url="https://acme.example/",  # type: ignore[arg-type]
        evidence_quote="q",
        created_at=NOW,
    )


def draft(recipient: str = EMAIL, status: str = "outboxed", days_ago: float = 1) -> OutreachDraft:
    return OutreachDraft(
        id="d1",
        lead_id="l1",
        user_id="u",
        run_id="r",
        channel="email",
        recipient=recipient,
        subject="Hi",
        body="b",
        status=status,  # type: ignore[arg-type]
        prompt_version="v1",
        created_at=NOW - timedelta(days=days_ago),
    )


FULL = TokenBucket.full(20, NOW)
EMPTY = TokenBucket(capacity=20, tokens=0.0, updated_at=NOW)


INEL, DUP, CAP = "skipped_ineligible", "skipped_duplicate", "skipped_cap"
ME = ["me@x.example"]

CHECKS = {
    "eligible": (lead(), [], FULL, "outbox", [], None),
    "no email": (lead(email=None, status="unchecked"), [], FULL, "outbox", [], INEL),
    "unverified": (lead(status="mx_failed"), [], FULL, "outbox", [], INEL),
    "not in source": (lead(status="not_in_source"), [], FULL, "outbox", [], INEL),
    "send, not allowlisted": (lead(), [], FULL, "gmail_send", ME, INEL),
    "send, allowlisted": (lead(), [], FULL, "gmail_send", ["BUYER@acme.example"], None),
    "gmail_draft ignores allowlist": (lead(), [], FULL, "gmail_draft", [], None),
    "duplicate": (lead(), [draft()], FULL, "outbox", [], DUP),
    "duplicate, other case": (
        lead(),
        [draft(recipient=" Buyer@ACME.example")],
        FULL,
        "outbox",
        [],
        DUP,
    ),
    "old contact is fine": (lead(), [draft(days_ago=31)], FULL, "outbox", [], None),
    "rejected does not count": (lead(), [draft(status="rejected")], FULL, "outbox", [], None),
    "other recipient": (lead(), [draft(recipient="x@y.example")], FULL, "outbox", [], None),
    "cap reached": (lead(), [], EMPTY, "outbox", [], CAP),
    "ineligible beats cap": (lead(status="invalid_format"), [], EMPTY, "outbox", [], INEL),
    "duplicate beats cap": (lead(), [draft()], EMPTY, "outbox", [], DUP),
}


@pytest.mark.parametrize(
    ("the_lead", "previous", "bucket", "mode", "allowlist", "expected"),
    list(CHECKS.values()),
    ids=list(CHECKS),
)
def test_check_lead(
    the_lead: Lead,
    previous: list[OutreachDraft],
    bucket: TokenBucket,
    mode: OutreachMode,
    allowlist: list[str],
    expected: str | None,
) -> None:
    status, new_bucket = check_lead(
        the_lead,
        previous=previous,
        bucket=bucket,
        now=NOW,
        mode=mode,
        allowlist=allowlist,
        dedupe_days=30,
    )
    assert status == expected
    # A token is spent only when the lead goes through.
    spent = bucket.refilled(NOW).tokens - new_bucket.tokens
    assert spent == (1 if expected is None else 0)


@pytest.mark.parametrize(
    ("status", "contacted"),
    [
        ("pending_approval", True),
        ("approved", True),
        ("sent", True),
        ("drafted_in_gmail", True),
        ("outboxed", True),
        ("rejected", False),
        ("failed", False),
        ("skipped_duplicate", False),
        ("skipped_ineligible", False),
        ("skipped_cap", False),
    ],
)
def test_dedupe_statuses(status: str, contacted: bool) -> None:
    assert is_duplicate(EMAIL, [draft(status=status)], NOW, 30) is contacted


def test_dedupe_window_edge() -> None:
    assert is_duplicate(EMAIL, [draft(days_ago=30)], NOW, 30)
    assert not is_duplicate(EMAIL, [draft(days_ago=30.01)], NOW, 30)


def test_allowlist_only_applies_to_send() -> None:
    assert is_allowlisted(EMAIL, "outbox", [])
    assert not is_allowlisted(EMAIL, "gmail_send", [])


def test_bucket_drains_then_refuses() -> None:
    bucket = TokenBucket.full(3, NOW)
    results = []
    for _ in range(4):
        allowed, bucket = bucket.take(NOW)
        results.append(allowed)
    assert results == [True, True, True, False]


def test_bucket_refills_with_time() -> None:
    bucket = TokenBucket(capacity=24, tokens=0.0, updated_at=NOW)
    assert bucket.take(NOW)[0] is False
    # 24 tokens per day = one per hour.
    assert bucket.take(NOW + timedelta(minutes=59))[0] is False
    allowed, after = bucket.take(NOW + timedelta(hours=1))
    assert allowed and after.tokens == pytest.approx(0.0)
    assert bucket.refilled(NOW + timedelta(hours=6)).tokens == pytest.approx(6.0)


def test_bucket_never_exceeds_capacity() -> None:
    bucket = TokenBucket(capacity=20, tokens=5.0, updated_at=NOW)
    assert bucket.refilled(NOW + timedelta(seconds=10 * DAY_S)).tokens == 20.0


def test_bucket_ignores_clock_going_back() -> None:
    bucket = TokenBucket(capacity=20, tokens=5.0, updated_at=NOW)
    later = bucket.refilled(NOW - timedelta(hours=3))
    assert later.tokens == 5.0 and later.updated_at == NOW


def test_zero_capacity_always_refuses() -> None:
    bucket = TokenBucket.full(0, NOW)
    assert bucket.take(NOW + timedelta(days=5))[0] is False


@pytest.mark.parametrize(
    ("body", "appended"),
    [
        ("Hi, I'm Priya from Crumb & Co.\n\nPriya", True),
        ("Hi, Priya here. Reply unsubscribe to stop.", False),
        ("Priya. To opt out, just say so.", False),
        ("Priya. You can OPT-OUT anytime.", False),
    ],
)
def test_opt_out_appended_when_missing(body: str, appended: bool) -> None:
    result = enforce_body(body, "Priya")
    assert result.body.endswith(OPT_OUT_LINE) is appended
    assert has_opt_out(result.body)
    assert result.ok


def test_body_too_long() -> None:
    body = "Priya " + "word " * MAX_BODY_WORDS
    result = enforce_body(body, "Priya")
    assert result.problems == ["too_long"]
    assert result.word_count > MAX_BODY_WORDS


def test_appended_opt_out_counts_toward_limit() -> None:
    opt_out_words = len(OPT_OUT_LINE.split())
    body = "Priya " + "word " * (MAX_BODY_WORDS - opt_out_words - 1)
    assert enforce_body(body, "Priya").ok
    assert enforce_body(body + "extra", "Priya").problems == ["too_long"]


def test_body_missing_sender() -> None:
    result = enforce_body("Hello there. Reply unsubscribe to stop.", "Priya Shah")
    assert result.problems == ["missing_sender"]
    assert enforce_body("hello from priya shah. unsubscribe", "Priya Shah").ok
