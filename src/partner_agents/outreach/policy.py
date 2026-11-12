"""Outreach policy as pure functions (HLD §6.9).

Every check maps a lead to a skip status or lets it through. Nothing here does I/O: the caller
loads the recent drafts and the user's token bucket, passes ``now`` in and stores the new bucket.
"""

import re
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict

from partner_agents.config import OutreachMode
from partner_agents.schemas import Lead, OutreachDraft

SkipStatus = Literal["skipped_ineligible", "skipped_duplicate", "skipped_cap"]

MAX_BODY_WORDS = 180
DAY_S = 86_400
OPT_OUT_LINE = (
    'If you would rather not hear from us, reply "unsubscribe" and we will not write again.'
)

# Drafts that count as "already contacted" for the dedupe window. Rejected, failed and skipped
# drafts never reached the recipient, so they do not block a later attempt.
_CONTACTED = frozenset({"pending_approval", "approved", "sent", "drafted_in_gmail", "outboxed"})
_OPT_OUT_RE = re.compile(r"unsubscribe|opt[\s-]?out", re.IGNORECASE)


def _norm(address: str) -> str:
    return address.strip().lower()


def is_eligible(lead: Lead) -> bool:
    """Only a lead with a verified email can be emailed."""
    return lead.outreach_eligible


def is_allowlisted(recipient: str, mode: OutreachMode, allowlist: Iterable[str]) -> bool:
    """``gmail_send`` reaches only allowlisted addresses; the other modes send nothing."""
    if mode != "gmail_send":
        return True
    return _norm(recipient) in {_norm(item) for item in allowlist}


def is_duplicate(
    recipient: str, previous: Iterable[OutreachDraft], now: datetime, dedupe_days: int
) -> bool:
    """True if this recipient was contacted within the last ``dedupe_days`` days."""
    since = now - timedelta(days=dedupe_days)
    wanted = _norm(recipient)
    return any(
        _norm(draft.recipient) == wanted
        and draft.status in _CONTACTED
        and draft.created_at >= since
        for draft in previous
    )


class TokenBucket(BaseModel):
    """A token bucket: ``capacity`` tokens, refilled evenly over ``period_s`` (24 h by default).

    Frozen and pure: ``take`` returns the new bucket. The outreach policy uses it for the daily
    send cap and ``api/ratelimit.py`` for the hourly discovery limit.
    """

    model_config = ConfigDict(frozen=True)

    capacity: int
    tokens: float
    updated_at: datetime
    period_s: float = DAY_S

    @classmethod
    def full(cls, capacity: int, now: datetime, period_s: float = DAY_S) -> "TokenBucket":
        return cls(capacity=capacity, tokens=float(capacity), updated_at=now, period_s=period_s)

    def refilled(self, now: datetime) -> "TokenBucket":
        elapsed = max(0.0, (now - self.updated_at).total_seconds())
        tokens = min(float(self.capacity), self.tokens + elapsed * self.capacity / self.period_s)
        return self.model_copy(update={"tokens": tokens, "updated_at": max(now, self.updated_at)})

    def take(self, now: datetime) -> tuple[bool, "TokenBucket"]:
        """Spend one token if there is one. Returns (allowed, new bucket)."""
        bucket = self.refilled(now)
        if bucket.tokens < 1:
            return False, bucket
        return True, bucket.model_copy(update={"tokens": bucket.tokens - 1})

    def wait_s(self, now: datetime) -> float:
        """Seconds until one token is available (0 if there is one now, inf if capacity is 0)."""
        tokens = self.refilled(now).tokens
        if tokens >= 1:
            return 0.0
        if self.capacity <= 0:
            return float("inf")
        return (1 - tokens) * self.period_s / self.capacity


def check_lead(
    lead: Lead,
    *,
    previous: Iterable[OutreachDraft],
    bucket: TokenBucket,
    now: datetime,
    mode: OutreachMode,
    allowlist: Iterable[str],
    dedupe_days: int,
) -> tuple[SkipStatus | None, TokenBucket]:
    """Run every check in order. A token is spent only when the lead passes all the others."""
    if not is_eligible(lead) or lead.email is None:
        return "skipped_ineligible", bucket
    if not is_allowlisted(lead.email, mode, allowlist):
        return "skipped_ineligible", bucket
    if is_duplicate(lead.email, previous, now, dedupe_days):
        return "skipped_duplicate", bucket
    allowed, bucket = bucket.take(now)
    return (None if allowed else "skipped_cap"), bucket


class BodyCheck(BaseModel):
    body: str  # with the opt-out line appended if it was missing
    word_count: int
    problems: list[Literal["too_long", "missing_sender"]]

    @property
    def ok(self) -> bool:
        return not self.problems


def has_opt_out(body: str) -> bool:
    return _OPT_OUT_RE.search(body) is not None


def enforce_body(body: str, sender_name: str) -> BodyCheck:
    """Append the opt-out line if missing, then check the word limit and the sender name."""
    text = body.rstrip()
    if not has_opt_out(text):
        text = f"{text}\n\n{OPT_OUT_LINE}"
    words = len(text.split())
    problems: list[Literal["too_long", "missing_sender"]] = []
    if words > MAX_BODY_WORDS:
        problems.append("too_long")
    if sender_name.strip().lower() not in text.lower():
        problems.append("missing_sender")
    return BodyCheck(body=text, word_count=words, problems=problems)
