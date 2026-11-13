"""Deterministic checks on every agent line (HLD §6.11, ADR-017). Pure functions.

Each check has a stable id (``C-NUM`` ...) that playbook rules point to through ``check_id``.
A check that does not apply to a turn (``C-AI`` outside the opening and ``disclose_ai``,
``C-SIGNOFF`` outside the closing line) is not reported. Details never quote the reply, so a
contact value cannot leak into logs or the UI through them (CLAUDE.md rule 5).
"""

import re
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from partner_agents.calls.scheduler import zone
from partner_agents.schemas import (
    ApprovedFact,
    BrandProfile,
    CallOutcome,
    CallStage,
    Directive,
    RuleCheck,
)
from partner_agents.utils.contacts import EMAIL_RE, OBFUSCATED_EMAIL_RE, PHONE_RE

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_AI = re.compile(r"\b(?:AI|A\.I\.|artificial intelligence)(?!\w)", re.IGNORECASE)
_BOOKING = re.compile(
    r"\b(?:book(?:ed|ing)?|schedul(?:e|ed|ing)|calendar|appointment|invite|"
    r"pencil (?:you |it )?in|lock (?:it |that )?in)\b",
    re.IGNORECASE,
)
_PRE_HANDOFF: frozenset[CallStage] = frozenset({"opening", "discovery", "qualification", "pitch"})
_TIME_ACTIONS = frozenset({"propose_slot", "offer_callback", "confirm_booking"})
_PUNCTUATION = re.compile(r"[^\w\s]")


class CheckContext(BaseModel):
    """What a reply is checked against: the turn's directive and the data it may use."""

    model_config = ConfigDict(frozen=True)

    stage: CallStage  # the stage the reply is spoken in (after ``advance``)
    directive: Directive
    brand: BrandProfile
    banned_phrases: Sequence[str]
    facts: Sequence[ApprovedFact] = ()  # the facts the directive allows
    slots: dict[str, str] = {}
    partner_last: str = ""  # the partner's last turn
    max_words: int = 60
    outcome: CallOutcome | None = None  # set when this reply ends the call

    @property
    def opted_out(self) -> bool:
        return self.outcome == "opted_out"


def run_checks(text: str, ctx: CheckContext) -> list[RuleCheck]:
    checks = [
        _num(text, ctx),
        _ban(text, ctx),
        _check("C-ONEQ", text.count("?") <= 1, "more than one question"),
        _check("C-LEN", len(text.split()) <= ctx.max_words, f"over {ctx.max_words} words"),
        _pii(text, ctx),
        _stage(text, ctx),
    ]
    if ctx.directive.action in ("open", "disclose_ai") and not ctx.opted_out:
        checks.append(_check("C-AI", bool(_AI.search(text)), "does not say it is an AI"))
    if ctx.directive.action == "close" and not ctx.opted_out:
        signed = _normal(ctx.brand.sign_off) in _normal(text)
        checks.append(_check("C-SIGNOFF", signed, "the sign-off line is missing"))
    return checks


def failures(checks: Sequence[RuleCheck]) -> list[RuleCheck]:
    return [check for check in checks if not check.passed]


# --- the checks ---------------------------------------------------------------------


def _check(rule_id: str, passed: bool, problem: str) -> RuleCheck:
    return RuleCheck(rule_id=rule_id, passed=passed, by="code", detail=None if passed else problem)


def _num(text: str, ctx: CheckContext) -> RuleCheck:
    """Every number must come from the allowed facts, the slots, the slot or the partner."""
    sources = [fact.text for fact in ctx.facts] + list(ctx.slots.values()) + [ctx.partner_last]
    allowed = {n for source in sources for n in _numbers(source)}
    if ctx.directive.proposed_slot is not None:
        allowed |= _slot_numbers(ctx.directive.proposed_slot, ctx.brand.timezone)
    unknown = sorted(n for n in _numbers(text) if n not in allowed)
    return _check("C-NUM", not unknown, f"{len(unknown)} number(s) not in the allowed sources")


def _ban(text: str, ctx: CheckContext) -> RuleCheck:
    hits = [
        phrase
        for phrase in ctx.banned_phrases
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE)
    ]
    return _check("C-BAN", not hits, "banned phrase: " + ", ".join(hits))


def _pii(text: str, ctx: CheckContext) -> RuleCheck:
    own = str(ctx.brand.sender_email).lower()
    emails = [m for m in EMAIL_RE.findall(text) if m.lower() != own]
    found = bool(emails or OBFUSCATED_EMAIL_RE.search(text) or PHONE_RE.search(text))
    return _check("C-PII", not found, "contains an email address or a phone number")


def _stage(text: str, ctx: CheckContext) -> RuleCheck:
    if ctx.opted_out:
        return _check("C-STAGE", "?" not in text, "asks a question after an opt-out")
    early = ctx.stage in _PRE_HANDOFF and ctx.directive.action not in _TIME_ACTIONS
    booking = early and bool(_BOOKING.search(text))
    return _check("C-STAGE", not booking, "booking language before the handoff stage")


# --- helpers ------------------------------------------------------------------------


def _numbers(text: str) -> set[str]:
    """Numbers in a canonical form: "1,000" -> "1000", "09" -> "9", "2.50" -> "2.5"."""
    found = set()
    for raw in _NUMBER.findall(text):
        value = raw.replace(",", "")
        try:
            found.add(f"{float(value):g}" if "." in value else str(int(value)))
        except ValueError:  # "1.2.3"
            found.add(value)
    return found


def _slot_numbers(slot: datetime, tz: str) -> set[str]:
    local = slot.astimezone(zone(tz))
    parts = (local.day, local.month, local.year, local.hour, local.hour % 12 or 12, local.minute)
    return {str(part) for part in parts}


def _normal(text: str) -> str:
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())
