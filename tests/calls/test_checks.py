"""The deterministic reply checks, table-driven (HLD §6.11, T-57)."""

from datetime import datetime
from typing import Any

import pytest

from partner_agents.calls.checks import CheckContext, failures, run_checks
from partner_agents.calls.playbook import load_playbook
from partner_agents.calls.scheduler import zone
from partner_agents.demo import SAMPLE_BRAND
from partner_agents.schemas import Directive

PLAYBOOK = load_playbook("partner_intro")
FACTS = {fact.id: fact for fact in SAMPLE_BRAND.approved_facts}
SLOT = datetime(2026, 11, 17, 11, 0, tzinfo=zone("America/Los_Angeles"))  # Tue 17 Nov 11:00


def ctx(action: str = "pitch", stage: str = "pitch", **fields: Any) -> CheckContext:
    directive = fields.pop("directive", None) or Directive(action=action)  # type: ignore[arg-type]
    return CheckContext(
        stage=stage,  # type: ignore[arg-type]
        directive=directive,
        brand=SAMPLE_BRAND,
        banned_phrases=PLAYBOOK.banned_phrases,
        facts=fields.pop("facts", [FACTS["F-3"], FACTS["F-4"]]),
        **fields,
    )


def failed_ids(text: str, context: CheckContext) -> list[str]:
    return [check.rule_id for check in failures(run_checks(text, context))]


CASES: list[tuple[str, str, CheckContext, list[str]]] = [
    # (id, reply, context, failed check ids)
    (
        "clean-pitch",
        "Our boxes come in 6-count and 12-count, and keep for 9 months. Does that fit your stores?",
        ctx(),
        [],
    ),
    ("invented-percent", "Retailers see 40% better sell-through with us.", ctx(), ["C-NUM"]),
    (
        "number-from-the-partner",
        "Got it, 15 stores across Oregon.",
        ctx("ask_slot", "qualification", partner_last="We have 15 stores."),
        [],
    ),
    (
        "number-from-a-slot",
        "So about 200 cases a month, right?",
        ctx("confirm_slot", "qualification", slots={"volume": "200 cases a month"}),
        [],
    ),
    (
        "numbers-of-the-proposed-slot",
        "Would Tuesday 17 November at 11:00 am work for you?",
        ctx(directive=Directive(action="propose_slot", proposed_slot=SLOT), stage="handoff"),
        [],
    ),
    ("guaranteed", "Our cookies are guaranteed to sell.", ctx(), ["C-BAN"]),
    ("banned-any-case", "That is a Special Offer for you.", ctx(), ["C-BAN"]),
    ("percent-phrase", "They are 100% natural.", ctx(), ["C-NUM", "C-BAN"]),
    ("banned-inside-a-word-is-fine", "That is a discountable point; no.", ctx(), []),
    (
        "two-questions",
        "Who decides on new lines? And what categories do you carry?",
        ctx("ask_slot", "qualification"),
        ["C-ONEQ"],
    ),
    ("too-long", " ".join(["word"] * 61), ctx(), ["C-LEN"]),
    ("email-address", "You can reach the team at sales@acme.example anytime.", ctx(), ["C-PII"]),
    ("obfuscated-email", "Write to sales [at] acme [dot] com.", ctx(), ["C-PII"]),
    ("phone-number", "Call us on (415) 555-0100.", ctx(), ["C-NUM", "C-PII"]),
    ("the-brands-own-email", "Maya is at maya@hearthside-crumbs.example.", ctx(), []),
    (
        "booking-before-handoff",
        "Shall I book you in with Maya?",
        ctx("ask_slot", "qualification"),
        ["C-STAGE"],
    ),
    (
        "booking-in-handoff",
        "Shall I book Tuesday with Maya?",
        ctx(directive=Directive(action="propose_slot"), stage="handoff"),
        [],
    ),
    (
        "question-after-opt-out",
        "Sorry about that. Can I ask who handles snacks?",
        ctx("exit_politely", "ended", outcome="opted_out"),
        ["C-STAGE"],
    ),
    (
        "clean-opt-out",
        "I am sorry. Nobody will call you again. Goodbye.",
        ctx("exit_politely", "ended", outcome="opted_out"),
        [],
    ),
    (
        "opening-without-ai",
        "Hi, this is Sam from Hearthside Crumbs. Is now a good time?",
        ctx("open", "opening"),
        ["C-AI"],
    ),
    (
        "opening-with-ai",
        "Hi, I am an AI assistant calling for Hearthside Crumbs. Is now good?",
        ctx("open", "opening"),
        [],
    ),
    (
        "disclose-ai",
        "Yes, I am an A.I. assistant working for the brand.",
        ctx("disclose_ai", "qualification"),
        [],
    ),
    ("missing-sign-off", "Thanks for your time, talk soon.", ctx("close", "ended"), ["C-SIGNOFF"]),
    (
        "sign-off",
        "We have you down for Tuesday. Thanks from all of us at Hearthside Crumbs!",
        ctx("close", "ended"),
        [],
    ),
]


@pytest.mark.parametrize(
    ("text", "context", "expected"), [pytest.param(*case[1:], id=case[0]) for case in CASES]
)
def test_checks(text: str, context: CheckContext, expected: list[str]) -> None:
    assert failed_ids(text, context) == expected


def test_only_the_checks_that_apply_are_reported() -> None:
    always = ["C-NUM", "C-BAN", "C-ONEQ", "C-LEN", "C-PII", "C-STAGE"]
    assert [c.rule_id for c in run_checks("Fine.", ctx())] == always
    assert [c.rule_id for c in run_checks("Fine.", ctx("open", "opening"))] == [*always, "C-AI"]
    assert [c.rule_id for c in run_checks("Fine.", ctx("close", "ended"))][-1] == "C-SIGNOFF"
    opted = ctx("exit_politely", "ended", outcome="opted_out")
    assert [c.rule_id for c in run_checks("Bye.", opted)] == always


def test_a_failure_detail_never_quotes_the_reply() -> None:
    checks = run_checks("Mail me at someone@private.example or 415-555-0100.", ctx())
    details = " ".join(c.detail or "" for c in checks)
    assert "someone" not in details and "555" not in details
    assert all(c.by == "code" for c in checks)


def test_every_code_check_a_playbook_rule_names_is_implemented() -> None:
    rules = list(PLAYBOOK.universal_rules) + [r for s in PLAYBOOK.stages for r in s.rules]
    named = {rule.check_id for rule in rules if rule.check == "code"} - {"C-FLOW"}
    implemented = {c.rule_id for c in run_checks("x", ctx("open", "opening"))}
    implemented |= {c.rule_id for c in run_checks("x", ctx("close", "ended"))}
    assert named == implemented
