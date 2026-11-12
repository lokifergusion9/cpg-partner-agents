"""``chase_select`` (LLM fallback, when to draft) and ``chase_draft`` (policy, limits).

HLD §6.9.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import END

from partner_agents import llm
from partner_agents.config import Settings
from partner_agents.fakes import ScriptedChatModel
from partner_agents.graph.chase import (
    NO_CALLS_YET,
    NO_PROFILE,
    NOT_UNDERSTOOD,
    make_chase_draft,
    make_chase_select,
    next_after_select,
)
from partner_agents.outreach.policy import MAX_BODY_WORDS, OPT_OUT_LINE, TokenBucket
from partner_agents.schemas import (
    BrandProfile,
    EmailDraft,
    Lead,
    LeadSelection,
    OutreachDraft,
    RouteDecision,
)
from partner_agents.storage.repo import InMemoryRepository
from partner_agents.utils.retry import RetryPolicy
from tests.helpers import ByPageModel

NOW = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "RETRY_POLICY", RetryPolicy(attempts=1, base_delay_s=0))


def run[T](coro: Any) -> T:
    return asyncio.run(coro)  # type: ignore[no-any-return]


def lead(
    key: str, location: str = "Los Angeles, CA", email: str | None = "", status: str = "verified"
) -> Lead:
    return Lead(
        id=key,
        run_id="r_0",
        user_id="u_1",
        company_name=f"{key.title()} Foods",
        partner_type="broker",
        location=location,
        website=None,
        email=f"buyer@{key}.example" if email == "" else email,
        email_status=status,  # type: ignore[arg-type]
        phone=None,
        phone_status="unchecked",
        source_url=f"https://{key}.example/",  # type: ignore[arg-type]
        evidence_quote="q",
        created_at=NOW,
    )


LEADS = [
    lead("acme"),
    lead("beta", "San Diego, CA"),
    lead("gamma", email=None, status="unchecked"),
    lead("delta", "Los Angeles, CA"),
]

PROFILE = BrandProfile(
    brand_name="Crumb & Co",
    product="gluten-free cookies",
    value_prop="small-batch cookies that sell through in natural grocery",
    sender_name="Priya Shah",
    sender_email="priya@crumb.example",
    founder_name="Priya Shah",
    category="gluten-free cookies",
    sign_off="Thanks from all of us at Crumb & Co",
)

GOOD_BODY = "Hi Acme team, we make gluten-free cookies. Could we talk for 15 minutes?\n\nPriya Shah"


def email(subject: str = "Gluten-free cookies for your LA accounts", body: str = GOOD_BODY) -> str:
    return EmailDraft(subject=subject, body=body).model_dump_json()


def route(selector: str, goal: str | None = None, channel: str = "email") -> RouteDecision:
    return RouteDecision(
        intent="outreach",
        confidence=0.9,
        lead_selector=selector,
        outreach_goal=goal,
        outreach_channel=channel,  # type: ignore[arg-type]
        reason="r",
    )


def state(text: str, decision: RouteDecision, **extra: Any) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=text)],
        "user_id": "u_1",
        "run_id": "r_1",
        "route": decision,
        "leads": LEADS,
        **extra,
    }


# --- chase_select -------------------------------------------------------------------


def select(text: str, decision: RouteDecision, model: Any = None) -> tuple[dict[str, Any], str]:
    node = make_chase_select(model or ScriptedChatModel.scripted())
    s = state(text, decision)
    update = run(node(s))
    return update, next_after_select({**s, **update})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "decision", "selected"),
    [
        ("Draft emails to the top 3", route("top 3"), ["acme", "beta", "gamma"]),
        ("email the LA ones", route("the LA ones"), ["acme", "gamma", "delta"]),
        (
            "the LA ones",
            route("the LA ones", goal="introduce our brand"),
            ["acme", "gamma", "delta"],
        ),
    ],
)
def test_a_drafting_turn_selects_deterministically_and_goes_to_chase_draft(
    text: str, decision: RouteDecision, selected: list[str]
) -> None:
    update, nxt = select(text, decision)
    assert update["selected_leads"] == selected
    assert "messages" not in update  # chase_draft replies
    assert nxt == "chase_draft"


def test_a_narrowing_turn_lists_the_leads_and_ends() -> None:
    update, nxt = select("only the ones in LA", route("the ones in LA"))
    assert update["selected_leads"] == ["acme", "gamma", "delta"]
    assert update["messages"][0].text.startswith('3 of 4 leads match "the ones in LA"')
    assert nxt == END


def test_a_call_request_does_not_draft_emails_yet() -> None:
    update, nxt = select("call the first one", route("the first one", channel="voice"))
    assert update["selected_leads"] == ["acme"]
    assert NO_CALLS_YET in update["messages"][0].text
    assert nxt == END


def test_an_unreadable_selector_falls_back_to_the_writer() -> None:
    model = ScriptedChatModel.scripted(LeadSelection(numbers=[4, 2, 9, 2]))
    update, nxt = select("email those ones", route("those ones"), model)
    assert update["selected_leads"] == ["beta", "delta"]  # list order, 9 dropped, no repeats
    prompt = model.received[0][1].text
    assert "<<<UNTRUSTED" in prompt and "4. Delta Foods (broker, Los Angeles, CA)" in prompt
    assert "buyer@" not in prompt  # no contact values in the prompt
    assert nxt == "chase_draft"


def test_a_failed_fallback_says_it_could_not_tell_and_records_the_error() -> None:
    update, nxt = select("email those ones", route("those ones"))  # the model has no reply
    assert update["selected_leads"] == []
    assert update["messages"][0].text == NOT_UNDERSTOOD
    assert [e.stage for e in update["errors"]] == ["chase_select"]
    assert nxt == END


# --- chase_draft --------------------------------------------------------------------


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def draft_node(
    pages: dict[str, list[str | Exception]],
    *,
    profile: BrandProfile | None = PROFILE,
    buckets: dict[str, TokenBucket] | None = None,
    earlier: list[OutreachDraft] = (),  # type: ignore[assignment]
    **config: Any,
) -> tuple[Any, ByPageModel, InMemoryRepository]:
    repo = InMemoryRepository(clock=lambda: NOW)
    if profile is not None:
        run(repo.save_profile("u_1", profile))
    for previous in earlier:
        run(repo.save_draft(previous))
    model = ByPageModel(
        messages=iter([]), pages={f"https://{k}.example/": v for k, v in pages.items()}
    )
    node = make_chase_draft(
        model, repo, settings(**config), {} if buckets is None else buckets, lambda: NOW
    )
    return node, model, repo


def drafted(
    node: Any, selected: list[str], goal: str | None = "introduce our cookies"
) -> dict[str, Any]:
    return run(node(state("draft emails", route("x", goal=goal), selected_leads=selected)))  # type: ignore[no-any-return]


def statuses(update: dict[str, Any]) -> dict[str, str]:
    return {d.lead_id: d.status for d in update["drafts"]}


def sent_before(recipient: str, days_ago: int = 3) -> OutreachDraft:
    return OutreachDraft(
        id="d_old", lead_id="old", user_id="u_1", run_id="r_0", channel="email",
        recipient=recipient, subject="s", body="b", status="outboxed", prompt_version="v",
        created_at=NOW - timedelta(days=days_ago),
    )  # fmt: skip


def test_eligible_leads_get_drafts_and_the_others_keep_their_skip_status() -> None:
    twin = lead("delta", email="buyer@acme.example")  # same inbox as acme
    node, model, repo = draft_node({"acme": [email()]}, earlier=[sent_before("buyer@beta.example")])
    update = run(
        node(
            state(
                "draft emails",
                route("x", goal="introduce our cookies"),
                leads=[*LEADS[:3], twin],
                selected_leads=["acme", "beta", "gamma", "delta"],
            )
        )
    )
    assert statuses(update) == {
        "acme": "pending_approval",
        "beta": "skipped_duplicate",  # contacted 3 days ago
        "gamma": "skipped_ineligible",  # no verified email
        "delta": "skipped_duplicate",  # the same address earlier in this batch
    }
    assert len(model.prompts) == 1  # only the eligible lead reached the model
    saved = run(repo.list_drafts("u_1"))
    assert {d.lead_id: d.status for d in saved if d.run_id == "r_1"} == statuses(update)
    reply = update["messages"][0].text
    assert reply.startswith("I drafted 1 email for approval:\n1. Acme Foods: Gluten-free cookies")
    assert "Gamma Foods (no verified email)" in reply and "Nothing is sent" in reply


def test_a_draft_respects_the_limits_and_gets_the_opt_out_line() -> None:
    node, model, _ = draft_node({"acme": [email()]})
    [draft] = drafted(node, ["acme"])["drafts"]
    assert draft.status == "pending_approval" and draft.recipient == "buyer@acme.example"
    assert draft.subject is not None and len(draft.subject) <= 80
    assert len(draft.body.split()) <= MAX_BODY_WORDS
    assert draft.body.endswith(OPT_OUT_LINE) and "Priya Shah" in draft.body
    assert draft.prompt_version == "chase-v1"
    prompt = model.prompts[0]
    assert "introduce our cookies" in prompt and "Crumb & Co" in prompt
    assert 'source="https://acme.example/"' in prompt and "buyer@acme.example" not in prompt


def test_the_default_goal_is_used_when_the_turn_has_none() -> None:
    node, model, _ = draft_node({"acme": [email()]})
    drafted(node, ["acme"], goal=None)
    assert "introduce Crumb & Co (gluten-free cookies)" in model.prompts[0]


def test_a_long_body_is_revised_once() -> None:
    long = email(body="Priya Shah " + "word " * 200)
    node, model, _ = draft_node({"acme": [long, email()]})
    update = drafted(node, ["acme"])
    assert statuses(update) == {"acme": "pending_approval"}
    assert "keep it under 150" in model.prompts[1]
    assert update["errors"] == []


def test_a_body_that_stays_invalid_fails_with_a_policy_error() -> None:
    no_sender = email(body="Hello, we make cookies. Talk soon?")
    node, _, _ = draft_node({"acme": [no_sender, no_sender]})
    update = drafted(node, ["acme"])
    assert statuses(update) == {"acme": "failed"}
    assert [(e.stage, e.code) for e in update["errors"]] == [("chase_draft", "policy_violation")]
    assert "Could not write a valid draft for: Acme Foods" in update["messages"][0].text


def test_a_subject_over_80_characters_is_rejected_by_the_schema() -> None:
    long_subject = '{"subject": "' + "x" * 81 + '", "body": "Priya Shah"}'
    node, _, _ = draft_node({"acme": [long_subject, long_subject]})
    update = drafted(node, ["acme"])
    assert statuses(update) == {"acme": "failed"}
    assert update["errors"][0].code == "invalid_output"


def test_the_daily_cap_skips_and_spends_only_on_drafts() -> None:
    buckets = {"u_1": TokenBucket(capacity=20, tokens=1.0, updated_at=NOW)}
    node, _, _ = draft_node({"acme": [email()]}, buckets=buckets)
    update = drafted(node, ["acme", "gamma", "beta"])
    assert statuses(update) == {
        "acme": "pending_approval",
        "gamma": "skipped_ineligible",  # costs nothing
        "beta": "skipped_cap",
    }
    assert buckets["u_1"].tokens < 1


def test_without_a_brand_profile_nothing_is_drafted() -> None:
    node, model, _ = draft_node({}, profile=None)
    update = drafted(node, ["acme"])
    assert update["drafts"] == [] and update["messages"][0].text == NO_PROFILE
    assert model.prompts == []
