"""Voice outreach: the call plan, the mock caller and the allowlist (HLD §6.9)."""

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from partner_agents.config import Settings
from partner_agents.graph.call_plan import make_call_plan
from partner_agents.graph.chase import NO_PROFILE
from partner_agents.outreach.voice import (
    PLAN_FACTS,
    MockVoiceCaller,
    build_call_plan,
    dispatch_voice,
    is_voice_allowlisted,
)
from partner_agents.schemas import ApprovedFact, BrandProfile, Lead, OutreachDraft
from partner_agents.storage.repo import InMemoryRepository

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
PHONE = "+13105550100"


def draft(status: str = "approved", **overrides: Any) -> OutreachDraft:
    fields: dict[str, Any] = {
        "id": "d1", "lead_id": "l1", "user_id": "u", "run_id": "r", "channel": "voice",
        "recipient": PHONE, "subject": None, "body": "plan", "status": status,
        "prompt_version": "call-plan-v1", "created_at": NOW,
    }  # fmt: skip
    return OutreachDraft(**{**fields, **overrides})


def lead(phone: str | None = PHONE, status: str = "verified") -> Lead:
    return Lead(
        id="acme", run_id="r_0", user_id="u", company_name="Acme Foods", partner_type="broker",
        location="Los Angeles, CA", website=None, email=None, email_status="unchecked",
        phone=phone, phone_status=status,  # type: ignore[arg-type]
        source_url="https://acme.example/", evidence_quote="q", created_at=NOW,
    )  # fmt: skip


def profile(facts: int = 5) -> BrandProfile:
    return BrandProfile(
        brand_name="Crumb & Co", product="cookies", value_prop="v", sender_name="Priya Shah",
        sender_email="priya@crumb.example", founder_name="Priya Shah", category="cookies",
        sign_off="Thanks",
        approved_facts=[
            ApprovedFact(id=f"F-{n}", text=f"Fact {n}.", source="brand_profile")
            for n in range(1, facts + 1)
        ],
    )  # fmt: skip


class Caller:
    def __init__(self, fail: Exception | None = None) -> None:
        self.started: list[str] = []
        self.fail = fail

    async def start(self, draft: OutreachDraft) -> str:
        if self.fail:
            raise self.fail
        self.started.append(draft.id)
        return "c_test"


def dispatch(d: OutreachDraft, provider: str, caller: Any, allowlist: list[str] = ()):  # type: ignore[no-untyped-def,assignment]
    return asyncio.run(
        dispatch_voice(d, provider=provider, allowlist=allowlist, caller=caller, now=NOW)  # type: ignore[arg-type]
    )


# --- dispatch_voice -----------------------------------------------------------------


def test_the_mock_caller_creates_a_call_without_a_number_or_an_allowlist() -> None:
    caller = MockVoiceCaller()
    result, call_id, error = dispatch(draft(recipient=""), "mock", caller)
    assert (result.status, error) == ("call_created", None)
    assert call_id is not None and call_id.startswith("c_")


@pytest.mark.parametrize(
    ("recipient", "allowlist", "caller", "code"),
    [
        (PHONE, [], Caller(), "not_allowlisted"),  # an empty allowlist allows nobody
        (PHONE, ["+19998887777"], Caller(), "not_allowlisted"),
        ("", ["+19998887777"], Caller(), "not_allowlisted"),  # no number
        ("", [""], Caller(), "not_allowlisted"),  # a blank entry does not match a blank number
        (PHONE, [PHONE], None, "voice_not_configured"),  # allowed, but nothing can dial yet
    ],
)
def test_vapi_refuses_unless_the_number_is_allowlisted_and_a_caller_exists(
    recipient: str, allowlist: list[str], caller: Caller | None, code: str
) -> None:
    result, call_id, error = dispatch(draft(recipient=recipient), "vapi", caller, allowlist)
    assert result.status == "failed" and call_id is None
    assert error is not None and (error.stage, error.code) == ("chase_dispatch", code)
    if isinstance(caller, Caller):
        assert caller.started == []  # nothing was dialled


def test_vapi_calls_an_allowlisted_number() -> None:
    caller = Caller()
    result, call_id, error = dispatch(draft(), "vapi", caller, ["+1 310 555 0100"])
    assert (result.status, call_id, error) == ("call_created", "c_test", None)
    assert caller.started == ["d1"]


@pytest.mark.parametrize("status", ["pending_approval", "rejected", "call_created", "failed"])
def test_only_an_approved_voice_draft_is_started(status: str) -> None:
    caller = Caller()
    result, _, error = dispatch(draft(status), "mock", caller)
    assert result.status == "failed" and error is not None and error.code == "not_approved"
    assert caller.started == []


def test_an_email_draft_is_not_started_as_a_call() -> None:
    _, _, error = dispatch(draft(channel="email"), "mock", Caller())
    assert error is not None and error.code == "not_approved"


def test_a_caller_failure_is_recorded_without_its_message() -> None:
    result, _, error = dispatch(draft(), "mock", Caller(fail=RuntimeError("dial +13105550100")))
    assert result.status == "failed" and error is not None
    assert (error.code, error.message) == ("voice_error", "RuntimeError")


def test_the_allowlist_ignores_spacing_but_not_digits() -> None:
    assert is_voice_allowlisted("+1 310-555 0100".replace("-", ""), [PHONE])
    assert not is_voice_allowlisted("+13105550101", [PHONE])
    assert not is_voice_allowlisted("", [""])


# --- the plan -----------------------------------------------------------------------


def test_the_plan_names_the_playbook_the_goal_and_at_most_three_facts() -> None:
    plan = build_call_plan(lead(), profile(5), "mock")
    assert "Acme Foods (broker, Los Angeles, CA)" in plan
    assert "Playbook: partner_intro" in plan and "nobody is dialled" in plan
    assert "book a meeting with Priya Shah of Crumb & Co" in plan
    assert plan.count("- [F-") == PLAN_FACTS and "[F-4]" not in plan
    assert "Mode: live call" in build_call_plan(lead(), profile(), "vapi")


def test_a_profile_without_facts_has_no_facts_section() -> None:
    assert "Approved facts" not in build_call_plan(lead(), profile(0), "mock")


# --- the call_plan node -------------------------------------------------------------


def run_plan(provider: str, leads: list[Lead], with_profile: bool = True) -> dict[str, Any]:
    repo = InMemoryRepository(clock=lambda: NOW)
    if with_profile:
        asyncio.run(repo.save_profile("u", profile()))
    settings = Settings(_env_file=None, voice_provider=provider)  # type: ignore[call-arg]
    node = make_call_plan(repo, settings, lambda: NOW)
    state: Any = {
        "messages": [HumanMessage(content="call them")], "user_id": "u", "run_id": "r_1",
        "leads": leads, "selected_leads": [x.id for x in leads],
    }  # fmt: skip
    return asyncio.run(node(state))


def test_a_live_provider_skips_leads_without_a_verified_phone() -> None:
    other = lead(phone=None, status="unchecked").model_copy(update={"id": "beta"})
    update = run_plan("vapi", [lead(), other])
    assert {d.lead_id: d.status for d in update["drafts"]} == {
        "acme": "pending_approval",
        "beta": "skipped_ineligible",
    }
    assert "Skipped (no verified phone): " in update["messages"][0].text


def test_the_mock_provider_plans_a_call_even_without_a_phone() -> None:
    [d] = run_plan("mock", [lead(phone=None, status="unchecked")])["drafts"]
    assert (d.status, d.recipient, d.channel) == ("pending_approval", "", "voice")


def test_without_a_brand_profile_no_call_is_planned() -> None:
    update = run_plan("mock", [lead()], with_profile=False)
    assert update["drafts"] == [] and update["messages"][0].text == NO_PROFILE
