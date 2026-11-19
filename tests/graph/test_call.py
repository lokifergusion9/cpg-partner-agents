"""The call sub-graph with scripted models: one run per partner turn (T-58, HLD section 6.11)."""

import asyncio
from datetime import datetime
from typing import Any

import pytest

from partner_agents import llm
from partner_agents.calls.playbook import load_playbook
from partner_agents.calls.scheduler import zone
from partner_agents.config import Settings
from partner_agents.demo import SAMPLE_BRAND
from partner_agents.fakes import ScriptedChatModel
from partner_agents.graph.call import (
    CallDeps,
    CallNotActive,
    build_call_graph,
    run_call_turn,
)
from partner_agents.outreach.voice import new_call_record
from partner_agents.prompts.caller import CallerReply
from partner_agents.schemas import CallRecord, Lead, OutreachDraft, TurnSignals
from partner_agents.storage.repo import InMemoryRepository
from partner_agents.utils.retry import RetryPolicy

PLAYBOOK = load_playbook("partner_intro")
LA = zone("America/Los_Angeles")
NOW = datetime(2026, 11, 16, 11, 0, tzinfo=LA)  # Monday 11:00
PROPOSAL = datetime(2026, 11, 17, 11, 0, tzinfo=LA)
LINE = CallerReply(text="Thanks for that. Does that sound useful so far?")
BOOKING = [
    TurnSignals(right_person=True, good_time=True),
    TurnSignals(
        interest="positive",
        in_rush=True,
        slot_values={"role": "owner", "categories": "snacks", "territory": "Oregon"},
    ),
    TurnSignals(),  # the three shared slots are confirmed, one per turn
    TurnSignals(),
    TurnSignals(),
    TurnSignals(agrees_to_meeting=True),  # after the pitch: a meeting?
    TurnSignals(agrees_to_meeting=True),  # the proposed time
    TurnSignals(),  # closing
]
BOOKING_TEXTS = ["hello", "all three", "yes", "yes", "yes", "ok", "yes", "bye"]


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "RETRY_POLICY", RetryPolicy(attempts=1, base_delay_s=0))


def make_lead(email: str | None) -> Lead:
    return Lead(
        id="acme",
        run_id="r_0",
        user_id="u_1",
        company_name="Acme Foods",
        partner_type="broker",
        location="Portland",
        website=None,
        email=email,
        email_status="verified" if email else "unchecked",
        phone="+13105550100",
        phone_status="verified",
        source_url="https://acme.example/",
        evidence_quote="q",
        created_at=NOW,
    )


def setup(
    signals: list[TurnSignals], email: str | None = "buyer@acme.example"
) -> tuple[CallDeps, Any, str]:
    repo = InMemoryRepository()
    deps = CallDeps(
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
        repo=repo,
        playbook=PLAYBOOK,
        classifier=ScriptedChatModel.scripted(*signals),
        caller=ScriptedChatModel.repeating(LINE),
        clock=lambda: NOW,
    )
    draft = OutreachDraft(
        id="d_1",
        lead_id="acme",
        user_id="u_1",
        run_id="r_0",
        channel="voice",
        subject=None,
        recipient="+13105550100",
        body="plan",
        status="call_created",
        prompt_version="v",
        created_at=NOW,
    )
    record = new_call_record(draft, PLAYBOOK, NOW)

    async def seed() -> None:
        await repo.save_profile("u_1", SAMPLE_BRAND)
        await repo.save_leads([make_lead(email)])
        await repo.save_call(record)

    asyncio.run(seed())
    return deps, build_call_graph(deps), record.id


def play(deps: CallDeps, graph: Any, call_id: str, partner: list[str | None]) -> CallRecord:
    async def go() -> CallRecord:
        record = None
        for text in partner:
            record, _ = await run_call_turn(
                graph, deps, user_id="u_1", call_id=call_id, partner_text=text
            )
        assert record is not None
        return record

    return asyncio.run(go())


def test_the_opening_is_the_agents_first_turn_and_needs_no_classifier() -> None:
    deps, graph, call_id = setup([])
    record = play(deps, graph, call_id, [None])
    assert [t.speaker for t in record.turns] == ["agent"]
    assert record.turns[0].directive is not None and record.turns[0].directive.action == "open"
    assert record.progress.stage == "opening" and record.progress.outcome is None


def test_a_booked_meeting_has_a_time_and_a_pending_confirmation_draft() -> None:
    deps, graph, call_id = setup(BOOKING)
    record = play(deps, graph, call_id, [None, *BOOKING_TEXTS])
    assert record.progress.outcome == "meeting_booked"
    assert record.meeting_at == PROPOSAL and record.finished_at == NOW
    assert record.progress.slots["territory"] == "Oregon"
    assert record.followup_draft_id is not None
    [draft] = asyncio.run(deps.repo.list_drafts("u_1", status="pending_approval"))
    assert draft.id == record.followup_draft_id
    assert (draft.channel, draft.recipient) == ("email", "buyer@acme.example")
    assert "Tuesday 17 November at 11:00 am" in draft.body
    assert asyncio.run(deps.repo.get_call("u_1", call_id)) == record


def test_no_verified_email_means_no_confirmation_draft() -> None:
    deps, graph, call_id = setup(BOOKING, email=None)
    record = play(deps, graph, call_id, [None, *BOOKING_TEXTS])
    assert record.progress.outcome == "meeting_booked" and record.followup_draft_id is None
    assert asyncio.run(deps.repo.list_drafts("u_1")) == []


def test_an_opt_out_mid_qualification_ends_the_call_at_once() -> None:
    deps, graph, call_id = setup(
        [
            TurnSignals(right_person=True, good_time=True),
            TurnSignals(interest="positive"),
            TurnSignals(opt_out=True),
        ]
    )
    record = play(deps, graph, call_id, [None, "hello", "sure", "stop calling me"])
    assert record.progress.outcome == "opted_out" and record.followup_draft_id is None
    assert record.turns[-1].directive is not None
    assert record.turns[-1].directive.action == "exit_politely"
    assert asyncio.run(deps.repo.list_drafts("u_1")) == []


def test_the_wrong_person_in_the_opening_ends_the_call() -> None:
    deps, graph, call_id = setup([TurnSignals(right_person=False)])
    record = play(deps, graph, call_id, [None, "no, wrong number"])
    assert record.progress.outcome == "wrong_person" and record.finished_at == NOW


def test_a_classifier_failure_is_survived_with_no_signals() -> None:
    deps, graph, call_id = setup([])  # the classifier has no replies: every call raises
    record = play(deps, graph, call_id, [None, "hello"])
    assert len(record.turns) == 3 and record.turns[1].speaker == "partner"
    assert record.progress.outcome is None  # the call goes on


def test_turns_that_do_not_fit_the_state_of_the_call_are_refused() -> None:
    deps, graph, call_id = setup([TurnSignals(right_person=False)])

    async def code(text: str | None) -> str:
        with pytest.raises(CallNotActive) as caught:
            await run_call_turn(graph, deps, user_id="u_1", call_id=call_id, partner_text=text)
        return caught.value.code

    assert asyncio.run(code("hello")) == "call_not_started"
    play(deps, graph, call_id, [None, "wrong number"])
    assert asyncio.run(code(None)) == "call_ended"
    assert asyncio.run(code("again")) == "call_ended"


def test_an_unknown_call_is_not_found_and_neither_is_another_users() -> None:
    deps, graph, call_id = setup([])
    with pytest.raises(LookupError):
        asyncio.run(run_call_turn(graph, deps, user_id="u_1", call_id="c_nope", partner_text=None))
    with pytest.raises(LookupError):
        asyncio.run(run_call_turn(graph, deps, user_id="u_2", call_id=call_id, partner_text=None))
