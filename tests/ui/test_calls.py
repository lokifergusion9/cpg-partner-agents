"""The Call tab's handlers (no browser) and its layout (T-60, PRD section 10 steps 8 and 9)."""

import asyncio
from typing import Any

from qdrant_client import AsyncQdrantClient

from partner_agents.demo import SAMPLE_BRAND
from partner_agents.fakes import ScriptedChatModel
from partner_agents.graph.build import Context
from partner_agents.graph.call import CallDeps
from partner_agents.outreach.voice import new_call_record
from partner_agents.prompts.simulator import SimulatorReply
from partner_agents.schemas import CallTurn, OutreachDraft, RuleCheck, TurnSignals
from partner_agents.ui.app import build_ui
from partner_agents.ui.calls import (
    call_options,
    call_summary,
    check_badge,
    list_call_options,
    persona_options,
    play_turn,
    simulate_selected,
    transcript_messages,
    view_call,
)
from tests.graph.test_build import (  # noqa: F401  (fixtures are used by name)
    client,
    instant_retries,
    make_deps,
)
from tests.graph.test_call import BOOKING, BOOKING_TEXTS, LINE, NOW, PLAYBOOK, make_lead


def make_ctx(
    qdrant: AsyncQdrantClient,
    signals: list[TurnSignals],
    texts: list[str] | None = None,
) -> tuple[Context, str]:
    deps = make_deps(qdrant)
    simulator = (
        None
        if texts is None
        else ScriptedChatModel.scripted(*(SimulatorReply(text=t) for t in texts))
    )
    deps.calls = CallDeps(
        settings=deps.settings,
        repo=deps.repo,
        playbook=PLAYBOOK,
        classifier=ScriptedChatModel.scripted(*signals),
        caller=ScriptedChatModel.repeating(LINE),
        simulator=simulator,
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
        await deps.repo.save_profile("u_1", SAMPLE_BRAND)
        await deps.repo.save_leads([make_lead("buyer@acme.example")])
        await deps.repo.save_call(record)

    asyncio.run(seed())
    return Context(deps=deps, graph=None), record.id  # type: ignore[arg-type]


def turn(**fields: Any) -> CallTurn:
    base: dict[str, Any] = {"index": 0, "speaker": "agent", "text": "Hi", "stage": "pitch"}
    return CallTurn(**{**base, **fields})


# --- pure helpers -------------------------------------------------------------------


def test_the_check_badge_tells_green_regenerated_and_fallback_apart() -> None:
    assert check_badge(turn()) == "✅ checks passed"
    assert check_badge(turn(regenerated=True)) == "🔁 regenerated once, then passed"
    assert check_badge(turn(regenerated=True, used_fallback=True)) == "⚠️ fallback line used"
    failed = [RuleCheck(rule_id="C-NUM", passed=False, by="code")]
    assert check_badge(turn(checks=failed)).endswith("❌ C-NUM")


def test_the_transcript_puts_the_partner_on_the_user_side_with_stage_chips() -> None:
    from tests.graph.test_call import setup

    deps, graph, call_id = setup([TurnSignals(right_person=True)])
    from tests.graph.test_call import play

    record = play(deps, graph, call_id, [None, "speaking"])
    messages = transcript_messages(record)
    assert [m["role"] for m in messages] == ["assistant", "user", "assistant"]
    assert messages[0]["content"].startswith("`opening` ")
    # the scripted line has no AI disclosure, so the opening used the playbook fallback line
    assert messages[0]["content"].endswith("*⚠️ fallback line used*")
    assert messages[2]["content"].endswith("*✅ checks passed*")
    assert messages[1]["content"] == "`opening` speaking"


def test_the_picker_names_the_company_and_the_state_of_each_call() -> None:
    from tests.graph.test_call import play, setup

    deps, graph, call_id = setup([])
    calls = asyncio.run(deps.repo.list_calls("u_1"))
    leads = asyncio.run(deps.repo.list_leads("u_1"))
    assert call_options(calls, leads) == [(f"Acme Foods · not started · {call_id}", call_id)]
    play(deps, graph, call_id, [None])
    calls = asyncio.run(deps.repo.list_calls("u_1"))
    assert "in progress" in call_options(calls, leads)[0][0]


def test_the_persona_picker_lists_the_demo_personas_by_title() -> None:
    options = dict(persona_options())
    assert options["Interested broker who books easily"] == "interested_broker"
    assert len(options) == 6


def test_a_summary_with_no_call_is_empty_and_a_fresh_call_says_so() -> None:
    from tests.graph.test_call import setup

    assert call_summary(None) == ""
    deps, _, call_id = setup([])
    record = asyncio.run(deps.repo.get_call("u_1", call_id))
    assert call_summary(record).startswith("### Not started")


# --- the handlers -------------------------------------------------------------------


def test_simulate_shows_the_transcript_the_outcome_the_slots_and_the_followup(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    ctx, call_id = make_ctx(client, BOOKING, BOOKING_TEXTS)
    assert [value for _, value in asyncio.run(list_call_options(ctx, "u_1"))] == [call_id]
    view = asyncio.run(simulate_selected(ctx, "u_1", call_id, "interested_broker"))
    assert view.notice == ""
    assert view.history[0]["role"] == "assistant" and len(view.history) == 17
    assert "### ✅ Outcome: `meeting_booked`" in view.summary
    assert "| territory | Oregon |" in view.summary
    assert "Tuesday 17 November at 11:00 am (America/Los_Angeles time)" in view.summary
    assert "to buyer@acme.example" in view.summary and "It is not sent." in view.summary
    assert "persona `interested_broker`" in view.summary


def test_a_live_call_opens_then_takes_typed_lines_and_ends_on_an_opt_out(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    signals = [TurnSignals(right_person=True, good_time=True), TurnSignals(opt_out=True)]
    ctx, call_id = make_ctx(client, signals)

    async def go() -> tuple[Any, ...]:
        opened = await play_turn(ctx, "u_1", call_id, None)
        talking = await play_turn(ctx, "u_1", call_id, "yes, speaking")
        ended = await play_turn(ctx, "u_1", call_id, "stop calling me")
        after = await play_turn(ctx, "u_1", call_id, "hello?")
        return opened, talking, ended, after

    opened, talking, ended, after = asyncio.run(go())
    assert len(opened.history) == 1 and "In progress" in opened.summary
    assert len(talking.history) == 3 and "stage `discovery`" in talking.summary
    assert "### 🛑 Outcome: `opted_out`" in ended.summary
    assert after.notice == "This call has ended. Pick another one."


def test_refusals_are_notices_not_errors(client: AsyncQdrantClient) -> None:  # noqa: F811
    ctx, call_id = make_ctx(client, [], texts=None)  # no simulator configured

    async def go() -> list[str]:
        return [
            (await simulate_selected(ctx, "u_1", None, "gatekeeper")).notice,
            (await simulate_selected(ctx, "u_1", call_id, None)).notice,
            (await simulate_selected(ctx, "u_1", call_id, "gatekeeper")).notice,
            (await play_turn(ctx, "u_1", call_id, "hello")).notice,  # before the opening
            (await play_turn(ctx, "u_1", "c_nope", "hello")).notice,
            (await view_call(ctx, "u_1", "c_nope")).notice,
        ]

    assert asyncio.run(go()) == [
        "Pick a call first. A call appears here once you approve its plan in the Approvals tab.",
        "Pick a persona first.",
        "The call agent is not configured.",
        "Start the call first.",
        "No such call.",
        "No such call.",
    ]


def test_a_simulator_that_fails_keeps_the_turns_played(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    ctx, call_id = make_ctx(client, BOOKING, texts=["Speaking."])  # then it runs dry
    view = asyncio.run(simulate_selected(ctx, "u_1", call_id, "interested_broker"))
    assert view.notice.startswith("The simulator failed") and len(view.history) == 3


def test_a_blank_live_line_changes_nothing(client: AsyncQdrantClient) -> None:  # noqa: F811
    ctx, call_id = make_ctx(client, [])
    asyncio.run(play_turn(ctx, "u_1", call_id, None))
    view = asyncio.run(play_turn(ctx, "u_1", call_id, "   "))
    assert len(view.history) == 1 and view.notice == ""


# --- layout -------------------------------------------------------------------------


def test_the_ui_has_a_call_tab(client: AsyncQdrantClient) -> None:  # noqa: F811
    ctx, _ = make_ctx(client, [])
    ui = build_ui(lambda: ctx)
    labels = [getattr(block, "label", None) for block in ui.blocks.values()]
    assert "Call" in labels and "Transcript" in labels and "Persona (for Simulate)" in labels
