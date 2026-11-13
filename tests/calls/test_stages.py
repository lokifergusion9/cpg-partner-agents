"""The call stage machine, table-driven (HLD §6.11, T-55). Pure: no model, fixed clock."""

from datetime import datetime
from typing import Any

import pytest

from partner_agents.calls.playbook import load_playbook
from partner_agents.calls.scheduler import zone
from partner_agents.calls.stages import (
    CallEnded,
    advance,
    first_directive,
    matching_facts,
    select_facts,
    territory_fits,
)
from partner_agents.demo import SAMPLE_BRAND
from partner_agents.schemas import CallProgress, Directive, TurnSignals

PLAYBOOK = load_playbook("partner_intro")
BRAND = SAMPLE_BRAND  # regions California / Oregon / Washington, Los Angeles time
LA = zone("America/Los_Angeles")
NOW = datetime(2026, 11, 16, 11, 0, tzinfo=LA)  # Monday 11:00
PROPOSAL = datetime(2026, 11, 17, 11, 0, tzinfo=LA)  # first slot >= 24 h ahead
STAGES = ("opening", "discovery", "qualification", "pitch", "handoff", "closing")
ALL_SLOTS = ["role", "categories", "territory", "accounts", "volume"]


def at(stage: str = "opening", **fields: Any) -> CallProgress:
    return CallProgress(stage=stage, **fields)  # type: ignore[arg-type]


def turn(progress: CallProgress, **signals: Any) -> tuple[CallProgress, Directive]:
    return advance(progress, TurnSignals(**signals), PLAYBOOK, BRAND, now=NOW)


def handoff(**fields: Any) -> CallProgress:
    return at("handoff", proposed_slot=PROPOSAL, **fields)


# --- the stage table, row by row ----------------------------------------------------

ROWS: list[tuple[str, CallProgress, dict[str, Any], str, str, str | None]] = [
    # (id, progress, signals, expected stage, expected action, expected outcome)
    ("opening-ok", at(), {"right_person": True, "good_time": True}, "discovery", "discover", None),
    (
        "opening-wrong-person",
        at(),
        {"right_person": False},
        "ended",
        "exit_politely",
        "wrong_person",
    ),
    (
        "opening-bad-time",
        at(),
        {"right_person": True, "good_time": False},
        "handoff",
        "offer_callback",
        None,
    ),
    ("opening-unclear", at(), {}, "opening", "confirm_person", None),
    ("opening-unclear-twice", at(retries_used={"opening": 1}), {}, "discovery", "discover", None),
    (
        "discovery-positive",
        at("discovery"),
        {"interest": "positive"},
        "qualification",
        "ask_slot",
        None,
    ),
    (
        "discovery-unsure",
        at("discovery"),
        {"interest": "unsure"},
        "qualification",
        "ask_slot",
        None,
    ),
    ("discovery-silent", at("discovery"), {}, "discovery", "discover", None),
    (
        "discovery-negative",
        at("discovery"),
        {"interest": "negative"},
        "discovery",
        "retry_once",
        None,
    ),
    (
        "discovery-negative-again",
        at("discovery", retries_used={"discovery": 1}),
        {"interest": "negative"},
        "ended",
        "exit_politely",
        "not_interested",
    ),
    (
        "qualification-next-slot",
        at("qualification", asked_slots=["role"]),
        {"slot_values": {"role": "owner"}},
        "qualification",
        "ask_slot",
        None,
    ),
    (
        "qualification-done",
        at("qualification", asked_slots=ALL_SLOTS),
        {"slot_values": {"volume": "two pallets"}},
        "pitch",
        "pitch",
        None,
    ),
    (
        "qualification-territory-stop",
        at("qualification", asked_slots=["role", "categories", "territory"]),
        {"slot_values": {"territory": "Texas and Oklahoma"}},
        "ended",
        "exit_politely",
        "disqualified",
    ),
    (
        "qualification-no-new-lines",
        at("qualification", asked_slots=["role"]),
        {"interest": "negative"},
        "ended",
        "exit_politely",
        "disqualified",
    ),
    ("pitch-agrees", at("pitch"), {"agrees_to_meeting": True}, "handoff", "propose_slot", None),
    ("pitch-positive", at("pitch"), {"interest": "positive"}, "handoff", "propose_slot", None),
    ("pitch-negative", at("pitch"), {"interest": "negative"}, "pitch", "retry_once", None),
    (
        "pitch-negative-again",
        at("pitch", retries_used={"pitch": 1}),
        {"agrees_to_meeting": False},
        "ended",
        "exit_politely",
        "not_interested",
    ),
    ("pitch-callback", at("pitch"), {"wants_callback": True}, "handoff", "offer_callback", None),
    ("handoff-agrees", handoff(), {"agrees_to_meeting": True}, "closing", "confirm_booking", None),
    (
        "handoff-own-time",
        handoff(),
        {"proposed_time": "Thursday at 2pm"},
        "closing",
        "confirm_booking",
        None,
    ),
    ("handoff-later", handoff(), {"proposed_time": "later"}, "handoff", "retry_once", None),
    (
        "handoff-later-again",
        handoff(retries_used={"handoff": 1}),
        {"proposed_time": "later"},
        "ended",
        "exit_politely",
        "incomplete",
    ),
    (
        "handoff-declines-again",
        handoff(retries_used={"handoff": 1}),
        {"agrees_to_meeting": False},
        "ended",
        "exit_politely",
        "not_interested",
    ),
    (
        "closing-done",
        at("closing", closing_outcome="meeting_booked"),
        {},
        "ended",
        "close",
        "meeting_booked",
    ),
    (
        "closing-question",
        at("closing", closing_outcome="meeting_booked"),
        {"question": "Will she bring samples?"},
        "closing",
        "answer_question",
        None,
    ),
]


@pytest.mark.parametrize(
    ("progress", "signals", "stage", "action", "outcome"),
    [pytest.param(*row[1:], id=row[0]) for row in ROWS],
)
def test_stage_table(
    progress: CallProgress,
    signals: dict[str, Any],
    stage: str,
    action: str,
    outcome: str | None,
) -> None:
    after, directive = turn(progress, **signals)
    assert (after.stage, directive.action, after.outcome) == (stage, action, outcome)
    assert after.turns == progress.turns + 1


def test_the_first_line_is_the_opening() -> None:
    assert first_directive() == Directive(action="open")


# --- overrides ----------------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
def test_opt_out_overrides_everything_in_every_stage(stage: str) -> None:
    progress = at(stage, closing_outcome="meeting_booked" if stage == "closing" else None)
    after, directive = turn(
        progress,
        opt_out=True,
        asks_if_ai=True,
        wants_human=True,
        distressed=True,
        agrees_to_meeting=True,
        question="Who are you?",
        proposed_time="tomorrow 3pm",
    )
    assert (after.stage, after.outcome, directive.action) == ("ended", "opted_out", "exit_politely")


def test_opt_out_beats_the_turn_limit() -> None:
    after, _ = advance(at("pitch", turns=29), TurnSignals(opt_out=True), PLAYBOOK, BRAND, now=NOW)
    assert after.outcome == "opted_out"


@pytest.mark.parametrize("stage", STAGES)
def test_asks_if_ai_discloses_and_stays_in_the_stage(stage: str) -> None:
    after, directive = turn(at(stage), asks_if_ai=True, question="Am I talking to a robot?")
    assert directive.action == "disclose_ai"
    assert (after.stage, after.outcome) == (stage, None)


def test_retry_once_then_the_second_resistance_exits() -> None:
    first, directive = turn(at("discovery"), resistance=True)
    assert directive.action == "retry_once" and first.retries_used == {"discovery": 1}
    second, directive = turn(first, resistance=True)
    assert (second.outcome, directive.action) == ("not_interested", "exit_politely")


def test_a_retry_in_qualification_is_about_the_slot_just_asked() -> None:
    after, directive = turn(
        at("qualification", asked_slots=["role", "categories"]), resistance=True
    )
    assert (directive.action, directive.slot, after.stage) == (
        "retry_once",
        "categories",
        "qualification",
    )


def test_retries_are_counted_per_stage() -> None:
    progress = at("pitch", retries_used={"discovery": 1})
    _, directive = turn(progress, interest="negative")
    assert directive.action == "retry_once"


# --- stop rules ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "signals",
    [
        pytest.param({"slot_values": {"territory": "Texas"}}, id="territory"),
        pytest.param({"wants_other_category": "frozen meals"}, id="category"),
        pytest.param({"interest": "negative"}, id="not-taking-new-lines"),
    ],
)
def test_each_stop_rule_disqualifies(signals: dict[str, Any]) -> None:
    progress = at("qualification", asked_slots=["role", "categories", "territory"])
    after, directive = turn(progress, **signals)
    assert (after.outcome, directive.action) == ("disqualified", "exit_politely")


def test_a_territory_shared_early_is_checked_when_qualification_starts() -> None:
    progress, _ = turn(
        at("discovery"), interest="positive", slot_values={"territory": "Florida only"}
    )
    assert progress.outcome == "disqualified"


def test_wants_other_category_is_a_polite_exit_in_any_stage() -> None:
    after, directive = turn(at("discovery"), wants_other_category="hot sauces")
    assert (after.outcome, directive.action) == ("disqualified", "exit_politely")


# --- slots --------------------------------------------------------------------------


def test_slots_are_asked_one_per_turn_in_order_then_the_pitch() -> None:
    progress, directive = turn(at("discovery"), interest="positive")
    asked = [directive.slot]
    answers = {
        "role": "owner",
        "categories": "snacks",
        "territory": "Oregon",
        "accounts": "co-ops",
        "volume": "two pallets a month",
    }
    while directive.action == "ask_slot":
        progress, directive = turn(progress, slot_values={directive.slot: answers[directive.slot]})
        asked.append(directive.slot)
    assert asked[:-1] == ALL_SLOTS and directive.action == "pitch"
    assert progress.slots == answers and progress.stage == "pitch"


def test_an_already_shared_slot_is_confirmed_not_asked_again() -> None:
    progress, directive = turn(
        at("discovery"), interest="positive", slot_values={"territory": "Northern California"}
    )
    assert (directive.action, directive.slot) == ("ask_slot", "role")
    progress, directive = turn(progress, slot_values={"role": "buyer"})
    progress, directive = turn(progress, slot_values={"categories": "snacks"})
    assert (directive.action, directive.slot) == ("confirm_slot", "territory")
    progress, directive = turn(progress)
    assert (directive.action, directive.slot) == ("ask_slot", "accounts")


def test_in_rush_asks_only_the_required_slots() -> None:
    progress, directive = turn(
        at("qualification", asked_slots=["role"]), in_rush=True, slot_values={"role": "owner"}
    )
    assert directive.slot == "categories" and progress.in_rush
    progress, directive = turn(progress, slot_values={"categories": "snacks"})
    assert directive.slot == "territory"
    progress, directive = turn(progress, slot_values={"territory": "Washington"})
    assert directive.action == "pitch"
    assert "accounts" not in progress.asked_slots and "volume" not in progress.asked_slots


def test_unknown_or_blank_slot_values_are_ignored() -> None:
    progress, _ = turn(at("discovery"), slot_values={"budget": "lots", "role": "  "})
    assert progress.slots == {}


# --- handoff and closing ------------------------------------------------------------


def test_the_proposal_is_the_next_business_slot_and_a_yes_books_it() -> None:
    progress, directive = turn(at("pitch"), agrees_to_meeting=True)
    assert directive.proposed_slot == PROPOSAL == progress.proposed_slot
    progress, directive = turn(progress, agrees_to_meeting=True)
    assert (directive.action, directive.proposed_slot) == ("confirm_booking", PROPOSAL)
    assert (progress.stage, progress.closing_outcome) == ("closing", "meeting_booked")
    progress, directive = turn(progress)
    assert (progress.outcome, directive.action) == ("meeting_booked", "close")


def test_the_partners_own_time_replaces_the_proposal() -> None:
    progress, directive = turn(handoff(), proposed_time="Thursday at 2pm")
    assert progress.proposed_slot == datetime(2026, 11, 19, 14, 0, tzinfo=LA)
    assert directive.proposed_slot == progress.proposed_slot


def test_a_named_time_that_cannot_be_booked_becomes_a_callback() -> None:
    progress, directive = turn(handoff(), proposed_time="Saturday at 9am")
    assert directive.action == "retry_once" and progress.partner_time is not None
    progress, directive = turn(progress, proposed_time="sometime")
    assert progress.outcome == "callback_booked"
    assert progress.proposed_slot == datetime(2026, 11, 21, 9, 0, tzinfo=LA)


def test_a_vague_time_never_books_a_meeting() -> None:
    progress, _ = turn(handoff(), proposed_time="sometime next week")
    progress, _ = turn(progress, proposed_time="maybe next week, not sure")
    assert progress.outcome == "incomplete"


def test_not_a_good_time_with_a_time_books_the_callback_at_once() -> None:
    progress, directive = turn(
        at(), right_person=True, good_time=False, proposed_time="tomorrow at 3pm"
    )
    assert directive.action == "confirm_booking"
    assert (progress.closing_outcome, progress.proposed_slot) == (
        "callback_booked",
        datetime(2026, 11, 17, 15, 0, tzinfo=LA),
    )


def test_closing_waits_on_an_open_question() -> None:
    progress, directive = turn(
        handoff(), agrees_to_meeting=True, question="What is the shelf life?"
    )
    assert directive.action == "confirm_booking" and progress.open_question
    progress, directive = turn(progress)  # the partner just says "great"
    assert directive.action == "answer_question" and progress.stage == "closing"
    progress, directive = turn(progress)
    assert (progress.outcome, directive.action) == ("meeting_booked", "close")


# --- wants_human, distress, questions, limits ---------------------------------------


def test_wants_human_books_the_founder() -> None:
    progress, directive = turn(at("discovery"), wants_human=True)
    assert (progress.stage, progress.booking, directive.action) == (
        "handoff",
        "founder",
        "propose_slot",
    )
    progress, _ = turn(progress, agrees_to_meeting=True)
    progress, _ = turn(progress)
    assert progress.outcome == "escalated_to_human"


def test_wants_human_without_a_time_ends_incomplete() -> None:
    progress, _ = turn(at("qualification"), wants_human=True)
    progress, _ = turn(progress, proposed_time="whenever")
    progress, _ = turn(progress, proposed_time="no idea")
    assert progress.outcome == "incomplete"


def test_distress_is_acknowledged_then_a_negative_turn_ends_the_call() -> None:
    progress, directive = turn(at("discovery"), distressed=True)
    assert directive.action == "acknowledge" and progress.recovering
    progress, directive = turn(progress, resistance=True)
    assert progress.outcome == "not_interested"


def test_distress_then_engaging_carries_on() -> None:
    progress, _ = turn(at("discovery"), distressed=True)
    progress, directive = turn(progress, interest="positive")
    assert (progress.stage, directive.action, progress.recovering) == (
        "qualification",
        "ask_slot",
        False,
    )


def test_a_question_is_answered_before_the_next_step() -> None:
    progress = at("qualification", asked_slots=["role"])
    after, directive = turn(
        progress, slot_values={"role": "owner"}, question="What is the shelf life of the cookies?"
    )
    assert directive.action == "answer_question" and directive.fact_ids == ["F-3"]
    assert after.slots == {"role": "owner"} and after.asked_slots == ["role"]
    after, directive = turn(after)
    assert (directive.action, directive.slot) == ("ask_slot", "categories")


def test_a_question_no_fact_covers_gets_no_facts() -> None:
    _, directive = turn(at("pitch"), question="What margin do retailers make?")
    assert directive == Directive(action="answer_question", fact_ids=[])


def test_max_turns_ends_the_call_as_incomplete() -> None:
    after, directive = turn(at("qualification", turns=29), slot_values={"role": "owner"})
    assert (after.outcome, directive.action, after.turns) == ("incomplete", "close", 30)
    capped, _ = advance(
        at("discovery", turns=4), TurnSignals(), PLAYBOOK, BRAND, now=NOW, max_turns=5
    )
    assert capped.outcome == "incomplete"


def test_the_turn_limit_keeps_a_booked_outcome() -> None:
    after, _ = turn(
        at("closing", turns=29, closing_outcome="callback_booked"), question="One more thing?"
    )
    assert after.outcome == "callback_booked"


def test_an_ended_call_cannot_advance_and_the_input_is_not_changed() -> None:
    with pytest.raises(CallEnded):
        turn(at("ended", outcome="opted_out"))
    progress = at("qualification", asked_slots=["role"])
    before = progress.model_copy(deep=True)
    turn(progress, slot_values={"role": "owner"})
    assert progress == before


# --- facts and territory ------------------------------------------------------------


def test_select_facts_prefers_tags_that_match_the_slots() -> None:
    slots = {
        "role": "owner",
        "categories": "snacks, natural foods",
        "territory": "Northern California",
    }
    assert select_facts(BRAND, slots) == ["F-2", "F-8", "F-1"]


def test_select_facts_falls_back_to_the_first_facts() -> None:
    assert select_facts(BRAND, {}) == ["F-1", "F-2"]
    assert select_facts(BRAND.model_copy(update={"approved_facts": []}), {"role": "x"}) == []


def test_matching_facts_reads_plurals() -> None:
    assert matching_facts(BRAND, ["Can you send a sample?"]) == ["F-6"]


@pytest.mark.parametrize(
    ("territory", "regions", "fits"),
    [
        ("Northern California", ["California", "Oregon"], True),
        ("oregon and idaho", ["California", "Oregon"], True),
        ("Texas", ["California", "Oregon"], False),
        ("We sell nationwide", ["California"], True),
        ("Texas", [], True),
        ("OR", ["Oregon"], False),  # abbreviations are not guessed
    ],
)
def test_territory_fits(territory: str, regions: list[str], fits: bool) -> None:
    assert territory_fits(territory, regions) is fits
