"""Demo personas, the simulator prompt and a whole simulated call (T-59)."""

import asyncio
from pathlib import Path

import pytest

from partner_agents.calls.personas import PersonaError, list_personas, load_persona
from partner_agents.calls.simulator import partner_line, simulate_call
from partner_agents.fakes import ScriptedChatModel
from partner_agents.graph.call import CallNotActive
from partner_agents.prompts.simulator import (
    PROMPT_VERSION,
    SimulatorReply,
    build_simulator_messages,
)
from partner_agents.schemas import CallTurn, Persona, TurnSignals
from tests.graph.test_call import BOOKING, BOOKING_TEXTS, setup

SLOTS = {"role", "categories", "territory", "accounts", "volume"}


def test_the_six_demo_personas_load() -> None:
    personas = list_personas()
    assert [p.id for p in personas] == [
        "busy_retailer",
        "gatekeeper",
        "interested_broker",
        "stop_calling",
        "territory_mismatch",
        "wants_human",
    ]
    for persona in personas:
        assert persona.title and persona.card and persona.expected_outcome is not None
        assert set(persona.hidden_facts) <= SLOTS  # a fact names a qualification slot


def test_every_expected_outcome_is_different_where_the_archetype_differs() -> None:
    outcomes = {p.id: p.expected_outcome for p in list_personas()}
    assert outcomes["interested_broker"] == "meeting_booked"
    assert outcomes["stop_calling"] == "opted_out"
    assert outcomes["gatekeeper"] == "wrong_person"


@pytest.mark.parametrize("bad", ["../secret", "Upper", "", "a b", "x" * 65])
def test_a_persona_id_that_is_not_a_plain_name_is_refused(bad: str) -> None:
    with pytest.raises(PersonaError):
        load_persona(bad)


def test_a_missing_or_broken_persona_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(PersonaError, match="no persona file"):
        load_persona("ghost", tmp_path)
    (tmp_path / "bad.toml").write_text("id = [", encoding="utf-8")
    with pytest.raises(PersonaError, match="not valid TOML"):
        load_persona("bad", tmp_path)
    (tmp_path / "thin.toml").write_text('id = "thin"', encoding="utf-8")
    with pytest.raises(PersonaError, match="invalid persona"):
        load_persona("thin", tmp_path)
    (tmp_path / "other.toml").write_text('id = "x"\ntitle = "t"\ncard = "c"', encoding="utf-8")
    with pytest.raises(PersonaError, match="declares id"):
        load_persona("other", tmp_path)


def test_the_prompt_holds_the_card_the_hidden_facts_and_the_call_so_far() -> None:
    persona = load_persona("interested_broker")
    turns = [
        CallTurn(index=0, speaker="agent", text="Hello, this is an AI assistant.", stage="opening"),
        CallTurn(index=1, speaker="partner", text="Speaking.", stage="opening"),
    ]
    system, user = (str(m.content) for m in build_simulator_messages(persona, turns))
    assert "Dana Whitfield" in system and "territory: Oregon and Washington" in system
    assert "ONLY when you are asked" in system and "Never say you are an AI" in system
    assert "Caller: Hello, this is an AI assistant." in user and "You: Speaking." in user
    assert PROMPT_VERSION.startswith("simulator-")


def test_the_prompt_keeps_only_the_last_turns() -> None:
    persona = load_persona("gatekeeper")
    turns = [
        CallTurn(index=i, speaker="agent", text=f"line {i}", stage="opening") for i in range(20)
    ]
    user = str(build_simulator_messages(persona, turns)[1].content)
    assert "line 19" in user and "line 12" in user and "line 11" not in user


def test_partner_line_is_trimmed_and_capped() -> None:
    persona = load_persona("gatekeeper")
    model = ScriptedChatModel.scripted(SimulatorReply(text="  " + "x" * 900 + "  "))
    line = asyncio.run(partner_line(model, persona, []))
    assert line == "x" * 500


def test_a_simulated_call_reaches_the_expected_outcome() -> None:
    deps, graph, call_id = setup(BOOKING)
    persona = load_persona("interested_broker")
    simulator = ScriptedChatModel.scripted(*(SimulatorReply(text=t) for t in BOOKING_TEXTS))
    record = asyncio.run(
        simulate_call(graph, deps, simulator, persona, user_id="u_1", call_id=call_id)
    )
    assert record.progress.outcome == persona.expected_outcome == "meeting_booked"
    assert (record.mode, record.persona_id) == ("simulate", "interested_broker")
    assert [t.speaker for t in record.turns][:3] == ["agent", "partner", "agent"]
    assert len(simulator.received) == len(BOOKING_TEXTS)
    saved = asyncio.run(deps.repo.get_call("u_1", call_id))
    assert saved == record and record.followup_draft_id is not None


def test_a_persona_that_never_agrees_ends_at_the_turn_limit() -> None:
    deps, graph, call_id = setup([TurnSignals(right_person=True, good_time=True)])
    simulator = ScriptedChatModel.repeating(SimulatorReply(text="Hmm, I am not sure."))
    record = asyncio.run(
        simulate_call(
            graph, deps, simulator, load_persona("busy_retailer"), user_id="u_1", call_id=call_id
        )
    )
    assert record.progress.outcome == "incomplete"
    assert record.progress.turns <= deps.settings.call_max_turns


def test_a_simulator_failure_keeps_the_turns_played_so_far() -> None:
    deps, graph, call_id = setup(BOOKING)
    simulator = ScriptedChatModel.scripted(SimulatorReply(text="Speaking."))  # then it runs dry
    with pytest.raises(Exception):  # noqa: B017  (the provider error type varies)
        asyncio.run(
            simulate_call(
                graph,
                deps,
                simulator,
                load_persona("interested_broker"),
                user_id="u_1",
                call_id=call_id,
            )
        )
    saved = asyncio.run(deps.repo.get_call("u_1", call_id))
    assert saved is not None and len(saved.turns) == 3 and saved.progress.outcome is None


def test_only_a_call_that_has_not_started_can_be_simulated() -> None:
    deps, graph, call_id = setup([])
    simulator = ScriptedChatModel.repeating(SimulatorReply(text="Hello."))
    persona = Persona(id="p", title="t", card="c")
    asyncio.run(simulate_call(graph, deps, simulator, persona, user_id="u_1", call_id=call_id))
    with pytest.raises(CallNotActive) as caught:
        asyncio.run(simulate_call(graph, deps, simulator, persona, user_id="u_1", call_id=call_id))
    assert caught.value.code == "call_already_started"
    with pytest.raises(LookupError):
        asyncio.run(simulate_call(graph, deps, simulator, persona, user_id="u_1", call_id="c_nope"))
