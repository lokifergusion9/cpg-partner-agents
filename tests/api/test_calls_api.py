"""The /api/calls routes with scripted models (T-58, HLD section 7)."""

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from partner_agents.api.main import create_app
from partner_agents.demo import SAMPLE_BRAND
from partner_agents.fakes import ScriptedChatModel
from partner_agents.graph.call import CallDeps
from partner_agents.outreach.voice import new_call_record
from partner_agents.prompts.caller import CallerReply
from partner_agents.prompts.simulator import SimulatorReply
from partner_agents.schemas import OutreachDraft, TurnSignals
from tests.graph.test_build import (  # noqa: F401  (fixtures are used by name)
    client,
    instant_retries,
    make_deps,
)
from tests.graph.test_call import BOOKING, BOOKING_TEXTS, LINE, NOW, PLAYBOOK, make_lead


def call_deps(
    qdrant: AsyncQdrantClient,
    signals: list[TurnSignals],
    reply: CallerReply = LINE,
    simulator: ScriptedChatModel | None = None,
) -> tuple[TestClient, str]:
    deps = make_deps(qdrant)
    deps.calls = CallDeps(
        settings=deps.settings,
        repo=deps.repo,
        playbook=PLAYBOOK,
        classifier=ScriptedChatModel.scripted(*signals),
        caller=ScriptedChatModel.repeating(reply),
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
    return TestClient(create_app(deps, with_ui=False)), record.id


@pytest.fixture
def web(client: AsyncQdrantClient) -> Iterator[tuple[TestClient, str]]:  # noqa: F811
    test_client, call_id = call_deps(
        client,
        [TurnSignals(), TurnSignals(right_person=False)],  # an unclear answer, then "wrong person"
    )
    with test_client:
        yield test_client, call_id


def test_start_then_two_turns_then_the_call_has_ended(web: tuple[TestClient, str]) -> None:
    http, call_id = web
    params = {"user_id": "u_1"}

    started = http.post(f"/api/calls/{call_id}/start", params=params)
    assert started.status_code == 200
    body = started.json()
    assert (body["call_id"], body["stage"], body["ended"], body["outcome"]) == (
        call_id,
        "opening",
        False,
        None,
    )
    assert body["reply"] and body["checks"]

    first = http.post(f"/api/calls/{call_id}/turn", params=params, json={"text": "speaking"})
    assert first.status_code == 200 and first.json()["stage"] == "opening"
    assert first.json()["ended"] is False

    last = http.post(f"/api/calls/{call_id}/turn", params=params, json={"text": "wrong number"})
    assert last.status_code == 200
    assert (last.json()["ended"], last.json()["outcome"]) == (True, "wrong_person")

    record = http.get(f"/api/calls/{call_id}", params=params).json()
    assert record["progress"]["outcome"] == "wrong_person" and len(record["turns"]) == 5
    [listed] = http.get("/api/calls", params=params).json()
    assert listed["id"] == call_id and listed["turns"] == []  # the list leaves the turns out
    assert http.get("/api/calls", params={**params, "outcome": "opted_out"}).json() == []


def test_an_unknown_call_is_404_on_every_route(web: tuple[TestClient, str]) -> None:
    http, _ = web
    params = {"user_id": "u_1"}
    for response in (
        http.post("/api/calls/c_nope/start", params=params),
        http.post("/api/calls/c_nope/turn", params=params, json={"text": "hi"}),
        http.get("/api/calls/c_nope", params=params),
    ):
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "call_not_found"


def test_another_users_call_is_404(web: tuple[TestClient, str]) -> None:
    http, call_id = web
    response = http.get(f"/api/calls/{call_id}", params={"user_id": "u_2"})
    assert response.status_code == 404


def test_turns_in_the_wrong_state_are_409(web: tuple[TestClient, str]) -> None:
    http, call_id = web
    params = {"user_id": "u_1"}
    early = http.post(f"/api/calls/{call_id}/turn", params=params, json={"text": "hi"})
    assert (early.status_code, early.json()["error"]["code"]) == (409, "call_not_started")
    assert http.post(f"/api/calls/{call_id}/start", params=params).status_code == 200
    again = http.post(f"/api/calls/{call_id}/start", params=params)
    assert (again.status_code, again.json()["error"]["code"]) == (409, "call_already_started")


def test_the_calls_are_off_when_the_call_agent_is_not_configured(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    with TestClient(create_app(make_deps(client), with_ui=False)) as http:
        response = http.post("/api/calls/c_1/start", params={"user_id": "u_1"})
    assert (response.status_code, response.json()["error"]["code"]) == (503, "calls_unavailable")


# --- /simulate (T-59) ---------------------------------------------------------------


def simulating(
    qdrant: AsyncQdrantClient,
    signals: list[TurnSignals],
    texts: list[str] | None,
) -> tuple[TestClient, str]:
    simulator = (
        None
        if texts is None
        else ScriptedChatModel.scripted(*(SimulatorReply(text=t) for t in texts))
    )
    return call_deps(qdrant, signals, simulator=simulator)


def error_code(response: Any) -> str:
    return str(response.json()["error"]["code"])


def test_simulate_plays_a_persona_through_to_the_outcome(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    http, call_id = simulating(client, BOOKING, BOOKING_TEXTS)
    with http:
        response = http.post(
            f"/api/calls/{call_id}/simulate",
            params={"user_id": "u_1"},
            json={"persona_id": "interested_broker"},
        )
        assert response.status_code == 200
        record = response.json()
        assert record["progress"]["outcome"] == "meeting_booked"
        assert (record["mode"], record["persona_id"]) == ("simulate", "interested_broker")
        assert record["followup_draft_id"] and record["meeting_at"]
        again = http.post(
            f"/api/calls/{call_id}/simulate",
            params={"user_id": "u_1"},
            json={"persona_id": "interested_broker"},
        )
        assert (again.status_code, error_code(again)) == (409, "call_already_started")


@pytest.mark.parametrize(
    ("call", "persona", "texts", "status", "code"),
    [
        ("known", "nobody", [], 404, "persona_not_found"),
        ("known", "../etc", [], 422, None),
        ("c_nope", "gatekeeper", [], 404, "call_not_found"),
        ("known", "gatekeeper", None, 503, "simulator_unavailable"),
        ("known", "gatekeeper", [], 502, "simulator_error"),  # the scripted simulator runs dry
    ],
)
def test_simulate_refusals(
    client: AsyncQdrantClient,  # noqa: F811
    call: str,
    persona: str,
    texts: list[str] | None,
    status: int,
    code: str | None,
) -> None:
    http, call_id = simulating(client, BOOKING, texts)
    with http:
        response = http.post(
            f"/api/calls/{call_id if call == 'known' else call}/simulate",
            params={"user_id": "u_1"},
            json={"persona_id": persona},
        )
    assert response.status_code == status
    if code is not None:
        assert error_code(response) == code
