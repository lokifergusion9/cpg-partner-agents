"""The /api/calls routes with scripted models (T-58, HLD section 7)."""

import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from partner_agents.api.main import create_app
from partner_agents.demo import SAMPLE_BRAND
from partner_agents.fakes import ScriptedChatModel
from partner_agents.graph.call import CallDeps
from partner_agents.outreach.voice import new_call_record
from partner_agents.prompts.caller import CallerReply
from partner_agents.schemas import OutreachDraft, TurnSignals
from tests.graph.test_build import (  # noqa: F401  (fixtures are used by name)
    client,
    instant_retries,
    make_deps,
)
from tests.graph.test_call import LINE, NOW, PLAYBOOK, make_lead


def call_deps(
    qdrant: AsyncQdrantClient, signals: list[TurnSignals], reply: CallerReply = LINE
) -> tuple[TestClient, str]:
    deps = make_deps(qdrant)
    deps.calls = CallDeps(
        settings=deps.settings,
        repo=deps.repo,
        playbook=PLAYBOOK,
        classifier=ScriptedChatModel.scripted(*signals),
        caller=ScriptedChatModel.repeating(reply),
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
