"""Every API route, offline: discovery -> outreach -> pending -> approve -> outboxed (HLD §7)."""

from collections.abc import Iterator
from typing import Any

import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient

from partner_agents.api.main import create_app
from partner_agents.config import Settings
from partner_agents.graph.build import AppDeps
from partner_agents.schemas import EmailDraft, RouteDecision
from tests.graph.test_build import (  # noqa: F401  (fixtures are used by name)
    ACME,
    BETA,
    DISCOVERY,
    client,
    instant_retries,
    make_deps,
    serve_sites,
)
from tests.helpers import ByPageModel

PROFILE = {
    "brand_name": "Crumb & Co",
    "product": "gluten-free cookies",
    "value_prop": "small-batch cookies",
    "sender_name": "Priya Shah",
    "sender_email": "priya@crumb.example",
    "founder_name": "Priya Shah",
    "category": "gluten-free cookies",
    "sign_off": "Thanks",
}
TOP_TWO = RouteDecision(
    intent="outreach",
    confidence=0.9,
    lead_selector="top 2",
    outreach_goal="introduce our cookies",
    reason="email",
)
EMAIL = EmailDraft(subject="Gluten-free cookies", body="A short intro. Priya Shah")


def deps_for(qdrant: AsyncQdrantClient, *routes: RouteDecision, **settings: Any) -> AppDeps:
    deps = make_deps(qdrant, *routes)
    if settings:
        deps.settings = Settings(_env_file=None, **settings)  # type: ignore[call-arg]
    deps.writer = ByPageModel(
        messages=iter([]),
        pages={ACME: [EMAIL.model_dump_json()], BETA: [EMAIL.model_dump_json()]},
    )
    return deps


@pytest.fixture
def web(client: AsyncQdrantClient) -> Iterator[TestClient]:  # noqa: F811
    with TestClient(create_app(deps_for(client, DISCOVERY, TOP_TWO))) as test_client:
        yield test_client


def error_of(response: Any) -> dict[str, Any]:
    body = response.json()
    assert list(body) == ["error"]
    return body["error"]  # type: ignore[no-any-return]


# --- the whole flow -----------------------------------------------------------------


@respx.mock
def test_discovery_then_outreach_then_approve_ends_in_the_outbox(web: TestClient) -> None:
    serve_sites()
    assert web.put("/api/profile", json={"user_id": "u_1", **PROFILE}).status_code == 200

    found = web.post("/api/chat", json={"user_id": "u_1", "message": "find cookie brokers"})
    assert found.status_code == 200
    first = found.json()
    assert first["route"]["intent"] == "discovery" and first["pending_approval"] is None
    assert {x["company_name"] for x in first["leads"]} == {"Acme Foods", "Beta Brokerage"}
    assert all(x["outreach_eligible"] for x in first["leads"])

    drafted = web.post(
        "/api/chat",
        json={"user_id": "u_1", "thread_id": first["thread_id"], "message": "email the top 2"},
    ).json()
    assert drafted["thread_id"] == first["thread_id"] and drafted["run_id"] != first["run_id"]
    pending = drafted["pending_approval"]["drafts"]
    assert [d["status"] for d in pending] == ["pending_approval"] * 2
    listed = web.get("/api/outreach", params={"user_id": "u_1", "status": "pending_approval"})
    assert len(listed.json()) == 2

    # While drafts wait, another message is refused.
    blocked = web.post(
        "/api/chat",
        json={"user_id": "u_1", "thread_id": first["thread_id"], "message": "hello"},
    )
    assert blocked.status_code == 409 and error_of(blocked)["code"] == "approval_pending"

    done = web.post(
        f"/api/chat/{first['thread_id']}/approve",
        params={"user_id": "u_1"},
        json={"approve": [pending[0]["id"]], "reject": [pending[1]["id"]]},
    )
    assert done.status_code == 200
    assert done.json()["pending_approval"] is None
    assert done.json()["reply"].startswith("Dispatch: 1 saved to the outbox")
    outboxed = web.get("/api/outreach", params={"user_id": "u_1", "status": "outboxed"}).json()
    assert [d["id"] for d in outboxed] == [pending[0]["id"]]
    rejected = web.get("/api/outreach", params={"user_id": "u_1", "status": "rejected"}).json()
    assert [d["id"] for d in rejected] == [pending[1]["id"]]

    run = web.get(f"/api/runs/{drafted['run_id']}", params={"user_id": "u_1"})
    assert run.status_code == 200
    assert {"chase_draft", "chase_dispatch"} <= set(run.json()["timings_ms"])


@respx.mock
def test_leads_can_be_listed_by_run(web: TestClient) -> None:
    serve_sites()
    first = web.post("/api/chat", json={"user_id": "u_1", "message": "find brokers"}).json()
    by_run = web.get("/api/leads", params={"user_id": "u_1", "run_id": first["run_id"]}).json()
    assert len(by_run) == 2
    assert web.get("/api/leads", params={"user_id": "u_1", "run_id": "r_other"}).json() == []
    assert web.get("/api/leads", params={"user_id": "u_2"}).json() == []


# --- errors -------------------------------------------------------------------------


def test_an_unknown_run_is_404_with_the_error_shape(web: TestClient) -> None:
    response = web.get("/api/runs/r_nope", params={"user_id": "u_1"})
    assert response.status_code == 404
    assert error_of(response) == {"code": "run_not_found", "message": "No such run."}


@respx.mock
def test_a_run_of_another_user_looks_unknown(web: TestClient) -> None:
    serve_sites()
    run_id = web.post("/api/chat", json={"user_id": "u_1", "message": "find"}).json()["run_id"]
    assert web.get(f"/api/runs/{run_id}", params={"user_id": "u_1"}).status_code == 200
    other = web.get(f"/api/runs/{run_id}", params={"user_id": "u_2"})
    assert other.status_code == 404 and error_of(other)["code"] == "run_not_found"


@respx.mock
def test_another_users_thread_is_not_found(web: TestClient) -> None:
    serve_sites()
    thread = web.post("/api/chat", json={"user_id": "u_1", "message": "find"}).json()["thread_id"]
    stolen = web.post(
        "/api/chat", json={"user_id": "u_2", "thread_id": thread, "message": "the LA ones"}
    )
    assert stolen.status_code == 404 and error_of(stolen)["code"] == "thread_not_found"
    approve = web.post(f"/api/chat/{thread}/approve", params={"user_id": "u_2"}, json={})
    assert approve.status_code == 404 and error_of(approve)["code"] == "thread_not_found"


def test_approving_with_nothing_pending_is_404(web: TestClient) -> None:
    response = web.post("/api/chat/t_none/approve", params={"user_id": "u_1"}, json={})
    assert response.status_code == 404 and error_of(response)["code"] == "nothing_to_approve"


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"user_id": "u 1; drop", "message": "hi"}, "user_id"),
        ({"user_id": "u_1", "message": ""}, "message"),
        ({"user_id": "u_1"}, "message"),
        ({"user_id": "u_1", "message": "hi", "thread_id": "../etc"}, "thread_id"),
    ],
)
def test_a_bad_request_is_422_and_does_not_echo_the_value(
    web: TestClient, body: dict[str, str], field: str
) -> None:
    response = web.post("/api/chat", json=body)
    error = error_of(response)
    assert response.status_code == 422 and error["code"] == "invalid_request"
    assert field in error["message"] and "drop" not in error["message"]
    assert "../etc" not in error["message"]


def test_an_unknown_path_and_a_wrong_method_use_the_error_shape(web: TestClient) -> None:
    assert error_of(web.get("/api/nope"))["code"] == "not_found"
    wrong = web.delete("/api/chat")
    assert wrong.status_code == 405 and error_of(wrong)["code"] == "method_not_allowed"


def test_an_unexpected_failure_is_a_generic_500(client: AsyncQdrantClient) -> Any:  # noqa: F811
    deps = deps_for(client, DISCOVERY)

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("secret detail buyer@acme.example")

    deps.repo.list_drafts = boom  # type: ignore[method-assign]
    with TestClient(create_app(deps), raise_server_exceptions=False) as test_client:
        response = test_client.get("/api/outreach", params={"user_id": "u_1"})
    assert response.status_code == 500 and error_of(response)["code"] == "internal_error"
    assert "secret" not in response.text and "acme" not in response.text


# --- API key ------------------------------------------------------------------------


def keyed(client_: AsyncQdrantClient, **extra: Any) -> TestClient:
    settings: dict[str, Any] = {"app_api_key": SecretStr("s3cret"), **extra}
    return TestClient(create_app(deps_for(client_, DISCOVERY, **settings)))


def test_a_missing_or_wrong_api_key_is_401(client: AsyncQdrantClient) -> None:  # noqa: F811
    with keyed(client) as web:
        for headers in ({}, {"X-API-Key": "wrong"}):
            response = web.get("/api/outreach", params={"user_id": "u_1"}, headers=headers)
            assert response.status_code == 401 and error_of(response)["code"] == "unauthorized"
        post = web.post("/api/chat", json={"user_id": "u_1", "message": "hi"})
        assert post.status_code == 401
        ok = web.get("/api/outreach", params={"user_id": "u_1"}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200 and ok.json() == []


def test_health_is_open_and_reveals_no_secret(client: AsyncQdrantClient) -> None:  # noqa: F811
    with keyed(client) as web:
        response = web.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "llm": "gemini",
        "qdrant": "local",
        "mongo": "memory",
        "search": "tavily",
    }
    assert "s3cret" not in response.text


def test_demo_mode_does_not_ask_for_the_key(client: AsyncQdrantClient) -> None:  # noqa: F811
    with keyed(client, demo_mode=True) as web:
        assert web.get("/api/outreach", params={"user_id": "u_1"}).status_code == 200


# --- profile ------------------------------------------------------------------------


def test_the_profile_round_trips_and_is_per_user(web: TestClient) -> None:
    assert web.get("/api/profile", params={"user_id": "u_1"}).status_code == 404
    saved = web.put("/api/profile", json={"user_id": "u_1", **PROFILE})
    assert saved.status_code == 200 and "user_id" not in saved.json()
    assert web.get("/api/profile", params={"user_id": "u_1"}).json()["brand_name"] == "Crumb & Co"
    assert web.get("/api/profile", params={"user_id": "u_2"}).status_code == 404
    bad = web.put("/api/profile", json={"user_id": "u_1", **PROFILE, "sender_email": "nope"})
    assert bad.status_code == 422 and "sender_email" in error_of(bad)["message"]
