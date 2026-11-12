"""The UI handlers (no browser) and the /ui mount (PRD §10 steps 1-6 and 10)."""

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from partner_agents.api.main import create_app
from partner_agents.graph.build import Context
from partner_agents.schemas import (
    ApprovedFact,
    BrandProfile,
    Lead,
    OutreachDraft,
    RunRecord,
    StageError,
)
from partner_agents.ui.app import (
    APPROVE,
    EDIT,
    EMAIL_BADGES,
    PENDING_NOTICE,
    PROFILE_FIELDS,
    REJECT,
    build_decision,
    decide,
    draft_rows,
    lead_rows,
    pending_drafts,
    profile_from_form,
    profile_to_form,
    run_rows,
    save_profile,
    send_message,
)
from tests.graph.test_build import (  # noqa: F401  (fixtures are used by name)
    client,
    instant_retries,
    make_deps,
    seeded_approval,
)

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


def lead(**overrides: Any) -> Lead:
    fields: dict[str, Any] = {
        "id": "l1", "run_id": "r", "user_id": "u", "company_name": "Acme Foods",
        "partner_type": "broker", "location": "Los Angeles, CA", "website": None,
        "email": "sales@acme.example", "email_status": "verified", "phone": None,
        "phone_status": "unchecked", "source_url": "https://www.acme.example/contact",
        "evidence_quote": "q", "created_at": NOW,
    }  # fmt: skip
    return Lead(**{**fields, **overrides})


def draft(key: str = "d1", channel: str = "email", **overrides: Any) -> OutreachDraft:
    fields: dict[str, Any] = {
        "id": key, "lead_id": "l1", "user_id": "u", "run_id": "r", "channel": channel,
        "recipient": "sales@acme.example", "subject": "Hello", "body": "Body. Priya",
        "status": "pending_approval", "prompt_version": "v", "created_at": NOW,
    }  # fmt: skip
    return OutreachDraft(**{**fields, **overrides})


# --- tables -------------------------------------------------------------------------


def test_lead_rows_show_a_badge_and_a_source_link() -> None:
    [row] = lead_rows([lead()])
    assert row[:6] == [
        "Acme Foods", "broker", "Los Angeles, CA", "sales@acme.example",
        EMAIL_BADGES["verified"], "",
    ]  # fmt: skip
    assert row[6] == "[www.acme.example](https://www.acme.example/contact)"


@pytest.mark.parametrize("status", ["mx_failed", "not_in_source", "invalid_format"])
def test_a_failed_check_is_not_shown_as_verified(status: str) -> None:
    [row] = lead_rows([lead(email_status=status)])
    assert row[4] == EMAIL_BADGES[status] and "✅" not in row[4]


def test_a_lead_without_an_email_has_no_badge() -> None:
    [row] = lead_rows([lead(email=None, email_status="unchecked")])
    assert row[3] == "" and row[4] == ""


def test_draft_rows_label_calls_and_decided_time() -> None:
    rows = draft_rows(
        [draft(status="outboxed", decided_at=NOW), draft("d2", "voice", recipient="", subject=None)]
    )
    assert rows[0] == ["outboxed", "email", "sales@acme.example", "Hello", "2026-11-12 12:00"]
    assert rows[1] == ["pending_approval", "voice", "(simulated)", "(call plan)", ""]


def test_run_rows_show_timings_tokens_errors_and_the_trace() -> None:
    run = RunRecord(
        run_id="r_1", user_id="u", thread_id="t", intent="discovery", started_at=NOW,
        finished_at=NOW, timings_ms={"charlie_route": 120, "scout_search": 480},
        tokens={"router": 300}, prompt_versions={}, trace_url="https://trace.example/r_1",
        errors=[StageError(stage="scout_fetch", code="blocked_robots", message="m")],
    )  # fmt: skip
    [row] = run_rows([run])
    assert row[0] == "r_1" and row[3] == "600" and row[4] == "router 300"
    assert row[5] == "scout_fetch: blocked_robots"
    assert row[6] == "charlie_route 120 · scout_search 480"
    assert row[7] == "[open](https://trace.example/r_1)"
    bare = run.model_copy(update={"trace_url": None, "tokens": {}, "errors": []})
    assert run_rows([bare])[0][4:6] == ["—", "—"] and run_rows([bare])[0][7] == "—"


# --- decisions ----------------------------------------------------------------------


def test_the_form_becomes_an_approval_decision() -> None:
    drafts = [draft("a"), draft("b"), draft("c"), draft("v", "voice", subject=None)]
    fields = [
        APPROVE, "Hello", "Body",
        EDIT, " New subject ", "New body",
        REJECT, "", "",
        EDIT, "", "plan",
    ]  # fmt: skip
    decision = build_decision(drafts, fields)
    assert decision.approve == ["a", "v"]  # a call plan cannot be edited, so it is approved
    assert decision.reject == ["c"]
    assert list(decision.edits) == ["b"]
    assert (decision.edits["b"].subject, decision.edits["b"].body) == ("New subject", "New body")


def test_an_edit_the_schema_refuses_is_explained_without_the_body() -> None:
    with pytest.raises(ValueError, match=r"sales@acme\.example") as raised:
        build_decision([draft()], [EDIT, "x" * 81, "secret body text"])
    assert "secret body text" not in str(raised.value)


def test_pending_drafts_reads_the_interrupt_payload() -> None:
    payload = {"drafts": [draft().model_dump(mode="json")], "problems": {}}
    assert [d.id for d in pending_drafts(payload)] == ["d1"] and pending_drafts(None) == []


# --- the profile form ---------------------------------------------------------------


def brand(**overrides: Any) -> BrandProfile:
    fact = ApprovedFact(id="F-9", text="Gluten free.", source="brand_profile", tags=["diet"])
    fields: dict[str, Any] = {
        "brand_name": "Crumb & Co", "product": "cookies", "value_prop": "v", "category": "cookies",
        "sender_name": "Priya Shah", "sender_email": "priya@crumb.example",
        "founder_name": "Priya", "sign_off": "Thanks", "target_regions": ["CA", "OR"],
        "timezone": "America/Los_Angeles", "approved_facts": [fact],
    }  # fmt: skip
    return BrandProfile(**{**fields, **overrides})


def test_the_profile_form_round_trips_and_keeps_fact_ids_and_tags() -> None:
    original = brand()
    values = profile_to_form(original)
    assert len(values) == len(PROFILE_FIELDS)
    values[-1] = "Gluten free.\nMade in small batches."
    saved = profile_from_form(values, original)
    assert saved.approved_facts[0] == original.approved_facts[0]  # same id and tags
    assert saved.approved_facts[1].id == "F-2" and saved.approved_facts[1].tags == []
    assert saved.target_regions == ["CA", "OR"] and saved.timezone == "America/Los_Angeles"


def test_a_blank_form_defaults_to_utc() -> None:
    values = profile_to_form(None)
    assert len(values) == len(PROFILE_FIELDS) and values[PROFILE_FIELDS.index("timezone")] == "UTC"


def test_saving_a_bad_profile_or_user_id_says_why(client: AsyncQdrantClient) -> None:  # noqa: F811
    ctx = Context(deps=make_deps(client), graph=None)  # type: ignore[arg-type]
    values = profile_to_form(brand())
    values[PROFILE_FIELDS.index("sender_email")] = "not-an-email"
    assert asyncio.run(save_profile(ctx, "u_1", values)).startswith("Not saved: sender_email")
    invalid = asyncio.run(save_profile(ctx, "bad id!", profile_to_form(brand())))
    assert invalid.startswith("Invalid user id")
    assert asyncio.run(save_profile(ctx, "u_1", profile_to_form(brand()))) == "Brand profile saved."
    saved = asyncio.run(ctx.deps.repo.get_profile("u_1"))
    assert saved is not None
    # A first save has no earlier facts to take ids and tags from, so they are numbered afresh.
    assert saved.model_dump(exclude={"approved_facts"}) == brand().model_dump(
        exclude={"approved_facts"}
    )
    assert [(f.id, f.text) for f in saved.approved_facts] == [("F-1", "Gluten free.")]


# --- a conversation through the handlers --------------------------------------------


def test_chat_pauses_for_approval_then_the_decision_goes_to_the_outbox(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    deps, graph, _ = seeded_approval(client)
    ctx = Context(deps=deps, graph=graph)
    seen: list[str] = []

    async def go() -> Any:
        first = await send_message(
            ctx, user_id="u_1", thread_id="t_1", text="email the top 3", client_ip="1.2.3.4",
            on_node=seen.append,
        )  # fmt: skip
        blocked = await send_message(
            ctx, user_id="u_1", thread_id="t_1", text="hello", client_ip="1.2.3.4"
        )
        drafts = pending_drafts(first.pending)
        fields = [APPROVE, "", "", EDIT, "Edited", "My own words. Priya Shah", REJECT, "", ""]
        reply, still = await decide(
            ctx, user_id="u_1", thread_id="t_1", drafts=drafts, fields=fields
        )
        return first, blocked, reply, still

    first, blocked, reply, still = asyncio.run(go())
    assert seen == ["charlie_route", "chase_select", "chase_draft"]  # the progress the UI shows
    assert first.thread_id == "t_1" and len(pending_drafts(first.pending)) == 3
    assert [x.company_name for x in first.leads] == ["Acme Foods", "Beta Foods", "Gamma Foods"]
    assert blocked.reply == PENDING_NOTICE and blocked.pending is not None
    assert reply.startswith("Dispatch: 2 saved to the outbox") and still is None
    saved = {d.lead_id: (d.status, d.subject) for d in asyncio.run(deps.repo.list_drafts("u_1"))}
    assert saved["acme"][0] == "outboxed" and saved["beta"] == ("outboxed", "Edited")
    assert saved["gamma"][0] == "rejected"


def test_deciding_without_anything_pending_is_harmless(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    deps, graph, _ = seeded_approval(client)
    ctx = Context(deps=deps, graph=graph)
    reply, still = asyncio.run(
        decide(ctx, user_id="u_1", thread_id="t_1", drafts=[draft()], fields=[APPROVE, "", ""])
    )
    assert reply == "There is nothing to approve." and still is None
    nothing = asyncio.run(decide(ctx, user_id="u_1", thread_id=None, drafts=[], fields=[]))
    assert nothing == ("There is nothing to approve.", None)


def test_a_refused_edit_keeps_the_drafts_pending(client: AsyncQdrantClient) -> None:  # noqa: F811
    deps, graph, _ = seeded_approval(client)
    ctx = Context(deps=deps, graph=graph)

    async def go() -> Any:
        first = await send_message(
            ctx, user_id="u_1", thread_id="t_1", text="email the top 3", client_ip="ip"
        )
        drafts = pending_drafts(first.pending)
        return await decide(
            ctx, user_id="u_1", thread_id="t_1", drafts=drafts, fields=[EDIT, "x" * 90, "b"] * 3
        )

    reply, still = asyncio.run(go())
    assert reply.startswith("Could not use your edit") and still is not None
    assert [d.status for d in asyncio.run(deps.repo.list_drafts("u_1"))] == ["pending_approval"] * 3


# --- the mount ----------------------------------------------------------------------


def test_the_ui_is_served_at_ui_next_to_the_api(client: AsyncQdrantClient) -> None:  # noqa: F811
    with TestClient(create_app(make_deps(client))) as web:
        page = web.get("/ui/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert "CPG Partner Agents" in page.text
        assert web.get("/api/health").json()["status"] == "ok"  # the lifespan still ran
        home = web.get("/", follow_redirects=False)
        assert home.status_code == 307 and home.headers["location"] == "/ui/"
