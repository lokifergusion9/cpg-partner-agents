"""The main graph end to end, offline: two turns in one thread (discovery, then "only the ones
in LA"), run records, message trimming and the fixed-reply branches."""

import asyncio
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from langchain_core.messages import HumanMessage
from qdrant_client import AsyncQdrantClient

from partner_agents import llm
from partner_agents.config import Settings
from partner_agents.fakes import HashingEmbedder, ScriptedChatModel
from partner_agents.graph.build import MAX_MESSAGES, AppDeps, Graph, build_graph, run_turn
from partner_agents.graph.charlie import REFUSAL
from partner_agents.graph.scout import ScoutDeps
from partner_agents.retrieval.store import KnowledgeStore, VectorStore
from partner_agents.schemas import (
    DiscoveryIntent,
    ExtractedLead,
    ExtractionResult,
    RouteDecision,
    SearchHit,
)
from partner_agents.storage.repo import InMemoryRepository
from partner_agents.utils.retry import RetryPolicy
from partner_agents.web import fetch as fetch_module
from partner_agents.web.fetch import Fetcher
from partner_agents.web.search import SearchService
from tests.helpers import ByPageModel

ACME = "https://acme.example/"
BETA = "https://beta.example/"
SCOUT_NODES = {
    "scout_search",
    "scout_fetch",
    "scout_index",
    "scout_evidence",
    "scout_extract",
    "scout_verify",
    "scout_report",
}

DISCOVERY = RouteDecision(
    intent="discovery",
    confidence=0.95,
    discovery=DiscoveryIntent(
        partner_type="broker", product_category="cookies", location="California", raw_query="x"
    ),
    reason="search",
)
ONLY_LA = RouteDecision(
    intent="outreach", confidence=0.9, lead_selector="the ones in LA", reason="narrow"
)


def decision(intent: str) -> RouteDecision:
    return RouteDecision(intent=intent, confidence=0.9, reason="r")  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch_module, "RETRY_POLICY", RetryPolicy(attempts=1, base_delay_s=0))
    monkeypatch.setattr(llm, "RETRY_POLICY", RetryPolicy(attempts=1, base_delay_s=0))


class FakeProvider:
    name = "fake"

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        return [
            SearchHit(url=url, title=url, snippet="s", rank=i + 1, provider="fake")  # type: ignore[arg-type]
            for i, url in enumerate([ACME, BETA])
        ]


def lead_reply(name: str, email: str, location: str) -> str:
    lead = ExtractedLead(
        company_name=name, partner_type="broker", email=email, location=location,
        evidence_quote=email,
    )  # fmt: skip
    return ExtractionResult(leads=[lead]).model_dump_json()


@pytest.fixture
def client() -> Iterator[AsyncQdrantClient]:
    qdrant = AsyncQdrantClient(":memory:")
    yield qdrant
    asyncio.run(qdrant.close())


def make_deps(client: AsyncQdrantClient, *router_replies: RouteDecision) -> AppDeps:
    async def resolve(host: str) -> list[str]:
        return ["93.184.216.34"]

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    repo = InMemoryRepository()
    embedder = HashingEmbedder()
    extractor = ByPageModel(
        messages=iter([]),
        pages={
            ACME: [lead_reply("Acme Foods", "sales@acme.example", "Los Angeles, CA")],
            BETA: [lead_reply("Beta Brokerage", "hello@beta.example", "San Diego, CA")],
        },
    )
    scout = ScoutDeps(
        settings=settings,
        repo=repo,
        search=SearchService([FakeProvider()], repo, frozenset()),
        fetcher=Fetcher(settings, repo, resolver=resolve),
        store=VectorStore(client, embedder),
        model=extractor,
        count_tokens=lambda text: len(text.split()),
    )
    return AppDeps(
        settings=settings,
        repo=repo,
        scout=scout,
        kb=KnowledgeStore(client, embedder),
        router=ScriptedChatModel.scripted(*router_replies),
        writer=ScriptedChatModel.scripted(),
    )


def serve_sites() -> None:
    for host in ("acme", "beta"):
        respx.get(f"https://{host}.example/robots.txt").mock(return_value=httpx.Response(404))
    respx.get(ACME).respond(
        200,
        html="<html><body><p>Acme Foods brokers cookies in Los Angeles.</p>"
        "<footer>sales@acme.example</footer></body></html>",
    )
    respx.get(BETA).respond(
        200,
        html="<html><body><p>Beta Brokerage sells cookies in San Diego.</p>"
        "<footer>hello@beta.example</footer></body></html>",
    )


def turns(deps: AppDeps, *texts: str, thread: str = "t_1") -> tuple[Graph, list[Any]]:
    graph = build_graph(deps)

    async def go() -> list[Any]:
        try:
            return [
                await run_turn(graph, deps, thread_id=thread, user_id="u_1", text=text)
                for text in texts
            ]
        finally:
            await deps.scout.fetcher.aclose()

    return graph, asyncio.run(go())


# --- two turns in one thread --------------------------------------------------------


@respx.mock
def test_a_follow_up_resolves_against_the_leads_of_the_first_turn(
    client: AsyncQdrantClient,
) -> None:
    serve_sites()
    deps = make_deps(client, DISCOVERY, ONLY_LA)
    _, (first, second) = turns(deps, "find cookie brokers in California", "only the ones in LA")

    leads = {lead.company_name: lead for lead in first.state["leads"]}
    assert set(leads) == {"Acme Foods", "Beta Brokerage"}
    assert first.reply.startswith("Found 2 companies")

    # Turn 2 ran in the same thread: the router saw turn 1's leads, the selector picked LA.
    assert second.state["selected_leads"] == [leads["Acme Foods"].id]
    assert '1 of 2 leads match "the ones in LA"' not in second.reply  # singular wording
    assert second.reply.startswith('1 of 2 lead matches "the ones in LA"')
    assert "Acme Foods" in second.reply and "Beta Brokerage" not in second.reply
    router_prompt = deps.router.received[1][0].text  # type: ignore[attr-defined]
    assert "Acme Foods (broker, Los Angeles, CA; verified email)" in router_prompt
    assert [m.text for m in deps.router.received[1][1:]] == [  # type: ignore[attr-defined]
        "find cookie brokers in California",
        first.reply,
        "only the ones in LA",
    ]


@respx.mock
def test_each_turn_gets_its_own_run_record_with_the_timings_of_the_nodes_that_ran(
    client: AsyncQdrantClient,
) -> None:
    serve_sites()
    deps = make_deps(client, DISCOVERY, ONLY_LA)
    _, (first, second) = turns(deps, "find cookie brokers in California", "only the ones in LA")

    assert first.run.run_id != second.run.run_id
    assert set(first.run.timings_ms) == {"charlie_route", *SCOUT_NODES}
    assert set(second.run.timings_ms) == {"charlie_route", "chase_select"}
    assert (first.run.intent, second.run.intent) == ("discovery", "outreach")
    assert first.run.prompt_versions.keys() == {"charlie", "scout"}
    assert first.run.finished_at is not None and first.run.finished_at >= first.run.started_at

    saved = asyncio.run(deps.repo.get_run(second.run.run_id))
    assert saved == second.run
    assert [run.run_id for run in asyncio.run(deps.repo.list_runs("u_1"))] == [
        second.run.run_id,
        first.run.run_id,
    ]
    leads = asyncio.run(deps.repo.list_leads("u_1", run_id=first.run.run_id))
    assert len(leads) == 2  # leads are saved under the run that found them


@respx.mock
def test_another_thread_does_not_see_the_leads(client: AsyncQdrantClient) -> None:
    serve_sites()
    deps = make_deps(client, DISCOVERY, ONLY_LA)
    graph = build_graph(deps)

    async def go() -> Any:
        try:
            await run_turn(graph, deps, thread_id="t_a", user_id="u_1", text="find brokers")
            return await run_turn(graph, deps, thread_id="t_b", user_id="u_1", text="the LA ones")
        finally:
            await deps.scout.fetcher.aclose()

    other = asyncio.run(go())
    assert other.reply.startswith("I have no leads in this conversation yet")


@respx.mock
def test_an_outreach_turn_drafts_emails_for_the_selected_leads(client: AsyncQdrantClient) -> None:
    from partner_agents.schemas import BrandProfile, EmailDraft

    serve_sites()
    top_two = RouteDecision(
        intent="outreach",
        confidence=0.9,
        lead_selector="top 2",
        outreach_goal="introduce our cookies",
        reason="email",
    )
    deps = make_deps(client, DISCOVERY, top_two)
    reply = EmailDraft(subject="Gluten-free cookies", body="Hi, a short intro.\n\nPriya Shah")
    deps.writer = ByPageModel(
        messages=iter([]),
        pages={ACME: [reply.model_dump_json()], BETA: [reply.model_dump_json()]},
    )
    profile = BrandProfile(
        brand_name="Crumb & Co", product="cookies", value_prop="v", sender_name="Priya Shah",
        sender_email="priya@crumb.example", founder_name="Priya Shah", category="cookies",
        sign_off="Thanks",
    )  # fmt: skip
    asyncio.run(deps.repo.save_profile("u_1", profile))
    _, (_, second) = turns(deps, "find cookie brokers in California", "email the top 2")

    drafts = second.state["drafts"]
    assert [d.status for d in drafts] == ["pending_approval", "pending_approval"]
    assert {d.recipient for d in drafts} == {"sales@acme.example", "hello@beta.example"}
    assert set(second.run.timings_ms) == {"charlie_route", "chase_select", "chase_draft"}
    assert second.run.prompt_versions["chase"] == "chase-v1"
    assert second.reply.startswith("I drafted 2 emails for approval:")
    assert len(asyncio.run(deps.repo.list_drafts("u_1", status="pending_approval"))) == 2


# --- approval gate (T-24) -----------------------------------------------------------


def seeded_approval(client: AsyncQdrantClient) -> tuple[AppDeps, Graph, Any]:
    """A thread that already holds three verified leads and a brand profile."""
    from datetime import UTC, datetime

    from partner_agents.schemas import BrandProfile, EmailDraft, Lead

    reply = EmailDraft(subject="Hello", body="A short intro. Priya Shah").model_dump_json()
    top3 = RouteDecision(
        intent="outreach", confidence=0.9, lead_selector="top 3", outreach_goal="intro",
        reason="email",
    )  # fmt: skip
    deps = make_deps(client, top3)
    keys = ("acme", "beta", "gamma")
    deps.writer = ByPageModel(
        messages=iter([]), pages={f"https://{k}.example/": [reply] for k in keys}
    )
    asyncio.run(
        deps.repo.save_profile(
            "u_1",
            BrandProfile(
                brand_name="Crumb & Co",
                product="cookies",
                value_prop="v",
                sender_name="Priya Shah",
                sender_email="priya@crumb.example",
                founder_name="Priya",
                category="cookies",
                sign_off="Thanks",
            ),
        )
    )
    leads = [
        Lead(
            id=k,
            run_id="r_0",
            user_id="u_1",
            company_name=f"{k.title()} Foods",
            partner_type="broker",
            location="LA",
            website=None,
            email=f"buyer@{k}.example",
            email_status="verified",
            phone=None,
            phone_status="unchecked",
            source_url=f"https://{k}.example/",
            evidence_quote="q",
            created_at=datetime.now(UTC),
        )
        for k in keys
    ]
    graph = build_graph(deps)
    config: Any = {"configurable": {"thread_id": "t_1"}}
    asyncio.run(
        graph.aupdate_state(config, {"leads": leads, "user_id": "u_1"}, as_node="scout_report")
    )
    return deps, graph, config


def test_the_run_pauses_for_approval_then_resumes_with_approve_edit_and_reject(
    client: AsyncQdrantClient,
) -> None:
    from partner_agents.graph.build import resume_turn
    from partner_agents.schemas import ApprovalDecision, EmailDraft

    deps, graph, _ = seeded_approval(client)

    async def go() -> Any:
        paused = await run_turn(graph, deps, thread_id="t_1", user_id="u_1", text="email the top 3")
        ids = {d["lead_id"]: d["id"] for d in paused.pending_approval["drafts"]}
        decision = ApprovalDecision(
            approve=[ids["acme"]],
            reject=[ids["gamma"]],
            edits={ids["beta"]: EmailDraft(subject="Edited", body="My own words. Priya Shah")},
        )
        done = await resume_turn(graph, deps, thread_id="t_1", user_id="u_1", decision=decision)
        return paused, done

    paused, done = asyncio.run(go())

    # Paused: three drafts wait, nothing is decided, and the run record is already saved.
    assert paused.pending_approval is not None
    assert [d["status"] for d in paused.pending_approval["drafts"]] == ["pending_approval"] * 3
    assert paused.reply.startswith("I drafted 3 emails for approval:")

    # Resumed: two approved (one with the edited content), one rejected.
    by_lead = {d.lead_id: d for d in done.state["drafts"]}
    assert by_lead["acme"].status == "approved" and by_lead["acme"].subject == "Hello"
    assert by_lead["beta"].status == "approved" and by_lead["beta"].subject == "Edited"
    assert (
        by_lead["beta"].body.startswith("My own words.") and "unsubscribe" in by_lead["beta"].body
    )
    assert by_lead["gamma"].status == "rejected"
    assert done.pending_approval is None and done.reply == "Approved 2, rejected 1."

    saved = {d.lead_id: d.status for d in asyncio.run(deps.repo.list_drafts("u_1"))}
    assert saved == {"acme": "approved", "beta": "approved", "gamma": "rejected"}
    assert done.run.run_id == paused.run.run_id  # one run, finished after the resume
    assert done.run.finished_at is not None and paused.run.finished_at is not None
    assert done.run.finished_at >= paused.run.finished_at


def test_an_edit_that_breaks_the_policy_pauses_again(client: AsyncQdrantClient) -> None:
    from partner_agents.graph.build import resume_turn
    from partner_agents.schemas import ApprovalDecision, EmailDraft

    deps, graph, _ = seeded_approval(client)

    async def go() -> Any:
        paused = await run_turn(graph, deps, thread_id="t_1", user_id="u_1", text="email the top 3")
        ids = [d["id"] for d in paused.pending_approval["drafts"]]
        bad = EmailDraft(subject="s", body="No sender name here.")
        second = await resume_turn(
            graph, deps, thread_id="t_1", user_id="u_1",
            decision=ApprovalDecision(approve=ids[:2], edits={ids[2]: bad}),
        )  # fmt: skip
        third = await resume_turn(
            graph, deps, thread_id="t_1", user_id="u_1",
            decision=ApprovalDecision(approve=[ids[2]]),
        )  # fmt: skip
        return second, third

    second, third = asyncio.run(go())
    assert second.pending_approval is not None
    assert [d["lead_id"] for d in second.pending_approval["drafts"]] == ["gamma"]
    assert second.pending_approval["problems"] == {
        second.pending_approval["drafts"][0]["id"]: ["missing_sender"]
    }
    assert [d.status for d in third.state["drafts"]] == ["approved"] * 3
    assert third.pending_approval is None


def test_resume_needs_a_waiting_approval_for_the_same_user(client: AsyncQdrantClient) -> None:
    from partner_agents.graph.build import NothingToApprove, resume_turn
    from partner_agents.schemas import ApprovalDecision

    deps, graph, _ = seeded_approval(client)

    async def go(thread: str, user: str) -> None:
        await resume_turn(graph, deps, thread_id=thread, user_id=user, decision=ApprovalDecision())

    with pytest.raises(NothingToApprove):  # nothing was drafted in this thread yet
        asyncio.run(go("t_1", "u_1"))

    asyncio.run(run_turn(graph, deps, thread_id="t_1", user_id="u_1", text="email the top 3"))
    with pytest.raises(NothingToApprove):  # someone else's thread
        asyncio.run(go("t_1", "u_2"))
    with pytest.raises(NothingToApprove):  # unknown thread
        asyncio.run(go("t_nope", "u_1"))


# --- the other branches -------------------------------------------------------------


def test_out_of_scope_and_clarify_end_with_their_fixed_replies(client: AsyncQdrantClient) -> None:
    clarify = RouteDecision(
        intent="clarify", confidence=0.9, clarifying_question="Which products?", reason="r"
    )
    deps = make_deps(client, decision("out_of_scope"), clarify)
    _, (refused, asked) = turns(deps, "Write me a poem", "Find partners")
    assert refused.reply == REFUSAL
    assert set(refused.run.timings_ms) == {"charlie_route", "refuse"}
    assert asked.reply == "Which products?"
    assert set(asked.run.timings_ms) == {"charlie_route", "ask_clarification"}


def test_a_question_goes_to_the_knowledge_base(client: AsyncQdrantClient) -> None:
    deps = make_deps(client, decision("cpg_qa"))
    _, (answered,) = turns(deps, "what is a slotting fee?")
    assert set(answered.run.timings_ms) == {"charlie_route", "kb_answer"}
    assert "knowledge base" in answered.reply  # the KB is empty here, so the fallback


def test_a_router_failure_is_in_the_run_record_and_the_turn_still_answers(
    client: AsyncQdrantClient,
) -> None:
    deps = make_deps(client)  # no scripted router replies -> the model raises
    _, (result,) = turns(deps, "hello")
    assert result.run.intent == "clarify"
    assert [e.stage for e in result.run.errors] == ["charlie_route"]
    assert result.reply.startswith("Sorry, I could not work out what you need")


# --- memory -------------------------------------------------------------------------


def test_the_thread_keeps_only_the_last_messages(client: AsyncQdrantClient) -> None:
    count = MAX_MESSAGES  # 20 turns -> 40 messages without trimming
    deps = make_deps(client, *[decision("out_of_scope")] * count)
    _, results = turns(deps, *[f"poem {i}" for i in range(count)])
    messages = results[-1].state["messages"]
    assert len(messages) == MAX_MESSAGES
    assert isinstance(messages[-2], HumanMessage) and messages[-2].text == f"poem {count - 1}"
    assert messages[0].text == f"poem {count - MAX_MESSAGES // 2}"
    # The router never sees more than the window it asks for.
    assert all(len(sent) <= 1 + 6 for sent in deps.router.received)  # type: ignore[attr-defined]


def test_checkpoints_revive_our_models_without_unregistered_type_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from partner_agents.graph.build import checkpoint_serde
    from partner_agents.schemas import PageRef

    serde = checkpoint_serde()
    page = PageRef(url="https://acme.example/", title=None, status="ok")  # type: ignore[arg-type]
    with caplog.at_level("WARNING"):
        revived = serde.loads_typed(serde.dumps_typed([page, DISCOVERY]))
    assert revived == [page, DISCOVERY]
    assert isinstance(revived[0], PageRef)
    assert "unregistered" not in caplog.text
