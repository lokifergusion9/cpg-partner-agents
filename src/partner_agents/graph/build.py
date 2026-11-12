"""The main graph (HLD §5) and one user turn through it (§6.10).

``build_graph`` wires Charlie's router to its five branches. The graph is compiled with a
checkpointer keyed by ``thread_id`` (``InMemorySaver`` by default; the MongoDB saver arrives
with T-27), so a follow-up in the same thread sees the earlier messages and leads.

``run_turn`` runs one user message: a new ``run_id``, the thread's history trimmed to the
last ``MAX_MESSAGES`` messages, and a ``RunRecord`` (timings of every node that ran, errors,
tokens and providers per role, prompt versions) saved at the end.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel

from partner_agents import schemas
from partner_agents.config import Settings
from partner_agents.graph import approval, charlie, chase
from partner_agents.graph.scout import ScoutDeps, add_scout_nodes
from partner_agents.graph.state import AgentState
from partner_agents.llm import get_chat_model, track_providers, track_usage
from partner_agents.observability import run_context
from partner_agents.outreach.policy import TokenBucket
from partner_agents.prompts import charlie as charlie_prompts
from partner_agents.prompts import chase as chase_prompts
from partner_agents.prompts import scout as scout_prompts
from partner_agents.retrieval.embeddings import SentenceTransformerEmbedder
from partner_agents.retrieval.store import KnowledgeStore, VectorStore, make_client
from partner_agents.schemas import ApprovalDecision, RunRecord, StageError
from partner_agents.storage.repo import InMemoryRepository, Repository
from partner_agents.web.fetch import Fetcher
from partner_agents.web.search import SearchService, build_providers

Graph = CompiledStateGraph[AgentState, Any, Any, Any]

# Messages kept in a thread's checkpoint (HLD §5.1, §6.10); older ones are removed.
MAX_MESSAGES = 20

# Which prompt module each node uses, for RunRecord.prompt_versions.
_PROMPTS: dict[str, tuple[str, str]] = {
    "charlie_route": ("charlie", charlie_prompts.PROMPT_VERSION),
    "kb_answer": ("charlie", charlie_prompts.PROMPT_VERSION),
    "scout_extract": ("scout", scout_prompts.PROMPT_VERSION),
    "chase_draft": ("chase", chase_prompts.PROMPT_VERSION),
}


@dataclass
class AppDeps:
    """Everything the main graph talks to. ``scout`` carries the extractor model."""

    settings: Settings
    repo: Repository
    scout: ScoutDeps
    kb: KnowledgeStore
    router: BaseChatModel
    writer: BaseChatModel
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    # Each user's daily outreach token bucket. Process memory: a restart refills it (HLD §6.9).
    outreach_buckets: dict[str, TokenBucket] = field(default_factory=dict)

    async def aclose(self) -> None:
        await self.scout.fetcher.aclose()
        await self.scout.store.close()  # the knowledge store shares this Qdrant client


def build_app_deps(settings: Settings) -> AppDeps:
    """Real dependencies from settings. One Qdrant client serves both collections, because
    the local on-disk store allows only one client per folder."""
    repo = InMemoryRepository()  # MongoRepository arrives with T-27
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    client = make_client(settings)
    scout = ScoutDeps(
        settings=settings,
        repo=repo,
        search=SearchService(build_providers(settings), repo, settings.effective_search_blocklist),
        fetcher=Fetcher(settings, repo),
        store=VectorStore(client, embedder),
        model=get_chat_model("extractor", settings=settings),
    )
    return AppDeps(
        settings=settings,
        repo=repo,
        scout=scout,
        kb=KnowledgeStore(client, embedder),
        router=get_chat_model("router", settings=settings),
        writer=get_chat_model("writer", settings=settings),
    )


def checkpoint_serde() -> JsonPlusSerializer:
    """The checkpoint serializer with our schema models allowlisted by name.

    LangGraph revives only allowlisted types from a checkpoint (unlisted types are warned
    about now and will be refused later), so a tampered checkpoint cannot name an arbitrary
    class. The T-27 MongoDB saver must use the same serializer.
    """
    allowed = [
        (schemas.__name__, name)
        for name, obj in vars(schemas).items()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == schemas.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def build_graph(deps: AppDeps, checkpointer: BaseCheckpointSaver[Any] | None = None) -> Graph:
    graph = StateGraph(AgentState)
    nodes: dict[str, Any] = {
        "charlie_route": charlie.make_charlie_route(deps.router, deps.settings),
        "kb_answer": charlie.make_kb_answer(deps.kb, deps.writer, deps.settings),
        "ask_clarification": charlie.ask_clarification,
        "refuse": charlie.refuse,
        "chase_select": chase.make_chase_select(deps.writer),
        "chase_draft": chase.make_chase_draft(
            deps.writer, deps.repo, deps.settings, deps.outreach_buckets, deps.clock
        ),
        "approval_gate": approval.make_approval_gate(deps.repo, deps.clock),
    }
    for name, node in nodes.items():
        graph.add_node(name, cast(Any, node))  # see the note in graph/scout.py
    add_scout_nodes(graph, deps.scout)
    graph.add_edge(START, "charlie_route")
    graph.add_conditional_edges(
        "charlie_route", charlie.next_after_route, list(charlie.NEXT_NODE.values())
    )
    graph.add_conditional_edges("chase_select", chase.next_after_select, ["chase_draft", END])
    graph.add_conditional_edges("chase_draft", approval.next_after_draft, ["approval_gate", END])
    # Until T-25 adds chase_dispatch, the turn ends after the gate (approved drafts wait).
    for name in ("kb_answer", "ask_clarification", "refuse", "approval_gate"):
        graph.add_edge(name, END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver(serde=checkpoint_serde()))


@dataclass
class TurnResult:
    reply: str
    run: RunRecord
    state: dict[str, Any]
    # The payload of a pending approval ({"drafts": [...], "problems": {...}}), else None.
    pending_approval: dict[str, Any] | None = None


class NothingToApprove(ValueError):
    """``resume_turn`` was called for a thread that is not waiting for an approval."""


def new_run_id() -> str:
    return f"r_{uuid.uuid4().hex[:12]}"


@dataclass
class _Observed:
    timings: dict[str, int] = field(default_factory=dict)
    errors: list[StageError] = field(default_factory=list)
    prompt_versions: dict[str, str] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    providers: dict[str, str] = field(default_factory=dict)


async def _drive(graph: Graph, config: RunnableConfig, graph_input: Any, run_id: str) -> _Observed:
    """Stream the graph once (a new turn or a resume) and collect what the RunRecord needs."""
    seen = _Observed()
    with run_context(run_id), track_usage() as tokens, track_providers() as providers:
        async for chunk in graph.astream(graph_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue  # an interrupt marker, not a node update
                seen.timings.update(update.get("timings_ms", {}))
                seen.errors.extend(update.get("errors", []))
                if node in _PROMPTS:
                    module, version = _PROMPTS[node]
                    seen.prompt_versions[module] = version
    seen.tokens, seen.providers = dict(tokens), dict(providers)
    return seen


async def _result(graph: Graph, config: RunnableConfig, run: RunRecord) -> TurnResult:
    snapshot = await graph.aget_state(config)
    state = dict(snapshot.values)
    waiting = snapshot.interrupts[0].value if snapshot.interrupts else None
    last = state.get("messages", [])[-1:] or [AIMessage(content="")]
    reply = str(last[0].content) if isinstance(last[0], AIMessage) else ""
    return TurnResult(reply=reply, run=run, state=state, pending_approval=waiting)


async def run_turn(
    graph: Graph, deps: AppDeps, *, thread_id: str, user_id: str, text: str
) -> TurnResult:
    """Run one user message through the graph in ``thread_id`` and persist its ``RunRecord``.

    If the turn ends at the approval gate, ``TurnResult.pending_approval`` holds the drafts.
    """
    run_id = new_run_id()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    history = (await graph.aget_state(config)).values.get("messages", [])
    # A turn adds the user's message and one reply, so keep room for both.
    stale = history[: max(0, len(history) - (MAX_MESSAGES - 2))]
    turn_input: dict[str, Any] = {
        "messages": [*(RemoveMessage(id=str(m.id)) for m in stale), HumanMessage(content=text)],
        "user_id": user_id,
        "thread_id": thread_id,
        "run_id": run_id,
    }
    started = deps.clock()
    seen = await _drive(graph, config, turn_input, run_id)
    route = (await graph.aget_state(config)).values.get("route")
    run = RunRecord(
        run_id=run_id,
        user_id=user_id,
        thread_id=thread_id,
        intent=route.intent if route is not None else "clarify",
        started_at=started,
        finished_at=deps.clock(),
        timings_ms=seen.timings,
        tokens=seen.tokens,
        prompt_versions=seen.prompt_versions,
        errors=seen.errors,
        providers=seen.providers,
    )
    await deps.repo.save_run(run)
    return await _result(graph, config, run)


async def resume_turn(
    graph: Graph, deps: AppDeps, *, thread_id: str, user_id: str, decision: ApprovalDecision
) -> TurnResult:
    """Resume the thread's paused approval with ``decision``.

    The resumed part belongs to the run that paused: its RunRecord gets the new timings,
    errors, tokens and a later ``finished_at``.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    if "approval_gate" not in snapshot.next:
        raise NothingToApprove(f"thread {thread_id} has no drafts waiting for approval")
    if snapshot.values.get("user_id") != user_id:
        raise NothingToApprove(f"thread {thread_id} has no drafts waiting for approval")
    run_id = str(snapshot.values["run_id"])
    seen = await _drive(graph, config, Command(resume=decision), run_id)
    run = await deps.repo.get_run(run_id)
    if run is None:  # the first part was never saved; should not happen
        raise NothingToApprove(f"run {run_id} is unknown")
    run = run.model_copy(
        update={
            "finished_at": deps.clock(),
            "timings_ms": {**run.timings_ms, **seen.timings},
            "errors": [*run.errors, *seen.errors],
            "tokens": {
                k: run.tokens.get(k, 0) + seen.tokens.get(k, 0) for k in {*run.tokens, *seen.tokens}
            },
            "providers": {**run.providers, **seen.providers},
        }
    )
    await deps.repo.save_run(run)
    return await _result(graph, config, run)
