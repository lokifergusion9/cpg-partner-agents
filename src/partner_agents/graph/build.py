"""The main graph (HLD §5) and one user turn through it (§6.10).

``build_graph`` wires Charlie's router to its branches. The graph is compiled with a
checkpointer keyed by ``thread_id`` (``InMemorySaver`` by default, the MongoDB saver when
``MONGODB_URI`` is set), so a follow-up in the same thread sees the earlier messages and leads.

``run_turn`` runs one user message: a new ``run_id``, the thread's history trimmed to the
last ``MAX_MESSAGES`` messages, and a ``RunRecord`` (timings of every node that ran, errors,
tokens and providers per role, prompt versions) saved at the end.
"""

import asyncio
import logging
import uuid
import weakref
from collections.abc import Callable, Sequence
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
from partner_agents.calls.playbook import load_playbook
from partner_agents.calls.reply import CheckedReply
from partner_agents.calls.speech import SpeechBackend, SpeechUnavailable, build_speech
from partner_agents.config import Settings
from partner_agents.graph import approval, call_plan, charlie, chase, dispatch, rate_gate
from partner_agents.graph.call import CallDeps, CallGraph, build_call_graph
from partner_agents.graph.rate_gate import RunLimiter
from partner_agents.graph.scout import ScoutDeps, add_scout_nodes
from partner_agents.graph.state import AgentState
from partner_agents.llm import get_chat_model, track_providers, track_usage
from partner_agents.observability import run_context
from partner_agents.outreach.email import GmailClient, HttpxGmailClient
from partner_agents.outreach.policy import TokenBucket
from partner_agents.outreach.voice import MockVoiceCaller, VoiceCaller
from partner_agents.prompts import charlie as charlie_prompts
from partner_agents.prompts import chase as chase_prompts
from partner_agents.prompts import scout as scout_prompts
from partner_agents.retrieval.embeddings import SentenceTransformerEmbedder
from partner_agents.retrieval.store import KnowledgeStore, VectorStore, make_client
from partner_agents.schemas import ApprovalDecision, RunRecord, StageError
from partner_agents.storage.mongo import build_repository
from partner_agents.storage.repo import Repository
from partner_agents.tracing import (
    RunTrace,
    Tracing,
    TracingUnavailable,
    build_tracing,
    finish_trace,
    start_trace,
    traced_config,
)
from partner_agents.utils.cache_stats import track_cache
from partner_agents.web.fetch import Fetcher
from partner_agents.web.search import SearchService, build_providers

logger = logging.getLogger(__name__)

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
# Every prompt module and its version, for the trace metadata (T-33).
PROMPT_VERSIONS = dict(_PROMPTS.values())


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
    # Only the gmail_draft and gmail_send modes use it; outbox needs no client.
    gmail: GmailClient | None = None
    # A durable saver (MongoDB) when MONGODB_URI is set; otherwise build_graph uses InMemorySaver.
    checkpointer: BaseCheckpointSaver[Any] | None = None
    # The mock caller dials nobody. A live provider has no caller yet (stretch), so it fails.
    voice: VoiceCaller | None = None
    # Limits discovery runs per key. The API sets one; the CLI and tests run without.
    rate_limiter: RunLimiter | None = None
    # The call agent's models and playbook (HLD §6.11). Without it the /calls routes are off.
    calls: CallDeps | None = None
    # Local speech for ``live_voice`` (ADR-021); None unless SPEECH_ENABLED and it loaded.
    speech: SpeechBackend | None = None
    # Langfuse, when both keys are set and the tracing extra is installed (ADR-029).
    tracing: Tracing | None = None

    async def aclose(self) -> None:
        if self.tracing is not None:
            await asyncio.to_thread(self.tracing.shutdown)  # send what is still queued
        await self.scout.fetcher.aclose()
        await self.scout.store.close()  # the knowledge store shares this Qdrant client
        close_repo = getattr(self.repo, "aclose", None)
        if close_repo is not None:
            await close_repo()
        saver_client = getattr(self.checkpointer, "client", None)
        if saver_client is not None:
            saver_client.close()


def build_app_deps(settings: Settings) -> AppDeps:
    """Real dependencies from settings. One Qdrant client serves both collections, because
    the local on-disk store allows only one client per folder."""
    repo = build_repository(settings)  # MongoDB with MONGODB_URI, else in memory
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
    calls = CallDeps(
        settings=settings,
        repo=repo,
        playbook=load_playbook(settings.call_playbook),
        classifier=get_chat_model("classifier", settings=settings),
        caller=get_chat_model("caller", settings=settings),
        simulator=get_chat_model("simulator", settings=settings),
    )
    mock = MockVoiceCaller(repo, calls.playbook, calls.clock)
    try:
        speech = build_speech(settings)
    except SpeechUnavailable as exc:  # the app still runs, in text only
        logger.warning("speech is off: %s", exc)
        speech = None
    try:
        tracing = build_tracing(settings)
    except TracingUnavailable as exc:  # the app still runs, untraced
        logger.warning("tracing is off: %s", exc)
        tracing = None
    calls.tracing = tracing
    return AppDeps(
        settings=settings,
        repo=repo,
        scout=scout,
        kb=KnowledgeStore(client, embedder),
        router=get_chat_model("router", settings=settings),
        writer=get_chat_model("writer", settings=settings),
        gmail=HttpxGmailClient(settings) if settings.outreach_mode != "outbox" else None,
        voice=mock if settings.voice_provider == "mock" else None,
        checkpointer=build_checkpointer(settings),
        calls=calls,
        speech=speech,
        tracing=tracing,
    )


def build_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any] | None:
    """The MongoDB saver when ``MONGODB_URI`` is set (ADR-028), else ``None`` (in memory).

    It uses the same allowlisted serializer as the default saver, so a stored checkpoint
    cannot revive an arbitrary class. Checkpoints expire with ``DATA_TTL_DAYS``.
    """
    if settings.mongodb_uri is None:
        return None
    from langgraph.checkpoint.mongodb import MongoDBSaver
    from pymongo import MongoClient

    client: Any = MongoClient(settings.mongodb_uri.get_secret_value(), tz_aware=True)
    return MongoDBSaver(
        client,
        db_name=settings.mongodb_database,
        serde=checkpoint_serde(),
        ttl=settings.data_ttl_days * 86400,
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
    allowed.append((CheckedReply.__module__, CheckedReply.__name__))  # the call sub-graph state
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
        "rate_gate": rate_gate.make_rate_gate(deps.rate_limiter),
        "call_plan": call_plan.make_call_plan(deps.repo, deps.settings, deps.clock),
        "chase_dispatch": dispatch.make_chase_dispatch(
            deps.repo, deps.settings, deps.gmail, deps.voice, deps.clock
        ),
    }
    for name, node in nodes.items():
        graph.add_node(name, cast(Any, node))  # see the note in graph/scout.py
    add_scout_nodes(graph, deps.scout)
    graph.add_edge(START, "charlie_route")
    graph.add_conditional_edges(
        "charlie_route", charlie.next_after_route, list(charlie.NEXT_NODE.values())
    )
    graph.add_conditional_edges("rate_gate", rate_gate.next_after_gate, ["scout_search", END])
    graph.add_conditional_edges(
        "chase_select", chase.next_after_select, ["chase_draft", "call_plan", END]
    )
    for name in ("chase_draft", "call_plan"):
        graph.add_conditional_edges(name, approval.next_after_draft, ["approval_gate", END])
    graph.add_edge("approval_gate", "chase_dispatch")
    for name in ("kb_answer", "ask_clarification", "refuse", "chase_dispatch"):
        graph.add_edge(name, END)
    saver = checkpointer or deps.checkpointer or InMemorySaver(serde=checkpoint_serde())
    return graph.compile(checkpointer=saver)


@dataclass
class Context:
    """A running app: the dependencies, the compiled graph and per-thread locks.

    Shared by the API and the UI, so both serialise the turns of one thread the same way.
    """

    deps: AppDeps
    graph: Graph
    locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = field(
        default_factory=weakref.WeakValueDictionary
    )
    # The call sub-graph, compiled from ``deps.calls`` (its own checkpoint threads: call ids).
    call_graph: CallGraph | None = None

    def __post_init__(self) -> None:
        if self.call_graph is None and self.deps is not None and self.deps.calls is not None:
            self.call_graph = build_call_graph(self.deps.calls)

    def lock(self, thread_id: str) -> asyncio.Lock:
        """One turn at a time per thread, so two requests cannot interleave its state."""
        lock = self.locks.get(thread_id)
        if lock is None:
            lock = self.locks[thread_id] = asyncio.Lock()
        return lock


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
    cache: dict[str, int] = field(default_factory=dict)
    trace_url: str | None = None


def _start_trace(
    deps: "AppDeps", name: str, thread_id: str, user_id: str, run_id: str
) -> RunTrace | None:
    return start_trace(
        deps.tracing,
        name=name,
        session_id=thread_id,
        user_id=user_id,
        metadata={"run_id": run_id, "prompt_versions": PROMPT_VERSIONS},
    )


async def _drive(
    graph: Graph,
    config: RunnableConfig,
    graph_input: Any,
    run_id: str,
    on_node: Callable[[str], None] | None = None,
    trace: RunTrace | None = None,
) -> _Observed:
    """Stream the graph once (a new turn or a resume) and collect what the RunRecord needs.

    With a ``trace`` its callback handler is attached, so every node becomes a span (T-33).
    """
    seen = _Observed()
    traced = cast(RunnableConfig, traced_config(config, trace))
    with (
        run_context(run_id),
        track_usage() as tokens,
        track_providers() as providers,
        track_cache() as cache,
    ):
        async for chunk in graph.astream(graph_input, traced, stream_mode="updates"):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue  # an interrupt marker, not a node update
                if on_node is not None:
                    on_node(node)
                seen.timings.update(update.get("timings_ms", {}))
                seen.errors.extend(update.get("errors", []))
                if node in _PROMPTS:
                    module, version = _PROMPTS[node]
                    seen.prompt_versions[module] = version
    seen.tokens, seen.providers, seen.cache = dict(tokens), dict(providers), dict(cache)
    seen.trace_url = await finish_trace(trace)
    return seen


async def _result(graph: Graph, config: RunnableConfig, run: RunRecord) -> TurnResult:
    snapshot = await graph.aget_state(config)
    state = dict(snapshot.values)
    waiting = snapshot.interrupts[0].value if snapshot.interrupts else None
    last = state.get("messages", [])[-1:] or [AIMessage(content="")]
    reply = str(last[0].content) if isinstance(last[0], AIMessage) else ""
    return TurnResult(reply=reply, run=run, state=state, pending_approval=waiting)


async def run_turn(
    graph: Graph,
    deps: AppDeps,
    *,
    thread_id: str,
    user_id: str,
    text: str,
    rate_keys: Sequence[str] | None = None,
    on_node: Callable[[str], None] | None = None,
) -> TurnResult:
    """Run one user message through the graph in ``thread_id`` and persist its ``RunRecord``.

    If the turn ends at the approval gate, ``TurnResult.pending_approval`` holds the drafts.
    ``rate_keys`` name who a discovery run counts against (default: the user). ``on_node`` is
    called with the name of each node as it finishes (the UI shows it as progress).
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
        "rate_keys": list(rate_keys or [f"user:{user_id}"]),
        "retry_after_s": None,  # a refusal in an earlier turn must not stick
    }
    started = deps.clock()
    trace = _start_trace(deps, "chat_turn", thread_id, user_id, run_id)
    seen = await _drive(graph, config, turn_input, run_id, on_node, trace)
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
        cache=seen.cache,
        trace_url=seen.trace_url,
    )
    await deps.repo.save_run(run)
    return await _result(graph, config, run)


async def resume_turn(
    graph: Graph,
    deps: AppDeps,
    *,
    thread_id: str,
    user_id: str,
    decision: ApprovalDecision,
    on_node: Callable[[str], None] | None = None,
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
    trace = _start_trace(deps, "approval_resume", thread_id, user_id, run_id)
    seen = await _drive(graph, config, Command(resume=decision), run_id, on_node, trace)
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
            "cache": {
                k: run.cache.get(k, 0) + seen.cache.get(k, 0) for k in {*run.cache, *seen.cache}
            },
            "trace_url": run.trace_url or seen.trace_url,  # the resume has its own trace
        }
    )
    await deps.repo.save_run(run)
    return await _result(graph, config, run)
