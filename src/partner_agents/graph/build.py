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
from pydantic import BaseModel

from partner_agents import schemas
from partner_agents.config import Settings
from partner_agents.graph import charlie
from partner_agents.graph.chase import chase_select
from partner_agents.graph.scout import ScoutDeps, add_scout_nodes
from partner_agents.graph.state import AgentState
from partner_agents.llm import get_chat_model, track_providers, track_usage
from partner_agents.observability import run_context
from partner_agents.prompts import charlie as charlie_prompts
from partner_agents.prompts import scout as scout_prompts
from partner_agents.retrieval.embeddings import SentenceTransformerEmbedder
from partner_agents.retrieval.store import KnowledgeStore, VectorStore, make_client
from partner_agents.schemas import RunRecord, StageError
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
        "chase_select": chase_select,
    }
    for name, node in nodes.items():
        graph.add_node(name, cast(Any, node))  # see the note in graph/scout.py
    add_scout_nodes(graph, deps.scout)
    graph.add_edge(START, "charlie_route")
    graph.add_conditional_edges(
        "charlie_route", charlie.next_after_route, list(charlie.NEXT_NODE.values())
    )
    for name in ("kb_answer", "ask_clarification", "refuse", "chase_select"):
        graph.add_edge(name, END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver(serde=checkpoint_serde()))


@dataclass
class TurnResult:
    reply: str
    run: RunRecord
    state: dict[str, Any]


def new_run_id() -> str:
    return f"r_{uuid.uuid4().hex[:12]}"


async def run_turn(
    graph: Graph, deps: AppDeps, *, thread_id: str, user_id: str, text: str
) -> TurnResult:
    """Run one user message through the graph in ``thread_id`` and persist its ``RunRecord``."""
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
    timings: dict[str, int] = {}
    errors: list[StageError] = []
    prompt_versions: dict[str, str] = {}
    started = deps.clock()
    with run_context(run_id), track_usage() as tokens, track_providers() as providers:
        async for chunk in graph.astream(turn_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue  # an interrupt marker, not a node update
                timings.update(update.get("timings_ms", {}))
                errors.extend(update.get("errors", []))
                if node in _PROMPTS:
                    module, version = _PROMPTS[node]
                    prompt_versions[module] = version
    state = dict((await graph.aget_state(config)).values)
    route = state.get("route")
    run = RunRecord(
        run_id=run_id,
        user_id=user_id,
        thread_id=thread_id,
        intent=route.intent if route is not None else "clarify",
        started_at=started,
        finished_at=deps.clock(),
        timings_ms=timings,
        tokens=dict(tokens),
        prompt_versions=prompt_versions,
        errors=errors,
        providers=dict(providers),
    )
    await deps.repo.save_run(run)
    last = state.get("messages", [])[-1:] or [AIMessage(content="")]
    reply = str(last[0].content) if isinstance(last[0], AIMessage) else ""
    return TurnResult(reply=reply, run=run, state=state)
