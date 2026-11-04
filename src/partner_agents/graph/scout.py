"""Scout's graph nodes and subgraph (HLD §5, §6.3-§6.8).

``scout_search -> scout_fetch -> scout_index -> scout_evidence -> scout_extract ->
scout_verify -> scout_report``. Dependencies are passed in as ``ScoutDeps`` and bound by
``make_*`` factories, so every node stays a plain ``state -> update`` function that tests can
call directly. The subgraph short-circuits to ``scout_report`` when there is nothing to read.
"""

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from partner_agents.config import Settings
from partner_agents.graph.state import AgentState
from partner_agents.llm import StructuredOutputError, invoke_structured
from partner_agents.observability import NodeUpdate, redact_contacts, timed_node
from partner_agents.prompts.scout import PROMPT_VERSION, build_extraction_messages
from partner_agents.retrieval.chunking import TokenCounter, chunk_text
from partner_agents.retrieval.evidence import select_evidence
from partner_agents.retrieval.store import VectorStore
from partner_agents.schemas import (
    DiscoveryIntent,
    ExtractedLead,
    ExtractionResult,
    Lead,
    PageRef,
    RetrievedChunk,
    SearchHit,
    StageError,
)
from partner_agents.storage.repo import Repository
from partner_agents.verify.contacts import verify_candidates
from partner_agents.web.fetch import Fetcher
from partner_agents.web.search import SearchService

logger = logging.getLogger(__name__)

Node = Callable[[AgentState], Awaitable[NodeUpdate]]


@dataclass
class ScoutDeps:
    """Everything the Scout nodes talk to."""

    settings: Settings
    repo: Repository
    search: SearchService
    fetcher: Fetcher
    store: VectorStore
    model: BaseChatModel  # the extractor role
    count_tokens: TokenCounter | None = None  # None -> tiktoken
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


# --- extraction (pure of graph state) -----------------------------------------------


async def extract_candidates(
    model: BaseChatModel,
    intent: DiscoveryIntent,
    evidence: Mapping[str, Sequence[RetrievedChunk]],
) -> tuple[dict[str, list[ExtractedLead]], list[StageError]]:
    """One extraction call per page, run together (``LLM_MAX_CONCURRENCY`` bounds the calls).

    A page that fails becomes a ``StageError`` and the others carry on. A page with no
    matching company gives an empty list, which is a valid answer.
    """

    async def one(url: str, chunks: Sequence[RetrievedChunk]) -> list[ExtractedLead] | StageError:
        try:
            result = await invoke_structured(
                model,
                ExtractionResult,
                build_extraction_messages(intent, url, chunks),
                role="extractor",
                prompt_version=PROMPT_VERSION,
            )
        except StructuredOutputError as exc:
            return StageError(
                stage="scout_extract", code="invalid_output", message=exc.detail, url=url
            )
        except Exception as exc:
            logger.warning(
                "extraction call failed", extra={"url": url, "error": type(exc).__name__}
            )
            return StageError(
                stage="scout_extract",
                code="model_error",
                message=f"{type(exc).__name__}: {redact_contacts(str(exc))[:200]}",
                url=url,
            )
        return result.leads

    pages = [(url, chunks) for url, chunks in evidence.items() if chunks]
    outcomes = await asyncio.gather(*(one(url, chunks) for url, chunks in pages))
    candidates: dict[str, list[ExtractedLead]] = {}
    errors: list[StageError] = []
    for (url, _), outcome in zip(pages, outcomes, strict=True):
        if isinstance(outcome, StageError):
            errors.append(outcome)
        else:
            candidates[url] = outcome
    return candidates, errors


# --- report (pure) ------------------------------------------------------------------


def build_report(leads: Sequence[Lead], pages: Sequence[PageRef], hits: Sequence[SearchHit]) -> str:
    """The chat reply for a finished Scout run."""
    if not hits:
        return (
            "I could not find any search results for that. "
            "Try a broader query (fewer words, or a larger region)."
        )
    readable = [page for page in pages if page.status == "ok"]
    if not readable:
        reasons = Counter(page.status for page in pages)
        detail = ", ".join(f"{status}: {count}" for status, count in sorted(reasons.items()))
        return (
            f"I found {len(hits)} candidate sites but could not read any of them ({detail}). "
            "Try a broader query."
        )
    eligible = sum(lead.outreach_eligible for lead in leads)
    if not leads:
        return (
            f"I read {len(readable)} pages but found no matching companies on them. "
            "Try a broader query."
        )
    noun = "company" if len(leads) == 1 else "companies"
    verb = "has" if eligible == 1 else "have"
    text = f"Found {len(leads)} {noun}; {eligible} {verb} a verified email."
    unread = len(pages) - len(readable)
    if unread:
        text += f" {unread} of {len(pages)} pages could not be read."
    return text


# --- nodes --------------------------------------------------------------------------


def _intent(state: AgentState, node: str) -> DiscoveryIntent:
    route = state.get("route")
    if route is None or route.discovery is None:
        raise ValueError(f"{node} needs state['route'].discovery")
    return route.discovery


def make_scout_search(deps: ScoutDeps) -> Node:
    @timed_node("scout_search")
    async def scout_search(state: AgentState) -> NodeUpdate:
        outcome = await deps.search.discover(_intent(state, "scout_search"))
        return {"search_hits": outcome.hits, "errors": outcome.errors}

    return scout_search


def make_scout_fetch(deps: ScoutDeps) -> Node:
    @timed_node("scout_fetch")
    async def scout_fetch(state: AgentState) -> NodeUpdate:
        urls = [str(hit.url) for hit in state.get("search_hits", [])]
        outcomes = await deps.fetcher.fetch_many(urls, state["run_id"])
        return {
            "pages": [outcome.page for outcome in outcomes],
            "errors": [outcome.error for outcome in outcomes if outcome.error is not None],
        }

    return scout_fetch


def make_scout_index(deps: ScoutDeps) -> Node:
    @timed_node("scout_index")
    async def scout_index(state: AgentState) -> NodeUpdate:
        run_id = state["run_id"]
        errors: list[StageError] = []
        for page in state.get("pages", []):
            if page.status != "ok":
                continue
            url = str(page.url)
            record = await deps.repo.get_page(run_id, url)
            if record is None:
                errors.append(
                    StageError(
                        stage="scout_index", code="page_missing", message="page expired", url=url
                    )
                )
                continue
            text = f"{record.text}\n\n{record.contact_text}".strip()
            chunks = await asyncio.to_thread(
                chunk_text,
                text,
                max_tokens=deps.settings.chunk_tokens,
                overlap_tokens=deps.settings.chunk_overlap,
                count=deps.count_tokens,
            )
            await deps.store.upsert_chunks(run_id, url, page.title, chunks, record.fetched_at)
        return {"errors": errors}

    return scout_index


def make_scout_evidence(deps: ScoutDeps) -> Node:
    @timed_node("scout_evidence")
    async def scout_evidence(state: AgentState) -> NodeUpdate:
        intent = _intent(state, "scout_evidence")
        urls = [str(p.url) for p in state.get("pages", []) if p.status == "ok"]
        chosen = await asyncio.gather(
            *(
                select_evidence(
                    deps.store,
                    intent,
                    state["run_id"],
                    url,
                    k=deps.settings.evidence_k,
                    max_total=deps.settings.evidence_max,
                )
                for url in urls
            )
        )
        return {"evidence": {url: chunks for url, chunks in zip(urls, chosen, strict=True)}}

    return scout_evidence


def make_scout_extract(model: BaseChatModel) -> Node:
    """The ``scout_extract`` node: ``evidence`` + the discovery intent -> ``candidates``."""

    @timed_node("scout_extract")
    async def scout_extract(state: AgentState) -> NodeUpdate:
        intent = _intent(state, "scout_extract")
        candidates, errors = await extract_candidates(model, intent, state.get("evidence", {}))
        return {"candidates": candidates, "errors": errors}

    return scout_extract


def make_scout_verify(deps: ScoutDeps) -> Node:
    @timed_node("scout_verify")
    async def scout_verify(state: AgentState) -> NodeUpdate:
        intent = _intent(state, "scout_verify")
        result = await verify_candidates(
            state.get("candidates", {}),
            deps.repo,
            run_id=state["run_id"],
            user_id=state.get("user_id", "anonymous"),
            settings=deps.settings,
            intent_location=intent.location,
            now=deps.clock,
        )
        return {"leads": result.leads, "errors": result.errors}

    return scout_verify


def make_scout_report(deps: ScoutDeps) -> Node:
    @timed_node("scout_report")
    async def scout_report(state: AgentState) -> NodeUpdate:
        leads = state.get("leads", [])
        await deps.repo.save_leads(leads)
        reply = build_report(leads, state.get("pages", []), state.get("search_hits", []))
        return {"leads": leads, "messages": [AIMessage(content=reply)]}

    return scout_report


# --- subgraph -----------------------------------------------------------------------


def _after_search(state: AgentState) -> str:
    return "scout_fetch" if state.get("search_hits") else "scout_report"


def _after_fetch(state: AgentState) -> str:
    readable = any(page.status == "ok" for page in state.get("pages", []))
    return "scout_index" if readable else "scout_report"


def build_scout_graph(deps: ScoutDeps) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    """The compiled Scout subgraph over ``AgentState`` (mounted as a node by T-21)."""
    graph = StateGraph(AgentState)
    nodes: dict[str, Node] = {
        "scout_search": make_scout_search(deps),
        "scout_fetch": make_scout_fetch(deps),
        "scout_index": make_scout_index(deps),
        "scout_evidence": make_scout_evidence(deps),
        "scout_extract": make_scout_extract(deps.model),
        "scout_verify": make_scout_verify(deps),
        "scout_report": make_scout_report(deps),
    }
    for name, node in nodes.items():
        # LangGraph's overloads for add_node do not accept a plain async callable over a
        # TypedDict state under strict mypy, although it is exactly what they run.
        graph.add_node(name, cast(Any, node))
    graph.add_edge(START, "scout_search")
    graph.add_conditional_edges("scout_search", _after_search, ["scout_fetch", "scout_report"])
    graph.add_conditional_edges("scout_fetch", _after_fetch, ["scout_index", "scout_report"])
    graph.add_edge("scout_index", "scout_evidence")
    graph.add_edge("scout_evidence", "scout_extract")
    graph.add_edge("scout_extract", "scout_verify")
    graph.add_edge("scout_verify", "scout_report")
    graph.add_edge("scout_report", END)
    return graph.compile()
