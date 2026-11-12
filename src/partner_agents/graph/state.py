"""Graph state shared by Charlie, Scout and Chase (HLD §5.1).

The state holds references, not page bodies: page text lives in ``repo.pages`` so checkpoints
stay small.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from partner_agents.schemas import (
    ExtractedLead,
    Lead,
    OutreachDraft,
    PageRef,
    RetrievedChunk,
    RouteDecision,
    SearchHit,
    StageError,
)


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer for ``timings_ms``: later keys win, nothing is dropped."""
    return {**(left or {}), **(right or {})}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # trimmed to the last 20 before LLM calls
    user_id: str
    thread_id: str
    run_id: str  # new per user turn
    rate_keys: list[str]  # who this turn counts against ("user:<id>", "ip:<addr>"), set per turn
    retry_after_s: int | None  # set by rate_gate when the turn was refused; reset per turn
    route: RouteDecision
    search_hits: list[SearchHit]
    pages: list[PageRef]  # refs only; bodies in repo.pages
    evidence: dict[str, list[RetrievedChunk]]  # url -> selected chunks
    candidates: dict[str, list[ExtractedLead]]  # url -> what the extractor read on that page
    leads: list[Lead]  # verified AND rejected, for audit + eval
    selected_leads: list[str]  # lead ids chase_select resolved from route.lead_selector
    drafts: list[OutreachDraft]
    call_ids: list[str]  # calls created by chase_dispatch for approved voice drafts
    errors: Annotated[list[StageError], operator.add]
    timings_ms: Annotated[dict[str, int], merge_dicts]
