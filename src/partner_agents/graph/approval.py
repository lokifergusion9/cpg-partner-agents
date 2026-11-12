"""The approval gate (HLD §6.9): the graph pauses until a human decides on the drafts.

``approval_gate`` calls ``interrupt()`` with the pending drafts. The run resumes through
``Command(resume=ApprovalDecision)`` in the same thread (``build.resume_turn``). LangGraph
re-runs the node from its start on resume, so everything before ``interrupt()`` is cheap and
free of side effects; the repo is written only once every draft is decided.

This node is not wrapped in ``timed_node``: that wrapper turns exceptions into stage errors, and
a pause is an exception (``GraphInterrupt``). Waiting for a human is not node time either.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import interrupt

from partner_agents.graph.state import AgentState
from partner_agents.observability import NodeUpdate
from partner_agents.outreach.approval import resolve
from partner_agents.schemas import ApprovalDecision, OutreachDraft
from partner_agents.storage.repo import Repository


def pending_drafts(state: AgentState) -> list[OutreachDraft]:
    return [d for d in state.get("drafts", []) if d.status == "pending_approval"]


def next_after_draft(state: AgentState) -> str:
    """Conditional edge out of ``chase_draft``: pause for approval only if a draft is waiting."""
    return "approval_gate" if pending_drafts(state) else END


def approval_payload(drafts: list[OutreachDraft], problems: dict[str, list[str]]) -> dict[str, Any]:
    return {"drafts": [d.model_dump(mode="json") for d in drafts], "problems": problems}


def make_approval_gate(
    repo: Repository, clock: Callable[[], datetime]
) -> Callable[[AgentState], Awaitable[NodeUpdate]]:
    async def approval_gate(state: AgentState) -> NodeUpdate:
        pending = pending_drafts(state)
        if not pending:
            return {}
        profile = await repo.get_profile(state.get("user_id", ""))
        sender = profile.sender_name if profile else ""
        decided: list[OutreachDraft] = []
        problems: dict[str, list[str]] = {}
        # One round per interrupt. An edit that breaks the policy is asked about again.
        while pending:
            answer = interrupt(approval_payload(pending, problems))
            decision = ApprovalDecision.model_validate(answer)
            result = resolve(pending, decision, sender_name=sender, now=clock())
            decided += result.decided
            pending, problems = result.unresolved, result.problems
        for draft in decided:
            await repo.save_draft(draft)
        by_id = {d.id: d for d in decided}
        drafts = [by_id.get(d.id, d) for d in state.get("drafts", [])]
        approved = sum(d.status == "approved" for d in decided)
        reply = f"Approved {approved}, rejected {len(decided) - approved}."
        return {"drafts": drafts, "messages": [AIMessage(content=reply)]}

    return approval_gate
