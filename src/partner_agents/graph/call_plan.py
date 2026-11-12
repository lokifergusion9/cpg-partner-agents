"""``call_plan`` (HLD §5.2, §6.9): one voice draft per selected lead, for approval.

Deterministic: no model is involved. The draft body is the call plan a human reads before
approving. Nothing is started here; ``chase_dispatch`` does that after approval.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime

from langchain_core.messages import AIMessage

from partner_agents.config import Settings
from partner_agents.graph.chase import NO_PROFILE, new_draft_id
from partner_agents.graph.state import AgentState
from partner_agents.observability import NodeUpdate, timed_node
from partner_agents.outreach.voice import build_call_plan
from partner_agents.schemas import Lead, OutreachDraft
from partner_agents.storage.repo import Repository

PLAN_VERSION = "call-plan-v1"


def make_call_plan(
    repo: Repository, settings: Settings, clock: Callable[[], datetime]
) -> Callable[[AgentState], Awaitable[NodeUpdate]]:
    @timed_node("call_plan")
    async def call_plan(state: AgentState) -> NodeUpdate:
        user_id, run_id = state.get("user_id", ""), state.get("run_id", "")
        profile = await repo.get_profile(user_id)
        if profile is None:
            return {"drafts": [], "messages": [AIMessage(content=NO_PROFILE)]}
        by_id = {lead.id: lead for lead in state.get("leads", [])}
        selected = [by_id[i] for i in state.get("selected_leads", []) if i in by_id]
        now = clock()
        provider = settings.voice_provider

        def record(lead: Lead, status: str, body: str = "") -> OutreachDraft:
            return OutreachDraft(
                id=new_draft_id(),
                lead_id=lead.id,
                user_id=user_id,
                run_id=run_id,
                channel="voice",
                recipient=lead.phone or "",  # the mock caller needs no number
                subject=None,
                body=body,
                status=status,  # type: ignore[arg-type]
                prompt_version=PLAN_VERSION,
                created_at=now,
            )

        drafts = []
        for lead in selected:
            if provider != "mock" and lead.phone_status != "verified":
                drafts.append(record(lead, "skipped_ineligible"))  # a live call needs one
            else:
                drafts.append(
                    record(lead, "pending_approval", build_call_plan(lead, profile, provider))
                )
        for draft in drafts:
            await repo.save_draft(draft)
        ready = [d for d in drafts if d.status == "pending_approval"]
        lines = [f"I planned {len(ready)} call{'s' if len(ready) != 1 else ''} for approval:"]
        lines += [f"{i}. {by_id[d.lead_id].company_name}" for i, d in enumerate(ready, start=1)]
        skipped = [by_id[d.lead_id].company_name for d in drafts if d.status != "pending_approval"]
        if skipped:
            lines.append("Skipped (no verified phone): " + ", ".join(skipped))
        if ready:
            lines.append("No call starts until you approve it.")
        return {"drafts": drafts, "messages": [AIMessage(content="\n".join(lines))]}

    return call_plan
