"""``chase_dispatch`` (HLD §6.9): deliver the drafts a human approved, nothing else."""

from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import datetime

from langchain_core.messages import AIMessage

from partner_agents.config import Settings
from partner_agents.graph.state import AgentState
from partner_agents.observability import NodeUpdate, timed_node
from partner_agents.outreach.email import GmailClient, dispatch_email
from partner_agents.schemas import OutreachDraft, StageError
from partner_agents.storage.repo import Repository

_OUTCOME = {
    "outboxed": "saved to the outbox (nothing was sent)",
    "drafted_in_gmail": "created as drafts in your Gmail (nothing was sent)",
    "sent": "sent",
}


def make_chase_dispatch(
    repo: Repository,
    settings: Settings,
    gmail: GmailClient | None,
    clock: Callable[[], datetime],
) -> Callable[[AgentState], Awaitable[NodeUpdate]]:
    @timed_node("chase_dispatch")
    async def chase_dispatch(state: AgentState) -> NodeUpdate:
        drafts = state.get("drafts", [])
        approved = [d for d in drafts if d.status == "approved" and d.channel == "email"]
        if not approved:
            return {}
        profile = await repo.get_profile(state.get("user_id", ""))
        sender = str(profile.sender_email) if profile else ""
        done: Counter[str] = Counter()
        errors: list[StageError] = []
        updated: dict[str, OutreachDraft] = {}
        for draft in approved:
            result, error = await dispatch_email(
                draft,
                mode=settings.outreach_mode,
                allowlist=settings.outreach_allowlist,
                sender=sender,
                client=gmail,
                now=clock(),
            )
            await repo.save_draft(result)
            updated[result.id] = result
            if error is not None:
                errors.append(error)
            done[result.status] += 1
        parts = [f"{n} {_OUTCOME[s]}" for s, n in done.items() if s in _OUTCOME]
        if done["failed"]:
            codes = ", ".join(sorted({e.code for e in errors}))
            parts.append(f"{done['failed']} failed ({codes})")
        reply = "Dispatch: " + "; ".join(parts) + "."
        return {
            "drafts": [updated.get(d.id, d) for d in drafts],
            "messages": [AIMessage(content=reply)],
            "errors": errors,
        }

    return chase_dispatch
