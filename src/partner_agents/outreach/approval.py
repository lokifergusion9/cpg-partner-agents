"""Turning a human's ``ApprovalDecision`` into draft statuses (HLD §6.9, PRD F6). Pure.

Rules: a rejection beats an approval or an edit of the same draft; an edit implies approval; a
draft the decision does not mention is rejected; ids that are not pending are ignored. An edit
goes through ``policy.enforce_body`` (opt-out line, word limit, sender name), so approving cannot
bypass the policy. An edit that still breaks it leaves the draft pending, with the problems.
"""

from dataclasses import dataclass, field
from datetime import datetime

from partner_agents.outreach.policy import enforce_body
from partner_agents.schemas import ApprovalDecision, OutreachDraft


@dataclass
class Resolution:
    approved: list[OutreachDraft] = field(default_factory=list)
    rejected: list[OutreachDraft] = field(default_factory=list)
    unresolved: list[OutreachDraft] = field(default_factory=list)  # edits that broke the policy
    problems: dict[str, list[str]] = field(default_factory=dict)  # draft id -> problems

    @property
    def decided(self) -> list[OutreachDraft]:
        return [*self.approved, *self.rejected]


def resolve(
    pending: list[OutreachDraft], decision: ApprovalDecision, *, sender_name: str, now: datetime
) -> Resolution:
    out = Resolution()
    rejected, approved = set(decision.reject), set(decision.approve)
    for draft in pending:
        if draft.id in rejected:
            out.rejected.append(draft.model_copy(update={"status": "rejected", "decided_at": now}))
        elif draft.id in decision.edits:
            edit = decision.edits[draft.id]
            check = enforce_body(edit.body, sender_name)
            if check.problems:
                out.unresolved.append(draft)
                out.problems[draft.id] = list(check.problems)
                continue
            out.approved.append(
                draft.model_copy(
                    update={
                        "subject": edit.subject,
                        "body": check.body,
                        "status": "approved",
                        "decided_at": now,
                    }
                )
            )
        elif draft.id in approved:
            out.approved.append(draft.model_copy(update={"status": "approved", "decided_at": now}))
        else:
            out.rejected.append(draft.model_copy(update={"status": "rejected", "decided_at": now}))
    return out
