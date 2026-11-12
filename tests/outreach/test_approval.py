"""``resolve`` turns an ApprovalDecision into draft statuses (HLD §6.9)."""

from datetime import UTC, datetime

import pytest

from partner_agents.outreach.approval import resolve
from partner_agents.outreach.policy import OPT_OUT_LINE
from partner_agents.schemas import ApprovalDecision, EmailDraft, OutreachDraft

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


def draft(key: str) -> OutreachDraft:
    return OutreachDraft(
        id=key, lead_id=key, user_id="u", run_id="r", channel="email",
        recipient=f"{key}@x.example", subject=f"Subject {key}", body=f"Body {key}. Priya",
        status="pending_approval", prompt_version="v", created_at=NOW,
    )  # fmt: skip


PENDING = [draft("a"), draft("b"), draft("c")]
GOOD_EDIT = EmailDraft(subject="Edited", body="A better note. Priya")


def statuses(decision: ApprovalDecision) -> dict[str, str]:
    result = resolve(PENDING, decision, sender_name="Priya", now=NOW)
    return {d.id: d.status for d in [*result.approved, *result.rejected, *result.unresolved]}


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (
            ApprovalDecision(approve=["a"], reject=["c"], edits={"b": GOOD_EDIT}),
            {"a": "approved", "b": "approved", "c": "rejected"},
        ),
        (
            ApprovalDecision(approve=["a", "b", "c"]),
            {"a": "approved", "b": "approved", "c": "approved"},
        ),
        (ApprovalDecision(), {"a": "rejected", "b": "rejected", "c": "rejected"}),
        (ApprovalDecision(approve=["a"]), {"a": "approved", "b": "rejected", "c": "rejected"}),
        (
            ApprovalDecision(approve=["a"], reject=["a"]),  # a rejection beats an approval
            {"a": "rejected", "b": "rejected", "c": "rejected"},
        ),
        (
            ApprovalDecision(reject=["b"], edits={"b": GOOD_EDIT}),  # ... and an edit
            {"a": "rejected", "b": "rejected", "c": "rejected"},
        ),
        (
            ApprovalDecision(approve=["zzz"], reject=["yyy"]),  # unknown ids are ignored
            {"a": "rejected", "b": "rejected", "c": "rejected"},
        ),
    ],
)
def test_statuses(decision: ApprovalDecision, expected: dict[str, str]) -> None:
    assert statuses(decision) == expected


def test_an_edit_replaces_the_content_and_gets_the_opt_out_line() -> None:
    result = resolve(
        PENDING, ApprovalDecision(edits={"b": GOOD_EDIT}), sender_name="Priya", now=NOW
    )
    [edited] = result.approved
    assert (edited.id, edited.subject) == ("b", "Edited")
    assert edited.body.startswith("A better note.") and edited.body.endswith(OPT_OUT_LINE)
    assert edited.decided_at == NOW


def test_an_approval_keeps_the_content_and_stamps_the_time() -> None:
    result = resolve(PENDING, ApprovalDecision(approve=["a"]), sender_name="Priya", now=NOW)
    [approved] = result.approved
    assert approved.body == PENDING[0].body and approved.decided_at == NOW
    assert PENDING[0].status == "pending_approval"  # the input is not mutated


@pytest.mark.parametrize(
    ("body", "problem"),
    [("word " * 200 + "Priya", "too_long"), ("No sender here, unsubscribe.", "missing_sender")],
)
def test_an_edit_that_breaks_the_policy_stays_pending(body: str, problem: str) -> None:
    decision = ApprovalDecision(approve=["a"], edits={"b": EmailDraft(subject="s", body=body)})
    result = resolve(PENDING, decision, sender_name="Priya", now=NOW)
    assert [d.id for d in result.unresolved] == ["b"]
    assert result.unresolved[0].status == "pending_approval"
    assert result.problems == {"b": [problem]}
    assert [d.id for d in result.approved] == ["a"]
