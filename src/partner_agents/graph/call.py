"""The call sub-graph (HLD §6.11): one invocation per partner turn.

``call_classify → call_advance → call_reply → call_persist``. The first agent turn (the
opening) skips ``call_classify``. ``call_reply`` writes the line and runs the checks with the
regenerate-once loop (``calls/reply.write_reply``). The graph is compiled with its own
checkpointer and run with ``thread_id = call_id``; the ``CallRecord`` in the repository is the
source of truth and is passed in on every turn, so a lost checkpoint loses nothing.

Partner text is untrusted (CLAUDE.md rule 11): it only reaches the classifier, wrapped, and can
only produce ``TurnSignals``. Nothing here sends anything: a booking creates a confirmation
email *draft* that waits for approval (ADR-012).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from partner_agents.calls.checks import CheckContext
from partner_agents.calls.classify import classify_turn
from partner_agents.calls.reply import CheckedReply, write_reply
from partner_agents.calls.stages import advance, first_directive
from partner_agents.config import Settings
from partner_agents.graph.chase import new_draft_id
from partner_agents.llm import track_providers, track_usage
from partner_agents.outreach.policy import enforce_body
from partner_agents.prompts.caller import format_slot
from partner_agents.schemas import (
    BrandProfile,
    CallOutcome,
    CallPlaybook,
    CallProgress,
    CallRecord,
    CallTurn,
    Directive,
    OutreachDraft,
    StageError,
    TurnSignals,
)
from partner_agents.storage.repo import Repository

logger = logging.getLogger(__name__)

FOLLOWUP_VERSION = "call-followup-v1"
# Outcomes that agree on a time with the founder and get a confirmation email draft.
BOOKED: frozenset[CallOutcome] = frozenset(
    {"meeting_booked", "callback_booked", "escalated_to_human"}
)
_ASKS_A_SLOT = frozenset({"ask_slot", "confirm_slot", "retry_once"})


class CallState(TypedDict, total=False):
    record: CallRecord
    brand: BrandProfile
    partner_text: str | None  # None for the opening turn
    signals: TurnSignals | None
    progress: CallProgress | None
    directive: Directive | None
    reply: CheckedReply | None
    errors: list[StageError]


CallGraph = CompiledStateGraph[CallState, Any, Any, Any]


@dataclass
class CallDeps:
    settings: Settings
    repo: Repository
    playbook: CallPlaybook
    classifier: BaseChatModel
    caller: BaseChatModel
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)


class CallNotActive(ValueError):
    """The call has not started yet, has already started, or has ended."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def build_call_graph(
    deps: CallDeps, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CallGraph:
    async def call_classify(state: CallState) -> dict[str, Any]:
        record, text = state["record"], state.get("partner_text") or ""
        last = next((t.directive for t in reversed(record.turns) if t.directive), None)
        asked = last.slot if last is not None and last.action in _ASKS_A_SLOT else None
        errors: list[StageError] = []
        try:
            signals = await classify_turn(
                deps.classifier, record.progress.stage, record.turns, text, asked_slot=asked
            )
        except Exception as exc:  # no signal is safer than a crashed call
            logger.warning("call classify failed", extra={"error_type": type(exc).__name__})
            errors.append(StageError(stage="call_classify", code="model_error",
                                     message=type(exc).__name__))  # fmt: skip
            signals = TurnSignals()
        turn = CallTurn(index=len(record.turns), speaker="partner", text=text,
                        stage=record.progress.stage, signals=signals)  # fmt: skip
        updated = record.model_copy(update={"turns": [*record.turns, turn]})
        return {"record": updated, "signals": signals, "errors": errors}

    async def call_advance(state: CallState) -> dict[str, Any]:
        record, signals = state["record"], state.get("signals")
        if signals is None:
            return {"progress": record.progress, "directive": first_directive()}
        progress, directive = advance(
            record.progress,
            signals,
            deps.playbook,
            state["brand"],
            now=deps.clock(),
            max_turns=deps.settings.call_max_turns,
        )
        return {"progress": progress, "directive": directive}

    async def call_reply(state: CallState) -> dict[str, Any]:
        brand, progress = state["brand"], state["progress"]
        directive = state["directive"]
        assert progress is not None and directive is not None
        ctx = CheckContext(
            stage=progress.stage,
            directive=directive,
            brand=brand,
            banned_phrases=deps.playbook.banned_phrases,
            facts=[f for f in brand.approved_facts if f.id in directive.fact_ids],
            slots=progress.slots,
            partner_last=state.get("partner_text") or "",
            max_words=deps.settings.call_reply_max_words,
            outcome=progress.outcome,
        )
        reply = await write_reply(deps.caller, deps.playbook, ctx, state["record"].turns)
        return {"reply": reply}

    async def call_persist(state: CallState) -> dict[str, Any]:
        record, progress = state["record"], state["progress"]
        reply, directive = state["reply"], state["directive"]
        assert progress is not None and reply is not None
        turn = CallTurn(
            index=len(record.turns),
            speaker="agent",
            text=reply.text,
            stage=progress.stage,
            directive=directive,
            checks=reply.checks,
            regenerated=reply.regenerated,
            used_fallback=reply.used_fallback,
        )
        update: dict[str, Any] = {"turns": [*record.turns, turn], "progress": progress}
        if progress.outcome is not None:
            update["finished_at"] = deps.clock()
            if progress.outcome in BOOKED and progress.proposed_slot is not None:
                update["meeting_at"] = progress.proposed_slot
        record = record.model_copy(update=update)
        if progress.outcome in BOOKED:
            draft = await followup_draft(deps.repo, record, state["brand"], deps.clock())
            if draft is not None:
                await deps.repo.save_draft(draft)
                record = record.model_copy(update={"followup_draft_id": draft.id})
        return {"record": record}

    graph = StateGraph(CallState)
    for name, node in (
        ("call_classify", call_classify),
        ("call_advance", call_advance),
        ("call_reply", call_reply),
        ("call_persist", call_persist),
    ):
        graph.add_node(name, cast(Any, node))  # see the note in graph/scout.py
    graph.add_conditional_edges(
        START,
        lambda s: "call_classify" if s.get("partner_text") is not None else "call_advance",
        ["call_classify", "call_advance"],
    )
    graph.add_edge("call_classify", "call_advance")
    graph.add_edge("call_advance", "call_reply")
    graph.add_edge("call_reply", "call_persist")
    graph.add_edge("call_persist", END)
    from partner_agents.graph.build import checkpoint_serde  # avoid an import cycle

    return graph.compile(checkpointer=checkpointer or InMemorySaver(serde=checkpoint_serde()))


async def followup_draft(
    repo: Repository, record: CallRecord, brand: BrandProfile, now: datetime
) -> OutreachDraft | None:
    """A confirmation email for a booked call, pending approval. Only to a verified email."""
    lead = await repo.get_lead(record.user_id, record.lead_id)
    if lead is None or not lead.outreach_eligible or lead.email is None:
        return None
    what = (
        "a call back"
        if record.progress.outcome == "callback_booked"
        else (f"a meeting with {brand.founder_name}")
    )
    when = ""
    if record.meeting_at is not None:
        when = f" on {format_slot(record.meeting_at, brand.timezone)} ({brand.timezone} time)"
    body = (
        f"Hello,\n\nThank you for speaking with our AI assistant today on behalf of "
        f"{brand.brand_name}. This email confirms {what}{when}. If the time no longer suits "
        f"you, just reply with one that does.\n\n{brand.sign_off},\n{brand.sender_name}\n"
        f"{brand.brand_name}"
    )
    checked = enforce_body(body, brand.sender_name)
    return OutreachDraft(
        id=new_draft_id(),
        lead_id=lead.id,
        user_id=record.user_id,
        run_id=record.run_id,
        channel="email",
        recipient=lead.email,
        subject=f"Confirming {what} with {brand.brand_name}"[:80],
        body=checked.body,
        status="pending_approval",
        prompt_version=FOLLOWUP_VERSION,
        created_at=now,
    )


async def run_call_turn(
    graph: CallGraph,
    deps: CallDeps,
    *,
    user_id: str,
    call_id: str,
    partner_text: str | None,
) -> tuple[CallRecord, list[StageError]]:
    """Run the opening (``partner_text=None``) or one partner turn, and save the record.

    Raises ``LookupError`` for an unknown call and ``CallNotActive`` when the turn does not fit
    the call's state (opening twice, a partner turn before the opening, or after the end).
    """
    record = await deps.repo.get_call(user_id, call_id)
    if record is None:
        raise LookupError(call_id)
    if record.progress.outcome is not None:
        raise CallNotActive("call_ended", "This call has already ended.")
    if partner_text is None and record.turns:
        raise CallNotActive("call_already_started", "This call has already started.")
    if partner_text is not None and not record.turns:
        raise CallNotActive("call_not_started", "Start the call before sending a turn.")
    brand = await deps.repo.get_profile(user_id)
    if brand is None:
        raise CallNotActive("profile_not_found", "Save a brand profile before calling.")
    config: RunnableConfig = {"configurable": {"thread_id": call_id}}
    graph_input: CallState = {
        "record": record,
        "brand": brand,
        "partner_text": partner_text,
        "signals": None,
        "progress": None,
        "directive": None,
        "reply": None,
        "errors": [],
    }
    with track_usage() as tokens, track_providers() as providers:
        final = await graph.ainvoke(graph_input, config)
    done: CallRecord = final["record"]
    done = done.model_copy(
        update={
            "tokens": {
                k: done.tokens.get(k, 0) + tokens.get(k, 0) for k in {*done.tokens, *tokens}
            },
            "providers": {**done.providers, **providers},
        }
    )
    await deps.repo.save_call(done)
    return done, list(final.get("errors", []))
