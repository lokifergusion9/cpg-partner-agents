"""The single-prompt baseline call for eval ablation A8 (``CALL_CONTROL=prompt_only``).

The same simulator persona talks to one free-form prompt instead of the stage machine (see
``prompts/baseline.py``). To make the numbers comparable, the partner's turns are still
classified and the agent's lines are still checked, but only to record what happened: the
classification never steers the call and a failed check never changes a line.
Only the eval uses this module.
"""

import logging
from datetime import UTC, datetime

from langchain_core.language_models import BaseChatModel

from partner_agents.calls.checks import CheckContext, run_checks
from partner_agents.calls.classify import classify_turn
from partner_agents.calls.simulator import partner_line
from partner_agents.graph.call import CallDeps
from partner_agents.llm import invoke_structured
from partner_agents.prompts.baseline import PROMPT_VERSION, BaselineReply, build_baseline_messages
from partner_agents.schemas import (
    BrandProfile,
    CallOutcome,
    CallProgress,
    CallRecord,
    CallStage,
    CallTurn,
    Directive,
    DirectiveAction,
    Persona,
    TurnSignals,
)

logger = logging.getLogger(__name__)


def observe_line(
    text: str,
    deps: CallDeps,
    brand: BrandProfile,
    *,
    first: bool,
    outcome: CallOutcome | None,
    partner_last: str,
    slots: dict[str, str],
) -> CheckContext:
    """The context in which a baseline line is checked. The baseline has no stage or directive,
    so a neutral one is made up: ``open`` for the first line (``C-AI``), ``close`` for the last
    (``C-SIGNOFF``), and the ``handoff`` stage otherwise so that talk of a meeting is not
    counted as booking too early. Every approved fact is allowed (the baseline sees them all)."""
    action: DirectiveAction = "open" if first else ("close" if outcome is not None else "pitch")
    stage: CallStage = "opening" if first else ("closing" if outcome is not None else "handoff")
    return CheckContext(
        stage=stage,
        directive=Directive(action=action),
        brand=brand,
        banned_phrases=deps.playbook.banned_phrases,
        facts=brand.approved_facts,
        slots=slots,
        partner_last=partner_last,
        max_words=deps.settings.call_reply_max_words,
        outcome=outcome,
    )


async def run_prompt_only_call(
    deps: CallDeps,
    simulator: BaseChatModel,
    persona: Persona,
    brand: BrandProfile,
    *,
    user_id: str,
    call_id: str,
) -> CallRecord:
    """Play ``persona`` against the baseline until it ends the call or the turn limit is hit."""
    limit = min(deps.settings.call_max_turns, deps.playbook.max_turns)
    turns: list[CallTurn] = []
    slots: dict[str, str] = {}
    outcome: CallOutcome | None = None
    partner_text = ""
    for partner_turns in range(limit + 1):
        reply = await invoke_structured(
            deps.caller,
            BaselineReply,
            build_baseline_messages(
                deps.playbook, brand, turns, max_words=deps.settings.call_reply_max_words
            ),
            role="caller",
            prompt_version=PROMPT_VERSION,
        )
        text = " ".join(reply.text.split()) or "..."
        last = reply.outcome if partner_turns < limit else (reply.outcome or "incomplete")
        ctx = observe_line(
            text, deps, brand, first=not turns, outcome=last, partner_last=partner_text, slots=slots
        )
        turns.append(
            CallTurn(
                index=len(turns),
                speaker="agent",
                text=text,
                stage=ctx.stage,
                checks=run_checks(text, ctx),
            )
        )
        if last is not None:
            outcome = last
            break
        partner_text = await partner_line(simulator, persona, turns)
        try:  # a measurement only: it never steers the call
            signals = await classify_turn(deps.classifier, "discovery", turns, partner_text)
        except Exception as exc:
            logger.warning("baseline classify failed", extra={"error_type": type(exc).__name__})
            signals = TurnSignals()
        slots.update(signals.slot_values)
        turns.append(
            CallTurn(
                index=len(turns),
                speaker="partner",
                text=partner_text,
                stage="discovery",
                signals=signals,
            )
        )
    now = datetime.now(UTC)
    return CallRecord(
        id=call_id,
        user_id=user_id,
        lead_id="eval_lead",
        run_id=f"eval_{call_id}",
        draft_id="eval_draft",
        playbook_id=deps.playbook.id,
        playbook_version=deps.playbook.version,
        prompt_version=PROMPT_VERSION,
        mode="simulate",
        persona_id=persona.id,
        progress=CallProgress(
            stage="ended",
            slots=slots,
            turns=sum(turn.speaker == "partner" for turn in turns),
            outcome=outcome or "incomplete",
        ),
        turns=turns,
        started_at=now,
        finished_at=now,
    )
