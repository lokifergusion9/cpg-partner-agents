"""The Call tab's handlers (HLD section 6.11, PRD section 10 steps 8 and 9).

Plain functions with no Gradio import, so they are tested without a browser. The layout in
``ui/app.py`` only wires them to components. A call is created when its plan is approved in the
Approvals tab (``MockVoiceCaller``); this tab plays it, either against a demo persona
(``simulate``) or with the user typing the partner's lines (``live_text``). Nothing is dialled.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from partner_agents.calls.personas import PersonaError, list_personas, load_persona
from partner_agents.calls.simulator import simulate_call
from partner_agents.graph.build import Context
from partner_agents.graph.call import CallNotActive, run_call_turn
from partner_agents.prompts.caller import format_slot
from partner_agents.schemas import CallRecord, CallTurn, Lead

logger = logging.getLogger(__name__)

OUTCOME_BADGES = {
    "meeting_booked": "✅",
    "callback_booked": "📞",
    "escalated_to_human": "🙋",
    "opted_out": "🛑",
    "wrong_person": "🔀",
    "not_interested": "⚪",
    "disqualified": "⚪",
    "incomplete": "⏳",
}
NOT_CONFIGURED = "The call agent is not configured."
NO_CALL = "Pick a call first. A call appears here once you approve its plan in the Approvals tab."
SIMULATOR_FAILED = "The simulator failed mid-call. The turns played so far are kept."


@dataclass
class CallView:
    """What the Call tab shows after an action."""

    history: list[dict[str, str]]
    summary: str
    notice: str = ""


def check_badge(turn: CallTurn) -> str:
    """How an agent line got through the checks (HLD 6.11): green, regenerated or fallback."""
    if turn.used_fallback:
        badge = "⚠️ fallback line used"
    elif turn.regenerated:
        badge = "🔁 regenerated once, then passed"
    else:
        badge = "✅ checks passed"
    failed = [check.rule_id for check in turn.checks if not check.passed]
    if failed:
        badge += f" · ❌ {', '.join(failed)}"
    return badge


def transcript_messages(record: CallRecord) -> list[dict[str, str]]:
    """The transcript as chatbot messages: the partner on the user side, with stage chips."""
    messages = []
    for turn in record.turns:
        chip = f"`{turn.stage}`"
        if turn.speaker == "agent":
            messages.append(
                {"role": "assistant", "content": f"{chip} {turn.text}\n\n*{check_badge(turn)}*"}
            )
        else:
            messages.append({"role": "user", "content": f"{chip} {turn.text}"})
    return messages


def call_summary(
    record: CallRecord | None, followup_to: str | None = None, timezone: str = "UTC"
) -> str:
    """The outcome badge, the slots panel, the meeting time (in ``timezone``) and the draft."""
    if record is None:
        return ""
    progress = record.progress
    if progress.outcome is not None:
        badge = OUTCOME_BADGES.get(progress.outcome, "⚪")
        lines = [f"### {badge} Outcome: `{progress.outcome}`"]
    elif record.turns:
        lines = [f"### In progress · stage `{progress.stage}`"]
    else:
        lines = ["### Not started"]
    mode = record.mode + (f" · persona `{record.persona_id}`" if record.persona_id else "")
    lines.append(f"Mode: {mode} · partner turns handled: {progress.turns}")
    if progress.slots:
        lines += ["", "| Slot | Answer |", "|---|---|"]
        lines += [f"| {slot} | {value} |" for slot, value in progress.slots.items()]
    else:
        lines += ["", "No qualification answers yet."]
    if record.meeting_at is not None:
        when = format_slot(record.meeting_at, timezone)
        lines += ["", f"**Time agreed:** {when} ({timezone} time)"]
    if record.followup_draft_id:
        target = f" to {followup_to}" if followup_to else ""
        lines += [
            "",
            f"**Confirmation email{target}:** draft `{record.followup_draft_id}`, waiting for "
            "approval. It is not sent.",
        ]
    return "\n".join(lines)


def call_options(calls: Sequence[CallRecord], leads: Sequence[Lead]) -> list[tuple[str, str]]:
    """Dropdown choices ``(label, call id)``, newest call first."""
    names = {lead.id: lead.company_name for lead in leads}
    options = []
    for call in calls:
        state = call.progress.outcome or ("in progress" if call.turns else "not started")
        options.append((f"{names.get(call.lead_id, call.lead_id)} · {state} · {call.id}", call.id))
    return options


def persona_options() -> list[tuple[str, str]]:
    return [(persona.title, persona.id) for persona in list_personas()]


async def list_call_options(ctx: Context, user_id: str) -> list[tuple[str, str]]:
    repo = ctx.deps.repo
    return call_options(await repo.list_calls(user_id), await repo.list_leads(user_id))


async def view_call(ctx: Context, user_id: str, call_id: str | None, notice: str = "") -> CallView:
    record = await ctx.deps.repo.get_call(user_id, call_id) if call_id else None
    if record is None:
        return CallView([], "", notice or (NO_CALL if not call_id else "No such call."))
    followup_to = None
    if record.followup_draft_id:
        draft = await ctx.deps.repo.get_draft(user_id, record.followup_draft_id)
        followup_to = draft.recipient if draft else None
    brand = await ctx.deps.repo.get_profile(user_id)
    summary = call_summary(record, followup_to, brand.timezone if brand else "UTC")
    return CallView(transcript_messages(record), summary, notice)


def _refusal(exc: CallNotActive) -> str:
    return {
        "call_already_started": "This call has already started. Pick one that has not.",
        "call_not_started": "Start the call first.",
        "call_ended": "This call has ended. Pick another one.",
        "profile_not_found": "Save a brand profile in Settings before calling.",
    }.get(exc.code, str(exc))


async def simulate_selected(
    ctx: Context, user_id: str, call_id: str | None, persona_id: str | None
) -> CallView:
    """Play a demo persona against the selected call that has not started."""
    if not call_id:
        return await view_call(ctx, user_id, None)
    if not persona_id:
        return await view_call(ctx, user_id, call_id, "Pick a persona first.")
    deps = ctx.deps.calls
    if ctx.call_graph is None or deps is None or deps.simulator is None:
        return await view_call(ctx, user_id, call_id, NOT_CONFIGURED)
    try:
        persona = load_persona(persona_id)
    except PersonaError:
        return await view_call(ctx, user_id, call_id, "No such persona.")
    notice = ""
    async with ctx.lock(f"call:{call_id}"):
        try:
            await simulate_call(
                ctx.call_graph, deps, deps.simulator, persona, user_id=user_id, call_id=call_id
            )
        except LookupError:
            notice = "No such call."
        except CallNotActive as exc:
            notice = _refusal(exc)
        except Exception as exc:  # the turns played so far stay saved
            logger.warning("simulation failed", extra={"error_type": type(exc).__name__})
            notice = SIMULATOR_FAILED
    return await view_call(ctx, user_id, call_id, notice)


async def play_turn(
    ctx: Context, user_id: str, call_id: str | None, partner_text: str | None
) -> CallView:
    """The opening (``partner_text=None``) or one typed partner line of a live text call."""
    if not call_id:
        return await view_call(ctx, user_id, None)
    deps = ctx.deps.calls
    if ctx.call_graph is None or deps is None:
        return await view_call(ctx, user_id, call_id, NOT_CONFIGURED)
    if partner_text is not None:
        partner_text = partner_text.strip()[:1000]
        if not partner_text:
            return await view_call(ctx, user_id, call_id)
    notice = ""
    async with ctx.lock(f"call:{call_id}"):
        try:
            await run_call_turn(
                ctx.call_graph, deps, user_id=user_id, call_id=call_id, partner_text=partner_text
            )
        except LookupError:
            notice = "No such call."
        except CallNotActive as exc:
            notice = _refusal(exc)
    return await view_call(ctx, user_id, call_id, notice)
