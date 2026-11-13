"""Voice outreach (HLD §6.9, §6.11): the call plan, the caller protocol and the mock caller.

A voice ``OutreachDraft`` holds the call plan as its body and goes through the same approval
gate as an email. ``dispatch_voice`` then starts the call, and only for an approved draft.
``MockVoiceCaller`` (the default) dials nobody: T-58 makes it create and run a real
``CallRecord``. Any other provider may call only numbers on ``VOICE_ALLOWLIST``.
"""

import logging
import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

from partner_agents.config import VoiceProvider
from partner_agents.schemas import BrandProfile, Lead, OutreachDraft, StageError

logger = logging.getLogger(__name__)

DEFAULT_PLAYBOOK = "partner_intro"  # Settings.call_playbook picks it (CALL_PLAYBOOK)
PLAN_FACTS = 3


class VoiceCaller(Protocol):
    async def start(self, draft: OutreachDraft) -> str:
        """Start the call for an approved voice draft. Returns the call id."""
        ...


class MockVoiceCaller:
    """Creates a call id and dials nothing."""

    async def start(self, draft: OutreachDraft) -> str:
        return f"c_{uuid.uuid4().hex[:12]}"


def build_call_plan(
    lead: Lead,
    profile: BrandProfile,
    provider: VoiceProvider,
    playbook: str = DEFAULT_PLAYBOOK,
) -> str:
    """The plan a human reads before approving a call. Deterministic: no model is involved."""
    details = ", ".join(part for part in (lead.partner_type, lead.location) if part)
    mode = "simulate (nobody is dialled)" if provider == "mock" else "live call"
    lines = [
        f"Call plan for {lead.company_name} ({details or 'details unknown'})",
        f"Playbook: {playbook}",
        f"Mode: {mode}",
        f"Goal: qualify the partner and, if there is a fit, book a meeting with "
        f"{profile.founder_name} of {profile.brand_name}.",
        "The agent says it is an AI, claims only the approved facts below and stops at once "
        "if the partner asks not to be called.",
    ]
    facts = profile.approved_facts[:PLAN_FACTS]
    if facts:
        lines.append("Approved facts it may state:")
        lines += [f"- [{fact.id}] {fact.text}" for fact in facts]
    return "\n".join(lines)


def is_voice_allowlisted(number: str, allowlist: Iterable[str]) -> bool:
    def norm(value: str) -> str:
        return "".join(value.split())

    return bool(number.strip()) and norm(number) in {norm(item) for item in allowlist}


def _failed(
    draft: OutreachDraft, code: str, message: str, now: datetime
) -> tuple[OutreachDraft, str | None, StageError]:
    error = StageError(stage="chase_dispatch", code=code, message=message)
    update = {"status": "failed", "decided_at": draft.decided_at or now}
    return draft.model_copy(update=update), None, error


async def dispatch_voice(
    draft: OutreachDraft,
    *,
    provider: VoiceProvider,
    allowlist: Iterable[str],
    caller: VoiceCaller | None,
    now: datetime,
) -> tuple[OutreachDraft, str | None, StageError | None]:
    """Start the call for one approved voice draft. Returns (draft, call id, error)."""
    if draft.status != "approved" or draft.channel != "voice":
        return _failed(draft, "not_approved", "only an approved voice draft can be started", now)
    if provider != "mock" and not is_voice_allowlisted(draft.recipient, allowlist):
        return _failed(draft, "not_allowlisted", "the number is not on VOICE_ALLOWLIST", now)
    if caller is None:
        return _failed(draft, "voice_not_configured", f"no caller for {provider}", now)
    try:
        call_id = await caller.start(draft)
    except Exception as exc:
        logger.warning("voice start failed", extra={"error_type": type(exc).__name__})
        return _failed(draft, "voice_error", type(exc).__name__, now)
    return draft.model_copy(update={"status": "call_created"}), call_id, None
