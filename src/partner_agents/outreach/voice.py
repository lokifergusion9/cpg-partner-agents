"""Voice outreach (HLD §6.9, §6.11): the call plan, the caller protocol and the mock caller.

A voice ``OutreachDraft`` holds the call plan as its body and goes through the same approval
gate as an email. ``dispatch_voice`` then starts the call, and only for an approved draft.
``MockVoiceCaller`` (the default) dials nobody: T-58 makes it create and run a real
``CallRecord``. Any other provider may call only numbers on ``VOICE_ALLOWLIST``.
"""

import logging
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Protocol

from partner_agents.config import VoiceProvider
from partner_agents.prompts.caller import PROMPT_VERSION as CALLER_PROMPT_VERSION
from partner_agents.schemas import (
    BrandProfile,
    CallMode,
    CallPlaybook,
    CallProgress,
    CallRecord,
    Lead,
    OutreachDraft,
    StageError,
)

logger = logging.getLogger(__name__)

DEFAULT_PLAYBOOK = "partner_intro"  # Settings.call_playbook picks it (CALL_PLAYBOOK)
PLAN_FACTS = 3


class VoiceCaller(Protocol):
    async def start(self, draft: OutreachDraft) -> str:
        """Start the call for an approved voice draft. Returns the call id."""
        ...


class MockVoiceCaller:
    """Dials nobody: saves a new ``CallRecord`` that the Call tab or the API then runs as text
    (``POST /api/calls/{id}/start``, ``/turn`` or ``/simulate``)."""

    def __init__(
        self, repo: "CallStore", playbook: CallPlaybook, clock: Callable[[], datetime]
    ) -> None:
        self._repo, self._playbook, self._clock = repo, playbook, clock

    async def start(self, draft: OutreachDraft) -> str:
        record = new_call_record(draft, self._playbook, self._clock())
        await self._repo.save_call(record)
        return record.id


class CallStore(Protocol):
    async def save_call(self, call: CallRecord) -> None: ...


def new_call_record(
    draft: OutreachDraft, playbook: CallPlaybook, now: datetime, mode: CallMode = "live_text"
) -> CallRecord:
    """A call that has not started: opening stage, no turns."""
    return CallRecord(
        id=f"c_{uuid.uuid4().hex[:12]}",
        user_id=draft.user_id,
        lead_id=draft.lead_id,
        run_id=draft.run_id,
        draft_id=draft.id,
        playbook_id=playbook.id,
        playbook_version=playbook.version,
        prompt_version=CALLER_PROMPT_VERSION,
        mode=mode,
        progress=CallProgress(),
        started_at=now,
    )


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
