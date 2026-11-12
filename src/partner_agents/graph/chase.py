"""Chase's graph nodes (HLD §6.9): ``chase_select`` and ``chase_draft``.

``resolve_selector`` turns ``route.lead_selector`` ("top 3", "the first one", "lead 2", "the
LA ones", a company name) into leads of the thread's latest run, deterministically. When it
cannot read the selector, the writer model picks numbers from the list.

``chase_select`` goes on to ``chase_draft`` only when the turn asks for outreach (a goal, or a
drafting verb): by email to ``chase_draft``, by voice to ``call_plan``. Otherwise it replies with
the matching leads, so follow-ups such as "only the ones in LA" work (PRD F5).

``chase_draft`` runs every selected lead through ``outreach.policy`` in order, then writes the
eligible ones concurrently. Each draft is saved as ``pending_approval``; nothing is sent here.
"""

import asyncio
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END

from partner_agents.config import Settings
from partner_agents.graph.charlie import last_user_text
from partner_agents.graph.state import AgentState
from partner_agents.llm import StructuredOutputError, invoke_structured
from partner_agents.observability import NodeUpdate, redact_contacts, timed_node
from partner_agents.outreach.policy import TokenBucket, check_lead, enforce_body
from partner_agents.prompts.chase import (
    PROMPT_VERSION,
    build_draft_messages,
    build_selection_messages,
    revision_note,
)
from partner_agents.schemas import (
    BrandProfile,
    EmailDraft,
    Lead,
    LeadSelection,
    OutreachDraft,
    RouteDecision,
    StageError,
)
from partner_agents.storage.repo import Repository

logger = logging.getLogger(__name__)

Node = Callable[[AgentState], Awaitable[NodeUpdate]]

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": -1}
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
_TOP_N = re.compile(r"\b(?:top|first)\s+(\d+|one|two|three|four|five|ten)\b")
_NTH = re.compile(r"\b(?:lead|number|no\.?|#)\s*#?\s*(\d+)\b|#(\d+)\b")
_ORDINAL = re.compile(r"\b(first|second|third|fourth|fifth|last)\b")
_ALL = re.compile(r"\b(?:all|every|everyone|everybody)\b")
# Place shorthands people type; the leads carry the full name.
_ALIASES = {
    "la": "los angeles",
    "sf": "san francisco",
    "nyc": "new york",
    "ny": "new york",
    "dc": "washington",
    "socal": "california",
    "norcal": "california",
}
_FILLER = frozenset(
    [
        "the",
        "a",
        "an",
        "only",
        "just",
        "ones",
        "one",
        "those",
        "these",
        "them",
        "lead",
        "leads",
        "in",
        "at",
        "from",
        "of",
        "based",
        "near",
        "around",
        "and",
        "or",
        "that",
        "are",
        "is",
        "which",
        "who",
        "with",
        "company",
        "companies",
        "partner",
        "partners",
    ]
)


def _number(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


def resolve_selector(leads: Sequence[Lead], selector: str | None) -> list[Lead] | None:
    """The leads ``selector`` picks, in list order, or ``None`` when it cannot be read.

    Positions follow the numbering the router shows (the order of ``state["leads"]``). An
    empty list means the selector was understood and nothing matched.
    """
    text = " ".join((selector or "").lower().split())
    if not text:
        return None
    if match := _TOP_N.search(text):
        return list(leads[: _number(match.group(1))])
    if match := _NTH.search(text):
        position = int(match.group(1) or match.group(2))
        return [leads[position - 1]] if 1 <= position <= len(leads) else []
    if match := _ORDINAL.search(text):
        position = _ORDINALS[match.group(1)]
        index = position - 1 if position > 0 else len(leads) - 1
        return [leads[index]] if 0 <= index < len(leads) else []
    if _ALL.search(text):
        return list(leads)
    terms = [_ALIASES.get(word, word) for word in re.findall(r"[a-z0-9&'-]+", text)]
    terms = [term for term in terms if term not in _FILLER]
    if not terms:
        return None
    return [lead for lead in leads if all(term in _haystack(lead) for term in terms)]


def _haystack(lead: Lead) -> str:
    return f"{lead.company_name} {lead.location or ''} {lead.partner_type or ''}".lower()


def _describe(lead: Lead) -> str:
    details = ", ".join(part for part in (lead.partner_type, lead.location) if part)
    email = "verified email" if lead.outreach_eligible else "no verified email"
    return f"{lead.company_name} ({details or 'details unknown'}; {email})"


def _error(stage: str, exc: Exception) -> StageError:
    if isinstance(exc, StructuredOutputError):
        return StageError(stage=stage, code="invalid_output", message=exc.detail)
    logger.warning("%s model call failed", stage, extra={"error_type": type(exc).__name__})
    return StageError(
        stage=stage,
        code="model_error",
        message=f"{type(exc).__name__}: {redact_contacts(str(exc))[:200]}",
    )


# --- chase_select -------------------------------------------------------------------

NO_LEADS = "I have no leads in this conversation yet. Ask me to find some partners first."
NOT_UNDERSTOOD = (
    'I could not tell which leads you mean. Try "the top 3", "lead 2" or "the ones in Los Angeles".'
)

# A turn asks for outreach when it states a goal or uses one of these verbs ("the LA ones"
# alone only narrows the list).
_OUTREACH_VERB = re.compile(
    r"\b(?:draft|e-?mail|mail|write|send|reach(?:\s+out)?|contact|message|pitch|introduce|"
    r"call|phone|ring)\b",
    re.IGNORECASE,
)


def wants_drafts(route: RouteDecision | None, user_text: str) -> bool:
    if route is None or route.intent != "outreach":
        return False
    return bool(route.outreach_goal) or _OUTREACH_VERB.search(user_text) is not None


def next_after_select(state: AgentState) -> str:
    """Conditional edge out of ``chase_select``: plan the outreach, or end with its reply."""
    route = state.get("route")
    if (
        state.get("selected_leads")
        and route is not None
        and wants_drafts(route, last_user_text(list(state.get("messages", []))))
    ):
        return "call_plan" if route.outreach_channel == "voice" else "chase_draft"
    return END


async def _llm_select(
    model: BaseChatModel, leads: Sequence[Lead], selector: str, user_text: str
) -> list[Lead]:
    picked = await invoke_structured(
        model,
        LeadSelection,
        build_selection_messages(selector, user_text, leads),
        role="writer",
        prompt_version=PROMPT_VERSION,
    )
    positions = sorted({n for n in picked.numbers if 1 <= n <= len(leads)})
    return [leads[n - 1] for n in positions]


def make_chase_select(model: BaseChatModel) -> Node:
    """The ``chase_select`` node: ``route.lead_selector`` + thread leads -> ``selected_leads``."""

    @timed_node("chase_select")
    async def chase_select(state: AgentState) -> NodeUpdate:
        leads = state.get("leads", [])
        route = state.get("route")
        selector = route.lead_selector if route else None
        user_text = last_user_text(list(state.get("messages", [])))
        if not leads:
            return {"selected_leads": [], "messages": [AIMessage(content=NO_LEADS)]}
        errors: list[StageError] = []
        chosen = resolve_selector(leads, selector)
        if chosen is None and (selector or "").strip():
            try:
                chosen = await _llm_select(model, leads, str(selector), user_text)
            except Exception as exc:
                errors.append(_error("chase_select", exc))
        if chosen is None:
            reply = NOT_UNDERSTOOD
            return {"selected_leads": [], "messages": [AIMessage(content=reply)], "errors": errors}
        update: NodeUpdate = {"selected_leads": [lead.id for lead in chosen], "errors": errors}
        drafting = wants_drafts(route, user_text)
        if chosen and drafting:
            return update  # chase_draft or call_plan replies
        if not chosen:
            reply = f'None of the {len(leads)} leads match "{selector}".'
        else:
            noun = "lead matches" if len(chosen) == 1 else "leads match"
            lines = [f"{i}. {_describe(lead)}" for i, lead in enumerate(chosen, start=1)]
            reply = f'{len(chosen)} of {len(leads)} {noun} "{selector}":\n' + "\n".join(lines)
        return {**update, "messages": [AIMessage(content=reply)]}

    return chase_select


# --- chase_draft --------------------------------------------------------------------

NO_PROFILE = (
    "Set up your brand profile first (brand, product, sender and sign-off), so the emails "
    "can say who they are from."
)
_SKIP_REASON = {
    "skipped_ineligible": "no verified email",
    "skipped_duplicate": "already contacted recently",
    "skipped_cap": "daily outreach limit reached",
}


def default_goal(profile: BrandProfile) -> str:
    return f"introduce {profile.brand_name} ({profile.product}) and ask for a short intro call"


def new_draft_id() -> str:
    return f"d_{uuid.uuid4().hex[:12]}"


async def write_email(
    model: BaseChatModel, profile: BrandProfile, goal: str, lead: Lead
) -> tuple[EmailDraft, list[str]]:
    """One draft, revised once if the policy rejects the body. Returns the draft (body with
    the opt-out line) and the problems that are left."""
    messages: list[BaseMessage] = build_draft_messages(profile, goal, lead)
    for attempt in (1, 2):
        draft = await invoke_structured(
            model, EmailDraft, messages, role="writer", prompt_version=PROMPT_VERSION
        )
        check = enforce_body(draft.body, profile.sender_name)
        fixed = draft.model_copy(update={"body": check.body})
        if check.ok or attempt == 2:
            return fixed, list(check.problems)
        note = revision_note(check.problems, check.word_count)
        messages = [*messages, AIMessage(content=draft.model_dump_json()), HumanMessage(note)]
    raise AssertionError("unreachable")


def _reply(drafts: Sequence[OutreachDraft], leads: dict[str, Lead]) -> str:
    def name(draft: OutreachDraft) -> str:
        return leads[draft.lead_id].company_name

    ready = [d for d in drafts if d.status == "pending_approval"]
    lines = []
    if ready:
        lines.append(f"I drafted {len(ready)} email{'s' if len(ready) != 1 else ''} for approval:")
        lines += [f"{i}. {name(d)}: {d.subject}" for i, d in enumerate(ready, start=1)]
    else:
        lines.append("I did not draft any emails.")
    skipped = [d for d in drafts if d.status in _SKIP_REASON]
    failed = [d for d in drafts if d.status == "failed"]
    if skipped:
        lines.append(
            "Skipped: " + "; ".join(f"{name(d)} ({_SKIP_REASON[d.status]})" for d in skipped)
        )
    if failed:
        lines.append("Could not write a valid draft for: " + ", ".join(name(d) for d in failed))
    if ready:
        lines.append("Nothing is sent until you approve it.")
    return "\n".join(lines)


def make_chase_draft(
    model: BaseChatModel,
    repo: Repository,
    settings: Settings,
    buckets: dict[str, TokenBucket],
    clock: Callable[[], datetime],
) -> Node:
    """The ``chase_draft`` node: selected leads + brand profile -> ``drafts``.

    ``buckets`` holds each user's daily token bucket (process memory; HLD §6.9).
    """

    @timed_node("chase_draft")
    async def chase_draft(state: AgentState) -> NodeUpdate:
        user_id, run_id = state.get("user_id", ""), state.get("run_id", "")
        profile = await repo.get_profile(user_id)
        if profile is None:
            return {"drafts": [], "messages": [AIMessage(content=NO_PROFILE)]}
        route = state.get("route")
        goal = (route.outreach_goal if route else None) or default_goal(profile)
        by_id = {lead.id: lead for lead in state.get("leads", [])}
        selected = [by_id[i] for i in state.get("selected_leads", []) if i in by_id]
        now = clock()

        def record(lead: Lead, status: str, email: EmailDraft | None = None) -> OutreachDraft:
            return OutreachDraft(
                id=new_draft_id(),
                lead_id=lead.id,
                user_id=user_id,
                run_id=run_id,
                channel="email",
                recipient=lead.email or "",
                subject=email.subject if email else None,
                body=email.body if email else "",
                status=status,  # type: ignore[arg-type]
                prompt_version=PROMPT_VERSION,
                created_at=now,
            )

        # Policy first, in order, so a recipient listed twice is a duplicate of itself.
        bucket = buckets.get(user_id) or TokenBucket.full(settings.outreach_daily_cap, now)
        since = now - timedelta(days=settings.outreach_dedupe_days)
        batch: list[OutreachDraft] = []
        to_write: list[Lead] = []
        for lead in selected:
            previous = await repo.recent_drafts_to(user_id, lead.email, since) if lead.email else []
            skip, bucket = check_lead(
                lead,
                previous=[*previous, *batch],
                bucket=bucket,
                now=now,
                mode=settings.outreach_mode,
                allowlist=settings.outreach_allowlist,
                dedupe_days=settings.outreach_dedupe_days,
            )
            if skip is None:
                to_write.append(lead)
            # A passing lead gets a pending placeholder, so the in-batch dedupe sees it.
            batch.append(record(lead, skip or "pending_approval"))
        buckets[user_id] = bucket

        results = await asyncio.gather(
            *(write_email(model, profile, goal, lead) for lead in to_write),
            return_exceptions=True,
        )
        written = dict(zip((lead.id for lead in to_write), results, strict=True))
        drafts: list[OutreachDraft] = []
        errors: list[StageError] = []
        for placeholder, lead in zip(batch, selected, strict=True):
            if placeholder.status != "pending_approval":
                drafts.append(placeholder)
                continue
            result = written[lead.id]
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result
                errors.append(_error("chase_draft", result))
                drafts.append(record(lead, "failed"))
                continue
            email, problems = result
            if problems:
                errors.append(
                    StageError(
                        stage="chase_draft",
                        code="policy_violation",
                        message=f"{lead.id}: {', '.join(problems)}",
                    )
                )
            drafts.append(record(lead, "failed" if problems else "pending_approval", email))
        for draft in drafts:
            await repo.save_draft(draft)
        return {
            "drafts": drafts,
            "messages": [AIMessage(content=_reply(drafts, by_id))],
            "errors": errors,
        }

    return chase_draft
