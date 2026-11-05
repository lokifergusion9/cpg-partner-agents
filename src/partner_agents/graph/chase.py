"""Chase's graph nodes (HLD §6.9). So far only ``chase_select``.

``resolve_selector`` turns ``route.lead_selector`` ("top 3", "the first one", "lead 2", "the
LA ones", a company name) into leads of the thread's latest run, deterministically. It returns
``None`` when it cannot read the selector; T-23 adds the LLM fallback for those, the
``policy.eligible()`` filter, and drafting. Until then ``chase_select`` replies with the
matching leads, so follow-ups such as "only the ones in LA" already work (PRD F5).
"""

import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage

from partner_agents.graph.state import AgentState
from partner_agents.observability import NodeUpdate, timed_node
from partner_agents.schemas import Lead

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


@timed_node("chase_select")
async def chase_select(state: AgentState) -> NodeUpdate:
    """Resolve the selector against the thread's latest leads and say which ones it means."""
    leads = state.get("leads", [])
    route = state.get("route")
    selector = route.lead_selector if route else None
    if not leads:
        reply = "I have no leads in this conversation yet. Ask me to find some partners first."
        return {"selected_leads": [], "messages": [AIMessage(content=reply)]}
    chosen = resolve_selector(leads, selector)
    if chosen is None:
        reply = (
            'I could not tell which leads you mean. Try "the top 3", "lead 2" or '
            '"the ones in Los Angeles".'
        )
        return {"selected_leads": [], "messages": [AIMessage(content=reply)]}
    if not chosen:
        reply = f'None of the {len(leads)} leads match "{selector}".'
    else:
        noun = "lead matches" if len(chosen) == 1 else "leads match"
        lines = [f"{i}. {_describe(lead)}" for i, lead in enumerate(chosen, start=1)]
        reply = f'{len(chosen)} of {len(leads)} {noun} "{selector}":\n' + "\n".join(lines)
    return {"selected_leads": [lead.id for lead in chosen], "messages": [AIMessage(content=reply)]}
