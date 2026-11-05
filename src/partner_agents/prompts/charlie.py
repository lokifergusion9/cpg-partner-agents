"""Charlie's router prompt (HLD §6.1). Bump ``PROMPT_VERSION`` on any wording change.

The few-shot examples are copies of rows from ``eval/datasets/routing.jsonl`` with
``split: dev`` (never ``test``, CLAUDE.md rule 9); ``tests/prompts/test_charlie.py`` checks
that each one still matches its dataset row.
"""

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage

from partner_agents.schemas import Lead

PROMPT_VERSION = "charlie-v1"

# Messages of the thread the router sees (HLD §6.1), the current request included.
ROUTER_HISTORY = 6
# Leads listed in the router prompt; enough for "the LA ones" or "lead 7".
ROUTER_MAX_LEADS = 25

ROUTER_SYSTEM = """\
You are Charlie, the router of an assistant that helps Consumer Packaged Goods (CPG) brands \
find B2B partners (retailers, distributors, brokers, co-packers), draft outreach that a human \
approves, and answer questions about how the CPG trade works.

Classify the user's LATEST message into exactly one intent:
- discovery: find partner companies on the web. Fill `discovery`: partner_type (retailer, \
distributor, broker, manufacturer or other; wholesalers are distributors, co-packers are \
manufacturers), product_category (the products, in a few words), location (null if none \
given), max_results (10 unless the user gives a number, at most 25) and raw_query (the \
message as written).
- cpg_qa: a question about the CPG industry: channels, partner types, fees, trade spend, \
retail calendars, labelling, food-safety basics, pitching to buyers.
- outreach: email or call partners that were already found. Set lead_selector to the words \
that pick them ("top 3", "the LA ones", "the first one"), outreach_goal to what the message \
should achieve (null if not stated) and outreach_channel to "voice" for a call, else "email".
- clarify: the request is in scope but too vague to act on, for example a discovery request \
without a product category or a partner type. Write one short clarifying_question.
- out_of_scope: anything else (creative writing, social media posts, general chit-chat, code, \
personal advice). Refuse politely; do not try to help.

Rules:
- Use earlier messages and the leads list only to understand follow-ups such as "only the \
ones in LA" or "email the top 3".
- Never invent a product category, partner type or location the user did not give or imply.
- The user may write in Hinglish or with typos; classify the meaning.
- confidence: how sure you are of the intent, from 0 to 1.
- reason: a few words, for logs only.
"""


def _example(query: str, **decision: Any) -> tuple[str, str]:
    return query, json.dumps(decision, ensure_ascii=False)


# Copied from routing.jsonl (split: dev). Keys are the dataset ids.
FEW_SHOT: dict[str, tuple[str, str]] = {
    "r002": _example(
        "Find snack food brokers in California",
        intent="discovery",
        confidence=0.95,
        discovery={
            "partner_type": "broker",
            "product_category": "snack foods",
            "location": "California",
            "max_results": 10,
            "raw_query": "Find snack food brokers in California",
        },
        reason="partner search with type, category and place",
    ),
    "r004": _example(
        "California me cookie brokers dhundo",
        intent="discovery",
        confidence=0.9,
        discovery={
            "partner_type": "broker",
            "product_category": "cookies",
            "location": "California",
            "max_results": 10,
            "raw_query": "California me cookie brokers dhundo",
        },
        reason="Hinglish partner search",
    ),
    "r008": _example(
        "What does a food broker do, and how is it different from a distributor?",
        intent="cpg_qa",
        confidence=0.95,
        reason="industry question",
    ),
    "r013": _example(
        "Draft emails to the top 3 for our gluten-free cookie brand",
        intent="outreach",
        confidence=0.9,
        lead_selector="top 3",
        outreach_goal="introduce our gluten-free cookie brand",
        outreach_channel="email",
        reason="email earlier leads",
    ),
    "r015": _example(
        "Call the first lead",
        intent="outreach",
        confidence=0.9,
        lead_selector="the first one",
        outreach_channel="voice",
        reason="call an earlier lead",
    ),
    "r016": _example(
        "Find partners",
        intent="clarify",
        confidence=0.85,
        clarifying_question=(
            "Happy to help. What kind of partner (retailer, distributor or broker), "
            "for which products, and where?"
        ),
        reason="no partner type, category or place",
    ),
    "r017": _example(
        "find me some brokers",
        intent="clarify",
        confidence=0.8,
        clarifying_question="Which products should the brokers handle, and in which region?",
        reason="discovery without a product category",
    ),
    "r019": _example(
        "Write me a poem", intent="out_of_scope", confidence=0.95, reason="creative writing"
    ),
}


def leads_summary(leads: Sequence[Lead]) -> str:
    """The thread's latest leads, numbered, without contact values."""
    if not leads:
        return "No leads have been found in this conversation yet."
    lines = []
    for number, lead in enumerate(leads[:ROUTER_MAX_LEADS], start=1):
        details = ", ".join(part for part in (lead.partner_type, lead.location) if part)
        email = "verified email" if lead.outreach_eligible else "no verified email"
        lines.append(f"{number}. {lead.company_name} ({details or 'details unknown'}; {email})")
    more = len(leads) - ROUTER_MAX_LEADS
    if more > 0:
        lines.append(f"... and {more} more")
    return "Leads from the last search in this conversation:\n" + "\n".join(lines)


def _few_shot_text() -> str:
    blocks = [f"User: {query}\nDecision: {decision}" for query, decision in FEW_SHOT.values()]
    return "Examples:\n\n" + "\n\n".join(blocks)


def build_router_messages(
    history: Sequence[AnyMessage], leads: Sequence[Lead]
) -> list[BaseMessage]:
    """System prompt (rules, examples, leads) followed by the last ``ROUTER_HISTORY`` messages."""
    system = f"{ROUTER_SYSTEM}\n{_few_shot_text()}\n\n{leads_summary(leads)}"
    return [SystemMessage(content=system), *history[-ROUTER_HISTORY:]]
