"""Charlie's graph nodes: the router (HLD §6.1).

``charlie_route`` asks the router model for a ``RouteDecision`` and then applies two rules in
code: a low-confidence decision becomes ``clarify`` (unless it is ``out_of_scope``), and a
discovery decision without a product category or partner type becomes ``clarify``.
"""

import logging
from collections.abc import Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage

from partner_agents.config import Settings
from partner_agents.graph.state import AgentState
from partner_agents.llm import StructuredOutputError, invoke_structured
from partner_agents.observability import NodeUpdate, redact_contacts, timed_node
from partner_agents.prompts.charlie import PROMPT_VERSION, build_router_messages
from partner_agents.schemas import Intent, RouteDecision, StageError

logger = logging.getLogger(__name__)

Node = Callable[[AgentState], Awaitable[NodeUpdate]]

# Where each intent goes next (HLD §5).
NEXT_NODE: dict[Intent, str] = {
    "cpg_qa": "kb_answer",
    "clarify": "ask_clarification",
    "out_of_scope": "refuse",
    "discovery": "scout_search",
    "outreach": "chase_select",
}

LOW_CONFIDENCE_QUESTION = (
    "I want to get this right. Are you looking for partner companies, help contacting "
    "partners you already found, or an answer about how the CPG trade works?"
)
MISSING_SLOTS_QUESTION = (
    "Happy to search. Which kind of partner (retailer, distributor or broker), "
    "for which products, and where?"
)
ROUTER_FAILED_QUESTION = (
    "Sorry, I could not work out what you need. Could you rephrase it in one sentence?"
)

# Values a model writes when it has no real product category.
_EMPTY_CATEGORY = frozenset({"", "unknown", "any", "none", "n/a", "na", "null", "products"})


def last_user_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def _clarify(decision: RouteDecision, question: str, reason: str) -> RouteDecision:
    return RouteDecision(
        intent="clarify",
        confidence=decision.confidence,
        clarifying_question=decision.clarifying_question or question,
        reason=f"{reason} (was {decision.intent}: {decision.reason})"[:200],
    )


def apply_route_rules(
    decision: RouteDecision, *, min_confidence: float, raw_query: str
) -> RouteDecision:
    """The coercion rules of HLD §6.1, plus tidying fields that belong to other intents."""
    if decision.intent not in ("out_of_scope", "clarify") and decision.confidence < min_confidence:
        return _clarify(decision, LOW_CONFIDENCE_QUESTION, "low confidence")
    if decision.intent == "discovery":
        intent = decision.discovery
        if intent is None or intent.product_category.strip().lower() in _EMPTY_CATEGORY:
            # A clarifying question the model wrote for a *different* intent is not reused.
            return _clarify(
                decision.model_copy(update={"clarifying_question": None}),
                MISSING_SLOTS_QUESTION,
                "missing discovery slots",
            )
        return decision.model_copy(
            update={
                "discovery": intent.model_copy(update={"raw_query": raw_query or intent.raw_query}),
                "clarifying_question": None,
                "outreach_goal": None,
                "lead_selector": None,
            }
        )
    if decision.intent == "clarify":
        return decision.model_copy(
            update={
                "discovery": None,
                "clarifying_question": decision.clarifying_question or MISSING_SLOTS_QUESTION,
            }
        )
    if decision.intent == "outreach":
        return decision.model_copy(update={"discovery": None, "clarifying_question": None})
    return decision.model_copy(
        update={"discovery": None, "clarifying_question": None, "lead_selector": None}
    )


def next_after_route(state: AgentState) -> str:
    """Conditional edge out of ``charlie_route``."""
    route = state.get("route")
    return NEXT_NODE[route.intent] if route is not None else "ask_clarification"


def make_charlie_route(model: BaseChatModel, settings: Settings) -> Node:
    """The ``charlie_route`` node: the thread's messages + latest leads -> ``route``."""

    @timed_node("charlie_route")
    async def charlie_route(state: AgentState) -> NodeUpdate:
        messages = list(state.get("messages", []))
        raw_query = last_user_text(messages)
        prompt = build_router_messages(messages, state.get("leads", []))
        try:
            decision = await invoke_structured(
                model, RouteDecision, prompt, role="router", prompt_version=PROMPT_VERSION
            )
        except StructuredOutputError as exc:
            error = StageError(stage="charlie_route", code="invalid_output", message=exc.detail)
        except Exception as exc:
            logger.warning("router call failed", extra={"error_type": type(exc).__name__})
            error = StageError(
                stage="charlie_route",
                code="model_error",
                message=f"{type(exc).__name__}: {redact_contacts(str(exc))[:200]}",
            )
        else:
            route = apply_route_rules(
                decision, min_confidence=settings.router_min_confidence, raw_query=raw_query
            )
            return {"route": route}
        fallback = RouteDecision(
            intent="clarify",
            confidence=0.0,
            clarifying_question=ROUTER_FAILED_QUESTION,
            reason="router failed",
        )
        return {"route": fallback, "errors": [error]}

    return charlie_route
