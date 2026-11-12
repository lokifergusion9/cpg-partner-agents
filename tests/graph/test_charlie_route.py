"""The ``charlie_route`` node with a scripted router model (HLD §6.1)."""

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from partner_agents import llm
from partner_agents.config import Settings
from partner_agents.fakes import ScriptedChatModel
from partner_agents.graph.charlie import (
    LOW_CONFIDENCE_QUESTION,
    MISSING_SLOTS_QUESTION,
    ROUTER_FAILED_QUESTION,
    apply_route_rules,
    make_charlie_route,
    next_after_route,
)
from partner_agents.graph.state import AgentState
from partner_agents.schemas import DiscoveryIntent, Lead, RouteDecision
from partner_agents.utils.retry import RetryPolicy

SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "RETRY_POLICY", RetryPolicy(attempts=1, base_delay_s=0))


def discovery(category: str = "cookies", **extra: Any) -> DiscoveryIntent:
    return DiscoveryIntent(
        partner_type="broker", product_category=category, raw_query="model wrote this", **extra
    )


def decision(intent: str, confidence: float = 0.9, **fields: Any) -> RouteDecision:
    return RouteDecision(intent=intent, confidence=confidence, reason="r", **fields)  # type: ignore[arg-type]


def route(
    *replies: RouteDecision | str, text: str = "find cookie brokers in California", **state: Any
) -> tuple[dict[str, Any], ScriptedChatModel]:
    model = ScriptedChatModel.scripted(*replies)
    node = make_charlie_route(model, SETTINGS)
    update = asyncio.run(node(AgentState(messages=[HumanMessage(content=text)], **state)))
    return update, model


# --- the five branches --------------------------------------------------------------

BRANCHES = [
    (decision("discovery", discovery=discovery()), "rate_gate"),
    (decision("cpg_qa"), "kb_answer"),
    (decision("outreach", lead_selector="top 3"), "chase_select"),
    (decision("clarify", clarifying_question="Which products?"), "ask_clarification"),
    (decision("out_of_scope"), "refuse"),
]


@pytest.mark.parametrize(("reply", "expected_next"), BRANCHES)
def test_each_intent_routes_to_its_branch(reply: RouteDecision, expected_next: str) -> None:
    update, _ = route(reply)
    assert update["route"].intent == reply.intent
    assert next_after_route(AgentState(route=update["route"])) == expected_next
    assert "errors" not in update
    assert "charlie_route" in update["timings_ms"]


def test_the_discovery_intent_keeps_the_users_own_words() -> None:
    update, _ = route(decision("discovery", discovery=discovery()), text="cookie brokers in CA")
    assert update["route"].discovery.raw_query == "cookie brokers in CA"


def test_outreach_keeps_selector_and_channel() -> None:
    reply = decision("outreach", lead_selector="the first one", outreach_channel="voice")
    update, _ = route(reply, text="Call the first lead")
    assert (update["route"].lead_selector, update["route"].outreach_channel) == (
        "the first one",
        "voice",
    )


# --- the coercion rules -------------------------------------------------------------


@pytest.mark.parametrize("intent", ["discovery", "cpg_qa", "outreach"])
def test_low_confidence_becomes_clarify(intent: str) -> None:
    fields = {"discovery": discovery()} if intent == "discovery" else {}
    update, _ = route(decision(intent, confidence=0.4, **fields))
    assert update["route"].intent == "clarify"
    assert update["route"].clarifying_question == LOW_CONFIDENCE_QUESTION
    assert update["route"].discovery is None


def test_low_confidence_out_of_scope_is_still_refused() -> None:
    update, _ = route(decision("out_of_scope", confidence=0.3), text="write me a poem")
    assert update["route"].intent == "out_of_scope"


def test_the_threshold_itself_is_confident_enough() -> None:
    update, _ = route(decision("cpg_qa", confidence=SETTINGS.router_min_confidence))
    assert update["route"].intent == "cpg_qa"


@pytest.mark.parametrize(
    "intent", [None, discovery(category=""), discovery(category="  "), discovery("unknown")]
)
def test_discovery_without_its_slots_becomes_clarify(intent: DiscoveryIntent | None) -> None:
    update, _ = route(decision("discovery", discovery=intent), text="find me some brokers")
    assert update["route"].intent == "clarify"
    assert update["route"].clarifying_question == MISSING_SLOTS_QUESTION


def test_a_clarify_without_a_question_gets_a_default_one() -> None:
    update, _ = route(decision("clarify"))
    assert update["route"].clarifying_question == MISSING_SLOTS_QUESTION


def test_a_models_own_clarifying_question_is_kept() -> None:
    update, _ = route(decision("clarify", confidence=0.2, clarifying_question="Which region?"))
    assert update["route"].clarifying_question == "Which region?"


def test_fields_of_other_intents_are_dropped() -> None:
    noisy = decision("cpg_qa", discovery=discovery(), clarifying_question="?", lead_selector="x")
    tidy = apply_route_rules(noisy, min_confidence=0.6, raw_query="q")
    assert (tidy.discovery, tidy.clarifying_question, tidy.lead_selector) == (None, None, None)


# --- failures -----------------------------------------------------------------------


def test_an_invalid_router_reply_twice_asks_the_user_to_rephrase() -> None:
    update, _ = route("not json", "still not json")
    assert update["route"].intent == "clarify"
    assert update["route"].clarifying_question == ROUTER_FAILED_QUESTION
    [error] = update["errors"]
    assert (error.stage, error.code) == ("charlie_route", "invalid_output")


def test_a_router_model_error_is_recorded_not_raised() -> None:
    update, _ = route()  # no scripted replies -> the model raises
    assert update["route"].intent == "clarify"
    assert update["errors"][0].code == "model_error"


# --- what the model sees ------------------------------------------------------------


def lead(name: str, location: str) -> Lead:
    return Lead(
        id=f"l_{name}",
        run_id="r_1",
        user_id="u",
        company_name=name,
        partner_type="broker",
        location=location,
        website=None,
        email="sales@acme.example",
        email_status="verified",
        phone=None,
        phone_status="unchecked",
        source_url="https://acme.example/",  # type: ignore[arg-type]
        evidence_quote="sales@acme.example",
        created_at=datetime.now(UTC),
    )


def test_the_prompt_has_the_last_six_messages_and_the_leads_without_contacts() -> None:
    history: list[Any] = []
    for i in range(5):
        history += [HumanMessage(content=f"question {i}"), AIMessage(content=f"answer {i}")]
    history.append(HumanMessage(content="only the ones in LA"))
    model = ScriptedChatModel.scripted(decision("outreach", lead_selector="the LA ones"))
    node = make_charlie_route(model, SETTINGS)
    asyncio.run(node(AgentState(messages=history, leads=[lead("Acme Foods", "Los Angeles, CA")])))
    [sent] = model.received
    assert len(sent) == 1 + 6  # system + the last six messages
    assert sent[-1].text == "only the ones in LA"
    assert sent[1].text == "answer 2"  # 11 messages in the thread, the oldest 5 dropped
    system = sent[0].text
    assert "1. Acme Foods (broker, Los Angeles, CA; verified email)" in system
    assert "sales@acme.example" not in system
