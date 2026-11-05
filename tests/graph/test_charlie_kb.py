"""``kb_answer``, ``refuse`` and ``ask_clarification`` (HLD §6.2), offline: Qdrant ``:memory:``
with a hashing embedder and a scripted writer model."""

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from qdrant_client import AsyncQdrantClient

from partner_agents import llm
from partner_agents.config import Settings
from partner_agents.fakes import HashingEmbedder, ScriptedChatModel
from partner_agents.graph.charlie import (
    NOT_IN_KB,
    REFUSAL,
    ask_clarification,
    cited_numbers,
    make_kb_answer,
    refuse,
)
from partner_agents.graph.state import AgentState
from partner_agents.retrieval.chunking import Chunk
from partner_agents.retrieval.store import KnowledgeStore
from partner_agents.schemas import KBAnswer, RouteDecision
from partner_agents.utils.retry import RetryPolicy

QUESTION = "What is a slotting fee and who pays the slotting fee?"
NOTES = {
    "slotting-fees": (
        "Slotting fees",
        "A slotting fee is a payment a brand makes to a retailer so the retailer puts a new "
        "product on its shelf. The brand pays the slotting fee, often as free product.",
    ),
    "food-brokers": (
        "Food brokers",
        "A food broker represents several brands to retail buyers and is paid a commission.",
    ),
}


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "RETRY_POLICY", RetryPolicy(attempts=1, base_delay_s=0))


@pytest.fixture
def store() -> Iterator[KnowledgeStore]:
    kb = KnowledgeStore(AsyncQdrantClient(":memory:"), HashingEmbedder())

    async def load() -> None:
        for doc_id, (title, body) in NOTES.items():
            text = f"{title}\n\n{body}"
            chunk = Chunk(
                index=0, text=text, start=0, end=len(text), tokens=50, has_contact_pattern=False
            )
            await kb.upsert_doc(
                doc_id, title, f"https://en.wikipedia.org/wiki/{doc_id}", "t", [chunk]
            )

    asyncio.run(load())
    yield kb
    asyncio.run(kb.close())


def answer(
    store: KnowledgeStore, *replies: Any, question: str = QUESTION, min_score: float = 0.2
) -> tuple[dict[str, Any], ScriptedChatModel]:
    model = ScriptedChatModel.scripted(*replies)
    settings = Settings(_env_file=None, kb_min_score=min_score)  # type: ignore[call-arg]
    node = make_kb_answer(store, model, settings)
    update = asyncio.run(node(AgentState(messages=[HumanMessage(content=question)])))
    return update, model


def reply_text(update: dict[str, Any]) -> str:
    [message] = update["messages"]
    return str(message.content)


# --- answers ------------------------------------------------------------------------


def test_the_citation_markers_map_to_the_sources_shown(store: KnowledgeStore) -> None:
    update, model = answer(store, KBAnswer(answer="The brand pays it to get shelf space [1]."))
    text = reply_text(update)
    assert text.startswith("The brand pays it to get shelf space [1].")
    # [1] is the best passage, which the hashing embedder ranks first: the slotting-fees note.
    assert "Sources:\n[1] Slotting fees: https://en.wikipedia.org/wiki/slotting-fees" in text
    assert "Food brokers" not in text.split("Sources:")[1]  # uncited passages are not listed
    prompt = model.received[0][1].text
    assert prompt.index("[1] Slotting fees") < prompt.index("Question: " + QUESTION)
    assert update["errors"] == []


def test_a_low_best_score_gives_the_fallback_without_calling_the_model(
    store: KnowledgeStore,
) -> None:
    update, model = answer(store, question="quantum chromodynamics gluon", min_score=0.5)
    assert reply_text(update) == NOT_IN_KB
    assert model.received == []


def test_an_answer_without_a_citation_gives_the_fallback(store: KnowledgeStore) -> None:
    update, _ = answer(store, KBAnswer(answer="Brands pay retailers for shelf space."))
    assert reply_text(update) == NOT_IN_KB


def test_an_answer_citing_a_passage_that_was_not_shown_gives_the_fallback(
    store: KnowledgeStore,
) -> None:
    update, _ = answer(store, KBAnswer(answer="Brands pay it [1][7]."))
    assert reply_text(update) == NOT_IN_KB


def test_an_empty_knowledge_base_gives_the_fallback() -> None:
    empty = KnowledgeStore(AsyncQdrantClient(":memory:"), HashingEmbedder())
    update, model = answer(empty)
    assert reply_text(update) == NOT_IN_KB
    assert model.received == []


def test_a_writer_failure_is_recorded_and_the_user_gets_the_fallback(
    store: KnowledgeStore,
) -> None:
    update, _ = answer(store, "not json", "not json either")
    assert reply_text(update) == NOT_IN_KB
    [error] = update["errors"]
    assert (error.stage, error.code) == ("kb_answer", "invalid_output")


@pytest.mark.parametrize(
    ("text", "passages", "expected"),
    [
        ("a [1] b [2]", 5, [1, 2]),
        ("a [2][2][1]", 5, [1, 2]),
        ("a [1, 3]", 3, [1, 3]),
        ("no citation", 5, None),
        ("a [0]", 5, None),
        ("a [6]", 5, None),
        ("a [1] and [9]", 5, None),
        ("see [note]", 5, None),
    ],
)
def test_cited_numbers(text: str, passages: int, expected: list[int] | None) -> None:
    assert cited_numbers(text, passages) == expected


# --- refusal and clarification ------------------------------------------------------


def test_out_of_scope_gets_a_refusal_that_lists_the_capabilities() -> None:
    update = asyncio.run(refuse(AgentState(messages=[HumanMessage(content="Write me a poem")])))
    text = reply_text(update)
    assert text == REFUSAL
    for capability in ("find retailers, distributors and brokers", "draft outreach", "CPG trade"):
        assert capability in text
    assert "refuse" in update["timings_ms"]


def test_clarify_asks_the_routers_question() -> None:
    route = RouteDecision(
        intent="clarify", confidence=0.8, clarifying_question="Which region?", reason="r"
    )
    update = asyncio.run(ask_clarification(AgentState(route=route)))
    assert reply_text(update) == "Which region?"
