import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient

from partner_agents.fakes import HashingEmbedder
from partner_agents.retrieval.chunking import Chunk
from partner_agents.retrieval.evidence import build_query, select_evidence
from partner_agents.retrieval.store import VectorStore
from partner_agents.schemas import DiscoveryIntent

NOW = datetime(2026, 11, 4, tzinfo=UTC)
URL = "https://acme.example/about"
INTENT = DiscoveryIntent(
    partner_type="broker",
    product_category="cookies",
    location="California",
    raw_query="cookie brokers in California",
)


def run[T](coro: Any) -> T:
    return asyncio.run(coro)  # type: ignore[no-any-return]


def chunk(index: int, text: str, contact: bool = False) -> Chunk:
    return Chunk(
        index=index,
        text=text,
        start=0,
        end=len(text),
        tokens=len(text.split()),
        has_contact_pattern=contact,
    )


def relevant(index: int) -> Chunk:
    return chunk(index, f"We are cookie brokers serving California retailers number {index}")


def unrelated(index: int) -> Chunk:
    return chunk(index, f"Annual picnic photos and holiday schedule {index} zebra")


def footer(index: int, address: str = "info@acme.example") -> Chunk:
    return chunk(index, f"Reach us at {address} zebra picnic", contact=True)


@pytest.fixture
def store() -> Iterator[VectorStore]:
    client = AsyncQdrantClient(":memory:")
    yield VectorStore(client, HashingEmbedder())
    asyncio.run(client.close())


def index_page(store: VectorStore, chunks: list[Chunk], url: str = URL) -> None:
    run(store.upsert_chunks("r_1", url, "Acme", chunks, NOW))


def pick(store: VectorStore, k: int = 4, max_total: int = 6, url: str = URL) -> Any:
    return run(select_evidence(store, INTENT, "r_1", url, k=k, max_total=max_total))


def test_the_query_names_the_intent_and_what_we_want() -> None:
    query = build_query(INTENT)
    assert query == "cookie brokers in California broker cookies California contact email phone"


def test_a_missing_location_leaves_no_stray_spaces() -> None:
    query = build_query(INTENT.model_copy(update={"location": None}))
    assert query == "cookie brokers in California broker cookies contact email phone"


def test_a_contact_chunk_with_a_low_vector_score_is_still_selected(store: VectorStore) -> None:
    page = [relevant(i) for i in range(8)] + [footer(8)]
    index_page(store, page)
    chosen = pick(store)
    footer_chunk = next(c for c in chosen if c.chunk_index == 8)
    assert footer_chunk.selected_by == "contact_pattern"
    vector_only = run(store.search(build_query(INTENT), "r_1", URL, k=4))
    assert 8 not in {c.chunk_index for c in vector_only}  # the vector search alone would miss it


def test_the_total_never_exceeds_the_cap(store: VectorStore) -> None:
    index_page(store, [relevant(i) for i in range(6)] + [footer(6 + i) for i in range(5)])
    chosen = pick(store, k=4, max_total=6)
    assert len(chosen) == 6


def test_contact_chunks_win_when_the_cap_forces_a_choice(store: VectorStore) -> None:
    index_page(store, [relevant(i) for i in range(6)] + [footer(6 + i) for i in range(3)])
    chosen = pick(store, k=4, max_total=6)
    contact = [c for c in chosen if c.selected_by == "contact_pattern"]
    assert len(contact) == 3  # 3 contact + the 3 best vector chunks
    assert len(chosen) - len(contact) == 3


def test_at_most_three_contact_chunks_are_added(store: VectorStore) -> None:
    index_page(store, [relevant(i) for i in range(4)] + [footer(4 + i) for i in range(6)])
    chosen = pick(store, k=4, max_total=20)
    assert sum(c.selected_by == "contact_pattern" for c in chosen) == 3


def test_the_result_is_sorted_by_chunk_index(store: VectorStore) -> None:
    index_page(
        store, [unrelated(0), relevant(1), footer(2), relevant(3), unrelated(4), relevant(5)]
    )
    indexes = [c.chunk_index for c in pick(store)]
    assert indexes == sorted(indexes)


def test_selected_by_is_set_per_chunk(store: VectorStore) -> None:
    index_page(store, [relevant(i) for i in range(5)] + [footer(5)])
    by_index = {c.chunk_index: c.selected_by for c in pick(store, k=4)}
    assert by_index[5] == "contact_pattern"
    assert [by_index[i] for i in sorted(by_index) if i != 5] == ["vector"] * 4


def test_a_contact_chunk_already_in_the_top_k_is_not_duplicated(store: VectorStore) -> None:
    contact_and_relevant = chunk(
        2, "Cookie brokers in California: email sales@acme.example", contact=True
    )
    index_page(store, [relevant(0), relevant(1), contact_and_relevant, unrelated(3)])
    chosen = pick(store, k=4)
    assert [c.chunk_index for c in chosen].count(2) == 1
    assert next(c for c in chosen if c.chunk_index == 2).selected_by == "vector"


def test_a_page_without_contact_chunks_gets_only_vector_results(store: VectorStore) -> None:
    index_page(store, [relevant(i) for i in range(10)])
    chosen = pick(store, k=4)
    assert len(chosen) == 4
    assert {c.selected_by for c in chosen} == {"vector"}


def test_a_short_page_returns_all_its_chunks(store: VectorStore) -> None:
    index_page(store, [relevant(0), footer(1)])
    assert [c.chunk_index for c in pick(store)] == [0, 1]


def test_only_the_requested_page_is_used(store: VectorStore) -> None:
    other = "https://beta.example/"
    index_page(store, [relevant(0), footer(1)], url=URL)
    index_page(store, [chunk(0, "Beta frozen pizza", True)] * 1, url=other)
    assert {str(c.url) for c in pick(store)} == {URL}
    assert {str(c.url) for c in pick(store, url=other)} == {other}


def test_an_unindexed_page_has_no_evidence(store: VectorStore) -> None:
    assert pick(store, url="https://nothing.example/") == []


def test_max_total_smaller_than_the_contact_allowance(store: VectorStore) -> None:
    index_page(store, [relevant(i) for i in range(4)] + [footer(4 + i) for i in range(4)])
    chosen = pick(store, k=4, max_total=2)
    assert len(chosen) == 2
    assert {c.selected_by for c in chosen} == {"contact_pattern"}


def test_one_chunk_of_room_goes_to_a_contact_chunk(store: VectorStore) -> None:
    index_page(store, [relevant(0), footer(1)])
    [only] = pick(store, k=1, max_total=1)  # k=1: the footer is not in the vector top-k
    assert only.chunk_index == 1
