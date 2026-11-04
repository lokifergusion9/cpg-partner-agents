import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient

from partner_agents.config import Settings
from partner_agents.fakes import HashingEmbedder
from partner_agents.retrieval.chunking import Chunk
from partner_agents.retrieval.store import (
    WEB_CHUNKS,
    VectorStore,
    make_client,
    point_id,
)

NOW = datetime(2026, 11, 4, 9, 0, tzinfo=UTC)
A = "https://acme.example/contact"
B = "https://beta.example/about"


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


A_CHUNKS = [
    chunk(0, "Acme Foods brokers cookies and snacks across California"),
    chunk(1, "Our founding story and team values"),
    chunk(2, "Email sales@acme.example or call 310 555 0100", contact=True),
]
B_CHUNKS = [
    chunk(0, "Beta Distribution ships frozen pizza nationwide"),
    chunk(1, "Reach Beta at orders@beta.example", contact=True),
]


@pytest.fixture
def store() -> Iterator[VectorStore]:
    client = AsyncQdrantClient(":memory:")
    yield VectorStore(client, HashingEmbedder())
    asyncio.run(client.close())


def fill(store: VectorStore) -> None:
    async def go() -> None:
        await store.upsert_chunks("r_1", A, "Acme", A_CHUNKS, NOW)
        await store.upsert_chunks("r_1", B, "Beta", B_CHUNKS, NOW)
        await store.upsert_chunks("r_2", A, "Acme", A_CHUNKS[:1], NOW)

    run(go())


def test_search_filtered_by_run_and_url_returns_only_that_pages_chunks(store: VectorStore) -> None:
    fill(store)
    hits = run(store.search("email contact", "r_1", A, k=10))
    assert {str(h.url) for h in hits} == {A}
    assert sorted(h.chunk_index for h in hits) == [0, 1, 2]  # all of page A in run 1, no more
    assert all(h.selected_by == "vector" for h in hits)


def test_search_never_crosses_runs(store: VectorStore) -> None:
    fill(store)
    hits = run(store.search("cookies", "r_2", A, k=10))
    assert [h.chunk_index for h in hits] == [0]
    assert run(store.search("cookies", "r_3", A, k=10)) == []


def test_search_never_crosses_pages(store: VectorStore) -> None:
    fill(store)
    hits = run(store.search("frozen pizza", "r_1", A, k=10))
    assert all("pizza" not in h.text for h in hits)


def test_search_orders_by_similarity_and_respects_k(store: VectorStore) -> None:
    fill(store)
    hits = run(store.search("email sales contact", "r_1", A, k=2))
    assert len(hits) == 2
    assert hits[0].chunk_index == 2
    assert hits[0].score >= hits[1].score


def test_results_carry_the_chunk_fields(store: VectorStore) -> None:
    fill(store)
    [top] = run(store.search("cookies snacks california", "r_1", A, k=1))
    assert top.text == A_CHUNKS[0].text
    assert top.chunk_id == point_id("r_1", A, 0)
    assert 0 < top.score <= 1.0001


def test_the_payload_has_the_documented_fields(store: VectorStore) -> None:
    fill(store)

    async def payloads() -> list[dict[str, Any]]:
        points, _ = await store._client.scroll(WEB_CHUNKS, limit=50, with_payload=True)
        return [p.payload or {} for p in points]

    rows = run(payloads())
    assert len(rows) == 6
    sample = next(
        r for r in rows if r["chunk_index"] == 2 and r["run_id"] == "r_1" and r["url"] == A
    )
    assert set(sample) == {
        "run_id",
        "url",
        "title",
        "chunk_index",
        "text",
        "has_contact_pattern",
        "fetched_at",
    }
    assert sample["has_contact_pattern"] is True
    assert sample["title"] == "Acme"
    assert datetime.fromisoformat(sample["fetched_at"]) == NOW


def test_indexing_a_page_twice_replaces_its_points(store: VectorStore) -> None:
    async def go() -> list[Any]:
        await store.upsert_chunks("r_1", A, "Acme", A_CHUNKS, NOW)
        await store.upsert_chunks("r_1", A, "Acme", A_CHUNKS, NOW)
        return await store.search("anything", "r_1", A, k=50)

    assert len(run(go())) == 3


def test_upserting_nothing_writes_nothing(store: VectorStore) -> None:
    assert run(store.upsert_chunks("r_1", A, None, [], NOW)) == 0


def test_upsert_reports_how_many_points_were_written(store: VectorStore) -> None:
    assert run(store.upsert_chunks("r_1", A, "Acme", A_CHUNKS, NOW)) == 3


def test_point_ids_are_stable_and_distinct() -> None:
    assert point_id("r", A, 0) == point_id("r", A, 0)
    assert (
        len({point_id("r", A, 0), point_id("r", A, 1), point_id("r2", A, 0), point_id("r", B, 0)})
        == 4
    )


def test_ensure_collection_is_idempotent_and_shared_between_stores() -> None:
    client = AsyncQdrantClient(":memory:")
    first = VectorStore(client, HashingEmbedder())
    second = VectorStore(client, HashingEmbedder())

    async def go() -> None:
        await first.ensure_collection()
        await first.upsert_chunks("r_1", A, "Acme", A_CHUNKS, NOW)
        await second.ensure_collection()  # must not recreate and wipe the collection
        assert len(await second.search("cookies", "r_1", A, k=10)) == 3
        await client.close()

    run(go())


def test_the_collection_uses_the_embedder_dimension_and_cosine(store: VectorStore) -> None:
    async def info() -> Any:
        await store.ensure_collection()
        return await store._client.get_collection(WEB_CHUNKS)

    params = run(info()).config.params.vectors
    assert params.size == 384
    assert params.distance.value == "Cosine"


# --- TTL cleanup --------------------------------------------------------------------


def test_cleanup_removes_old_points_and_keeps_new_ones(store: VectorStore) -> None:
    async def go() -> int:
        await store.upsert_chunks("r_old", A, "Acme", A_CHUNKS, NOW - timedelta(days=8))
        await store.upsert_chunks("r_new", B, "Beta", B_CHUNKS, NOW - timedelta(days=1))
        return await store.cleanup(NOW - timedelta(days=7))

    assert run(go()) == 3
    assert run(store.search("cookies", "r_old", A, k=10)) == []
    assert len(run(store.search("pizza", "r_new", B, k=10))) == 2


def test_cleanup_with_nothing_expired_removes_nothing(store: VectorStore) -> None:
    fill(store)
    assert run(store.cleanup(NOW - timedelta(days=7))) == 0
    assert len(run(store.search("cookies", "r_1", A, k=10))) == 3


def test_cleanup_on_an_empty_store_is_fine(store: VectorStore) -> None:
    assert run(store.cleanup(NOW)) == 0


def test_cleanup_cutoff_is_exclusive(store: VectorStore) -> None:
    cutoff = NOW - timedelta(days=7)
    run(store.upsert_chunks("r_1", A, "Acme", A_CHUNKS, cutoff))
    assert run(store.cleanup(cutoff)) == 0


# --- client selection ---------------------------------------------------------------


def test_an_empty_qdrant_url_selects_the_local_store(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, qdrant_local_path=str(tmp_path / "q"))  # type: ignore[call-arg]
    client = make_client(settings)

    async def go() -> None:
        store = VectorStore(client, HashingEmbedder())
        await store.upsert_chunks("r_1", A, "Acme", A_CHUNKS, NOW)
        await client.close()

    run(go())
    assert (tmp_path / "q").exists()


def test_a_qdrant_url_selects_the_remote_client() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, qdrant_url="http://qdrant.invalid:6333", qdrant_api_key="secret-key"
    )
    client = make_client(settings)
    assert type(client._client).__name__ != "AsyncQdrantLocal"
