"""The startup job that removes expired web chunks (T-35, HLD section 8.2)."""

import asyncio
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from partner_agents.api.main import create_app
from partner_agents.retrieval.chunking import Chunk
from tests.graph.test_build import (  # noqa: F401  (fixtures are used by name)
    client,
    instant_retries,
    make_deps,
)

NOW = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
OLD, NEW = "https://old.example/", "https://new.example/"


def one_chunk(text: str) -> list[Chunk]:
    return [Chunk(index=0, text=text, start=0, end=len(text), tokens=3, has_contact_pattern=False)]


def test_starting_the_app_removes_expired_chunks_and_keeps_fresh_ones(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    deps = make_deps(client)
    deps.clock = lambda: NOW
    ttl = deps.settings.data_ttl_days
    store = deps.scout.store

    async def seed() -> None:
        await store.upsert_chunks(
            "r_old", OLD, "Old", one_chunk("old cookies page"), NOW - timedelta(days=ttl, hours=1)
        )
        await store.upsert_chunks(
            "r_new", NEW, "New", one_chunk("new cookies page"), NOW - timedelta(days=1)
        )

    asyncio.run(seed())
    with TestClient(create_app(deps, with_ui=False)):
        old = asyncio.run(store.search("cookies", "r_old", OLD, k=5))
        new = asyncio.run(store.search("cookies", "r_new", NEW, k=5))
    assert old == [] and len(new) == 1


def test_a_vector_store_that_fails_does_not_stop_the_app(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    deps = make_deps(client)

    async def broken(older_than: datetime) -> int:
        raise ConnectionError("qdrant is down")

    deps.scout.store.cleanup = broken  # type: ignore[method-assign]
    with TestClient(create_app(deps, with_ui=False)) as http:
        assert http.get("/api/health").status_code == 200
