"""Search and embedding caches and their per-run counters (T-34, HLD sections 6.3 and 6.5)."""

import asyncio
from typing import Any

import respx
from qdrant_client import AsyncQdrantClient

from partner_agents.graph.build import build_graph, run_turn
from partner_agents.retrieval.embeddings import SentenceTransformerEmbedder
from partner_agents.retrieval.store import VectorStore
from partner_agents.schemas import DiscoveryIntent, SearchHit
from partner_agents.storage.repo import InMemoryRepository
from partner_agents.utils.cache_stats import record_cache, track_cache
from partner_agents.web.search import SearchService
from tests.graph.test_build import (  # noqa: F401  (fixtures are used by name)
    ACME,
    BETA,
    DISCOVERY,
    client,
    instant_retries,
    make_deps,
    serve_sites,
)

INTENT = DiscoveryIntent(
    partner_type="broker", product_category="cookies", location="California", raw_query="x"
)


class CountingProvider:
    name = "counting"

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.queries.append(query)
        return [
            SearchHit(url=url, title=url, snippet="s", rank=i + 1, provider=self.name)  # type: ignore[arg-type]
            for i, url in enumerate([ACME, BETA])
        ]


class CountingModel:
    """A stand-in sentence-transformer that counts the texts it really encodes."""

    def __init__(self) -> None:
        self.encoded = 0

    def encode(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.encoded += len(texts)
        return [[float(len(text) % 7 + 1)] * 8 for text in texts]


def make_embedder() -> tuple[SentenceTransformerEmbedder, CountingModel]:
    model = CountingModel()
    return SentenceTransformerEmbedder("stub", dimension=8, model_factory=lambda _: model), model


# --- the counter --------------------------------------------------------------------


def test_the_counter_collects_hits_and_misses_and_ignores_zero() -> None:
    with track_cache() as stats:
        record_cache("search", hits=1)
        record_cache("search", hits=2, misses=1)
        record_cache("embed", misses=0)
    assert stats == {"search_hits": 3, "search_misses": 1}


def test_recording_outside_a_run_does_nothing() -> None:
    record_cache("search", hits=1)  # no counter is open: no error and no state
    with track_cache() as stats:
        pass
    assert stats == {}


def test_counters_of_two_runs_are_separate() -> None:
    with track_cache() as first:
        record_cache("embed", hits=1)
    with track_cache() as second:
        record_cache("embed", misses=1)
    assert first == {"embed_hits": 1} and second == {"embed_misses": 1}


# --- the search cache ---------------------------------------------------------------


def test_a_repeated_discovery_makes_no_search_api_calls() -> None:
    provider = CountingProvider()
    service = SearchService([provider], InMemoryRepository(), frozenset())

    async def go() -> tuple[dict[str, int], dict[str, int]]:
        with track_cache() as first:
            await service.discover(INTENT)
        calls_after_first = len(provider.queries)
        with track_cache() as second:
            again = await service.discover(INTENT)
        assert again.hits and again.cache_hits == calls_after_first
        return first, second

    first, second = asyncio.run(go())
    assert len(provider.queries) == 2  # one call for each of the two queries, only the first time
    assert first == {"search_misses": 2} and second == {"search_hits": 2}


def test_a_query_that_differs_only_in_case_and_spaces_is_served_from_the_cache() -> None:
    provider = CountingProvider()
    service = SearchService([provider], InMemoryRepository(), frozenset())
    shouting = INTENT.model_copy(
        update={"product_category": "  COOKIES ", "location": "california"}
    )

    async def go() -> None:
        await service.discover(INTENT)
        await service.discover(shouting)

    asyncio.run(go())
    assert len(provider.queries) == 2


# --- the embedding cache ------------------------------------------------------------


def test_identical_chunks_are_not_embedded_again() -> None:
    embedder, model = make_embedder()
    chunks = [f"chunk number {i} about cookies" for i in range(5)]

    async def go() -> tuple[dict[str, int], dict[str, int]]:
        with track_cache() as first:
            await embedder.embed_documents(chunks)
        encoded_first = model.encoded
        with track_cache() as second:
            await embedder.embed_documents([*chunks, chunks[0]])
        assert model.encoded == encoded_first  # not one more text was encoded
        return first, second

    first, second = asyncio.run(go())
    assert first == {"embed_misses": 5} and second == {"embed_hits": 6}
    assert (embedder.cache_hits, embedder.cache_misses) == (6, 5)  # the lifetime counters stay


def test_a_query_and_a_document_with_the_same_text_are_different_entries() -> None:
    embedder, model = make_embedder()

    async def go() -> None:
        await embedder.embed_documents(["slotting fee"])
        await embedder.embed_query("slotting fee")  # carries the BGE prefix: a miss

    asyncio.run(go())
    assert model.encoded == 2


# --- in a whole run -----------------------------------------------------------------


@respx.mock
def test_a_second_identical_discovery_costs_no_search_call_and_no_new_embedding(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    serve_sites()
    deps = make_deps(client, DISCOVERY, DISCOVERY)
    provider = CountingProvider()
    embedder, model = make_embedder()
    deps.scout.search = SearchService([provider], deps.repo, frozenset())
    deps.scout.store = VectorStore(client, embedder)
    graph = build_graph(deps)

    async def go() -> Any:
        try:
            first = await run_turn(graph, deps, thread_id="t_1", user_id="u_1", text="find")
            encoded = model.encoded
            second = await run_turn(graph, deps, thread_id="t_2", user_id="u_1", text="find")
            return first, second, encoded
        finally:
            await deps.scout.fetcher.aclose()

    first, second, encoded_after_first = asyncio.run(go())
    assert first.run.cache["search_misses"] == 2 and first.run.cache["embed_misses"] > 0
    assert "search_hits" not in first.run.cache
    assert len(provider.queries) == 2  # the second run made 0 search API calls
    assert second.run.cache["search_hits"] == 2 and "search_misses" not in second.run.cache
    assert "embed_misses" not in second.run.cache and second.run.cache["embed_hits"] > 0
    assert model.encoded == encoded_after_first  # 0 re-embeddings of identical chunks
    saved = asyncio.run(deps.repo.get_run(second.run.run_id))
    assert saved is not None and saved.cache == second.run.cache
