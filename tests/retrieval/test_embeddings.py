import asyncio
import math
from typing import Any

import pytest

from partner_agents.fakes import HashingEmbedder
from partner_agents.retrieval.embeddings import (
    BGE_QUERY_PREFIX,
    SentenceTransformerEmbedder,
)

DIM = 4


class FakeModel:
    """Stands in for SentenceTransformer: records every ``encode`` call."""

    def __init__(self, dimension: int = DIM) -> None:
        self.dimension = dimension
        self.batches: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.batches.append(list(texts))
        self.kwargs.append(kwargs)
        return [[float(len(text)), 0.0, 0.0, float(i)] for i, text in enumerate(texts)]


def embedder(
    model: FakeModel | None = None, **kwargs: Any
) -> tuple[SentenceTransformerEmbedder, FakeModel]:
    fake = model or FakeModel()
    options: dict[str, Any] = {"dimension": DIM, "model_factory": lambda name: fake}
    return SentenceTransformerEmbedder("fake-model", **{**options, **kwargs}), fake


def run[T](coro: Any) -> T:
    return asyncio.run(coro)  # type: ignore[no-any-return]


def test_documents_are_embedded_in_batches_of_32() -> None:
    emb, model = embedder()
    texts = [f"document number {n}" for n in range(70)]
    vectors = run(emb.embed_documents(texts))
    assert [len(batch) for batch in model.batches] == [32, 32, 6]
    assert len(vectors) == 70
    assert all(len(v) == DIM for v in vectors)


def test_the_model_is_asked_for_normalised_numpy_output_without_a_progress_bar() -> None:
    emb, model = embedder()
    run(emb.embed_documents(["a"]))
    assert model.kwargs[0] == {
        "batch_size": 32,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }


def test_the_model_loads_lazily_and_once() -> None:
    loads: list[str] = []
    fake = FakeModel()

    def factory(name: str) -> FakeModel:
        loads.append(name)
        return fake

    emb = SentenceTransformerEmbedder("bge", dimension=DIM, model_factory=factory)
    assert loads == []
    run(emb.embed_documents(["a"]))
    run(emb.embed_documents(["b"]))
    assert loads == ["bge"]


def test_order_is_preserved_when_some_texts_are_cached() -> None:
    emb, model = embedder()
    first = run(emb.embed_documents(["alpha", "gamma"]))
    mixed = run(emb.embed_documents(["gamma", "beta", "alpha"]))
    assert mixed[0] == first[1]
    assert mixed[2] == first[0]
    assert model.batches[-1] == ["beta"]  # only the unseen text reached the model


def test_identical_text_is_cached() -> None:
    emb, model = embedder()
    run(emb.embed_documents(["same text"]))
    run(emb.embed_documents(["same text"]))
    assert len(model.batches) == 1
    assert (emb.cache_hits, emb.cache_misses) == (1, 1)


def test_duplicates_inside_one_request_are_encoded_once() -> None:
    emb, model = embedder()
    vectors = run(emb.embed_documents(["dup", "dup", "other", "dup"]))
    assert model.batches == [["dup", "other"]]
    assert vectors[0] == vectors[1] == vectors[3]


def test_least_recently_used_text_is_evicted() -> None:
    emb, model = embedder(cache_size=2)
    run(emb.embed_documents(["a"]))
    run(emb.embed_documents(["b"]))
    run(emb.embed_documents(["a"]))  # a is now the most recent
    run(emb.embed_documents(["c"]))  # evicts b
    calls_before = len(model.batches)
    run(emb.embed_documents(["a"]))
    assert len(model.batches) == calls_before  # still cached
    run(emb.embed_documents(["b"]))
    assert len(model.batches) == calls_before + 1  # was evicted


def test_a_request_bigger_than_the_cache_still_returns_every_vector() -> None:
    emb, _ = embedder(cache_size=2)
    texts = [f"t{n}" for n in range(10)]
    assert len(run(emb.embed_documents(texts))) == 10


def test_returned_vectors_are_copies() -> None:
    emb, _ = embedder()
    [vector] = run(emb.embed_documents(["x"]))
    vector[0] = 999.0
    [again] = run(emb.embed_documents(["x"]))
    assert again[0] != 999.0


def test_queries_get_the_bge_instruction_prefix() -> None:
    emb, model = embedder()
    run(emb.embed_query("cookie brokers contact"))
    assert model.batches == [[BGE_QUERY_PREFIX + "cookie brokers contact"]]


def test_a_query_and_a_document_with_the_same_words_are_cached_separately() -> None:
    emb, model = embedder()
    run(emb.embed_documents(["cookie brokers"]))
    run(emb.embed_query("cookie brokers"))
    assert len(model.batches) == 2


def test_a_model_with_the_wrong_dimension_is_rejected() -> None:
    emb, _ = embedder(FakeModel(dimension=3), dimension=5)
    with pytest.raises(ValueError, match="expected 5"):
        run(emb.embed_documents(["x"]))


def test_empty_input_embeds_nothing() -> None:
    emb, model = embedder()
    assert run(emb.embed_documents([])) == []
    assert model.batches == []


# --- the hashing fake used by the store tests ----------------------------------------


def test_hashing_embedder_is_normalised_deterministic_and_topical() -> None:
    fake = HashingEmbedder()
    [a, b, c] = run(
        fake.embed_documents(["cookie broker california", "california cookie", "tires"])
    )
    assert fake.dimension == 384 and len(a) == 384
    assert math.isclose(sum(x * x for x in a), 1.0)
    assert a == run(fake.embed_query("cookie broker california"))
    dot = lambda u, v: sum(x * y for x, y in zip(u, v, strict=True))  # noqa: E731
    assert dot(a, b) > dot(a, c)


def test_hashing_embedder_handles_empty_text() -> None:
    [vector] = run(HashingEmbedder().embed_documents([""]))
    assert math.isclose(sum(x * x for x in vector), 1.0)


# --- the real model (downloads BAAI/bge-small-en-v1.5 on first use) -------------------


@pytest.mark.live
def test_real_bge_small_returns_384_dimensional_normalised_vectors() -> None:
    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    docs = run(
        emb.embed_documents(
            ["Contact our sales team at sales@acme-foods.com", "Tires and wheel alignment"]
        )
    )
    query = run(emb.embed_query("how do I email the company"))
    assert len(docs[0]) == emb.dimension == 384
    assert math.isclose(sum(x * x for x in docs[0]), 1.0, rel_tol=1e-3)
    similarity = [sum(q * d for q, d in zip(query, doc, strict=True)) for doc in docs]
    assert similarity[0] > similarity[1]
