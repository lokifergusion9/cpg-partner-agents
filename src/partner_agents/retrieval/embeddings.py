"""Text embeddings (HLD §6.5, ADR-003).

``SentenceTransformerEmbedder`` runs ``BAAI/bge-small-en-v1.5`` locally (384-d, normalised, so
cosine = dot product). Documents are embedded in batches of 32. Queries get the BGE
instruction prefix. An in-process LRU cache keyed by ``sha256(text)`` skips re-embedding the
same text. The model loads on first use and encodes in a worker thread.
"""

import asyncio
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Protocol

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 32
CACHE_SIZE = 4096


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def _load_sentence_transformer(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        dimension: int = EMBEDDING_DIMENSION,
        batch_size: int = BATCH_SIZE,
        cache_size: int = CACHE_SIZE,
        model_factory: Callable[[str], Any] = _load_sentence_transformer,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._cache_size = cache_size
        self._model_factory = model_factory
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(list(texts))

    async def embed_query(self, text: str) -> list[float]:
        [vector] = await self._embed([BGE_QUERY_PREFIX + text])
        return vector

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        missing: dict[str, str] = {}  # key -> text, each distinct text once
        for key, text in zip(keys, texts, strict=True):
            if key in self._cache:
                self._cache.move_to_end(key)
                self.cache_hits += 1
            elif key not in missing:
                missing[key] = text
                self.cache_misses += 1
            else:
                self.cache_hits += 1
        pending = list(missing.items())
        for offset in range(0, len(pending), self._batch_size):
            batch = pending[offset : offset + self._batch_size]
            vectors = await asyncio.to_thread(self._encode, [text for _, text in batch])
            for (key, _), vector in zip(batch, vectors, strict=True):
                self._cache[key] = vector
        # Evict only after the whole request is stored, so a big request cannot evict itself.
        while len(self._cache) > max(self._cache_size, len(set(keys))):
            self._cache.popitem(last=False)
        return [list(self._cache[key]) for key in keys]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        with self._load_lock:
            if self._model is None:
                self._model = self._model_factory(self._model_name)
        encoded = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors: list[list[float]] = [[float(x) for x in row] for row in encoded]
        if vectors and len(vectors[0]) != self._dimension:
            raise ValueError(
                f"{self._model_name} gave {len(vectors[0])}-d vectors, expected {self._dimension}"
            )
        return vectors
