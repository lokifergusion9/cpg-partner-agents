"""Qdrant vector store for fetched page chunks (HLD §6.5, §8.2).

Collection ``web_chunks``: 384-d cosine vectors with the payload ``run_id``, ``url``,
``title``, ``chunk_index``, ``text``, ``has_contact_pattern``, ``fetched_at``. Searches are
always filtered by ``run_id`` *and* ``url`` so one page can never crowd out another
(ADR-007). A point id is derived from ``(run_id, url, chunk_index)``, so indexing the same
page twice replaces its points.
"""

import logging
import uuid
import warnings
from collections.abc import Collection, Sequence
from datetime import UTC, datetime

from pydantic import HttpUrl, TypeAdapter
from qdrant_client import AsyncQdrantClient, models

from partner_agents.config import Settings
from partner_agents.retrieval.chunking import Chunk
from partner_agents.retrieval.embeddings import Embedder
from partner_agents.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

WEB_CHUNKS = "web_chunks"
_URL = TypeAdapter(HttpUrl)


def make_client(settings: Settings) -> AsyncQdrantClient:
    """Qdrant Cloud when ``QDRANT_URL`` is set, otherwise an on-disk local store."""
    if settings.qdrant_is_local:
        return AsyncQdrantClient(path=settings.qdrant_local_path)
    key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=key)


def point_id(run_id: str, url: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}|{url}|{chunk_index}"))


def page_filter(run_id: str, url: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key="run_id", match=models.MatchValue(value=run_id)),
            models.FieldCondition(key="url", match=models.MatchValue(value=url)),
        ]
    )


class VectorStore:
    def __init__(
        self, client: AsyncQdrantClient, embedder: Embedder, *, collection: str = WEB_CHUNKS
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._ready = False

    async def close(self) -> None:
        await self._client.close()

    async def ensure_collection(self) -> None:
        """Create the collection and its payload indexes if they do not exist."""
        if self._ready:
            return
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=self._embedder.dimension, distance=models.Distance.COSINE
                ),
            )
            with warnings.catch_warnings():
                # The local store accepts but ignores payload indexes, and says so.
                warnings.simplefilter("ignore", UserWarning)
                for field in ("run_id", "url"):
                    await self._client.create_payload_index(
                        self._collection, field, models.PayloadSchemaType.KEYWORD
                    )
                await self._client.create_payload_index(
                    self._collection, "fetched_at", models.PayloadSchemaType.DATETIME
                )
        self._ready = True

    async def upsert_chunks(
        self,
        run_id: str,
        url: str,
        title: str | None,
        chunks: Sequence[Chunk],
        fetched_at: datetime,
    ) -> int:
        """Embed and store ``chunks`` of one page. Returns the number of points written."""
        if not chunks:
            return 0
        await self.ensure_collection()
        vectors = await self._embedder.embed_documents([chunk.text for chunk in chunks])
        points = [
            models.PointStruct(
                id=point_id(run_id, url, chunk.index),
                vector=vector,
                payload={
                    "run_id": run_id,
                    "url": url,
                    "title": title,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "has_contact_pattern": chunk.has_contact_pattern,
                    "fetched_at": fetched_at.astimezone(UTC).isoformat(),
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self._client.upsert(self._collection, points=points)
        return len(points)

    async def search(
        self,
        query: str,
        run_id: str,
        url: str,
        k: int,
        *,
        contact_only: bool = False,
        exclude: Collection[int] = (),
    ) -> list[RetrievedChunk]:
        """The ``k`` chunks of one page (this run, this url) most similar to ``query``.

        ``contact_only`` keeps chunks flagged ``has_contact_pattern`` (they come back marked
        ``selected_by="contact_pattern"``); ``exclude`` skips chunk indexes already chosen.
        """
        await self.ensure_collection()
        vector = await self._embedder.embed_query(query)
        query_filter = page_filter(run_id, url)
        if contact_only:
            query_filter.must.append(  # type: ignore[union-attr]
                models.FieldCondition(
                    key="has_contact_pattern", match=models.MatchValue(value=True)
                )
            )
        if exclude:
            query_filter.must_not = [
                models.FieldCondition(key="chunk_index", match=models.MatchAny(any=sorted(exclude)))
            ]
        response = await self._client.query_points(
            self._collection,
            query=vector,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
        )
        selected_by = "contact_pattern" if contact_only else "vector"
        return [
            _to_chunk(point.id, point.payload or {}, point.score, selected_by)
            for point in response.points
        ]

    async def cleanup(self, older_than: datetime) -> int:
        """Delete points fetched before ``older_than``. Returns how many were removed."""
        await self.ensure_collection()
        stale = models.Filter(
            must=[
                models.FieldCondition(
                    key="fetched_at", range=models.DatetimeRange(lt=older_than.astimezone(UTC))
                )
            ]
        )
        count = (await self._client.count(self._collection, count_filter=stale, exact=True)).count
        if count:
            await self._client.delete(
                self._collection, points_selector=models.FilterSelector(filter=stale)
            )
            logger.info("expired chunks removed", extra={"count": count})
        return count


def _to_chunk(
    point: object, payload: dict[str, object], score: float, selected_by: str = "vector"
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(point),
        url=_URL.validate_python(payload["url"]),
        chunk_index=int(payload["chunk_index"]),  # type: ignore[call-overload]
        text=str(payload["text"]),
        score=score,
        selected_by=selected_by,  # type: ignore[arg-type]
    )
