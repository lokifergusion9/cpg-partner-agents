"""``MongoRepository``: the ``Repository`` protocol on MongoDB (HLD §8.1, ADR-028).

Each model is stored as its JSON form plus a few real BSON datetimes for the fields that are
queried, sorted or expired (``created_at``, ``started_at``, ``fetched_at``). Natural ids are the
document ``_id``, so saving an existing id replaces it. Expiry is decided at read time against
the injectable clock, as in ``InMemoryRepository``: a Mongo TTL index only sweeps about once a
minute, so it is a clean-up, not the rule. Lists with equal timestamps keep their save order
through ``_seq``.

Only a small surface of the driver is used (``replace_one``, ``find_one``, ``find`` with
``sort``/``limit``/``to_list``, ``create_index``) so the contract suite can run offline on a
small fake and, with ``-m live``, on a real server.
"""

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel
from pymongo import AsyncMongoClient
from pymongo.errors import OperationFailure

from partner_agents.config import Settings
from partner_agents.schemas import (
    BrandProfile,
    CallRecord,
    Lead,
    OutreachDraft,
    PageRecord,
    RunRecord,
    SearchHit,
)
from partner_agents.storage.repo import InMemoryRepository, Repository

_INTERNAL = ("_id", "_seq", "recipient_key", "outcome")
# MongoDB error codes for an existing index with the same keys but other options.
_INDEX_CONFLICT_CODES = frozenset({85, 86})


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _doc[M: BaseModel](model: M, doc_id: str, **extra: Any) -> dict[str, Any]:
    """The model as a document: JSON values, then ``extra`` (real datetimes and keys)."""
    data = model.model_dump(mode="json", include=set(type(model).model_fields))
    return {**data, "_id": doc_id, "_seq": time.time_ns(), **extra}


def _model[M: BaseModel](cls: type[M], doc: dict[str, Any]) -> M:
    return cls.model_validate({k: v for k, v in doc.items() if k not in _INTERNAL})


def _order(*keys: tuple[str, int]) -> list[tuple[str, int]]:
    return [*keys, ("_seq", keys[0][1])]  # ties keep save order in the direction of the sort


class MongoRepository:
    def __init__(
        self,
        database: Any,
        *,
        page_ttl: timedelta = timedelta(days=7),
        search_ttl: timedelta = timedelta(hours=24),
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._db = database
        self._page_ttl = page_ttl
        self._search_ttl = search_ttl
        self._clock = clock
        self._client: Any = None  # set by ``connect``; closed by ``aclose``

    @classmethod
    def connect(
        cls, uri: str, database: str, *, data_ttl_days: int, clock: Callable[[], datetime] = _utcnow
    ) -> "MongoRepository":
        """A repository on a real server. No I/O happens until the first call."""
        client: Any = AsyncMongoClient(uri, tz_aware=True, tzinfo=UTC)
        repo = cls(client[database], page_ttl=timedelta(days=data_ttl_days), clock=clock)
        repo._client = client
        return repo

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()

    # --- indexes (HLD §8.1) --------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create the indexes and TTLs. Safe to call on every start."""
        db = self._db
        await db["leads"].create_index([("user_id", 1), ("run_id", 1)])
        await db["leads"].create_index([("user_id", 1), ("email", 1)])
        await db["outreach"].create_index([("user_id", 1), ("recipient_key", 1), ("created_at", 1)])
        await db["outreach"].create_index([("user_id", 1), ("status", 1)])
        await db["runs"].create_index([("user_id", 1), ("started_at", -1)])
        await db["calls"].create_index([("user_id", 1), ("started_at", -1)])
        await db["calls"].create_index([("user_id", 1), ("outcome", 1)])
        await self._ttl_index("pages", "fetched_at", self._page_ttl)
        await self._ttl_index("search_cache", "created_at", self._search_ttl)

    async def _ttl_index(self, collection: str, field: str, ttl: timedelta) -> None:
        seconds = int(ttl.total_seconds())
        try:
            await self._db[collection].create_index([(field, 1)], expireAfterSeconds=seconds)
        except OperationFailure as exc:
            if exc.code not in _INDEX_CONFLICT_CODES:
                raise
            # DATA_TTL_DAYS changed: update the existing index instead of failing the start.
            await self._db.command(
                {
                    "collMod": collection,
                    "index": {"keyPattern": {field: 1}, "expireAfterSeconds": seconds},
                }
            )

    # --- leads ---------------------------------------------------------------------

    async def save_leads(self, leads: Sequence[Lead]) -> None:
        for lead in leads:
            doc = _doc(lead, lead.id, created_at=lead.created_at)
            await self._db["leads"].replace_one({"_id": lead.id}, doc, upsert=True)

    async def get_lead(self, user_id: str, lead_id: str) -> Lead | None:
        doc = await self._db["leads"].find_one({"_id": lead_id, "user_id": user_id})
        return _model(Lead, doc) if doc else None

    async def list_leads(self, user_id: str, run_id: str | None = None) -> list[Lead]:
        query: dict[str, Any] = {"user_id": user_id}
        if run_id is not None:
            query["run_id"] = run_id
        docs = await self._db["leads"].find(query).sort(_order(("created_at", 1))).to_list()
        return [_model(Lead, d) for d in docs]

    # --- outreach ------------------------------------------------------------------

    async def save_draft(self, draft: OutreachDraft) -> None:
        doc = _doc(
            draft,
            draft.id,
            created_at=draft.created_at,
            recipient_key=draft.recipient.strip().lower(),
        )
        await self._db["outreach"].replace_one({"_id": draft.id}, doc, upsert=True)

    async def get_draft(self, user_id: str, draft_id: str) -> OutreachDraft | None:
        doc = await self._db["outreach"].find_one({"_id": draft_id, "user_id": user_id})
        return _model(OutreachDraft, doc) if doc else None

    async def list_drafts(self, user_id: str, status: str | None = None) -> list[OutreachDraft]:
        query: dict[str, Any] = {"user_id": user_id}
        if status is not None:
            query["status"] = status
        cursor = self._db["outreach"].find(query).sort(_order(("created_at", 1)))
        return [_model(OutreachDraft, d) for d in await cursor.to_list()]

    async def recent_drafts_to(
        self, user_id: str, recipient: str, since: datetime
    ) -> list[OutreachDraft]:
        query = {
            "user_id": user_id,
            "recipient_key": recipient.strip().lower(),
            "created_at": {"$gte": since},
        }
        cursor = self._db["outreach"].find(query).sort(_order(("created_at", 1)))
        return [_model(OutreachDraft, d) for d in await cursor.to_list()]

    # --- runs ----------------------------------------------------------------------

    async def save_run(self, run: RunRecord) -> None:
        doc = _doc(run, run.run_id, started_at=run.started_at)
        await self._db["runs"].replace_one({"_id": run.run_id}, doc, upsert=True)

    async def get_run(self, run_id: str) -> RunRecord | None:
        doc = await self._db["runs"].find_one({"_id": run_id})
        return _model(RunRecord, doc) if doc else None

    async def list_runs(self, user_id: str, limit: int = 20) -> list[RunRecord]:
        cursor = (
            self._db["runs"]
            .find({"user_id": user_id})
            .sort(_order(("started_at", -1)))
            .limit(limit)
        )
        return [_model(RunRecord, d) for d in await cursor.to_list()]

    # --- pages and the search cache (expire at read time) --------------------------

    @staticmethod
    def _page_id(run_id: str, url: str) -> str:
        return f"{run_id}|{url}"

    async def save_page(self, page: PageRecord) -> None:
        doc_id = self._page_id(page.run_id, page.url)
        doc = _doc(page, doc_id, fetched_at=page.fetched_at)
        await self._db["pages"].replace_one({"_id": doc_id}, doc, upsert=True)

    async def get_page(self, run_id: str, url: str) -> PageRecord | None:
        fresh = {"$gt": self._clock() - self._page_ttl}
        doc = await self._db["pages"].find_one(
            {"_id": self._page_id(run_id, url), "fetched_at": fresh}
        )
        return _model(PageRecord, doc) if doc else None

    async def get_cached_search(self, query_key: str) -> list[SearchHit] | None:
        fresh = {"$gt": self._clock() - self._search_ttl}
        doc = await self._db["search_cache"].find_one({"_id": query_key, "created_at": fresh})
        return [SearchHit.model_validate(hit) for hit in doc["hits"]] if doc else None

    async def put_cached_search(self, query_key: str, hits: Sequence[SearchHit]) -> None:
        doc = {
            "_id": query_key,
            "hits": [hit.model_dump(mode="json") for hit in hits],
            "created_at": self._clock(),
        }
        await self._db["search_cache"].replace_one({"_id": query_key}, doc, upsert=True)

    # --- profiles ------------------------------------------------------------------

    async def save_profile(self, user_id: str, profile: BrandProfile) -> None:
        doc = _doc(profile, user_id, user_id=user_id)
        await self._db["profiles"].replace_one({"_id": user_id}, doc, upsert=True)

    async def get_profile(self, user_id: str) -> BrandProfile | None:
        doc = await self._db["profiles"].find_one({"_id": user_id})
        if not doc:
            return None
        return BrandProfile.model_validate(
            {k: v for k, v in doc.items() if k not in (*_INTERNAL, "user_id")}
        )

    # --- calls ---------------------------------------------------------------------

    async def save_call(self, call: CallRecord) -> None:
        doc = _doc(call, call.id, started_at=call.started_at, outcome=call.progress.outcome)
        await self._db["calls"].replace_one({"_id": call.id}, doc, upsert=True)

    async def get_call(self, user_id: str, call_id: str) -> CallRecord | None:
        doc = await self._db["calls"].find_one({"_id": call_id, "user_id": user_id})
        return _model(CallRecord, doc) if doc else None

    async def list_calls(self, user_id: str, outcome: str | None = None) -> list[CallRecord]:
        query: dict[str, Any] = {"user_id": user_id}
        if outcome is not None:
            query["outcome"] = outcome
        cursor = self._db["calls"].find(query).sort(_order(("started_at", -1)))
        return [_model(CallRecord, d) for d in await cursor.to_list()]


def build_repository(settings: Settings) -> Repository:
    """MongoDB when ``MONGODB_URI`` is set, otherwise the in-memory repository."""
    if settings.mongodb_uri is None:
        return InMemoryRepository()
    return MongoRepository.connect(
        settings.mongodb_uri.get_secret_value(),
        settings.mongodb_database,
        data_ttl_days=settings.data_ttl_days,
    )
