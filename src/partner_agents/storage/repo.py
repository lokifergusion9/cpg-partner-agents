"""Storage contract and the in-memory implementation (HLD §8.1).

Everything the graph persists goes through the ``Repository`` protocol. Two
implementations exist: ``InMemoryRepository`` (dev, tests, no ``MONGODB_URI``) and the Mongo
one (T-27). Both pass the same suite, ``tests/storage/repo_contract.py``.

Reads return copies, so a caller mutating a model never changes what is stored.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from partner_agents.schemas import (
    BrandProfile,
    CallRecord,
    Lead,
    OutreachDraft,
    PageRecord,
    RunRecord,
    SearchHit,
)


class Repository(Protocol):
    """Persistence for leads, outreach, runs, pages, the search cache, profiles and calls."""

    # leads: oldest first, ties in save order
    async def save_leads(self, leads: Sequence[Lead]) -> None: ...
    async def get_lead(self, user_id: str, lead_id: str) -> Lead | None: ...
    async def list_leads(self, user_id: str, run_id: str | None = None) -> list[Lead]: ...

    # outreach drafts (saving an existing id replaces it, which is how status changes)
    async def save_draft(self, draft: OutreachDraft) -> None: ...
    async def get_draft(self, user_id: str, draft_id: str) -> OutreachDraft | None: ...
    async def list_drafts(self, user_id: str, status: str | None = None) -> list[OutreachDraft]: ...
    async def recent_drafts_to(
        self, user_id: str, recipient: str, since: datetime
    ) -> list[OutreachDraft]:
        """Drafts to ``recipient`` (case-insensitive) created at or after ``since``."""
        ...

    # runs
    async def save_run(self, run: RunRecord) -> None: ...
    async def get_run(self, run_id: str) -> RunRecord | None: ...
    async def list_runs(self, user_id: str, limit: int = 20) -> list[RunRecord]:
        """The user's runs, newest first."""
        ...

    # fetched pages, expire ``DATA_TTL_DAYS`` after ``fetched_at``
    async def save_page(self, page: PageRecord) -> None: ...
    async def get_page(self, run_id: str, url: str) -> PageRecord | None: ...

    # search cache, expires 24 h after it was written
    async def get_cached_search(self, query_key: str) -> list[SearchHit] | None: ...
    async def put_cached_search(self, query_key: str, hits: Sequence[SearchHit]) -> None: ...

    # brand profiles, one per user
    async def save_profile(self, user_id: str, profile: BrandProfile) -> None: ...
    async def get_profile(self, user_id: str) -> BrandProfile | None: ...

    # calls (saving an existing id replaces it: every turn saves the whole record)
    async def save_call(self, call: CallRecord) -> None: ...
    async def get_call(self, user_id: str, call_id: str) -> CallRecord | None: ...
    async def list_calls(self, user_id: str, outcome: str | None = None) -> list[CallRecord]:
        """The user's calls, newest first."""
        ...


def _utcnow() -> datetime:
    return datetime.now(UTC)


class InMemoryRepository:
    """A ``Repository`` held in dicts. ``clock`` is injectable so TTLs can be tested."""

    def __init__(
        self,
        *,
        page_ttl: timedelta = timedelta(days=7),
        search_ttl: timedelta = timedelta(hours=24),
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._page_ttl = page_ttl
        self._search_ttl = search_ttl
        self._clock = clock
        self._leads: dict[str, Lead] = {}
        self._drafts: dict[str, OutreachDraft] = {}
        self._runs: dict[str, RunRecord] = {}
        self._pages: dict[tuple[str, str], PageRecord] = {}
        self._searches: dict[str, tuple[datetime, list[SearchHit]]] = {}
        self._profiles: dict[str, BrandProfile] = {}
        self._calls: dict[str, CallRecord] = {}

    # leads

    async def save_leads(self, leads: Sequence[Lead]) -> None:
        for lead in leads:
            self._leads[lead.id] = lead.model_copy(deep=True)

    async def get_lead(self, user_id: str, lead_id: str) -> Lead | None:
        lead = self._leads.get(lead_id)
        return lead.model_copy(deep=True) if lead and lead.user_id == user_id else None

    async def list_leads(self, user_id: str, run_id: str | None = None) -> list[Lead]:
        found = [
            lead
            for lead in self._leads.values()
            if lead.user_id == user_id and (run_id is None or lead.run_id == run_id)
        ]
        found.sort(key=lambda lead: lead.created_at)  # stable: ties keep save order
        return [lead.model_copy(deep=True) for lead in found]

    # outreach

    async def save_draft(self, draft: OutreachDraft) -> None:
        self._drafts[draft.id] = draft.model_copy(deep=True)

    async def get_draft(self, user_id: str, draft_id: str) -> OutreachDraft | None:
        draft = self._drafts.get(draft_id)
        return draft.model_copy(deep=True) if draft and draft.user_id == user_id else None

    async def list_drafts(self, user_id: str, status: str | None = None) -> list[OutreachDraft]:
        found = [
            draft
            for draft in self._drafts.values()
            if draft.user_id == user_id and (status is None or draft.status == status)
        ]
        found.sort(key=lambda draft: draft.created_at)
        return [draft.model_copy(deep=True) for draft in found]

    async def recent_drafts_to(
        self, user_id: str, recipient: str, since: datetime
    ) -> list[OutreachDraft]:
        wanted = recipient.strip().lower()
        found = [
            draft
            for draft in self._drafts.values()
            if draft.user_id == user_id
            and draft.recipient.strip().lower() == wanted
            and draft.created_at >= since
        ]
        found.sort(key=lambda draft: draft.created_at)
        return [draft.model_copy(deep=True) for draft in found]

    # runs

    async def save_run(self, run: RunRecord) -> None:
        self._runs[run.run_id] = run.model_copy(deep=True)

    async def get_run(self, run_id: str) -> RunRecord | None:
        run = self._runs.get(run_id)
        return run.model_copy(deep=True) if run else None

    async def list_runs(self, user_id: str, limit: int = 20) -> list[RunRecord]:
        found = [run for run in self._runs.values() if run.user_id == user_id]
        found.sort(key=lambda run: run.started_at, reverse=True)
        return [run.model_copy(deep=True) for run in found[:limit]]

    # pages

    async def save_page(self, page: PageRecord) -> None:
        self._pages[(page.run_id, page.url)] = page.model_copy(deep=True)

    async def get_page(self, run_id: str, url: str) -> PageRecord | None:
        page = self._pages.get((run_id, url))
        if page is None:
            return None
        if self._clock() >= page.fetched_at + self._page_ttl:
            del self._pages[(run_id, url)]
            return None
        return page.model_copy(deep=True)

    # search cache

    async def get_cached_search(self, query_key: str) -> list[SearchHit] | None:
        entry = self._searches.get(query_key)
        if entry is None:
            return None
        created_at, hits = entry
        if self._clock() >= created_at + self._search_ttl:
            del self._searches[query_key]
            return None
        return [hit.model_copy(deep=True) for hit in hits]

    async def put_cached_search(self, query_key: str, hits: Sequence[SearchHit]) -> None:
        self._searches[query_key] = (self._clock(), [hit.model_copy(deep=True) for hit in hits])

    # profiles

    async def save_profile(self, user_id: str, profile: BrandProfile) -> None:
        self._profiles[user_id] = profile.model_copy(deep=True)

    async def get_profile(self, user_id: str) -> BrandProfile | None:
        profile = self._profiles.get(user_id)
        return profile.model_copy(deep=True) if profile else None

    # calls

    async def save_call(self, call: CallRecord) -> None:
        self._calls[call.id] = call.model_copy(deep=True)

    async def get_call(self, user_id: str, call_id: str) -> CallRecord | None:
        call = self._calls.get(call_id)
        return call.model_copy(deep=True) if call and call.user_id == user_id else None

    async def list_calls(self, user_id: str, outcome: str | None = None) -> list[CallRecord]:
        found = [
            call
            for call in self._calls.values()
            if call.user_id == user_id and (outcome is None or call.progress.outcome == outcome)
        ]
        found.sort(key=lambda call: call.started_at, reverse=True)
        return [call.model_copy(deep=True) for call in found]
