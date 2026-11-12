"""``MongoRepository`` against the shared contract: offline on a fake, and live on a server.

The offline class checks the repository's logic (documents, ordering, read-time expiry). The
live class (``pytest -m live``, needs ``MONGODB_URI``) checks the real server: BSON, indexes
and time zones. Both run the same ``RepositoryContract``.
"""

import asyncio
import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from typing import Any

import pytest
from pymongo.errors import OperationFailure

from partner_agents.config import get_settings
from partner_agents.graph.build import build_checkpointer
from partner_agents.storage.mongo import MongoRepository, build_repository
from partner_agents.storage.repo import InMemoryRepository, Repository
from tests.storage.fake_mongo import FakeDatabase
from tests.storage.repo_contract import (
    PAGE_TTL,
    SEARCH_TTL,
    FakeClock,
    RepositoryContract,
    lead,
    page,
    run,
)


class TestMongoRepositoryOffline(RepositoryContract):
    @pytest.fixture
    def repo(self, clock: FakeClock) -> Repository:
        return MongoRepository(FakeDatabase(), clock=clock)


# --- what the offline fake lets us look at --------------------------------------------


def fresh() -> tuple[MongoRepository, FakeDatabase, FakeClock]:
    clock, database = FakeClock(), FakeDatabase()
    return MongoRepository(database, clock=clock), database, clock


def test_documents_use_natural_ids_and_real_datetimes() -> None:
    repo, database, _ = fresh()
    run(repo.save_leads([lead("l_7")]))
    doc = database["leads"].docs["l_7"]
    assert doc["_id"] == "l_7" and isinstance(doc["created_at"], datetime)
    assert doc["created_at"].tzinfo is not None
    assert "outreach_eligible" not in doc  # a computed field is not stored
    assert doc["source_url"].startswith("https://")  # JSON form: a string, not a Url object


def test_saving_the_same_id_twice_keeps_one_document() -> None:
    repo, database, _ = fresh()
    run(repo.save_leads([lead("l_1", company_name="Old")]))
    run(repo.save_leads([lead("l_1", company_name="New")]))
    assert len(database["leads"].docs) == 1
    assert run(repo.get_lead("u_1", "l_1")).company_name == "New"  # type: ignore[union-attr]


def test_the_recipient_is_stored_normalised_for_the_dedupe_query() -> None:
    from tests.storage.repo_contract import draft

    repo, database, _ = fresh()
    run(repo.save_draft(draft("d_1", recipient="  Buyer@Acme.EXAMPLE ")))
    assert database["outreach"].docs["d_1"]["recipient_key"] == "buyer@acme.example"
    assert "recipient_key" not in run(repo.get_draft("u_1", "d_1")).model_dump()  # type: ignore[union-attr]


def test_a_page_is_filtered_by_the_injected_clock_not_by_the_server() -> None:
    repo, database, clock = fresh()
    run(repo.save_page(page(fetched_at=clock())))
    clock.advance(PAGE_TTL)
    assert run(repo.get_page("r_1", page().url)) is None
    assert len(database["pages"].docs) == 1  # still stored: the TTL index sweeps it later


def test_the_search_cache_is_filtered_by_the_injected_clock() -> None:
    repo, _, clock = fresh()
    run(repo.put_cached_search("k", []))
    clock.advance(SEARCH_TTL - timedelta(seconds=1))
    assert run(repo.get_cached_search("k")) == []
    clock.advance(timedelta(seconds=1))
    assert run(repo.get_cached_search("k")) is None


def test_ensure_indexes_creates_the_indexes_and_ttls_of_the_design() -> None:
    repo, database, _ = fresh()
    run(repo.ensure_indexes())
    made = {
        name: [(dict(keys), opts) for keys, opts in col.indexes]
        for name, col in database.collections.items()
    }
    assert ({"user_id": 1, "run_id": 1}, {}) in made["leads"]
    assert ({"user_id": 1, "recipient_key": 1, "created_at": 1}, {}) in made["outreach"]
    assert ({"user_id": 1, "status": 1}, {}) in made["outreach"]
    assert ({"user_id": 1, "started_at": -1}, {}) in made["runs"]
    assert ({"fetched_at": 1}, {"expireAfterSeconds": 7 * 86400}) in made["pages"]
    assert ({"created_at": 1}, {"expireAfterSeconds": 86400}) in made["search_cache"]


def test_a_changed_ttl_updates_the_existing_index_instead_of_failing() -> None:
    repo, database, _ = fresh()
    database["pages"].index_conflict = 85
    run(repo.ensure_indexes())
    assert database.commands == [
        {
            "collMod": "pages",
            "index": {"keyPattern": {"fetched_at": 1}, "expireAfterSeconds": 7 * 86400},
        }
    ]


def test_any_other_index_error_is_not_hidden() -> None:
    repo, database, _ = fresh()
    database["pages"].index_conflict = 13  # unauthorized
    with pytest.raises(OperationFailure):
        run(repo.ensure_indexes())


# --- choosing the stores --------------------------------------------------------------


def test_without_a_uri_the_repository_and_the_saver_stay_in_memory() -> None:
    from partner_agents.config import Settings

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert isinstance(build_repository(settings), InMemoryRepository)
    assert build_checkpointer(settings) is None


def test_with_a_uri_both_stores_are_built_lazily_for_mongodb() -> None:
    from partner_agents.config import Settings

    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, mongodb_uri="mongodb://localhost:1/?serverSelectionTimeoutMS=50"
    )
    repo = build_repository(settings)  # connecting is lazy: nothing is contacted yet
    assert isinstance(repo, MongoRepository)
    asyncio.run(repo.aclose())


def test_the_mongodb_saver_gets_our_serializer_and_the_data_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from langgraph.checkpoint.mongodb import MongoDBSaver

    from partner_agents.config import Settings

    fake_client = MagicMock()
    monkeypatch.setattr("pymongo.MongoClient", lambda *args, **kwargs: fake_client)
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, mongodb_uri="mongodb://example.invalid/", mongodb_database="app_db"
    )
    saver = build_checkpointer(settings)
    assert isinstance(saver, MongoDBSaver) and saver.client is fake_client
    assert saver.db.name == fake_client["app_db"].name
    assert type(saver.serde).__name__ == "JsonPlusSerializer"
    ttl_calls = [c for c in fake_client.mock_calls if "expireAfterSeconds" in str(c)]
    assert ttl_calls and "604800" in str(ttl_calls)  # DATA_TTL_DAYS=7 in seconds


# --- live: a real server --------------------------------------------------------------


class _OnOneLoop:
    """Runs a repository's coroutines on one background event loop.

    The contract calls ``asyncio.run`` for every operation, which makes a new loop each time;
    a real driver connection belongs to the loop that opened it.
    """

    def __init__(self, repo: MongoRepository) -> None:
        self._repo = repo
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def call(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=30)

    def stop(self) -> None:
        self.call(self._repo.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        method = getattr(self._repo, name)

        async def forward(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(method(*args, **kwargs), self._loop)
            )

        return forward


@pytest.mark.live
class TestMongoRepositoryLive(RepositoryContract):
    """Needs ``MONGODB_URI`` (Atlas M0 or a local server). Uses a throwaway database."""

    @pytest.fixture
    def repo(self, clock: FakeClock) -> Iterator[Repository]:
        uri = get_settings().mongodb_uri
        if uri is None:
            pytest.skip("MONGODB_URI is not set")
        name = f"partner_agents_test_{uuid.uuid4().hex[:10]}"
        real = MongoRepository.connect(uri.get_secret_value(), name, data_ttl_days=7, clock=clock)
        wrapped = _OnOneLoop(real)
        wrapped.call(real.ensure_indexes())
        try:
            yield wrapped  # type: ignore[misc]
        finally:
            wrapped.call(real._client.drop_database(name))  # only the database made above
            wrapped.stop()

    def test_stored_timestamps_come_back_timezone_aware(self, repo: Repository) -> None:
        run(repo.save_leads([lead("l_tz")]))
        stored = run(repo.get_lead("u_1", "l_tz"))
        assert stored is not None and stored.created_at.tzinfo is not None
        assert stored.created_at.utcoffset() == timedelta(0)
