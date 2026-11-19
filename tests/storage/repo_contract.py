"""The behaviour every ``Repository`` must have (HLD §8.1).

Run it against an implementation by subclassing and overriding the ``repo`` fixture:

    class TestInMemoryRepository(RepositoryContract):
        @pytest.fixture
        def repo(self, clock: FakeClock) -> Repository:
            return InMemoryRepository(clock=clock)

The Mongo repository (T-27) reuses this class with ``-m live``. It must honour the same
injected clock when filtering expired documents, because a Mongo TTL index lags.
"""

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from partner_agents.schemas import (
    BrandProfile,
    CallProgress,
    CallRecord,
    CallTurn,
    Lead,
    OutreachDraft,
    PageRecord,
    RunRecord,
    SearchHit,
    StageError,
)
from partner_agents.storage.repo import Repository

T0 = datetime(2026, 11, 1, 9, 0, tzinfo=UTC)
PAGE_TTL = timedelta(days=7)
SEARCH_TTL = timedelta(hours=24)


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


def run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(_await(coro))


async def _await[T](coro: Awaitable[T]) -> T:
    return await coro


def lead(id: str = "l_1", **overrides: Any) -> Lead:
    fields: dict[str, Any] = {
        "id": id,
        "run_id": "r_1",
        "user_id": "u_1",
        "company_name": "Acme Foods",
        "partner_type": "broker",
        "location": "Los Angeles, CA",
        "website": "https://acme-foods.example",
        "email": "sales@acme-foods.example",
        "email_status": "verified",
        "phone": None,
        "phone_status": "unchecked",
        "source_url": "https://acme-foods.example/contact",
        "evidence_quote": "Email sales@acme-foods.example",
        "created_at": T0,
    }
    return Lead(**{**fields, **overrides})


def draft(id: str = "d_1", **overrides: Any) -> OutreachDraft:
    fields: dict[str, Any] = {
        "id": id,
        "lead_id": "l_1",
        "user_id": "u_1",
        "run_id": "r_1",
        "channel": "email",
        "recipient": "sales@acme-foods.example",
        "subject": "Hello",
        "body": "Body",
        "status": "pending_approval",
        "prompt_version": "chase-v1",
        "created_at": T0,
    }
    return OutreachDraft(**{**fields, **overrides})


def run_record(run_id: str = "r_1", **overrides: Any) -> RunRecord:
    fields: dict[str, Any] = {
        "run_id": run_id,
        "user_id": "u_1",
        "thread_id": "t_1",
        "intent": "discovery",
        "started_at": T0,
        "finished_at": None,
        "timings_ms": {"scout_search": 400},
        "tokens": {"extractor": 900},
        "prompt_versions": {"scout": "scout-v1"},
        "errors": [StageError(stage="scout_fetch", code="timeout", message="slow")],
    }
    return RunRecord(**{**fields, **overrides})


def page(url: str = "https://acme-foods.example/contact", **overrides: Any) -> PageRecord:
    fields: dict[str, Any] = {
        "run_id": "r_1",
        "url": url,
        "text": "Contact us",
        "contact_text": "mailto:sales@acme-foods.example",
        "fetched_at": T0,
    }
    return PageRecord(**{**fields, **overrides})


def hit(rank: int = 1) -> SearchHit:
    return SearchHit(
        url="https://acme-foods.example/",  # type: ignore[arg-type]
        title="Acme",
        snippet="Brokers",
        rank=rank,
        provider="tavily",
    )


def brand(**overrides: Any) -> BrandProfile:
    fields: dict[str, Any] = {
        "brand_name": "Crumb & Co",
        "product": "Cookies",
        "value_prop": "Small batch",
        "sender_name": "Asha Rao",
        "sender_email": "asha@crumbandco.example",
        "founder_name": "Asha Rao",
        "category": "gluten-free cookies",
        "sign_off": "Thanks",
    }
    return BrandProfile(**{**fields, **overrides})


def call(id: str = "c_1", **overrides: Any) -> CallRecord:
    fields: dict[str, Any] = {
        "id": id,
        "user_id": "u_1",
        "lead_id": "l_1",
        "run_id": "r_1",
        "draft_id": "d_1",
        "playbook_id": "partner_intro",
        "playbook_version": "1",
        "prompt_version": "caller-v1",
        "mode": "live_text",
        "progress": CallProgress(proposed_slot=T0 + timedelta(days=1)),
        "turns": [CallTurn(index=0, speaker="agent", text="Hello", stage="opening")],
        "started_at": T0,
    }
    return CallRecord(**{**fields, **overrides})


class RepositoryContract:
    @pytest.fixture
    def clock(self) -> FakeClock:
        return FakeClock()

    @pytest.fixture
    def repo(self, clock: FakeClock) -> Repository:
        raise NotImplementedError("subclasses build the repository under test")

    # --- leads ----------------------------------------------------------------------

    def test_lead_round_trip(self, repo: Repository) -> None:
        run(repo.save_leads([lead()]))
        assert run(repo.get_lead("u_1", "l_1")) == lead()

    def test_unknown_lead_is_none(self, repo: Repository) -> None:
        assert run(repo.get_lead("u_1", "nope")) is None

    def test_leads_are_isolated_per_user(self, repo: Repository) -> None:
        run(repo.save_leads([lead("l_1"), lead("l_2", user_id="u_2")]))
        assert run(repo.get_lead("u_2", "l_1")) is None
        assert [x.id for x in run(repo.list_leads("u_1"))] == ["l_1"]
        assert [x.id for x in run(repo.list_leads("u_2"))] == ["l_2"]

    def test_leads_filter_by_run_and_come_oldest_first(self, repo: Repository) -> None:
        later = T0 + timedelta(hours=1)
        run(
            repo.save_leads(
                [
                    lead("l_c", run_id="r_2", created_at=later),
                    lead("l_a", run_id="r_1"),
                    lead("l_b", run_id="r_1"),
                ]
            )
        )
        assert [x.id for x in run(repo.list_leads("u_1"))] == ["l_a", "l_b", "l_c"]
        assert [x.id for x in run(repo.list_leads("u_1", run_id="r_2"))] == ["l_c"]
        assert run(repo.list_leads("u_1", run_id="r_9")) == []

    def test_saving_a_lead_again_replaces_it(self, repo: Repository) -> None:
        run(repo.save_leads([lead()]))
        run(repo.save_leads([lead(company_name="Acme Foods Inc")]))
        [only] = run(repo.list_leads("u_1"))
        assert only.company_name == "Acme Foods Inc"

    def test_a_stored_lead_keeps_its_computed_eligibility(self, repo: Repository) -> None:
        run(repo.save_leads([lead(email_status="not_in_source")]))
        stored = run(repo.get_lead("u_1", "l_1"))
        assert stored is not None
        assert stored.outreach_eligible is False

    def test_mutating_a_returned_lead_does_not_change_the_store(self, repo: Repository) -> None:
        run(repo.save_leads([lead()]))
        fetched = run(repo.get_lead("u_1", "l_1"))
        assert fetched is not None
        fetched.company_name = "Changed"
        again = run(repo.get_lead("u_1", "l_1"))
        assert again is not None
        assert again.company_name == "Acme Foods"

    # --- outreach -------------------------------------------------------------------

    def test_draft_round_trip_and_status_update(self, repo: Repository) -> None:
        run(repo.save_draft(draft()))
        assert run(repo.get_draft("u_1", "d_1")) == draft()
        run(repo.save_draft(draft(status="outboxed", decided_at=T0)))
        updated = run(repo.get_draft("u_1", "d_1"))
        assert updated is not None
        assert updated.status == "outboxed"
        assert len(run(repo.list_drafts("u_1"))) == 1

    def test_drafts_are_isolated_per_user(self, repo: Repository) -> None:
        run(repo.save_draft(draft()))
        assert run(repo.get_draft("u_2", "d_1")) is None
        assert run(repo.list_drafts("u_2")) == []

    def test_drafts_filter_by_status(self, repo: Repository) -> None:
        run(repo.save_draft(draft("d_1", status="pending_approval")))
        run(repo.save_draft(draft("d_2", status="outboxed")))
        assert [x.id for x in run(repo.list_drafts("u_1", status="outboxed"))] == ["d_2"]
        assert len(run(repo.list_drafts("u_1"))) == 2

    def test_recent_drafts_to_a_recipient_respects_the_window(self, repo: Repository) -> None:
        old = T0 - timedelta(days=40)
        run(repo.save_draft(draft("d_old", created_at=old)))
        run(repo.save_draft(draft("d_new", created_at=T0 - timedelta(days=3))))
        run(repo.save_draft(draft("d_other", recipient="other@x.example")))
        found = run(
            repo.recent_drafts_to("u_1", "sales@acme-foods.example", T0 - timedelta(days=30))
        )
        assert [x.id for x in found] == ["d_new"]

    def test_recent_drafts_match_the_recipient_case_insensitively(self, repo: Repository) -> None:
        run(repo.save_draft(draft("d_1", recipient="Sales@Acme-Foods.example")))
        found = run(
            repo.recent_drafts_to("u_1", " sales@acme-foods.EXAMPLE ", T0 - timedelta(days=1))
        )
        assert [x.id for x in found] == ["d_1"]

    def test_recent_drafts_are_per_user(self, repo: Repository) -> None:
        run(repo.save_draft(draft("d_1", user_id="u_2")))
        assert (
            run(repo.recent_drafts_to("u_1", "sales@acme-foods.example", T0 - timedelta(days=1)))
            == []
        )

    # --- runs -----------------------------------------------------------------------

    def test_run_round_trip(self, repo: Repository) -> None:
        run(repo.save_run(run_record()))
        assert run(repo.get_run("r_1")) == run_record()
        assert run(repo.get_run("r_404")) is None

    def test_saving_a_run_again_replaces_it(self, repo: Repository) -> None:
        run(repo.save_run(run_record()))
        run(repo.save_run(run_record(finished_at=T0 + timedelta(seconds=9))))
        finished = run(repo.get_run("r_1"))
        assert finished is not None
        assert finished.finished_at == T0 + timedelta(seconds=9)

    def test_runs_list_newest_first_and_per_user(self, repo: Repository) -> None:
        run(repo.save_run(run_record("r_1", started_at=T0)))
        run(repo.save_run(run_record("r_2", started_at=T0 + timedelta(minutes=5))))
        run(repo.save_run(run_record("r_3", user_id="u_2")))
        assert [x.run_id for x in run(repo.list_runs("u_1"))] == ["r_2", "r_1"]
        assert [x.run_id for x in run(repo.list_runs("u_1", limit=1))] == ["r_2"]

    # --- pages (TTL) ----------------------------------------------------------------

    def test_page_round_trip(self, repo: Repository) -> None:
        run(repo.save_page(page()))
        assert run(repo.get_page("r_1", "https://acme-foods.example/contact")) == page()
        assert run(repo.get_page("r_1", "https://other.example/")) is None
        assert run(repo.get_page("r_2", "https://acme-foods.example/contact")) is None

    def test_page_is_readable_just_before_the_ttl(self, repo: Repository, clock: FakeClock) -> None:
        run(repo.save_page(page()))
        clock.advance(PAGE_TTL - timedelta(seconds=1))
        assert run(repo.get_page("r_1", "https://acme-foods.example/contact")) is not None

    def test_page_expires_after_the_ttl(self, repo: Repository, clock: FakeClock) -> None:
        run(repo.save_page(page()))
        clock.advance(PAGE_TTL)
        assert run(repo.get_page("r_1", "https://acme-foods.example/contact")) is None

    def test_page_ttl_counts_from_fetched_at_not_from_saving(
        self, repo: Repository, clock: FakeClock
    ) -> None:
        run(repo.save_page(page(fetched_at=T0 - timedelta(days=6, hours=23))))
        assert run(repo.get_page("r_1", "https://acme-foods.example/contact")) is not None
        clock.advance(timedelta(hours=2))
        assert run(repo.get_page("r_1", "https://acme-foods.example/contact")) is None

    def test_a_refetched_page_gets_a_fresh_ttl(self, repo: Repository, clock: FakeClock) -> None:
        run(repo.save_page(page()))
        clock.advance(timedelta(days=6))
        run(repo.save_page(page(text="newer", fetched_at=clock())))
        clock.advance(timedelta(days=6))
        got = run(repo.get_page("r_1", "https://acme-foods.example/contact"))
        assert got is not None
        assert got.text == "newer"

    # --- search cache (TTL 24 h) ----------------------------------------------------

    def test_search_cache_round_trip(self, repo: Repository) -> None:
        assert run(repo.get_cached_search("cookie broker ca")) is None
        run(repo.put_cached_search("cookie broker ca", [hit(1), hit(2)]))
        assert run(repo.get_cached_search("cookie broker ca")) == [hit(1), hit(2)]
        assert run(repo.get_cached_search("another query")) is None

    def test_an_empty_result_list_is_cached_as_empty_not_missing(self, repo: Repository) -> None:
        run(repo.put_cached_search("nothing here", []))
        assert run(repo.get_cached_search("nothing here")) == []

    def test_search_cache_expires_after_24_hours(self, repo: Repository, clock: FakeClock) -> None:
        run(repo.put_cached_search("q", [hit()]))
        clock.advance(SEARCH_TTL - timedelta(seconds=1))
        assert run(repo.get_cached_search("q")) is not None
        clock.advance(timedelta(seconds=1))
        assert run(repo.get_cached_search("q")) is None

    def test_putting_a_search_again_restarts_its_ttl(
        self, repo: Repository, clock: FakeClock
    ) -> None:
        run(repo.put_cached_search("q", [hit()]))
        clock.advance(timedelta(hours=20))
        run(repo.put_cached_search("q", [hit(2)]))
        clock.advance(timedelta(hours=20))
        assert run(repo.get_cached_search("q")) == [hit(2)]

    # --- profiles -------------------------------------------------------------------

    def test_profile_round_trip_and_overwrite(self, repo: Repository) -> None:
        assert run(repo.get_profile("u_1")) is None
        run(repo.save_profile("u_1", brand()))
        assert run(repo.get_profile("u_1")) == brand()
        run(repo.save_profile("u_1", brand(brand_name="Crumb Two")))
        updated = run(repo.get_profile("u_1"))
        assert updated is not None
        assert updated.brand_name == "Crumb Two"

    def test_profiles_are_per_user(self, repo: Repository) -> None:
        run(repo.save_profile("u_1", brand()))
        assert run(repo.get_profile("u_2")) is None

    # --- calls ----------------------------------------------------------------------

    def test_call_round_trip_and_replace(self, repo: Repository) -> None:
        record = call()
        run(repo.save_call(record))
        assert run(repo.get_call("u_1", "c_1")) == record
        ended = record.model_copy(
            update={"progress": CallProgress(stage="ended", outcome="opted_out")}
        )
        run(repo.save_call(ended))
        stored = run(repo.get_call("u_1", "c_1"))
        assert stored is not None and stored.progress.outcome == "opted_out"

    def test_calls_are_per_user_and_list_newest_first(self, repo: Repository) -> None:
        run(repo.save_call(call("c_old")))
        run(repo.save_call(call("c_new", started_at=T0 + timedelta(hours=1))))
        run(repo.save_call(call("c_other", user_id="u_2")))
        assert run(repo.get_call("u_2", "c_old")) is None
        assert [c.id for c in run(repo.list_calls("u_1"))] == ["c_new", "c_old"]

    def test_calls_filter_by_outcome(self, repo: Repository) -> None:
        run(repo.save_call(call("c_1")))
        booked = CallProgress(stage="ended", outcome="meeting_booked")
        run(repo.save_call(call("c_2", progress=booked)))
        assert [c.id for c in run(repo.list_calls("u_1", outcome="meeting_booked"))] == ["c_2"]
