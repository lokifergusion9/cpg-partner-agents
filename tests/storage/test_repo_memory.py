from datetime import timedelta

import pytest

from partner_agents.storage.repo import InMemoryRepository, Repository
from tests.storage.repo_contract import (
    PAGE_TTL,
    SEARCH_TTL,
    FakeClock,
    RepositoryContract,
    page,
    run,
)


class TestInMemoryRepository(RepositoryContract):
    @pytest.fixture
    def repo(self, clock: FakeClock) -> Repository:
        return InMemoryRepository(page_ttl=PAGE_TTL, search_ttl=SEARCH_TTL, clock=clock)


def test_the_page_ttl_is_configurable() -> None:
    clock = FakeClock()
    repo = InMemoryRepository(page_ttl=timedelta(hours=1), clock=clock)
    run(repo.save_page(page()))
    clock.advance(timedelta(minutes=59))
    assert run(repo.get_page("r_1", "https://acme-foods.example/contact")) is not None
    clock.advance(timedelta(minutes=1))
    assert run(repo.get_page("r_1", "https://acme-foods.example/contact")) is None
