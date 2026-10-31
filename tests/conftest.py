from collections.abc import Iterator

import pytest

from partner_agents.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Offline tests never see the developer's real env vars (CLAUDE.md rule 4)."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
