from collections.abc import Iterator

import pytest

from partner_agents.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_settings_env(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[None]:
    """Offline tests never see the developer's env vars or .env file (CLAUDE.md rule 4).

    A ``live`` test is the exception for the .env: it exists to use the real keys.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
