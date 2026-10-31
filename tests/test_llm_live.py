"""Calls the configured provider for real. Run with ``uv run pytest -m live`` and a .env."""

import pytest

from partner_agents.config import Settings
from partner_agents.llm import get_chat_model
from partner_agents.schemas import RouteDecision

pytestmark = pytest.mark.live


def test_router_returns_a_valid_route_decision() -> None:
    settings = Settings()
    if settings.llm_provider == "fake":
        pytest.skip("LLM_PROVIDER=fake: set a real provider in .env")
    model = get_chat_model("router", settings=settings).with_structured_output(RouteDecision)
    decision = model.invoke(
        "Classify this user query for a CPG partner-finding assistant. "
        "Query: find cookie brokers in California"
    )
    assert isinstance(decision, RouteDecision)
    assert decision.intent in {"cpg_qa", "discovery", "outreach", "clarify", "out_of_scope"}
    assert 0 <= decision.confidence <= 1
