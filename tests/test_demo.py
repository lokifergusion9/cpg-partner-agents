"""DEMO_MODE: every override, the sample brand and the banner (HLD §13, T-31)."""

import asyncio
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from qdrant_client import AsyncQdrantClient

from partner_agents.config import DEMO_MAX_RESULTS, Settings
from partner_agents.demo import SAMPLE_BRAND
from partner_agents.fakes import ScriptedChatModel
from partner_agents.graph.build import Context
from partner_agents.graph.charlie import apply_route_rules, make_charlie_route
from partner_agents.schemas import DiscoveryIntent, RouteDecision
from partner_agents.ui.app import DEMO_BANNER, USER_ID, build_ui, init_session, profile_to_form
from tests.graph.test_build import client, make_deps  # noqa: F401  (fixture used by name)


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def wide_search(max_results: int) -> RouteDecision:
    return RouteDecision(
        intent="discovery",
        confidence=0.95,
        discovery=DiscoveryIntent(
            partner_type="broker",
            product_category="cookies",
            max_results=max_results,
            raw_query="q",
        ),
        reason="search",
    )


# --- settings -----------------------------------------------------------------------


def test_demo_mode_forces_the_outbox_and_the_mock_voice() -> None:
    s = settings(
        demo_mode=True,
        outreach_mode="gmail_send",
        outreach_allowlist=["me@x.example"],
        voice_provider="vapi",
        voice_allowlist=["+13105550100"],
    )
    assert (s.outreach_mode, s.voice_provider) == ("outbox", "mock")


def test_demo_mode_does_not_need_a_send_allowlist() -> None:
    # gmail_send without an allowlist is a startup error, but the demo never sends.
    assert settings(demo_mode=True, outreach_mode="gmail_send").outreach_mode == "outbox"
    with pytest.raises(ValueError, match="OUTREACH_ALLOWLIST"):
        settings(outreach_mode="gmail_send")


def test_outside_demo_mode_nothing_is_forced() -> None:
    s = settings(outreach_mode="gmail_draft", voice_provider="vapi")
    assert (s.outreach_mode, s.voice_provider) == ("gmail_draft", "vapi")


# --- max_results --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("asked", "cap", "expected"),
    [
        (25, DEMO_MAX_RESULTS, 5),
        (10, DEMO_MAX_RESULTS, 5),
        (3, DEMO_MAX_RESULTS, 3),
        (25, None, 25),
    ],
)
def test_the_demo_caps_how_many_results_a_search_may_ask_for(
    asked: int, cap: int | None, expected: int
) -> None:
    route = apply_route_rules(
        wide_search(asked), min_confidence=0.6, raw_query="q", max_results_cap=cap
    )
    assert route.discovery is not None and route.discovery.max_results == expected


@pytest.mark.parametrize(("demo", "expected"), [(True, 5), (False, 25)])
def test_the_router_node_applies_the_cap_only_in_demo_mode(demo: bool, expected: int) -> None:
    node = make_charlie_route(ScriptedChatModel.scripted(wide_search(25)), settings(demo_mode=demo))
    update = asyncio.run(node({"messages": [HumanMessage(content="find brokers")]}))  # type: ignore[typeddict-item]
    assert update["route"].discovery.max_results == expected


# --- the sample brand ---------------------------------------------------------------


def test_the_sample_brand_is_fictional_and_complete() -> None:
    facts = SAMPLE_BRAND.approved_facts
    assert 6 <= len(facts) <= 10
    assert len({f.id for f in facts}) == len(facts)
    assert str(SAMPLE_BRAND.sender_email).endswith(".example")  # a reserved, undeliverable domain
    assert SAMPLE_BRAND.sender_name == SAMPLE_BRAND.founder_name != ""


# --- the session --------------------------------------------------------------------


def test_a_demo_session_gets_an_anonymous_id_and_the_sample_profile(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    ctx = Context(deps=make_deps(client), graph=None)  # type: ignore[arg-type]

    async def go() -> Any:
        first = await init_session(ctx, demo_mode=True)
        second = await init_session(ctx, demo_mode=True)
        return first, second, await ctx.deps.repo.get_profile(first[0])

    (first_id, form), (second_id, _), saved = asyncio.run(go())
    assert USER_ID.match(first_id) and first_id != second_id  # one id per session
    assert saved == SAMPLE_BRAND and form == profile_to_form(SAMPLE_BRAND)
    assert form[0] == "Hearthside Crumbs"


def test_a_normal_session_gets_an_empty_form_and_no_saved_profile(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    ctx = Context(deps=make_deps(client), graph=None)  # type: ignore[arg-type]
    user_id, form = asyncio.run(init_session(ctx, demo_mode=False))
    assert asyncio.run(ctx.deps.repo.get_profile(user_id)) is None
    assert form == profile_to_form(None)


# --- the banner and the id box ------------------------------------------------------


def components(ui: Any) -> list[Any]:
    return list(ui.blocks.values())


def banner_shown(ui: Any) -> bool:
    return any(getattr(block, "value", None) == DEMO_BANNER for block in components(ui))


def user_box(ui: Any) -> Any:
    return next(b for b in components(ui) if getattr(b, "label", None) == "User id")


def test_the_banner_is_in_the_ui_only_in_demo_mode() -> None:
    def no_ctx() -> Context:
        raise AssertionError("not called while building")

    demo, normal = build_ui(no_ctx, demo_mode=True), build_ui(no_ctx, demo_mode=False)
    assert banner_shown(demo) and not banner_shown(normal)
    assert "Demo mode" in DEMO_BANNER and "sandbox outbox" in DEMO_BANNER
    # The visitor cannot pick someone else's id in the demo.
    assert user_box(demo).visible is False and user_box(normal).visible is True
