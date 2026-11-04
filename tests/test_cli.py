import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from partner_agents import cli
from partner_agents.config import Settings
from partner_agents.schemas import Lead, StageError

NOW = datetime(2026, 11, 4, tzinfo=UTC)


@pytest.mark.parametrize(
    ("text", "partner_type", "category", "location"),
    [
        ("cookie brokers in California", "broker", "cookie", "California"),
        (
            "snack food distributors in Los Angeles, CA",
            "distributor",
            "snack food",
            "Los Angeles, CA",
        ),
        ("Find me independent retailers near Austin", "retailer", "independent", "Austin"),
        ("organic grocery stores around Portland", "retailer", "organic grocery", "Portland"),
        ("gluten-free manufacturers", "manufacturer", "gluten-free", None),
        ("co-packers in Ohio", "manufacturer", "food", "Ohio"),
        ("snack companies in Texas", "other", "snack companies", "Texas"),
        ("brokers", "broker", "food", None),
        ("  cookie   BROKERS,  in   california ", "broker", "cookie", "california"),
    ],
)
def test_parse_query(text: str, partner_type: str, category: str, location: str | None) -> None:
    intent = cli.parse_query(text)
    assert (intent.partner_type, intent.product_category, intent.location) == (
        partner_type,
        category,
        location,
    )
    assert intent.raw_query == " ".join(text.split())


def test_parse_query_flags_override_the_guess() -> None:
    intent = cli.parse_query(
        "cookie brokers in California",
        max_results=5,
        partner_type="distributor",
        category="biscuits",
        location="Nevada",
    )
    assert (intent.partner_type, intent.product_category, intent.location, intent.max_results) == (
        "distributor",
        "biscuits",
        "Nevada",
        5,
    )


def lead(name: str, email: str | None, status: str = "verified", **kw: Any) -> Lead:
    fields: dict[str, Any] = {
        "id": name,
        "run_id": "r",
        "user_id": "u",
        "company_name": name,
        "partner_type": "broker",
        "location": None,
        "website": None,
        "email": email,
        "email_status": status,
        "phone": None,
        "phone_status": "unchecked",
        "source_url": "https://acme.example/contact",
        "evidence_quote": "q",
        "created_at": NOW,
    }
    return Lead(**{**fields, **kw})


def test_the_lead_table_lists_verified_leads_first() -> None:
    table = cli.format_leads(
        [
            lead("Zed Foods", "z@zed.example"),
            lead("Beta", "ghost@beta.example", "not_in_source"),
            lead("Acme", "sales@acme.example", phone="+13105550100", phone_status="verified"),
        ]
    )
    rows = table.splitlines()
    assert rows[0].split()[:2] == ["#", "Company"]
    assert set(rows[1]) <= {"-", " "}
    assert [r.split()[1] for r in rows[2:]] == ["Acme", "Zed", "Beta"]
    assert "sales@acme.example [verified]" in rows[2]
    assert "+13105550100 [verified]" in rows[2]
    assert "ghost@beta.example [not_in_source]" in rows[4]


def test_missing_contacts_print_a_dash_and_long_sources_are_cut() -> None:
    long_url = "https://acme.example/" + "a" * 80
    table = cli.format_leads([lead("Acme", None, "unchecked", source_url=long_url)])
    row = table.splitlines()[2]
    assert " - " in row
    assert row.rstrip().endswith("...")
    assert len(row) < 200


def test_no_leads() -> None:
    assert cli.format_leads([]) == "(no leads)"


def test_errors_are_listed() -> None:
    errors = [
        StageError(
            stage="scout_fetch",
            code="blocked_robots",
            message="disallowed",
            url="https://g.example/",
        ),
        StageError(stage="scout_search", code="no_results", message="empty"),
    ]
    text = cli.format_errors(errors)
    assert "scout_fetch" in text and "blocked_robots" in text and "https://g.example/" in text
    assert text.splitlines()[1].split()[:2] == ["scout_search", "no_results"]
    assert cli.format_errors([]) == "(none)"


def test_timings_follow_the_pipeline_order_and_add_up() -> None:
    text = cli.format_timings(
        {"scout_verify": 5, "scout_search": 120, "scout_fetch": 800, "extra": 1}
    )
    names = [line.split()[0] for line in text.splitlines()]
    assert names == ["scout_search", "scout_fetch", "scout_verify", "extra", "total"]
    assert text.splitlines()[-1].split() == ["total", "926", "ms"]


def test_render_puts_the_reply_first() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    out = cli.render(
        {
            "messages": [HumanMessage(content="q"), AIMessage(content="Found 1 company.")],
            "leads": [lead("Acme", "sales@acme.example")],
            "errors": [],
            "timings_ms": {"scout_search": 10},
            "tokens": {"extractor": 321},
            "providers": {"extractor": "gemini:flash"},
        }
    )
    sections = out.split("\n\n")
    assert sections[0] == "Found 1 company."
    assert "Tokens: extractor: 321" in out
    assert "Models: extractor=gemini:flash" in out


def test_render_copes_with_an_empty_state() -> None:
    out = cli.render({})
    assert "(no leads)" in out and "Errors:\n(none)" in out


def test_the_parser_accepts_the_documented_commands() -> None:
    args = cli.build_parser().parse_args(
        ["scout", "cookie brokers in California", "--max-results", "5", "--partner-type", "broker"]
    )
    assert (args.command, args.query, args.max_results, args.partner_type) == (
        "scout",
        "cookie brokers in California",
        5,
        "broker",
    )


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["scout"],
        ["scout", "q", "--max-results", "99"],
        ["scout", "q", "--partner-type", "nope"],
    ],
)
def test_the_parser_rejects_bad_input(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


def test_main_prints_the_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_scout(args: Any, settings: Settings) -> str:
        return f"scouted: {args.query}"

    monkeypatch.setattr(cli, "_scout", fake_scout)
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert cli.main(["scout", "cookie brokers in California"]) == 0
    assert capsys.readouterr().out.strip() == "scouted: cookie brokers in California"


def test_build_deps_wires_the_real_services_without_touching_the_network(tmp_path: Any) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        llm_provider="fake",
        qdrant_local_path=str(tmp_path / "q"),
        tavily_api_key="k",
    )
    deps = cli.build_deps(settings)

    async def close() -> None:
        await deps.fetcher.aclose()
        await deps.store.close()

    asyncio.run(close())
    assert deps.settings is settings
