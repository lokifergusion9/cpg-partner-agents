"""Schema and mix tests for the evaluation datasets (T-36 onwards, docs/EVAL.md section 3)."""

from collections import Counter

import pytest
from eval.schemas import RoutingRow, load_routing
from pydantic import ValidationError

from partner_agents.prompts.charlie import FEW_SHOT

# EVAL 3.1: discovery 35%, cpg_qa 25%, outreach 15%, clarify 15%, out_of_scope 10% per split.
SHARES = {
    "discovery": 0.35,
    "cpg_qa": 0.25,
    "outreach": 0.15,
    "clarify": 0.15,
    "out_of_scope": 0.10,
}
SIZES = {"dev": 20, "test": 60}
HINGLISH = ("me ", " ke ", "dhundo", "chahiye", "kya ", "kuch ", "hota hai", "maangte")


def rows(split: str | None = None) -> list[RoutingRow]:
    return [row for row in load_routing() if split in (None, row.split)]


def normal(query: str) -> str:
    return " ".join(query.lower().split())


# --- routing ------------------------------------------------------------------------


def test_there_are_80_rows_20_dev_and_60_test_with_unique_sequential_ids() -> None:
    everything = rows()
    assert Counter(row.split for row in everything) == SIZES
    assert [row.id for row in everything] == [f"r{n:03d}" for n in range(1, 81)]


def test_no_query_appears_twice_so_no_row_is_shared_between_splits() -> None:
    queries = [normal(row.query) for row in rows()]
    assert len(set(queries)) == len(queries)


@pytest.mark.parametrize("split", ["dev", "test"])
def test_each_split_has_the_documented_intent_distribution(split: str) -> None:
    counts = Counter(row.intent for row in rows(split))
    expected = {intent: round(share * SIZES[split]) for intent, share in SHARES.items()}
    assert dict(counts) == expected


def test_the_test_split_counts_are_the_ones_in_the_docs() -> None:
    counts = Counter(row.intent for row in rows("test"))
    assert counts == {"discovery": 21, "cpg_qa": 15, "outreach": 9, "clarify": 9, "out_of_scope": 6}


@pytest.mark.parametrize("split", ["dev", "test"])
def test_about_a_third_of_the_outreach_rows_are_call_requests(split: str) -> None:
    outreach = [row for row in rows(split) if row.intent == "outreach"]
    voice = [row for row in outreach if row.outreach_channel == "voice"]
    assert len(voice) * 3 == len(outreach)
    assert all(row.context_leads for row in outreach)


def test_the_test_split_has_the_hard_cases_the_docs_ask_for() -> None:
    test = rows("test")
    hinglish = [row for row in test if any(word in row.query.lower() for word in HINGLISH)]
    assert len(hinglish) >= 4  # discovery and a cpg_qa and a clarify among them
    assert {row.intent for row in hinglish} >= {"discovery", "cpg_qa", "clarify"}
    # A discovery-looking request that misses a slot is expected to be a clarify.
    assert any(row.intent == "clarify" and row.query.startswith("find") for row in test)
    # CPG-adjacent but out of scope.
    assert any(row.intent == "out_of_scope" and "brand" in row.query for row in test)
    assert any(row.discovery and row.discovery.location is None for row in test)
    typos = {"r023", "r034"}  # "Califronia", "distributers"
    assert typos <= {row.id for row in test}


def test_the_discovery_rows_cover_every_partner_type_in_both_splits() -> None:
    for split in ("dev", "test"):
        types = {row.discovery.partner_type for row in rows(split) if row.discovery}
        assert {"broker", "distributor", "retailer", "manufacturer"} <= types


def test_no_few_shot_example_of_the_router_prompt_comes_from_the_test_split() -> None:
    """CLAUDE.md rule 9: the prompt uses dev rows only, by id and by query text."""
    by_id = {row.id: row for row in rows()}
    test_queries = {normal(row.query) for row in rows("test")}
    for example_id, (query, _) in FEW_SHOT.items():
        assert by_id[example_id].split == "dev"
        assert normal(query) not in test_queries


# --- the row model itself -----------------------------------------------------------


def valid(**fields: object) -> dict[str, object]:
    base: dict[str, object] = {"id": "r001", "split": "dev", "query": "q", "intent": "clarify"}
    return {**base, **fields}


@pytest.mark.parametrize(
    "bad",
    [
        valid(intent="discovery"),  # a discovery row without slots
        valid(discovery={"partner_type": "broker", "product_category": "x"}),  # slots, wrong intent
        valid(intent="outreach"),  # an outreach row without a channel
        valid(intent="outreach", outreach_channel="voice", lead_selector="x"),  # no context
        valid(outreach_channel="email"),  # a channel on a non-outreach row
        valid(intent="maybe"),
        valid(split="train"),
        valid(id="x1"),
        valid(query=""),
        valid(surprise=1),
    ],
)
def test_a_malformed_row_is_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RoutingRow.model_validate(bad)


def test_a_well_formed_discovery_row_is_accepted() -> None:
    row = RoutingRow.model_validate(
        valid(
            intent="discovery",
            discovery={"partner_type": "broker", "product_category": "cookies", "location": None},
        )
    )
    assert row.discovery is not None and row.discovery.location is None
