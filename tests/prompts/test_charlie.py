"""The router prompt uses dev-split examples only (CLAUDE.md rule 9)."""

import json
from pathlib import Path
from typing import Any

from partner_agents.prompts.charlie import FEW_SHOT, PROMPT_VERSION, build_router_messages
from partner_agents.schemas import RouteDecision

DATASET = Path(__file__).parents[2] / "eval" / "datasets" / "routing.jsonl"


def rows() -> dict[str, dict[str, Any]]:
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    return {row["id"]: row for row in map(json.loads, filter(None, lines))}


def test_every_few_shot_example_is_a_dev_row_with_the_same_query_and_intent() -> None:
    dataset = rows()
    for example_id, (query, decision_json) in FEW_SHOT.items():
        row = dataset[example_id]
        assert row["split"] == "dev", example_id
        assert row["query"] == query
        assert RouteDecision.model_validate_json(decision_json).intent == row["intent"]


def test_few_shot_discovery_slots_match_the_gold_labels() -> None:
    dataset = rows()
    for example_id, (_, decision_json) in FEW_SHOT.items():
        decision = RouteDecision.model_validate_json(decision_json)
        if decision.discovery is None:
            continue
        gold = dataset[example_id]["discovery"]
        got = decision.discovery.model_dump(
            include={"partner_type", "product_category", "location"}
        )
        assert got == gold


def test_every_intent_has_an_example() -> None:
    intents = {RouteDecision.model_validate_json(d).intent for _, d in FEW_SHOT.values()}
    assert intents == {"discovery", "cpg_qa", "outreach", "clarify", "out_of_scope"}


def test_the_dev_split_has_20_rows_with_the_documented_mix() -> None:
    dev = [row for row in rows().values() if row["split"] == "dev"]
    counts: dict[str, int] = {}
    for row in dev:
        counts[row["intent"]] = counts.get(row["intent"], 0) + 1
    # EVAL §3.1: discovery 35%, cpg_qa 25%, outreach 15%, clarify 15%, out_of_scope 10%.
    assert counts == {"discovery": 7, "cpg_qa": 5, "outreach": 3, "clarify": 3, "out_of_scope": 2}


def test_the_prompt_version_is_set_and_the_system_prompt_lists_the_examples() -> None:
    assert PROMPT_VERSION.startswith("charlie-")
    [system] = build_router_messages([], [])
    assert "California me cookie brokers dhundo" in system.text
    assert "No leads have been found" in system.text
