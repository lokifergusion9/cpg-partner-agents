"""Call schemas and the playbook loader (HLD §4.1, §6.11, T-54)."""

import copy
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from partner_agents.calls.playbook import (
    CHECK_IDS,
    DEFAULT_PLAYBOOK_DIR,
    STAGE_ORDER,
    PlaybookError,
    load_playbook,
    parse_playbook,
)
from partner_agents.config import Settings
from partner_agents.demo import SAMPLE_BRAND
from partner_agents.schemas import (
    CallProgress,
    CallRecord,
    CallTurn,
    Directive,
    RuleCheck,
    TurnSignals,
)

NOW = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)
REAL = DEFAULT_PLAYBOOK_DIR / "partner_intro.toml"


def real_data() -> dict[str, Any]:
    return tomllib.loads(REAL.read_text(encoding="utf-8"))


def all_rules(data: dict[str, Any]) -> list[dict[str, Any]]:
    return list(data["universal_rules"]) + [r for s in data["stages"] for r in s["rules"]]


def broken(change: Any) -> str:
    """Apply ``change`` to a copy of the real playbook and return the loader's error."""
    data = copy.deepcopy(real_data())
    change(data)
    with pytest.raises(PlaybookError) as excinfo:
        parse_playbook(data)
    return str(excinfo.value)


# --- the real playbook --------------------------------------------------------------


def test_the_real_playbook_loads_with_six_stages_in_order() -> None:
    playbook = load_playbook("partner_intro")
    assert playbook.id == "partner_intro" and playbook.version
    assert tuple(stage.name for stage in playbook.stages) == STAGE_ORDER
    assert all(stage.fallback_line.strip() and stage.goal.strip() for stage in playbook.stages)
    assert playbook.banned_phrases and playbook.max_turns == 30


def test_the_real_playbook_has_the_rule_ids_the_docs_rely_on() -> None:
    playbook = load_playbook("partner_intro")
    ids = [rule.id for rule in playbook.universal_rules]
    ids += [rule.id for stage in playbook.stages for rule in stage.rules]
    assert len(ids) == len(set(ids))
    # ADR-019, EVAL §3.7 and HLD §4.1 name these.
    assert {"U-AI-1", "U-AI-2", "U-OPT-1", "U-RETRY-1", "Q-2", "Q-STOP-1"} <= set(ids)


def test_every_code_rule_maps_to_a_known_check_and_every_reply_check_is_used() -> None:
    playbook = load_playbook("partner_intro")
    rules = list(playbook.universal_rules) + [r for s in playbook.stages for r in s.rules]
    code_rules = [rule for rule in rules if rule.check == "code"]
    assert all(rule.check_id in CHECK_IDS for rule in code_rules)
    assert {rule.check_id for rule in code_rules} == CHECK_IDS  # no check goes unreferenced
    assert any(rule.check == "judge" for rule in rules)  # soft rules exist (ADR-020)


def test_qualification_has_the_slots_in_order_and_the_required_ones() -> None:
    stage = next(s for s in load_playbook("partner_intro").stages if s.name == "qualification")
    assert stage.slots == ["role", "categories", "territory", "accounts", "volume"]
    assert stage.required_slots == ["role", "categories", "territory"]


def test_the_fallback_lines_fill_from_the_sample_brand() -> None:
    fields = {
        "brand_name": SAMPLE_BRAND.brand_name,
        "founder_name": SAMPLE_BRAND.founder_name,
        "category": SAMPLE_BRAND.category,
        "sign_off": SAMPLE_BRAND.sign_off,
    }
    for stage in load_playbook("partner_intro").stages:
        line = stage.fallback_line.format(**fields)
        assert "{" not in line and line.count("?") <= 1
    opening = load_playbook("partner_intro").stages[0].fallback_line
    assert "AI assistant" in opening  # a fallback opening still discloses (U-AI-1)


def test_the_default_setting_names_a_playbook_that_exists() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.call_playbook == "partner_intro"
    assert (settings.call_max_turns, settings.call_reply_max_words) == (30, 60)
    load_playbook(settings.call_playbook)


def test_a_playbook_setting_that_could_leave_the_directory_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, call_playbook="../secrets")  # type: ignore[call-arg]


# --- rejected playbooks -------------------------------------------------------------


def test_a_missing_stage_is_rejected() -> None:
    message = broken(lambda d: d["stages"].pop(3))  # no pitch
    assert "stages must be" in message


def test_stages_out_of_order_are_rejected() -> None:
    def swap(d: dict[str, Any]) -> None:
        d["stages"][1], d["stages"][2] = d["stages"][2], d["stages"][1]

    assert "in this order" in broken(swap)


def test_a_duplicate_rule_id_is_rejected() -> None:
    def duplicate(d: dict[str, Any]) -> None:
        d["stages"][2]["rules"][0]["id"] = "U-OPT-1"

    assert "duplicate rule id U-OPT-1" in broken(duplicate)


def test_an_unknown_check_id_is_rejected() -> None:
    def unknown(d: dict[str, Any]) -> None:
        d["universal_rules"][0]["check_id"] = "C-MAGIC"

    assert "unknown check id C-MAGIC" in broken(unknown)


def test_a_code_rule_without_a_check_id_is_rejected() -> None:
    def missing(d: dict[str, Any]) -> None:
        del d["universal_rules"][0]["check_id"]

    assert "no check_id" in broken(missing)


def test_a_check_id_on_a_judge_rule_is_rejected() -> None:
    def judged(d: dict[str, Any]) -> None:
        rule = next(r for r in all_rules(d) if r["check"] == "judge")
        rule["check_id"] = "C-NUM"

    assert "has a check_id but check=judge" in broken(judged)


def test_slots_outside_qualification_are_rejected() -> None:
    assert "only qualification has slots" in broken(lambda d: d["stages"][0].update(slots=["role"]))


def test_a_required_slot_that_is_not_a_slot_is_rejected() -> None:
    assert "requires unknown slots" in broken(
        lambda d: d["stages"][2].update(required_slots=["budget"])
    )


def test_a_fallback_line_with_an_unknown_field_is_rejected() -> None:
    assert "unknown fields ['partner_email']" in broken(
        lambda d: d["stages"][4].update(fallback_line="Mail {partner_email}?")
    )


def test_a_malformed_rule_is_rejected_by_the_schema() -> None:
    assert "invalid playbook" in broken(lambda d: d["universal_rules"][0].update(kind="maybe"))


def test_a_blank_banned_phrase_is_rejected() -> None:
    assert "banned phrases" in broken(lambda d: d["banned_phrases"].append("  "))


def test_a_missing_file_invalid_toml_and_a_wrong_id_are_clear_errors(tmp_path: Path) -> None:
    with pytest.raises(PlaybookError, match="no playbook file"):
        load_playbook("nope", tmp_path)
    (tmp_path / "bad.toml").write_text("id = [", encoding="utf-8")
    with pytest.raises(PlaybookError, match="not valid TOML"):
        load_playbook("bad", tmp_path)
    (tmp_path / "other.toml").write_text(REAL.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(PlaybookError, match="declares id 'partner_intro'"):
        load_playbook("other", tmp_path)


@pytest.mark.parametrize("playbook_id", ["../partner_intro", "Partner", "a/b", ""])
def test_a_playbook_id_that_is_not_a_plain_name_is_refused(playbook_id: str) -> None:
    with pytest.raises(PlaybookError, match="must match"):
        load_playbook(playbook_id)


# --- the call record ----------------------------------------------------------------


def make_record() -> CallRecord:
    slot = NOW + timedelta(days=1)
    return CallRecord(
        id="c_1",
        user_id="u_1",
        lead_id="l_1",
        run_id="r_1",
        draft_id="d_1",
        playbook_id="partner_intro",
        playbook_version="1",
        prompt_version="caller-v1",
        mode="simulate",
        persona_id="demo_easy",
        progress=CallProgress(
            stage="ended",
            slots={"territory": "Northern California"},
            retries_used={"discovery": 1},
            proposed_slot=slot,
            turns=2,
            outcome="meeting_booked",
        ),
        turns=[
            CallTurn(
                index=0,
                speaker="agent",
                text="Hello, this is an AI assistant.",
                stage="opening",
                directive=Directive(action="propose_slot", fact_ids=["F-1"], proposed_slot=slot),
                checks=[RuleCheck(rule_id="C-AI", passed=True, by="code")],
                regenerated=True,
            ),
            CallTurn(
                index=1,
                speaker="partner",
                text="Sure, Thursday at 11.",
                stage="handoff",
                signals=TurnSignals(agrees_to_meeting=True, slot_values={"role": "owner"}),
            ),
        ],
        meeting_at=slot,
        followup_draft_id="d_2",
        started_at=NOW,
        finished_at=NOW + timedelta(minutes=4),
        tokens={"caller": 120},
        providers={"caller": "fake"},
    )


def test_a_call_record_round_trips_through_json() -> None:
    record = make_record()
    assert CallRecord.model_validate_json(record.model_dump_json()) == record


def test_call_timestamps_must_be_timezone_aware() -> None:
    naive = datetime(2026, 11, 17, 10, 0)
    with pytest.raises(ValidationError):
        CallProgress(proposed_slot=naive)
    with pytest.raises(ValidationError):
        make_record().model_validate({**make_record().model_dump(), "meeting_at": naive})


def test_a_new_call_starts_in_the_opening_with_nothing_filled() -> None:
    progress = CallProgress()
    assert (progress.stage, progress.slots, progress.turns, progress.outcome) == (
        "opening",
        {},
        0,
        None,
    )
    assert TurnSignals().opt_out is False and TurnSignals().slot_values == {}
