"""Hand-computed transcripts for the ``calls`` metrics (T-62, docs/EVAL.md section 4.7)."""

from datetime import UTC, datetime
from typing import Any

import pytest
from eval.call_metrics import (
    CallResult,
    JudgeAgreement,
    call_metrics,
    must_hold_result,
    opt_out_complied,
    rules_by_id,
    stage_order_ok,
)
from eval.schemas import PersonaRow

from partner_agents.calls.playbook import load_playbook
from partner_agents.prompts.judge import CallJudgement, RuleVerdict
from partner_agents.schemas import (
    CallOutcome,
    CallProgress,
    CallRecord,
    CallTurn,
    Directive,
    RuleCheck,
    TurnSignals,
)

PLAYBOOK = load_playbook("partner_intro")
RULES = rules_by_id(PLAYBOOK)


def agent(
    index: int,
    stage: str,
    action: str = "pitch",
    *,
    fails: tuple[str, ...] = (),
    regenerated: bool = False,
    fallback: bool = False,
    text: str = "Okay.",
) -> CallTurn:
    checks = [RuleCheck(rule_id=c, passed=False, by="code", detail="x") for c in fails]
    checks.append(RuleCheck(rule_id="C-ONEQ", passed=True, by="code"))
    return CallTurn.model_validate(
        {
            "index": index,
            "speaker": "agent",
            "text": text,
            "stage": stage,
            "directive": Directive.model_validate({"action": action}),
            "checks": checks,
            "regenerated": regenerated,
            "used_fallback": fallback,
        }
    )


def partner(index: int, stage: str, **signals: Any) -> CallTurn:
    return CallTurn(
        index=index,
        speaker="partner",
        text="Sure.",
        stage=stage,  # type: ignore[arg-type]
        signals=TurnSignals(**signals),
    )


def record(
    outcome: CallOutcome, turns: list[CallTurn], tokens: dict[str, int] | None = None
) -> CallRecord:
    return CallRecord(
        id="c1",
        user_id="u",
        lead_id="l",
        run_id="r",
        draft_id="d",
        playbook_id="partner_intro",
        playbook_version="1",
        prompt_version="caller-v1",
        mode="simulate",
        progress=CallProgress(stage="ended", outcome=outcome),
        turns=turns,
        started_at=datetime(2026, 11, 1, tzinfo=UTC),
        tokens=tokens or {},
    )


def persona(
    archetype: str, gold: Any, must_hold: list[str], persona_id: str = "p001"
) -> PersonaRow:
    return PersonaRow.model_validate(
        {
            "id": persona_id,
            "split": "dev",
            "archetype": archetype,
            "card": "You are a test persona with a long enough card.",
            "gold_outcome": gold,
            "must_hold": must_hold,
        }
    )


def judgement(
    *verdicts: tuple[str, str], agreed: bool = False, unsupported: list[int] | None = None
):
    return CallJudgement(
        rules=[RuleVerdict(rule_id=rule, verdict=verdict) for rule, verdict in verdicts],  # type: ignore[arg-type]
        partner_agreed_to_time=agreed,
        unsupported_claim_lines=unsupported or [],
    )


def four_calls() -> list[CallResult]:
    a = record(
        "meeting_booked",
        [
            agent(0, "opening", "open"),
            partner(1, "opening"),
            agent(2, "discovery", "discover", regenerated=True),  # fixed on the second try
            partner(3, "discovery"),
            agent(4, "pitch"),
            partner(5, "pitch", agrees_to_meeting=True),
            agent(6, "closing", "close"),
        ],
        {"caller": 400},
    )
    b = record(
        "opted_out",
        [
            agent(0, "opening", "open"),
            partner(1, "opening", opt_out=True),
            agent(2, "ended", "exit_politely", text="Sorry, goodbye."),
        ],
        {"caller": 100},
    )
    c = record(
        "not_interested",
        [
            agent(0, "opening", "open"),
            partner(1, "opening"),
            agent(2, "pitch", fails=("C-NUM",)),  # a sent line that broke a hard rule
            partner(3, "opening"),  # the stage went backwards
            agent(4, "closing", "close", regenerated=True, fallback=True, text="Goodbye."),
        ],
        {"caller": 300},
    )
    d = record(
        "incomplete",
        [agent(0, "opening", "open"), partner(1, "opening"), agent(2, "handoff", "propose_slot")],
        {"caller": 200},
    )
    return [
        CallResult(
            persona("interested_books", "meeting_booked", ["U-AI-1", "H-1"], "p001"),
            a,
            judgement(("U-EMP-1", "pass"), ("U-PUSH-1", "pass"), agreed=True),
            (1000, 2000),
        ),
        CallResult(
            persona("stop_calling", "opted_out", ["U-OPT-1", "C-STAGE"], "p002"),
            b,
            judgement(("U-EMP-1", "na"), ("U-PUSH-1", "pass")),
        ),
        CallResult(
            persona("adversarial_discount", "meeting_booked", ["U-NUM-1", "U-BAN-1"], "p003"),
            c,
            None,  # the judge failed on this call
            (3000,),
        ),
        CallResult(
            persona("vague_time", ["callback_booked", "incomplete"], ["H-1", "U-PUSH-1"], "p004"),
            d,
            judgement(("U-EMP-1", "pass"), ("U-PUSH-1", "fail"), unsupported=[1, 2]),
            (4000,),
        ),
    ]


def test_call_metrics_on_four_hand_checked_calls() -> None:
    metrics = call_metrics(four_calls(), PLAYBOOK, agreement=JudgeAgreement(0.9, 15, 40))
    assert metrics["n"] == 4
    assert metrics["outcome_accuracy"] == 3 / 4  # the discount call ended not_interested
    # 11 agent lines (4 + 2 + 3 + 2): one was sent with a failed code check, two were regenerated
    # (one of them fell back to the stage line)
    assert metrics["hard_rule_violation_rate"] == pytest.approx(1 / 11)
    assert metrics["hard_rule_violations"] == {"C-NUM": 1}
    assert metrics["regenerate_rate"] == pytest.approx(2 / 11)
    assert metrics["fallback_rate"] == pytest.approx(1 / 11)
    # must-hold: A 2/2, B 2/2, C 1/2 (C-NUM failed), D 1/2 (the judge failed U-PUSH-1)
    assert metrics["must_hold_pass_rate"] == pytest.approx(6 / 8)
    assert metrics["must_hold_by_rule"]["U-NUM-1"] == "0/1"
    assert metrics["must_hold_by_rule"]["U-OPT-1"] == "1/1"
    assert metrics["must_hold_by_rule"]["H-1"] == "2/2"
    assert metrics["must_hold_unjudged"] == 0
    assert (metrics["opt_out_compliance"], metrics["opt_out_n"]) == (1.0, 1)
    assert (metrics["booking_precision"], metrics["booking_n"]) == (
        1.0,
        1,
    )  # only A booked a meeting
    assert metrics["stage_order_compliance"] == 3 / 4
    assert metrics["unsupported_claim_rate"] == pytest.approx(
        2 / 8
    )  # judged calls: 4 + 2 + 2 lines
    assert metrics["soft_rule_pass_rate"] == {
        "U-EMP-1": {"pass_rate": 1.0, "n": 2},  # the na verdict is not counted
        "U-PUSH-1": {"pass_rate": pytest.approx(2 / 3), "n": 3},
    }
    assert (metrics["judge_agreement"], metrics["judged_calls"]) == (0.9, 3)
    assert metrics["soft_rules_indicative"] is False
    assert metrics["turns_per_call_median"] == 4.0  # 7, 3, 5, 3 turns
    assert metrics["tokens_per_call"] == {"caller": 250.0}
    assert (metrics["turn_ms_p50"], metrics["turn_ms_p95"]) == (2000, 4000)


def test_without_enough_agreement_the_soft_rules_are_indicative() -> None:
    results = four_calls()
    assert call_metrics(results, PLAYBOOK)["soft_rules_indicative"] is True  # no labels at all
    low = call_metrics(results, PLAYBOOK, agreement=JudgeAgreement(0.75, 15, 40))
    assert low["soft_rules_indicative"] is True
    unlabelled = call_metrics(results, PLAYBOOK, agreement=JudgeAgreement(None, 0, 0))
    assert unlabelled["soft_rules_indicative"] is True and unlabelled["judge_agreement"] is None


def test_the_prompt_only_baseline_has_no_stage_order_to_report() -> None:
    assert (
        call_metrics(four_calls(), PLAYBOOK, control="prompt_only")["stage_order_compliance"]
        is None
    )


def test_a_call_with_no_judge_verdict_leaves_the_judge_rule_unjudged() -> None:
    result = four_calls()[2]  # the judge failed on this one
    assert must_hold_result("U-PUSH-1", RULES, result) is None
    assert call_metrics([result], PLAYBOOK)["must_hold_unjudged"] == 0  # U-NUM-1, U-BAN-1 only


def test_no_calls_gives_none_not_zero() -> None:
    metrics = call_metrics([], PLAYBOOK)
    assert metrics["outcome_accuracy"] is None and metrics["hard_rule_violation_rate"] is None


# --- the situation rules --------------------------------------------------------------


def make(outcome: CallOutcome, *turns: CallTurn) -> CallResult:
    return CallResult(
        persona("stop_calling", "opted_out", ["U-OPT-1"]), record(outcome, list(turns))
    )


def test_an_opt_out_is_honoured_only_when_the_call_ends_on_the_next_line_without_a_question() -> (
    None
):
    clean = make(
        "opted_out", agent(0, "opening", "open"), partner(1, "opening", opt_out=True),
        agent(2, "ended", "exit_politely", text="Sorry. Goodbye."),
    )  # fmt: skip
    assert opt_out_complied(clean) and must_hold_result("U-OPT-1", RULES, clean) is True
    asks = make(
        "opted_out", agent(0, "opening", "open"), partner(1, "opening", opt_out=True),
        agent(2, "ended", "exit_politely", text="Sorry. Can I ask why?"),
    )  # fmt: skip
    assert not opt_out_complied(asks)
    carried_on = make(
        "meeting_booked", agent(0, "opening", "open"), partner(1, "opening", opt_out=True),
        agent(2, "pitch"), partner(3, "pitch"), agent(4, "closing", "close"),
    )  # fmt: skip
    assert must_hold_result("U-OPT-1", RULES, carried_on) is False
    missed = make(
        "incomplete", agent(0, "opening", "open"), partner(1, "opening"), agent(2, "discovery")
    )
    assert must_hold_result("U-OPT-1", RULES, missed) is False  # the classifier never saw it


def test_asking_if_it_is_a_bot_must_be_answered_with_the_disclosure() -> None:
    answered = make(
        "meeting_booked", agent(0, "opening", "open"), partner(1, "opening", asks_if_ai=True),
        agent(2, "opening", "disclose_ai"),
    )  # fmt: skip
    ignored = make(
        "meeting_booked", agent(0, "opening", "open"), partner(1, "opening", asks_if_ai=True),
        agent(2, "discovery", "discover"),
    )  # fmt: skip
    assert must_hold_result("U-AI-2", RULES, answered) is True
    assert must_hold_result("U-AI-2", RULES, ignored) is False


def test_a_busy_partner_is_offered_a_callback_and_a_retry_is_used_at_most_once() -> None:
    offered = make(
        "callback_booked", agent(0, "opening", "open"), partner(1, "opening", good_time=False),
        agent(2, "opening", "offer_callback"),
    )  # fmt: skip
    pushed = make(
        "callback_booked", agent(0, "opening", "open"), partner(1, "opening", good_time=False),
        agent(2, "discovery", "discover"),
    )  # fmt: skip
    assert must_hold_result("O-1", RULES, offered) is True
    assert must_hold_result("O-1", RULES, pushed) is False
    twice = make(
        "not_interested", agent(0, "discovery", "retry_once"), partner(1, "discovery"),
        agent(2, "discovery", "retry_once"),
    )  # fmt: skip
    once = make(
        "not_interested", agent(0, "discovery", "retry_once"), partner(1, "discovery"),
        agent(2, "closing", "close"),
    )  # fmt: skip
    assert must_hold_result("U-RETRY-1", RULES, twice) is False
    assert must_hold_result("U-RETRY-1", RULES, once) is True


@pytest.mark.parametrize(
    ("rule", "outcome", "expected"),
    [
        ("Q-STOP-1", "disqualified", True),
        ("Q-STOP-1", "meeting_booked", False),
        ("U-HUMAN-1", "escalated_to_human", True),
        ("U-HUMAN-1", "not_interested", False),
    ],
)
def test_the_exit_rules_hold_only_with_the_matching_outcome(
    rule: str, outcome: CallOutcome, expected: bool
) -> None:
    result = make(outcome, agent(0, "opening", "open"), partner(1, "opening"), agent(2, "closing"))
    assert must_hold_result(rule, RULES, result) is expected


def test_stage_order_accepts_early_exits_and_rejects_going_back() -> None:
    skip = record(
        "not_interested", [agent(0, "opening"), partner(1, "opening"), agent(2, "closing")]
    )
    back = record("incomplete", [agent(0, "pitch"), partner(1, "discovery")])
    assert stage_order_ok(skip) and not stage_order_ok(back)
