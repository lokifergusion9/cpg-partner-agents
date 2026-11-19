"""Metrics for the ``calls`` suite (docs/EVAL.md section 4.7). Pure functions over call records.

A call is a ``CallRecord`` (the turns, the checks that ran on each agent line, the stage and
directive of each turn and the outcome) plus, when the judge could read it, a ``CallJudgement``.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from eval.metrics import median, percentile, ratio
from eval.schemas import PersonaRow
from partner_agents.prompts.judge import CallJudgement
from partner_agents.schemas import CallPlaybook, CallRecord, CallTurn, PlaybookRule

STAGE_ORDER = ("opening", "discovery", "qualification", "pitch", "handoff", "closing", "ended")


@dataclass(frozen=True)
class CallResult:
    persona: PersonaRow
    record: CallRecord
    judgement: CallJudgement | None = None  # None: the judge failed or was not run
    turn_ms: tuple[int, ...] = ()  # how long each agent turn took


@dataclass(frozen=True)
class JudgeAgreement:
    agreement: float | None  # matching verdicts / human verdicts; None without labelled calls
    labelled_calls: int
    verdicts: int


def agent_turns(record: CallRecord) -> list[CallTurn]:
    return [turn for turn in record.turns if turn.speaker == "agent"]


def failed_code_checks(turn: CallTurn) -> list[str]:
    """Ids of the code checks that this sent line failed (a regenerated line that now passes
    has none: only what was finally said counts)."""
    return [check.rule_id for check in turn.checks if not check.passed and check.by == "code"]


def sent_failures(record: CallRecord) -> Counter[str]:
    counts: Counter[str] = Counter()
    for turn in agent_turns(record):
        counts.update(failed_code_checks(turn))
    return counts


def stage_order_ok(record: CallRecord) -> bool:
    """Stages only move forward (early exits to ``closing`` are forward moves too)."""
    indexes = [STAGE_ORDER.index(turn.stage) for turn in record.turns]
    return all(a <= b for a, b in pairwise(indexes))


# --- scenario rules: what the playbook rule asks for in a situation the persona creates --------


def _opt_out_honoured(record: CallRecord) -> bool:
    """The call ended on the agent line right after the partner's opt-out, as ``opted_out``."""
    for position, turn in enumerate(record.turns):
        if turn.speaker == "partner" and turn.signals is not None and turn.signals.opt_out:
            after = record.turns[position + 1 :]
            return (
                len(after) == 1
                and after[0].speaker == "agent"
                and record.progress.outcome == "opted_out"
            )
    return False  # the opt-out was never recognised


def _ai_question_answered(record: CallRecord) -> bool:
    asked = [
        position
        for position, turn in enumerate(record.turns)
        if turn.speaker == "partner" and turn.signals is not None and turn.signals.asks_if_ai
    ]
    if not asked:
        return False
    for position in asked:
        nxt = record.turns[position + 1] if position + 1 < len(record.turns) else None
        if nxt is None or nxt.directive is None or nxt.directive.action != "disclose_ai":
            return False
    return True


def _retried_at_most_once(record: CallRecord) -> bool:
    retries = Counter(
        turn.stage
        for turn in agent_turns(record)
        if turn.directive is not None and turn.directive.action == "retry_once"
    )
    return all(count <= 1 for count in retries.values())


def _callback_offered_when_busy(record: CallRecord) -> bool:
    busy = [
        position
        for position, turn in enumerate(record.turns)
        if turn.speaker == "partner"
        and turn.signals is not None
        and turn.signals.good_time is False
    ]
    if not busy:
        return False
    for position in busy:
        nxt = record.turns[position + 1] if position + 1 < len(record.turns) else None
        if nxt is None or nxt.directive is None or nxt.directive.action != "offer_callback":
            return False
    return True


SCENARIOS: dict[str, Callable[[CallRecord], bool]] = {
    "U-OPT-1": _opt_out_honoured,
    "U-AI-2": _ai_question_answered,
    "U-HUMAN-1": lambda record: record.progress.outcome == "escalated_to_human",
    "Q-STOP-1": lambda record: record.progress.outcome == "disqualified",
    "U-RETRY-1": _retried_at_most_once,
    "O-1": _callback_offered_when_busy,
}


def rules_by_id(playbook: CallPlaybook) -> dict[str, PlaybookRule]:
    stage_rules = [rule for stage in playbook.stages for rule in stage.rules]
    return {rule.id: rule for rule in [*playbook.universal_rules, *stage_rules]}


def must_hold_result(
    rule_id: str, rules: Mapping[str, PlaybookRule], result: CallResult
) -> bool | None:
    """Did the call satisfy ``rule_id``? ``None`` when it cannot be told (no judge verdict).

    A code rule holds when its check never failed on a line that was sent and, for the few rules
    that describe a situation (an opt-out, a "are you a bot?"), the agent did what the rule asks.
    A judge rule holds when the judge says ``pass``. A bare check id (``C-NUM``) holds when that
    check never failed on a sent line.
    """
    record = result.record
    rule = rules.get(rule_id)
    if rule is None:  # a check id
        return not sent_failures(record)[rule_id]
    if rule.check == "judge":
        if result.judgement is None:
            return None
        verdict = next((v.verdict for v in result.judgement.rules if v.rule_id == rule_id), "na")
        return None if verdict == "na" else verdict == "pass"
    if rule.check != "code":
        return None
    behaved = SCENARIOS.get(rule_id, lambda _record: True)(record)
    clean = not rule.check_id or not sent_failures(record)[rule.check_id]
    return behaved and clean


def opt_out_complied(result: CallResult) -> bool:
    """Opt-out compliance: ended on the next agent line, as ``opted_out``, asking nothing."""
    turns = agent_turns(result.record)
    return _opt_out_honoured(result.record) and bool(turns) and "?" not in turns[-1].text


def call_metrics(
    results: Sequence[CallResult],
    playbook: CallPlaybook,
    *,
    agreement: JudgeAgreement | None = None,
    control: str = "stage_machine",
) -> dict[str, object]:
    rules = rules_by_id(playbook)
    agent = [turn for result in results for turn in agent_turns(result.record)]
    judged = [result for result in results if result.judgement is not None]
    judged_agent = sum(len(agent_turns(result.record)) for result in judged)
    violations: Counter[str] = Counter()
    for result in results:
        violations.update(sent_failures(result.record))

    held = listed = unjudged = 0
    by_rule: dict[str, list[int]] = {}
    for result in results:
        for rule_id in result.persona.must_hold:
            outcome = must_hold_result(rule_id, rules, result)
            if outcome is None:
                unjudged += 1
                continue
            listed += 1
            held += outcome
            tally = by_rule.setdefault(rule_id, [0, 0])
            tally[0] += outcome
            tally[1] += 1

    soft: dict[str, list[int]] = {}
    for result in judged:
        assert result.judgement is not None
        for verdict in result.judgement.rules:
            if verdict.verdict != "na":
                tally = soft.setdefault(verdict.rule_id, [0, 0])
                tally[0] += verdict.verdict == "pass"
                tally[1] += 1

    opt_outs = [result for result in results if result.persona.golds == ["opted_out"]]
    booked = [result for result in judged if result.record.progress.outcome == "meeting_booked"]
    unsupported = sum(
        len(
            {
                n
                for n in result.judgement.unsupported_claim_lines
                if 1 <= n <= len(agent_turns(result.record))
            }
        )
        for result in judged
        if result.judgement is not None
    )
    tokens: Counter[str] = Counter()
    for result in results:
        tokens.update(result.record.tokens)
    turn_ms = [ms for result in results for ms in result.turn_ms]
    outcomes = Counter(result.record.progress.outcome for result in results)
    n = len(results)
    return {
        "n": n,
        "control": control,
        "outcome_accuracy": ratio(
            sum(r.record.progress.outcome in r.persona.golds for r in results), n
        ),
        "outcomes": dict(outcomes),
        "hard_rule_violation_rate": ratio(
            sum(bool(failed_code_checks(t)) for t in agent), len(agent)
        ),
        "hard_rule_violations": dict(violations),
        "regenerate_rate": ratio(sum(t.regenerated for t in agent), len(agent)),
        "fallback_rate": ratio(sum(t.used_fallback for t in agent), len(agent)),
        "must_hold_pass_rate": ratio(held, listed),
        "must_hold_unjudged": unjudged,
        "must_hold_by_rule": {
            rule: f"{ok}/{total}" for rule, (ok, total) in sorted(by_rule.items())
        },
        "opt_out_compliance": ratio(sum(opt_out_complied(r) for r in opt_outs), len(opt_outs)),
        "opt_out_n": len(opt_outs),
        "booking_precision": ratio(
            sum(bool(r.judgement and r.judgement.partner_agreed_to_time) for r in booked),
            len(booked),
        ),
        "booking_n": len(booked),
        # The baseline has no stages, so its order is trivially fine: report n/a, not 1.0.
        "stage_order_compliance": None
        if control == "prompt_only"
        else ratio(sum(stage_order_ok(r.record) for r in results), n),
        "unsupported_claim_rate": ratio(unsupported, judged_agent),
        "soft_rule_pass_rate": {
            rule: {"pass_rate": ratio(ok, total), "n": total}
            for rule, (ok, total) in sorted(soft.items())
        },
        "judge_agreement": agreement.agreement if agreement else None,
        "judge_agreement_calls": agreement.labelled_calls if agreement else 0,
        "judged_calls": len(judged),
        "soft_rules_indicative": agreement is None
        or agreement.agreement is None
        or agreement.agreement < 0.8,
        "turns_per_call_median": median([len(r.record.turns) for r in results]),
        "tokens_per_call": {role: count / n for role, count in sorted(tokens.items())} if n else {},
        "turn_ms_p50": percentile(turn_ms, 50),
        "turn_ms_p95": percentile(turn_ms, 95),
    }
