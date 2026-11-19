"""The ``calls`` suite (docs/EVAL.md sections 3.7 and 4.7, ablations A8 to A10).

Every persona of the chosen split is played by the ``simulator`` role against the call agent: the
stage machine and checks by default, or the single-prompt baseline with ``CALL_CONTROL=prompt_only``
(A8). The ``judge`` role then reads each call for the soft rules, the booked time and unsupported
claims. The judge is only as good as its agreement with the human labels in
``eval/datasets/call_labels.jsonl``; without them the soft-rule numbers are marked indicative.
"""

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel

from eval.call_metrics import (
    CallResult,
    JudgeAgreement,
    call_metrics,
    must_hold_result,
    rules_by_id,
    sent_failures,
)
from eval.metrics import SuiteResult
from eval.schemas import CallLabelRow, PersonaRow, load_call_labels, load_personas
from partner_agents.calls.baseline import run_prompt_only_call
from partner_agents.calls.playbook import load_playbook
from partner_agents.calls.simulator import simulate_call
from partner_agents.demo import SAMPLE_BRAND
from partner_agents.graph.call import CallDeps, build_call_graph
from partner_agents.llm import invoke_structured, track_providers, track_usage
from partner_agents.prompts import baseline as baseline_prompt
from partner_agents.prompts import caller as caller_prompt
from partner_agents.prompts import judge as judge_prompt
from partner_agents.prompts import simulator as simulator_prompt
from partner_agents.prompts.judge import CallJudgement, Line, build_judge_messages, soft_rules
from partner_agents.schemas import BrandProfile, CallPlaybook, CallProgress, CallRecord
from partner_agents.storage.repo import InMemoryRepository

if TYPE_CHECKING:
    from eval.run import Env

logger = logging.getLogger(__name__)

EVAL_USER = "eval"
CALL_CONCURRENCY = 2  # calls in flight; a local simulator model is the slow part
LABEL_TARGET = 15  # dev transcripts to label by hand (ADR-020)


class TimedGraph:
    """Wraps the call graph to record how long each agent turn takes (EVAL 4.7, p95 per turn)."""

    def __init__(self, graph: Any, timings: list[int]) -> None:
        self._graph, self._timings = graph, timings

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await self._graph.ainvoke(*args, **kwargs)
        finally:
            self._timings.append(round((time.perf_counter() - started) * 1000))


def new_call(persona_id: str, playbook: CallPlaybook, call_id: str) -> CallRecord:
    """A call that has not started, for the stage machine to run."""
    return CallRecord(
        id=call_id,
        user_id=EVAL_USER,
        lead_id="eval_lead",
        run_id=f"eval_{call_id}",
        draft_id="eval_draft",
        playbook_id=playbook.id,
        playbook_version=playbook.version,
        prompt_version=caller_prompt.PROMPT_VERSION,
        mode="simulate",
        persona_id=persona_id,
        progress=CallProgress(),
        started_at=datetime.now(UTC),
    )


def lines_of(record: CallRecord) -> list[Line]:
    return [(turn.speaker, turn.text) for turn in record.turns]


async def judge_call(
    model: BaseChatModel, playbook: CallPlaybook, brand: BrandProfile, lines: Sequence[Line]
) -> CallJudgement | None:
    """The judge's reading of a call, or ``None`` when it failed (the call stays unjudged)."""
    try:
        return await invoke_structured(
            model,
            CallJudgement,
            build_judge_messages(playbook, brand, lines),
            role="judge",
            prompt_version=judge_prompt.PROMPT_VERSION,
        )
    except Exception as exc:
        logger.warning("judge failed", extra={"error_type": type(exc).__name__})
        return None


async def judge_agreement(
    model: BaseChatModel,
    playbook: CallPlaybook,
    brand: BrandProfile,
    labels: Sequence[CallLabelRow],
) -> JudgeAgreement:
    """How often the judge gives the human's verdict on the labelled transcripts (ADR-020).

    Every human verdict (each soft rule and the "agreed to a time" flag) is one comparison; a
    call the judge could not read counts as all disagreements."""
    matches = total = 0
    for row in labels:
        lines: list[Line] = [(line.speaker, line.text) for line in row.transcript]
        got = await judge_call(model, playbook, brand, lines)
        verdicts = {v.rule_id: v.verdict for v in got.rules} if got else {}
        for rule_id, human in row.labels.items():
            total += 1
            matches += verdicts.get(rule_id) == human
        total += 1
        matches += got is not None and got.partner_agreed_to_time == row.partner_agreed_to_time
    return JudgeAgreement(matches / total if total else None, len(labels), total)


@dataclass
class Played:
    persona: PersonaRow
    record: CallRecord | None
    turn_ms: list[int]
    error: str | None = None


async def play(env: "Env", persona: PersonaRow, playbook: CallPlaybook) -> Played:
    """One persona against the call agent. A simulator failure leaves the call unplayed."""
    repo = InMemoryRepository()
    await repo.save_profile(EVAL_USER, SAMPLE_BRAND)
    simulator = env.model("simulator")
    deps = CallDeps(
        settings=env.settings,
        repo=repo,
        playbook=playbook,
        classifier=env.model("classifier"),
        caller=env.model("caller"),
        simulator=simulator,
    )
    call_id = f"c_{persona.id}"
    timings: list[int] = []
    try:
        with track_usage() as outer_tokens, track_providers() as providers:
            if env.settings.call_control == "prompt_only":
                record = await run_prompt_only_call(
                    deps, simulator, persona.to_persona(), SAMPLE_BRAND,
                    user_id=EVAL_USER, call_id=call_id,
                )  # fmt: skip
            else:
                await repo.save_call(new_call(persona.id, playbook, call_id))
                graph = TimedGraph(build_call_graph(deps), timings)
                record = await simulate_call(
                    graph,  # type: ignore[arg-type]
                    deps,
                    simulator,
                    persona.to_persona(),
                    user_id=EVAL_USER,
                    call_id=call_id,
                )
    except Exception as exc:
        logger.warning("simulated call failed", extra={"error_type": type(exc).__name__})
        return Played(persona, None, timings, f"{type(exc).__name__}: {str(exc)[:150]}")
    # The stage machine's own turns count their tokens inside the record; the simulator's (and all
    # of the baseline's) were counted here, outside it.
    tokens = dict(record.tokens)
    for role, count in outer_tokens.items():
        tokens[role] = tokens.get(role, 0) + count
    record = record.model_copy(
        update={"tokens": tokens, "providers": {**record.providers, **providers}}
    )
    return Played(persona, record, timings)


def item_of(result: CallResult, rules: dict[str, Any], played: Played) -> dict[str, Any]:
    record, persona = result.record, result.persona
    held = {rule_id: must_hold_result(rule_id, rules, result) for rule_id in persona.must_hold}
    judged = result.judgement
    return {
        "id": persona.id,
        "archetype": persona.archetype,
        "gold": persona.golds,
        "outcome": record.progress.outcome,
        "ok": record.progress.outcome in persona.golds
        and all(v is not False for v in held.values()),
        "turns": len(record.turns),
        "violations": dict(sent_failures(record)),
        "regenerated": sum(t.regenerated for t in record.turns),
        "fallbacks": sum(t.used_fallback for t in record.turns),
        "must_hold": held,
        "judge": None
        if judged is None
        else {
            "rules": {v.rule_id: v.verdict for v in judged.rules},
            "partner_agreed_to_time": judged.partner_agreed_to_time,
            "unsupported_claim_lines": judged.unsupported_claim_lines,
        },
        "transcript": [
            {"speaker": t.speaker, "stage": t.stage, "text": t.text} for t in record.turns
        ],
    }


def write_label_sheet(path: Path, played: Sequence[Played], playbook: CallPlaybook) -> int:
    """Dev transcripts for a human to label: the same row shape as ``call_labels.jsonl`` with
    every verdict left empty (``null``), which the loader refuses until it is filled in."""
    rules = [rule.id for rule in soft_rules(playbook)]
    rows = []
    for number, item in enumerate([p for p in played if p.record is not None][:LABEL_TARGET], 1):
        assert item.record is not None
        rows.append(
            {
                "id": f"L{number:02d}",
                "persona_id": item.persona.id,
                "transcript": [{"speaker": s, "text": t} for s, t in lines_of(item.record)],
                "labels": dict.fromkeys(rules),
                "partner_agreed_to_time": None,
            }
        )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return len(rows)


async def run_calls(
    env: "Env",
    personas: Sequence[PersonaRow] | None = None,
    labels: Sequence[CallLabelRow] | None = None,
) -> SuiteResult:
    rows = [
        row for row in (load_personas() if personas is None else personas) if row.split == env.split
    ]
    playbook = load_playbook(env.settings.call_playbook)
    rules = rules_by_id(playbook)
    slots = asyncio.Semaphore(CALL_CONCURRENCY)

    async def one(persona: PersonaRow) -> Played:
        async with slots:
            return await play(env, persona, playbook)

    played = list(await asyncio.gather(*(one(row) for row in rows)))
    judge_model = env.model("judge")
    results: list[CallResult] = []
    items: list[dict[str, Any]] = []
    notes = [
        f"split: {env.split}; control: {env.settings.call_control}",
        "the simulator's hidden-fact discipline is not checked here: read the transcripts for "
        "a persona that broke character and re-run it (EVAL section 9)",
    ]
    for entry in played:
        if entry.record is None:
            notes.append(f"{entry.persona.id} excluded: the call did not finish ({entry.error})")
            items.append({"id": entry.persona.id, "ok": None, "error": entry.error})
            continue
        judgement = await judge_call(judge_model, playbook, SAMPLE_BRAND, lines_of(entry.record))
        result = CallResult(entry.persona, entry.record, judgement, tuple(entry.turn_ms))
        results.append(result)
        items.append(item_of(result, rules, entry))
    labels = load_call_labels() if labels is None else list(labels)
    agreement = await judge_agreement(judge_model, playbook, SAMPLE_BRAND, labels)
    if not labels:
        notes.append(
            "no human-labelled transcripts yet (call_labels.jsonl is empty): judge agreement is "
            "unknown and the soft-rule, unsupported-claim and booking-precision numbers are "
            "indicative only"
        )
    if env.settings.call_control == "prompt_only":
        notes.append("per-turn timing and stage order are not measured for the baseline")
    if env.label_sheet is not None:
        written = write_label_sheet(env.label_sheet, played, playbook)
        notes.append(f"wrote {written} transcripts to label in {env.label_sheet}")
        if env.split != "dev":
            notes.append("warning: the label sheet should come from the dev split")
    versions = {
        "caller": caller_prompt.PROMPT_VERSION,
        "simulator": simulator_prompt.PROMPT_VERSION,
        "judge": judge_prompt.PROMPT_VERSION,
    }
    if env.settings.call_control == "prompt_only":
        versions["baseline"] = baseline_prompt.PROMPT_VERSION
    metrics = call_metrics(
        results, playbook, agreement=agreement, control=env.settings.call_control
    )
    return SuiteResult("calls", len(results), metrics, items, notes, prompt_versions=versions)
