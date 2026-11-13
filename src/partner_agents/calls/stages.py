"""The call agent's stage machine (HLD §6.11, ADR-017). Pure: no model, no I/O, no clock.

``advance()`` takes the call's progress and the classified partner turn and returns the new
progress plus a ``Directive`` that tells the writer what to say next. Stage order, stop rules,
retries, opt-outs and outcomes live here and nowhere else (CLAUDE.md rule 10).

Order of the rules for every partner turn:

1. ``opt_out`` ends the call as ``opted_out``. It overrides everything.
2. The turn limit ends the call as ``incomplete`` (or with the outcome already booked).
3. ``asks_if_ai`` -> ``disclose_ai``, and the stage stays.
4. ``wants_human`` -> handoff to book a meeting with the founder (``escalated_to_human``).
5. ``distressed`` -> ``acknowledge``; if the next turn is negative the call ends.
6. ``wants_other_category`` -> polite exit as ``disqualified``.
7. The stage's own transition (the table in HLD §6.11), with retry-once on resistance.
8. A partner ``question`` is answered before the next step is taken; the step is taken on the
   next turn. Exits and a booking confirmation are not delayed: after a booking the question
   stays open and closing answers it before signing off.
"""

import re
from collections.abc import Iterable
from datetime import datetime

from partner_agents.calls.scheduler import is_bookable, parse_time, propose_slot
from partner_agents.schemas import (
    BrandProfile,
    CallOutcome,
    CallPlaybook,
    CallProgress,
    Directive,
    DirectiveAction,
    PlaybookStage,
    TurnSignals,
)

PITCH_FACTS = 3  # at most this many facts in a pitch or an answer
MIN_PITCH_FACTS = 2  # a pitch is padded with the first facts up to this many

_BOOKED: dict[str, CallOutcome] = {
    "meeting": "meeting_booked",
    "callback": "callback_booked",
    "founder": "escalated_to_human",
}
# A territory answer like this fits any region list.
_EVERYWHERE = ("nationwide", "national", "all states", "across the us", "across the country")
_WORD = re.compile(r"[a-z]+")


class CallEnded(ValueError):
    """``advance`` was called on a call that already has an outcome."""


def first_directive() -> Directive:
    """The opening line, before the partner has said anything."""
    return Directive(action="open")


def advance(
    progress: CallProgress,
    signals: TurnSignals,
    playbook: CallPlaybook,
    brand: BrandProfile,
    *,
    now: datetime,
    max_turns: int | None = None,
) -> tuple[CallProgress, Directive]:
    """Apply one classified partner turn. ``now`` is the call time, for meeting proposals."""
    if progress.outcome is not None or progress.stage == "ended":
        raise CallEnded("the call has already ended")
    p = progress.model_copy(deep=True)
    p.turns += 1
    if signals.opt_out:
        return _end(p, "opted_out")
    if p.turns >= min(max_turns or playbook.max_turns, playbook.max_turns):
        return _end(p, p.closing_outcome or "incomplete", "close")

    slots = _stage(playbook, "qualification")
    for name, value in signals.slot_values.items():
        if name in slots.slots and value.strip():
            p.slots[name] = value.strip()
    p.in_rush = p.in_rush or signals.in_rush
    if p.recovering:
        p.recovering = False
        if _negative(signals):
            return _end(p, "not_interested")

    if signals.asks_if_ai:
        return p, Directive(action="disclose_ai")
    if signals.wants_human and p.stage not in ("handoff", "closing"):
        p.booking = "founder"
        return _enter_handoff(p, signals, brand, now, "propose_slot")
    if signals.wants_human and p.stage == "handoff":
        p.booking = "founder"
    if signals.distressed:
        p.recovering = True
        return p, Directive(action="acknowledge")
    if signals.wants_other_category and p.stage != "closing":
        return _end(p, "disqualified")

    step = _Step(signals, playbook, brand, now)
    after, directive = step.run(p.model_copy(deep=True))
    if signals.question and after.outcome is None and p.stage != "closing":
        if directive.action == "confirm_booking":
            after.open_question = True
        else:
            return p, _answer(brand, signals.question)
    return after, directive


class _Step:
    """The stage-specific transition for one turn (rule 7)."""

    def __init__(
        self, signals: TurnSignals, playbook: CallPlaybook, brand: BrandProfile, now: datetime
    ) -> None:
        self.s, self.playbook, self.brand, self.now = signals, playbook, brand, now

    def run(self, p: CallProgress) -> tuple[CallProgress, Directive]:
        s = self.s
        if s.wants_callback and p.stage in ("opening", "discovery", "qualification", "pitch"):
            p.booking = "callback"
            return _enter_handoff(p, s, self.brand, self.now, "offer_callback")
        handler = {
            "opening": self.opening,
            "discovery": self.discovery,
            "qualification": self.qualification,
            "pitch": self.pitch,
            "handoff": self.handoff,
            "closing": self.closing,
        }[p.stage]
        return handler(p)

    def opening(self, p: CallProgress) -> tuple[CallProgress, Directive]:
        s = self.s
        if s.right_person is False:
            return _end(p, "wrong_person")
        if s.good_time is False:
            p.booking = "callback"
            return _enter_handoff(p, s, self.brand, self.now, "offer_callback")
        if _negative(s):
            return _retry_or_end(p, "not_interested")
        if s.right_person is None and s.good_time is not True and not p.retries_used.get("opening"):
            p.retries_used["opening"] = 1
            return p, Directive(action="confirm_person")
        p.stage = "discovery"
        return p, Directive(action="discover")

    def discovery(self, p: CallProgress) -> tuple[CallProgress, Directive]:
        if self.s.interest in ("positive", "unsure"):
            p.stage = "qualification"
            return self.next_slot(p)
        if _negative(self.s):
            return _retry_or_end(p, "not_interested")
        return p, Directive(action="discover")

    def qualification(self, p: CallProgress) -> tuple[CallProgress, Directive]:
        if self.s.interest == "negative":
            return _end(p, "disqualified")  # not taking new lines (Q-STOP-1)
        if self.s.resistance:
            slot = p.asked_slots[-1] if p.asked_slots else None
            return _retry_or_end(p, "not_interested", slot=slot)
        return self.next_slot(p)

    def next_slot(self, p: CallProgress) -> tuple[CallProgress, Directive]:
        """Stop rules first, then the next slot in order, then the pitch."""
        territory = p.slots.get("territory")
        if territory is not None and not territory_fits(territory, self.brand.target_regions):
            return _end(p, "disqualified")  # outside the brand's regions (Q-STOP-1)
        stage = _stage(self.playbook, "qualification")
        order = stage.required_slots if p.in_rush else stage.slots
        for slot in order:
            if slot not in p.asked_slots:
                p.asked_slots.append(slot)
                action: DirectiveAction = "confirm_slot" if slot in p.slots else "ask_slot"
                return p, Directive(action=action, slot=slot)
        p.stage = "pitch"
        return p, Directive(action="pitch", fact_ids=select_facts(self.brand, p.slots))

    def pitch(self, p: CallProgress) -> tuple[CallProgress, Directive]:
        if self.s.agrees_to_meeting is False or _negative(self.s):
            facts = select_facts(self.brand, p.slots)
            return _retry_or_end(p, "not_interested", fact_ids=facts)
        # A positive or neutral reply moves on: the meeting is offered, and can be declined.
        return _enter_handoff(p, self.s, self.brand, self.now, "propose_slot")

    def handoff(self, p: CallProgress) -> tuple[CallProgress, Directive]:
        s = self.s
        if s.proposed_time:
            slot = parse_time(s.proposed_time, self.now, self.brand.timezone)
            if slot is not None and is_bookable(slot, self.brand.timezone):
                p.proposed_slot = slot
                return _book(p)
            if slot is not None:
                p.partner_time = slot
            return _vague_time(p)
        if s.agrees_to_meeting is True or s.interest == "positive":
            return _book(p)  # our proposal is accepted
        if s.agrees_to_meeting is False or _negative(s):
            return _retry_or_end(p, "not_interested", proposed_slot=p.proposed_slot)
        return _vague_time(p)

    def closing(self, p: CallProgress) -> tuple[CallProgress, Directive]:
        if self.s.question or p.open_question:
            p.open_question = False
            return p, _answer(self.brand, self.s.question or "")
        return _end(p, p.closing_outcome or "incomplete", "close")


# --- helpers -------------------------------------------------------------------------


def _stage(playbook: CallPlaybook, name: str) -> PlaybookStage:
    return next(stage for stage in playbook.stages if stage.name == name)


def _negative(s: TurnSignals) -> bool:
    return s.resistance or s.interest == "negative"


def _end(
    p: CallProgress, outcome: CallOutcome, action: DirectiveAction = "exit_politely"
) -> tuple[CallProgress, Directive]:
    p.stage, p.outcome = "ended", outcome
    return p, Directive(action=action)


def _retry_or_end(
    p: CallProgress,
    outcome: CallOutcome,
    *,
    slot: str | None = None,
    fact_ids: list[str] | None = None,
    proposed_slot: datetime | None = None,
) -> tuple[CallProgress, Directive]:
    """Retry once per stage after resistance (U-RETRY-1); the second time, exit."""
    if p.retries_used.get(p.stage, 0) >= 1:
        return _end(p, outcome)
    p.retries_used[p.stage] = 1
    directive = Directive(
        action="retry_once", slot=slot, fact_ids=fact_ids or [], proposed_slot=proposed_slot
    )
    return p, directive


def _enter_handoff(
    p: CallProgress,
    s: TurnSignals,
    brand: BrandProfile,
    now: datetime,
    action: DirectiveAction,
) -> tuple[CallProgress, Directive]:
    p.stage = "handoff"
    p.proposed_slot = propose_slot(now, brand.timezone)
    if s.proposed_time:  # "not now, call me tomorrow at 3": book it in the same turn
        slot = parse_time(s.proposed_time, now, brand.timezone)
        if slot is not None and is_bookable(slot, brand.timezone):
            p.proposed_slot = slot
            return _book(p)
    return p, Directive(action=action, proposed_slot=p.proposed_slot)


def _book(p: CallProgress) -> tuple[CallProgress, Directive]:
    p.stage = "closing"
    p.closing_outcome = _BOOKED[p.booking]
    return p, Directive(action="confirm_booking", proposed_slot=p.proposed_slot)


def _vague_time(p: CallProgress) -> tuple[CallProgress, Directive]:
    """No usable time: ask once more; then a callback if the partner named a time, else stop."""
    if p.retries_used.get("handoff", 0) >= 1:
        if p.partner_time is not None:
            p.proposed_slot = p.partner_time
            return _end(p, "callback_booked")
        return _end(p, "incomplete")
    p.retries_used["handoff"] = 1
    return p, Directive(action="retry_once", proposed_slot=p.proposed_slot)


def _answer(brand: BrandProfile, question: str) -> Directive:
    """Answer with matching facts only; none means the founder answers it in the meeting."""
    return Directive(action="answer_question", fact_ids=matching_facts(brand, [question]))


def _stems(words: Iterable[str]) -> set[str]:
    """Lower case with a plural "s" dropped, so "samples" matches "sample"."""
    lowered = (w.lower() for w in words)
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in lowered}


def matching_facts(brand: BrandProfile, texts: list[str], names: tuple[str, ...] = ()) -> list[str]:
    """Fact ids whose tags match ``names`` (slot names) or words in ``texts``, best first."""
    words = _stems([w for text in texts for w in _WORD.findall(text.lower())]) | _stems(names)
    scored = []
    for index, fact in enumerate(brand.approved_facts):
        score = len(_stems(fact.tags) & words)
        if score:
            scored.append((-score, index, fact.id))
    return [fact_id for _, _, fact_id in sorted(scored)][:PITCH_FACTS]


def select_facts(brand: BrandProfile, slots: dict[str, str]) -> list[str]:
    """2-3 facts for the pitch: tags matching the filled slots and their values, best first,
    padded with the first facts in profile order (HLD §6.11)."""
    chosen = matching_facts(brand, list(slots.values()), tuple(slots))
    for fact in brand.approved_facts:
        if len(chosen) >= MIN_PITCH_FACTS:
            break
        if fact.id not in chosen:
            chosen.append(fact.id)
    return chosen


def territory_fits(territory: str, regions: list[str]) -> bool:
    """The partner sells in at least one of the brand's regions. No regions means anywhere.

    Regions are matched as written ("Northern California" fits "California"); the classifier
    is asked to spell regions out, so abbreviations are not guessed.
    """
    if not regions:
        return True
    text = territory.lower()
    if any(word in text for word in _EVERYWHERE):
        return True
    return any(region.lower() in text for region in regions)
