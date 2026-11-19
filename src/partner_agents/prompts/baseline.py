"""The single-prompt baseline caller for eval ablation A8 (EVAL §7, ``CALL_CONTROL=prompt_only``).

One free-form prompt holds the whole playbook, every approved fact and the transcript, and the
model decides everything: what to say, which stage it is in and when the call ends. There is no
stage machine, no scheduler and no reply check, which is exactly what A8 takes away to see what
they are worth. Only the eval reads this module; the app never does. Partner turns still go
through ``wrap_untrusted`` (CLAUDE.md rule 11), so the comparison is about control, not about
safety plumbing. Bump ``PROMPT_VERSION`` on any wording change.
"""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from partner_agents.prompts.caller import (
    _DATA_RULE,
    SLOT_TOPICS,
    _rule_lines,
    _transcript,
    brand_fields,
)
from partner_agents.schemas import BrandProfile, CallOutcome, CallPlaybook, CallTurn

PROMPT_VERSION = "baseline-v1"
BASELINE_TURNS = 12  # transcript lines the baseline sees (the stage machine's caller sees 6)

_OUTCOMES = (
    "meeting_booked (the partner agreed to a specific day and time with the founder), "
    "callback_booked (agreed to a call back at a specific time), not_interested, disqualified "
    "(outside the brand's regions or categories), opted_out (asked not to be called again), "
    "wrong_person, escalated_to_human (wants the founder or a person), incomplete"
)


class BaselineReply(BaseModel):  # the baseline's output: its next line and, at the end, the outcome
    text: str
    outcome: CallOutcome | None = None  # set only on the last line; the call ends after it


def _playbook_text(playbook: CallPlaybook, brand: BrandProfile) -> str:
    fields = brand_fields(brand)
    parts = [f"Rules for every turn:\n{_rule_lines(playbook.universal_rules)}"]
    for stage in playbook.stages:
        slots = f" Details to learn, one per turn: {', '.join(stage.slots)}." if stage.slots else ""
        parts.append(
            f"Stage {stage.name}. Goal: {stage.goal.format(**fields)}{slots}\n"
            f"{_rule_lines(stage.rules)}"
        )
    return "\n\n".join(parts)


def build_baseline_messages(
    playbook: CallPlaybook,
    brand: BrandProfile,
    turns: Sequence[CallTurn],
    *,
    max_words: int,
) -> list[BaseMessage]:
    facts = "\n".join(f"- [{fact.id}] {fact.text}" for fact in brand.approved_facts)
    topics = "\n".join(f"- {name}: {topic}" for name, topic in SLOT_TOPICS.items())
    system = (
        f"You are an AI assistant on a phone call, calling on behalf of {brand.brand_name}, a "
        f"brand of {brand.category}. You speak for {brand.founder_name}, the founder, and your "
        "aim is a short meeting between the partner and the founder when there is a fit. You "
        "never pretend to be human.\n\n"
        f"{_DATA_RULE}\n\n"
        f"Follow this playbook. Move through its stages in order yourself:\n\n"
        f"{_playbook_text(playbook, brand)}\n\n"
        f"Details you may learn about a partner:\n{topics}\n\n"
        f"Facts you may state (nothing else):\n{facts}\n\n"
        f"The brand's region and meeting times: the founder meets partners on business days "
        f"between 10:00 and 17:00 {brand.timezone} time, at least 24 hours ahead.\n\n"
        f"Reply with `text`, what you say next (plain spoken English, at most {max_words} "
        "words, no lists, at most one question), and `outcome`. Leave `outcome` empty while "
        f"the call goes on. Set it on your last line, after which the call ends: {_OUTCOMES}. "
        f'When you end the call, sign off with: "{brand.sign_off}".'
    )
    if turns:
        user = f"The call so far:\n{_transcript(turns, BASELINE_TURNS)}\n\nWrite your next line."
    else:
        user = "The call is starting. Write your opening line."
    return [SystemMessage(content=system), HumanMessage(content=user)]
