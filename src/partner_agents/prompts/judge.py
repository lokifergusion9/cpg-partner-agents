"""The call judge's prompt (EVAL §4.7, ADR-020).

The ``judge`` role reads a whole call and grades what code cannot: the playbook's soft rules
(``check = "judge"``), whether the partner really agreed to a specific time, and whether any
agent line states a fact that is not among the approved facts. It never decides a call's
outcome or flow. Transcript lines from the partner are untrusted (CLAUDE.md rule 11) and are
wrapped. Bump ``PROMPT_VERSION`` on any wording change.
"""

from collections.abc import Sequence
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from partner_agents.prompts.caller import _DATA_RULE
from partner_agents.schemas import BrandProfile, CallPlaybook, PlaybookRule
from partner_agents.web.safety import wrap_untrusted

PROMPT_VERSION = "judge-v1"

Verdict = Literal["pass", "fail", "na"]
Line = tuple[Literal["agent", "partner"], str]  # who spoke, and what they said


class RuleVerdict(BaseModel):
    rule_id: str
    verdict: Verdict  # "na": the call gave no occasion to apply the rule
    reason: str = Field(default="", max_length=300)


class CallJudgement(BaseModel):
    rules: list[RuleVerdict]
    partner_agreed_to_time: bool  # the partner accepted one specific day and time
    unsupported_claim_lines: list[int] = []  # numbers N of assistant lines "A<N>" with a claim
    # that is not among the approved facts and not something the partner said


def soft_rules(playbook: CallPlaybook) -> list[PlaybookRule]:
    """The rules the judge grades: every rule that code cannot check."""
    stage_rules = [rule for stage in playbook.stages for rule in stage.rules]
    return [rule for rule in [*playbook.universal_rules, *stage_rules] if rule.check == "judge"]


def numbered(lines: Sequence[Line]) -> str:
    """``A1: ...`` for the assistant, ``P1: ...`` for the partner (wrapped as untrusted data)."""
    agent = partner = 0
    out = []
    for who, text in lines:
        if who == "agent":
            agent += 1
            out.append(f"A{agent}: {text}")
        else:
            partner += 1
            out.append(f"P{partner}:\n" + wrap_untrusted(text, kind="call_partner"))
    return "\n".join(out)


def build_judge_messages(
    playbook: CallPlaybook, brand: BrandProfile, lines: Sequence[Line]
) -> list[BaseMessage]:
    rules = "\n".join(
        f"- [{rule.id}] {rule.kind.upper()}: {rule.text}" for rule in soft_rules(playbook)
    )
    facts = "\n".join(f"- [{fact.id}] {fact.text}" for fact in brand.approved_facts)
    system = (
        "You grade a recorded phone call between an AI assistant (A lines), which calls for the "
        f"brand {brand.brand_name}, and a possible business partner (P lines). You report; you "
        "never take part in the call.\n\n"
        f"{_DATA_RULE}\n\n"
        "Grade each rule below as pass, fail or na (na only when the call gave no occasion to "
        "apply it). A MUST rule fails if the assistant did not do it when it was called for; a "
        "NEVER rule fails if the assistant did it. Judge only what the transcript shows.\n"
        f"Rules:\n{rules}\n\n"
        "Also report partner_agreed_to_time: true only if the partner accepted one specific day "
        "and time (not 'later' or 'sometime').\n"
        "And report unsupported_claim_lines: the numbers N of assistant lines A<N> that state a "
        "fact about the brand or its products that is not in the approved facts below and was "
        "not said by the partner. Greetings, questions and the proposed meeting time are not "
        f"claims.\nApproved facts:\n{facts}"
    )
    user = f"The call:\n{numbered(lines)}"
    return [SystemMessage(content=system), HumanMessage(content=user)]
