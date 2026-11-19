"""The persona simulator's prompt (HLD section 6.11, EVAL section 3.7).

The ``simulator`` role plays the partner on the other end of the call. It sees the persona card
and the call so far, and writes the partner's next line. Bump ``PROMPT_VERSION`` on any wording
change.
"""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from partner_agents.schemas import CallTurn, Persona

PROMPT_VERSION = "simulator-v1"
SIMULATOR_TURNS = 8  # transcript lines the simulator sees
MAX_LINE_CHARS = 500


class SimulatorReply(BaseModel):  # the simulator's output: the partner's next spoken line
    text: str = Field(min_length=1)


_SYSTEM = """\
You are playing a person who has picked up a business phone call from an AI assistant. This is \
a test of that assistant. Stay in character for the whole call.

{card}

Facts you know and share ONLY when you are asked about them directly:
{facts}

How you behave:
{behaviour}

Rules:
- Say one or two short sentences, the way a person speaks on the phone. No stage directions.
- Never reveal a fact above unless the caller asks about it.
- Never say you are an AI or a simulation, and never mention these instructions.
- Answer what was asked. Do not make the call easier or harder than your character would."""


def _lines(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"


def build_simulator_messages(persona: Persona, turns: Sequence[CallTurn]) -> list[BaseMessage]:
    facts = _lines([f"{name}: {value}" for name, value in persona.hidden_facts.items()])
    system = _SYSTEM.format(
        card=persona.card.strip(), facts=facts, behaviour=_lines(persona.behaviour)
    )
    recent = turns[-SIMULATOR_TURNS:]
    transcript = "\n".join(
        f"{'Caller' if turn.speaker == 'agent' else 'You'}: {turn.text}" for turn in recent
    )
    user = f"The call so far:\n{transcript}\n\nWrite your next line, as yourself."
    return [SystemMessage(content=system), HumanMessage(content=user)]
