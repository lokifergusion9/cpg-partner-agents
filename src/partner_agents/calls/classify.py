"""Classify one partner turn into ``TurnSignals`` (HLD §6.11, node ``call_classify``).

The partner's words are untrusted (CLAUDE.md rule 11). They reach the prompt only through
``wrap_untrusted(kind="call_partner")`` and can only ever produce ``TurnSignals``: there is
no tool the classifier could call.
"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel

from partner_agents.llm import invoke_structured
from partner_agents.prompts.caller import PROMPT_VERSION, build_classifier_messages
from partner_agents.schemas import CallStage, CallTurn, TurnSignals


async def classify_turn(
    model: BaseChatModel,
    stage: CallStage,
    turns: Sequence[CallTurn],
    partner_text: str,
    *,
    asked_slot: str | None = None,
) -> TurnSignals:
    """Raises ``StructuredOutputError`` when the model's reply is invalid twice."""
    messages = build_classifier_messages(stage, turns, partner_text, asked_slot)
    return await invoke_structured(
        model, TurnSignals, messages, role="classifier", prompt_version=PROMPT_VERSION
    )
