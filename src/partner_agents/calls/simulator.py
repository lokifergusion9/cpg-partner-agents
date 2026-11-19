"""Run a whole call against a persona (HLD section 6.11, mode ``simulate``).

The ``simulator`` role plays the partner and the call sub-graph plays the agent, turn about,
until the call ends. The stage machine ends every call at ``CALL_MAX_TURNS`` as ``incomplete``,
so the loop is bounded even when the persona never agrees to anything.

The persona's words are untrusted like any partner turn (CLAUDE.md rule 11): they reach the
classifier only through the call sub-graph, which wraps them.
"""

from langchain_core.language_models import BaseChatModel

from partner_agents.graph.call import CallDeps, CallGraph, CallNotActive, run_call_turn
from partner_agents.llm import invoke_structured
from partner_agents.prompts.simulator import (
    MAX_LINE_CHARS,
    PROMPT_VERSION,
    SimulatorReply,
    build_simulator_messages,
)
from partner_agents.schemas import CallRecord, CallTurn, Persona


async def partner_line(model: BaseChatModel, persona: Persona, turns: list[CallTurn]) -> str:
    """The persona's next line. Raises ``StructuredOutputError`` when the model fails twice."""
    reply = await invoke_structured(
        model,
        SimulatorReply,
        build_simulator_messages(persona, turns),
        role="simulator",
        prompt_version=PROMPT_VERSION,
    )
    return reply.text.strip()[:MAX_LINE_CHARS] or "..."


async def simulate_call(
    graph: CallGraph,
    deps: CallDeps,
    simulator: BaseChatModel,
    persona: Persona,
    *,
    user_id: str,
    call_id: str,
) -> CallRecord:
    """Play ``persona`` against a call that has not started. Returns the finished record.

    Raises ``LookupError`` for an unknown call and ``CallNotActive`` when the call has already
    started. A simulator failure propagates; the turns played so far stay saved.
    """
    record = await deps.repo.get_call(user_id, call_id)
    if record is None:
        raise LookupError(call_id)
    if record.turns:
        raise CallNotActive("call_already_started", "This call has already started.")
    await deps.repo.save_call(
        record.model_copy(update={"mode": "simulate", "persona_id": persona.id})
    )
    record, _ = await run_call_turn(
        graph, deps, user_id=user_id, call_id=call_id, partner_text=None
    )
    for _ in range(deps.settings.call_max_turns + 1):  # a safety bound; the stage machine ends it
        if record.progress.outcome is not None:
            break
        text = await partner_line(simulator, persona, record.turns)
        record, _ = await run_call_turn(
            graph, deps, user_id=user_id, call_id=call_id, partner_text=text
        )
    return record
