"""Write the agent's next line and check it (HLD §6.11, nodes ``call_reply`` + ``call_check``).

The caller model writes a line for the directive; ``calls/checks.py`` checks it. A line that
fails is regenerated once with the failed check ids; if that fails too (or the model errors),
the stage's fallback line from the playbook is used instead. The fallback is checked as well,
so the record shows honestly whether it passed.
"""

import logging
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from partner_agents.calls.checks import CheckContext, failures, run_checks
from partner_agents.llm import invoke_structured
from partner_agents.prompts.caller import (
    ACTION_FALLBACKS,
    OPT_OUT_LINE,
    PROMPT_VERSION,
    CallerReply,
    brand_fields,
    build_caller_messages,
)
from partner_agents.schemas import CallPlaybook, CallTurn, RuleCheck

logger = logging.getLogger(__name__)


class CheckedReply(BaseModel):
    text: str
    checks: list[RuleCheck]
    regenerated: bool = False
    used_fallback: bool = False


async def write_reply(
    model: BaseChatModel,
    playbook: CallPlaybook,
    ctx: CheckContext,
    turns: Sequence[CallTurn],
) -> CheckedReply:
    def messages(failed: Sequence[str] = ()) -> list[BaseMessage]:
        return build_caller_messages(
            playbook,
            ctx.stage,
            ctx.directive,
            ctx.brand,
            ctx.facts,
            ctx.slots,
            turns,
            max_words=ctx.max_words,
            opted_out=ctx.opted_out,
            failed=failed,
        )

    first = await _attempt(model, messages())
    if first is not None:
        checks = run_checks(first, ctx)
        if not failures(checks):
            return CheckedReply(text=first, checks=checks)
        failed = [f"{c.rule_id}: {c.detail}" for c in failures(checks)]
    else:
        failed = ["the previous attempt gave no usable line"]

    second = await _attempt(model, messages(failed))
    if second is not None:
        checks = run_checks(second, ctx)
        if not failures(checks):
            return CheckedReply(text=second, checks=checks, regenerated=True)

    line = fallback_line(playbook, ctx)
    return CheckedReply(
        text=line, checks=run_checks(line, ctx), regenerated=True, used_fallback=True
    )


def fallback_line(playbook: CallPlaybook, ctx: CheckContext) -> str:
    """The safe line for this turn: opt-out, an action-specific line, or the stage's line."""
    if ctx.opted_out:
        return OPT_OUT_LINE
    fields = brand_fields(ctx.brand)
    if ctx.directive.action in ACTION_FALLBACKS:
        return ACTION_FALLBACKS[ctx.directive.action].format(**fields)
    name = "closing" if ctx.stage == "ended" else ctx.stage
    stage = next(s for s in playbook.stages if s.name == name)
    return stage.fallback_line.format(**fields)


async def _attempt(model: BaseChatModel, messages: list[BaseMessage]) -> str | None:
    try:
        reply = await invoke_structured(
            model, CallerReply, messages, role="caller", prompt_version=PROMPT_VERSION
        )
    except Exception as exc:  # a model error must not end the call: the fallback covers it
        logger.warning("caller reply failed", extra={"error_type": type(exc).__name__})
        return None
    text = " ".join(reply.text.split())
    return text or None
