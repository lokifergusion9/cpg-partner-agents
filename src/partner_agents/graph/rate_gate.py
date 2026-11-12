"""``rate_gate`` (HLD §5.2, §7): the discovery limit, checked before any search is paid for.

The gate sits between the router and ``scout_search``, so only discovery turns are counted. A
refused turn ends with a reply and ``retry_after_s`` in the state, which the API turns into a
429. Without a limiter (the CLI, most tests) the gate lets everything through.
"""

import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from langchain_core.messages import AIMessage
from langgraph.graph import END

from partner_agents.graph.state import AgentState
from partner_agents.observability import NodeUpdate, timed_node
from partner_agents.schemas import StageError


class RunLimiter(Protocol):
    capacity: int

    def take(self, keys: Sequence[str]) -> int | None:
        """``None`` if allowed (a token was taken), else the seconds to wait."""
        ...


def next_after_gate(state: AgentState) -> str:
    return END if state.get("retry_after_s") else "scout_search"


def make_rate_gate(limiter: RunLimiter | None) -> Callable[[AgentState], Awaitable[NodeUpdate]]:
    @timed_node("rate_gate")
    async def rate_gate(state: AgentState) -> NodeUpdate:
        if limiter is None:
            return {}
        keys = state.get("rate_keys") or [f"user:{state.get('user_id', '')}"]
        wait = limiter.take(keys)
        if wait is None:
            return {}
        minutes = max(1, math.ceil(wait / 60))
        reply = (
            f"You have used your {limiter.capacity} partner searches for this hour. "
            f"Please try again in about {minutes} minute{'s' if minutes != 1 else ''}."
        )
        error = StageError(stage="rate_gate", code="rate_limited", message=f"retry in {wait}s")
        return {"retry_after_s": wait, "messages": [AIMessage(content=reply)], "errors": [error]}

    return rate_gate
