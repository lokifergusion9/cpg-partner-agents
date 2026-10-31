"""Shared test doubles for the LLM layer."""

from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from partner_agents.fakes import ScriptedChatModel


class RateLimited(Exception):
    status_code = 429


class ServerError(Exception):
    status_code = 500


class FaultyModel(ScriptedChatModel):
    """Raises the queued errors, one per transport call, before replaying its script."""

    faults: list[BaseException] = Field(default_factory=list)
    transport_calls: int = 0

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, Any]:
        real = super().with_structured_output(schema, include_raw=include_raw, **kwargs)

        async def call(messages: Any) -> Any:
            self.transport_calls += 1
            if self.faults:
                raise self.faults.pop(0)
            return await real.ainvoke(messages)

        return RunnableLambda(call)
