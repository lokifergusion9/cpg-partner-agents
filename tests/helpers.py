"""Shared test doubles for the LLM layer."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from partner_agents.fakes import ScriptedChatModel


class RateLimited(Exception):
    status_code = 429


class ServerError(Exception):
    status_code = 500


class ByPageModel(ScriptedChatModel):
    """Answers by page: the reply is taken from the queue of the URL named in the prompt."""

    pages: dict[str, list[str | Exception]] = Field(default_factory=dict)
    prompts: list[str] = Field(default_factory=list)

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        prompt = "\n".join(m.text for m in messages)
        self.prompts.append(prompt)
        for url, queue in self.pages.items():
            if f'source="{url}"' in prompt and queue:
                reply = queue.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])
        raise AssertionError("no scripted reply for this prompt")


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
