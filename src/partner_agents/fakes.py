"""A scripted chat model for offline tests and ``LLM_PROVIDER=fake`` (CLAUDE.md rule 4).

``ScriptedChatModel`` replays a fixed list of replies and records every conversation it
receives. Its ``with_structured_output(..., include_raw=True)`` mirrors the real providers:
it returns ``{"raw", "parsed", "parsing_error"}`` and does not raise on invalid output.
"""

from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field, ValidationError

Reply = str | AIMessage | BaseModel


def _as_message(reply: Reply) -> AIMessage:
    if isinstance(reply, AIMessage):
        return reply
    if isinstance(reply, BaseModel):
        return AIMessage(content=reply.model_dump_json())
    return AIMessage(content=reply)


class ScriptedChatModel(GenericFakeChatModel):
    """Replies in order. A ``BaseModel`` reply is sent as its JSON."""

    received: list[list[BaseMessage]] = Field(default_factory=list)

    @classmethod
    def scripted(cls, *replies: Reply) -> "ScriptedChatModel":
        return cls(messages=iter([_as_message(reply) for reply in replies]))

    @classmethod
    def repeating(cls, reply: Reply) -> "ScriptedChatModel":
        return cls(messages=_forever(_as_message(reply)))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except StopIteration:
            # A StopIteration raised inside a worker thread would surface as a confusing error.
            raise RuntimeError("ScriptedChatModel ran out of scripted replies") from None

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[LanguageModelInput, Any]:
        def parse(raw: AIMessage) -> dict[str, Any]:
            try:
                return {
                    "raw": raw,
                    "parsed": schema.model_validate_json(raw.text),
                    "parsing_error": None,
                }
            except ValidationError as exc:
                return {"raw": raw, "parsed": None, "parsing_error": exc}

        chain: Runnable[LanguageModelInput, Any] = self | RunnableLambda(parse)
        if include_raw:
            return chain
        return chain | RunnableLambda(_parsed_or_raise)


def _parsed_or_raise(result: dict[str, Any]) -> Any:
    if result["parsing_error"] is not None:
        raise result["parsing_error"]
    return result["parsed"]


def _forever(message: AIMessage) -> Iterator[AIMessage]:
    while True:
        yield message.model_copy()
