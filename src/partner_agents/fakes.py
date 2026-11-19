"""A scripted chat model for offline tests and ``LLM_PROVIDER=fake`` (CLAUDE.md rule 4).

``ScriptedChatModel`` replays a fixed list of replies and records every conversation it
receives. Its ``with_structured_output(..., include_raw=True)`` mirrors the real providers:
it returns ``{"raw", "parsed", "parsing_error"}`` and does not raise on invalid output.
"""

import hashlib
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
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


class HashingEmbedder:
    """A deterministic bag-of-words embedder: texts sharing words get similar vectors.

    No model download, so vector-store tests stay offline. Satisfies the ``Embedder`` protocol.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            vector[0], norm = 1.0, 1.0
        return [x / norm for x in vector]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class RecordingHandler(BaseCallbackHandler):
    """A LangChain callback handler that remembers the names of the chains (graph nodes) run."""

    def __init__(self) -> None:
        self.chains: list[str] = []

    def on_chain_start(self, serialized: dict[str, Any] | None, inputs: Any, **kwargs: Any) -> None:
        if kwargs.get("name"):
            self.chains.append(str(kwargs["name"]))


class FakeRunTrace:
    def __init__(self, url: str | None, metadata: dict[str, Any]) -> None:
        self._url, self._metadata = url, metadata
        self.handler = RecordingHandler()  # stands in for Langfuse's callback handler
        self.finished = False

    @property
    def callbacks(self) -> list[Any]:
        return [self.handler]

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def finish(self) -> str | None:
        self.finished = True
        return self._url


class FakeTracing:
    """Offline stand-in for ``tracing.LangfuseTracing``: records each trace it starts."""

    def __init__(self, *, fail_start: bool = False, fail_finish: bool = False) -> None:
        self.runs: list[FakeRunTrace] = []
        self.fail_start, self.fail_finish = fail_start, fail_finish
        self.shut_down = False

    def start(
        self, *, name: str, session_id: str, user_id: str, metadata: Mapping[str, Any]
    ) -> FakeRunTrace:
        if self.fail_start:
            raise RuntimeError("tracing is down")
        url = None if self.fail_finish else f"https://trace.example/{name}/{len(self.runs) + 1}"
        trace = FakeRunTrace(
            url, {"langfuse_session_id": session_id, "langfuse_user_id": user_id, **metadata}
        )
        trace.metadata["name"] = name
        self.runs.append(trace)
        return trace

    def shutdown(self) -> None:
        self.shut_down = True


class FakeSpeech:
    """Offline stand-in for ``calls.speech.LocalSpeech``: scripted transcripts, silent audio."""

    def __init__(self, transcripts: Sequence[str] = ()) -> None:
        self._transcripts = iter(transcripts)
        self.heard: list[int] = []  # the number of samples of each recording
        self.spoken: list[str] = []

    def transcribe(self, samples: npt.NDArray[np.float32]) -> str:
        self.heard.append(int(samples.size))
        return next(self._transcripts, "")

    def synthesize(self, text: str) -> tuple[int, npt.NDArray[Any]]:
        self.spoken.append(text)
        return 22_050, np.zeros(2_205, dtype=np.int16)


def _parsed_or_raise(result: dict[str, Any]) -> Any:
    if result["parsing_error"] is not None:
        raise result["parsing_error"]
    return result["parsed"]


def _forever(message: AIMessage) -> Iterator[AIMessage]:
    while True:
        yield message.model_copy()
