"""Langfuse tracing (T-33, ADR-029): the protocol, the real class over stub modules, the graph."""

import asyncio
import sys
import types
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient

from partner_agents.api.main import create_app
from partner_agents.config import Settings
from partner_agents.fakes import FakeTracing, ScriptedChatModel
from partner_agents.graph.build import PROMPT_VERSIONS, build_graph, resume_turn, run_turn
from partner_agents.graph.call import build_call_graph, run_call_turn
from partner_agents.schemas import ApprovalDecision, TurnSignals
from partner_agents.tracing import (
    LangfuseTracing,
    TracingUnavailable,
    build_tracing,
    finish_trace,
    start_trace,
    trace_metadata,
    traced_config,
)
from tests.graph.test_build import (  # noqa: F401  (fixtures are used by name)
    client,
    decision,
    instant_retries,
    make_deps,
    seeded_approval,
)
from tests.graph.test_call import setup as call_setup


def keys(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "langfuse_public_key": SecretStr("pk-test"),
        "langfuse_secret_key": SecretStr("sk-test"),
    }
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[call-arg]


# --- configuration ------------------------------------------------------------------


def test_tracing_is_off_without_both_keys() -> None:
    assert build_tracing(Settings(_env_file=None)) is None  # type: ignore[call-arg]
    one = Settings(_env_file=None, langfuse_public_key=SecretStr("pk"))  # type: ignore[call-arg]
    assert build_tracing(one) is None


def test_keys_without_the_extra_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langfuse", None)  # `import langfuse` now fails
    with pytest.raises(TracingUnavailable, match="--extra tracing"):
        build_tracing(keys())


def test_trace_metadata_carries_the_keys_the_handler_reads() -> None:
    metadata = trace_metadata("chat_turn", "t_1", "u_1", {"run_id": "r_1"})
    assert metadata == {
        "langfuse_session_id": "t_1",
        "langfuse_user_id": "u_1",
        "langfuse_trace_name": "chat_turn",
        "langfuse_tags": ["cpg-partner-agents"],
        "run_id": "r_1",
    }


def test_traced_config_adds_the_callbacks_and_keeps_what_was_there() -> None:
    trace = FakeTracing().start(name="n", session_id="s", user_id="u", metadata={})
    base = {"configurable": {"thread_id": "t"}, "callbacks": ["existing"], "metadata": {"a": 1}}
    merged = traced_config(base, trace)
    assert merged["callbacks"] == ["existing", trace.handler]
    assert merged["metadata"]["a"] == 1 and merged["metadata"]["langfuse_session_id"] == "s"
    assert merged["configurable"] == {"thread_id": "t"} and base["callbacks"] == ["existing"]
    assert traced_config(base, None) == base


# --- the real class over stub modules -----------------------------------------------


class StubClient:
    instances: ClassVar[list["StubClient"]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.shut_down = False
        StubClient.instances.append(self)

    def get_trace_url(self, *, trace_id: str) -> str | None:
        return f"https://cloud.example/project/p1/traces/{trace_id}" if trace_id != "gone" else None

    def shutdown(self) -> None:
        self.shut_down = True


class StubHandler:
    last_trace_id: str | None = "abc123"


@pytest.fixture
def stub_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    StubClient.instances = []
    package = types.ModuleType("langfuse")
    package.Langfuse = StubClient  # type: ignore[attr-defined]
    langchain = types.ModuleType("langfuse.langchain")
    langchain.CallbackHandler = StubHandler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", package)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", langchain)


@pytest.mark.usefixtures("stub_langfuse")
def test_the_real_class_builds_a_client_a_handler_per_run_and_the_trace_url() -> None:
    tracing = build_tracing(keys(langfuse_host="https://lf.example"))
    assert isinstance(tracing, LangfuseTracing)
    assert StubClient.instances[0].kwargs == {
        "public_key": "pk-test",
        "secret_key": "sk-test",
        "base_url": "https://lf.example",
    }
    first = tracing.start(name="chat_turn", session_id="t_1", user_id="u_1", metadata={"k": 1})
    second = tracing.start(name="chat_turn", session_id="t_1", user_id="u_1", metadata={})
    assert first.callbacks[0] is not second.callbacks[0]  # a handler for each run
    assert first.metadata["langfuse_session_id"] == "t_1" and first.metadata["k"] == 1
    assert first.finish() == "https://cloud.example/project/p1/traces/abc123"
    tracing.shutdown()
    assert StubClient.instances[0].shut_down


@pytest.mark.usefixtures("stub_langfuse")
def test_without_a_host_the_default_cloud_is_used_and_a_missing_trace_id_gives_no_url() -> None:
    tracing = build_tracing(keys())
    assert tracing is not None and "base_url" not in StubClient.instances[0].kwargs
    trace = tracing.start(name="n", session_id="s", user_id="u", metadata={})
    trace.callbacks[0].last_trace_id = None
    assert trace.finish() is None
    trace.callbacks[0].last_trace_id = "gone"
    assert trace.finish() is None


def test_start_and_finish_never_raise() -> None:
    assert start_trace(None, name="n", session_id="s", user_id="u", metadata={}) is None
    broken = FakeTracing(fail_start=True)
    assert start_trace(broken, name="n", session_id="s", user_id="u", metadata={}) is None
    assert asyncio.run(finish_trace(None)) is None

    class Boom:
        def __init__(self) -> None:
            self.callbacks: list[Any] = []
            self.metadata: dict[str, Any] = {}

        def finish(self) -> str | None:
            raise RuntimeError("network down")

    assert asyncio.run(finish_trace(Boom())) is None


# --- in the graph -------------------------------------------------------------------


def test_a_turn_has_a_trace_with_every_node_the_url_and_the_prompt_versions(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    deps = make_deps(client, decision("cpg_qa"))
    deps.tracing = FakeTracing()
    graph = build_graph(deps)
    result = asyncio.run(run_turn(graph, deps, thread_id="t_1", user_id="u_1", text="slotting?"))
    [trace] = deps.tracing.runs
    assert result.run.trace_url == "https://trace.example/chat_turn/1" and trace.finished
    assert {"charlie_route", "kb_answer"} <= set(trace.handler.chains)  # a span for each node
    assert trace.metadata["langfuse_session_id"] == "t_1"
    assert trace.metadata["run_id"] == result.run.run_id
    assert trace.metadata["prompt_versions"] == PROMPT_VERSIONS
    assert {"charlie", "scout", "chase"} <= set(PROMPT_VERSIONS)
    saved = asyncio.run(deps.repo.get_run(result.run.run_id))
    assert saved is not None and saved.trace_url == result.run.trace_url


def test_a_turn_without_tracing_has_no_url(client: AsyncQdrantClient) -> None:  # noqa: F811
    deps = make_deps(client, decision("cpg_qa"))
    graph = build_graph(deps)
    result = asyncio.run(run_turn(graph, deps, thread_id="t_1", user_id="u_1", text="slotting?"))
    assert result.run.trace_url is None


@pytest.mark.parametrize("failure", [{"fail_start": True}, {"fail_finish": True}])
def test_a_tracing_failure_never_fails_the_run(
    client: AsyncQdrantClient,  # noqa: F811
    failure: dict[str, bool],
) -> None:
    deps = make_deps(client, decision("cpg_qa"))
    deps.tracing = FakeTracing(**failure)
    graph = build_graph(deps)
    result = asyncio.run(run_turn(graph, deps, thread_id="t_1", user_id="u_1", text="slotting?"))
    assert result.run.trace_url is None and result.reply


def test_a_resumed_approval_keeps_the_first_trace_url(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    deps, graph, _ = seeded_approval(client)
    deps.tracing = FakeTracing()

    async def go() -> Any:
        paused = await run_turn(graph, deps, thread_id="t_1", user_id="u_1", text="email top 3")
        ids = [d["id"] for d in paused.pending_approval["drafts"]]
        done = await resume_turn(
            graph, deps, thread_id="t_1", user_id="u_1", decision=ApprovalDecision(reject=ids)
        )
        return paused, done

    paused, done = asyncio.run(go())
    assert [t.metadata["name"] for t in deps.tracing.runs] == ["chat_turn", "approval_resume"]
    assert paused.run.trace_url == "https://trace.example/chat_turn/1"
    assert done.run.trace_url == paused.run.trace_url  # the first trace stays the run's link
    assert "approval_gate" in deps.tracing.runs[1].handler.chains


def test_a_call_turn_is_traced_with_the_call_id_as_the_session() -> None:
    deps, _, call_id = call_setup([TurnSignals(right_person=True)])
    deps.tracing = FakeTracing()
    graph = build_call_graph(deps)
    asyncio.run(run_call_turn(graph, deps, user_id="u_1", call_id=call_id, partner_text=None))
    [trace] = deps.tracing.runs
    assert (
        trace.metadata["langfuse_session_id"] == call_id and trace.metadata["name"] == "call_turn"
    )
    assert {"call_advance", "call_reply", "call_persist"} <= set(trace.handler.chains)
    assert trace.metadata["prompt_versions"]["caller"].startswith("caller-")


def test_the_runs_endpoint_returns_the_trace_url_and_the_token_counts(
    client: AsyncQdrantClient,  # noqa: F811
) -> None:
    deps = make_deps(client, decision("cpg_qa"))
    deps.tracing = FakeTracing()
    deps.router = ScriptedChatModel.scripted(decision("cpg_qa"))
    with TestClient(create_app(deps, with_ui=False)) as http:
        chat = http.post("/api/chat", json={"user_id": "u_1", "message": "what is a slotting fee?"})
        run = http.get(f"/api/runs/{chat.json()['run_id']}", params={"user_id": "u_1"}).json()
    assert run["trace_url"] == "https://trace.example/chat_turn/1"
    assert isinstance(run["tokens"], dict) and "charlie_route" in run["timings_ms"]
