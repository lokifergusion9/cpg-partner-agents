import asyncio
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage

from partner_agents import llm
from partner_agents.config import Role
from partner_agents.fakes import ScriptedChatModel
from partner_agents.schemas import DiscoveryIntent, RouteDecision
from partner_agents.utils.retry import RetryPolicy
from tests.helpers import FaultyModel, RateLimited

VALID = DiscoveryIntent(
    partner_type="broker", product_category="cookies", raw_query="cookie brokers in CA"
)
PROMPT = [HumanMessage(content="Extract the intent")]


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def ask(model: ScriptedChatModel, role: Role = "router") -> DiscoveryIntent:
    return run(llm.invoke_structured(model, DiscoveryIntent, PROMPT, role=role))  # type: ignore[no-any-return]


def with_tokens(reply: str, total: int) -> AIMessage:
    return AIMessage(
        content=reply,
        usage_metadata={"input_tokens": total - 5, "output_tokens": 5, "total_tokens": total},
    )


# --- validate and retry -------------------------------------------------------------


def test_valid_reply_succeeds_on_the_first_call() -> None:
    model = ScriptedChatModel.scripted(VALID)
    assert ask(model) == VALID
    assert len(model.received) == 1


def test_invalid_then_valid_succeeds_after_one_retry() -> None:
    model = ScriptedChatModel.scripted('{"partner_type": "influencer"}', VALID)
    assert ask(model) == VALID
    assert len(model.received) == 2


def test_the_retry_carries_the_bad_reply_and_the_validation_error() -> None:
    model = ScriptedChatModel.scripted('{"partner_type": "influencer"}', VALID)
    ask(model)
    first, second = model.received
    assert first == PROMPT
    assert second[: len(PROMPT)] == PROMPT
    bad_reply, note = second[len(PROMPT) :]
    assert bad_reply.text == '{"partner_type": "influencer"}'
    assert isinstance(note, HumanMessage)
    assert "partner_type" in note.text
    assert "product_category: Field required" in note.text
    assert "influencer" not in note.text  # the error names fields, it does not echo inputs


def test_invalid_twice_raises_a_typed_error() -> None:
    model = ScriptedChatModel.scripted("not json", '{"partner_type": "broker"}', VALID)
    with pytest.raises(llm.StructuredOutputError) as excinfo:
        ask(model)
    assert excinfo.value.schema == "DiscoveryIntent"
    assert "product_category" in excinfo.value.detail
    assert len(model.received) == 2  # exactly one retry, the third reply is never requested


def test_a_non_json_reply_is_retried_too() -> None:
    model = ScriptedChatModel.scripted("Sure! Here you go.", VALID)
    assert ask(model) == VALID


def test_a_parser_exception_from_the_provider_counts_as_invalid() -> None:
    model = FaultyModel.scripted(VALID)
    model.faults.append(OutputParserException("bad json"))
    assert ask(model) == VALID
    assert model.transport_calls == 2
    assert "bad json" in model.received[0][-1].text  # the error reached the retry prompt


def test_a_complex_schema_round_trips() -> None:
    decision = RouteDecision(intent="discovery", confidence=0.9, discovery=VALID, reason="r")
    model = ScriptedChatModel.scripted(decision)
    assert run(llm.invoke_structured(model, RouteDecision, PROMPT, role="router")) == decision


# --- tokens -------------------------------------------------------------------------


def test_token_counts_are_captured_per_role() -> None:
    model = ScriptedChatModel.scripted(with_tokens(VALID.model_dump_json(), 100))
    with llm.track_usage() as usage:
        ask(model, "router")
    assert usage == {"router": 100}


def test_tokens_of_a_failed_attempt_are_counted_too() -> None:
    model = ScriptedChatModel.scripted(
        with_tokens("not json", 40), with_tokens(VALID.model_dump_json(), 60)
    )
    with llm.track_usage() as usage:
        ask(model, "extractor")
    assert usage == {"extractor": 100}


def test_tokens_accumulate_across_roles_and_calls() -> None:
    with llm.track_usage() as usage:
        ask(ScriptedChatModel.scripted(with_tokens(VALID.model_dump_json(), 10)), "router")
        ask(ScriptedChatModel.scripted(with_tokens(VALID.model_dump_json(), 20)), "router")
        ask(ScriptedChatModel.scripted(with_tokens(VALID.model_dump_json(), 7)), "writer")
    assert usage == {"router": 30, "writer": 7}


def test_input_plus_output_is_used_when_the_total_is_missing() -> None:
    reply = AIMessage(
        content=VALID.model_dump_json(),
        usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 0},
    )
    with llm.track_usage() as usage:
        ask(ScriptedChatModel.scripted(reply))
    assert usage == {"router": 15}


def test_replies_without_usage_metadata_record_nothing() -> None:
    with llm.track_usage() as usage:
        ask(ScriptedChatModel.scripted(VALID))
    assert usage == {}


def test_usage_is_shared_with_tasks_started_inside_the_block() -> None:
    async def main() -> dict[str, int]:
        with llm.track_usage() as usage:
            await asyncio.gather(
                *(
                    llm.invoke_structured(
                        ScriptedChatModel.scripted(with_tokens(VALID.model_dump_json(), 10)),
                        DiscoveryIntent,
                        PROMPT,
                        role="extractor",
                    )
                    for _ in range(3)
                )
            )
        return usage

    assert run(main()) == {"extractor": 30}


def test_no_tracker_means_no_error() -> None:
    ask(ScriptedChatModel.scripted(with_tokens(VALID.model_dump_json(), 10)))


# --- transport errors are not validation errors -------------------------------------


@pytest.fixture
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "RETRY_POLICY", RetryPolicy(base_delay_s=0))


def test_a_rate_limit_is_retried_by_the_retry_utility(instant_retries: None) -> None:
    model = FaultyModel.scripted(VALID)
    model.faults.append(RateLimited())
    assert ask(model) == VALID
    assert model.transport_calls == 2
    assert len(model.received) == 1  # the model saw one conversation: no validation note


def test_a_rate_limit_never_counts_as_the_validation_retry(instant_retries: None) -> None:
    model = FaultyModel.scripted("not json", VALID)
    model.faults.append(RateLimited())
    assert ask(model) == VALID  # one transport retry plus the one validation retry


def test_a_non_retryable_error_propagates_unchanged() -> None:
    model = FaultyModel.scripted(VALID)
    model.faults.append(KeyError("boom"))
    with pytest.raises(KeyError):
        ask(model)
    assert model.transport_calls == 1
