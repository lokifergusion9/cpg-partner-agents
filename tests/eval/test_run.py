"""The eval runner end to end, offline (T-39): scripted "oracle" models answer from gold labels or
are fooled on purpose, pages are synthetic, and the numbers are checked by hand."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from eval.metrics import SuiteResult
from eval.run import Env
from eval.schemas import ExtractionRow, load_injection, load_kb, load_routing
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from eval import run as runner
from partner_agents import llm
from partner_agents.config import Role, Settings
from partner_agents.fakes import HashingEmbedder, ScriptedChatModel
from partner_agents.schemas import ExtractedLead, ExtractionResult, KBAnswer, RouteDecision
from partner_agents.utils.retry import RetryPolicy
from tests.helpers import ByPageModel


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "RETRY_POLICY", RetryPolicy(attempts=1, base_delay_s=0))


def make_env(
    models: dict[Role, BaseChatModel] | None = None,
    *,
    fixtures: Path = Path("."),
    split: Any = "test",
    **fields: object,
) -> Env:
    given = models or {}
    return Env(
        settings=Settings(_env_file=None, llm_provider="fake", **fields),  # type: ignore[call-arg]
        model=lambda role: given[role],
        embedder=HashingEmbedder(),
        split=split,
        count_tokens=runner.words,
        fixtures=fixtures,
    )


def result_of(coro: Any) -> SuiteResult:
    result: SuiteResult = asyncio.run(coro)
    return result


# --- routing ------------------------------------------------------------------------


class OracleRouter(ScriptedChatModel):
    """Replies with the prepared RouteDecision for the query in the last message."""

    answers: dict[str, str] = Field(default_factory=dict)

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        reply = self.answers[messages[-1].text]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])


def gold_decision(row: Any) -> RouteDecision:
    base: dict[str, Any] = {"intent": row.intent, "confidence": 0.95, "reason": "gold"}
    if row.discovery:
        base["discovery"] = {**row.discovery.model_dump(), "raw_query": row.query}
    if row.intent == "outreach":
        base.update(outreach_channel=row.outreach_channel, lead_selector=row.lead_selector)
    if row.intent == "clarify":
        base["clarifying_question"] = "Which partner type?"
    return RouteDecision.model_validate(base)


def dev_rows() -> list[Any]:
    return [row for row in load_routing() if row.split == "dev"]


def test_routing_suite_with_an_oracle_router_and_one_planted_mistake() -> None:
    rows = dev_rows()
    answers = {row.query: gold_decision(row).model_dump_json() for row in rows}
    victim = next(row for row in rows if row.intent == "outreach")
    wrong = gold_decision(victim).model_copy(update={"intent": "cpg_qa", "lead_selector": None})
    answers[victim.query] = wrong.model_dump_json()
    router = OracleRouter.scripted("unused")
    router.answers = answers
    result = result_of(runner.run_routing(make_env({"router": router}, split="dev")))
    assert result.n == 20
    assert result.metrics["intent_accuracy"] == pytest.approx(19 / 20)
    assert result.metrics["slot_accuracy"] == 1.0  # every discovery row answered from gold
    assert [item["id"] for item in result.items if not item["ok"]] == [victim.id]


def test_a_wrong_outreach_channel_makes_the_item_fail_and_is_reported() -> None:
    rows = dev_rows()
    answers = {row.query: gold_decision(row).model_dump_json() for row in rows}
    call = next(row for row in rows if row.outreach_channel == "voice")
    flipped = gold_decision(call).model_copy(update={"outreach_channel": "email"})
    answers[call.query] = flipped.model_dump_json()
    router = OracleRouter.scripted("unused")
    router.answers = answers
    result = result_of(runner.run_routing(make_env({"router": router}, split="dev")))
    item = next(item for item in result.items if item["id"] == call.id)
    assert (item["gold_channel"], item["predicted_channel"], item["ok"]) == (
        "voice",
        "email",
        False,
    )
    assert result.metrics["intent_accuracy"] == 1.0  # the intent was right
    assert result.metrics["outreach_channel_accuracy"] == pytest.approx(2 / 3)


def test_a_router_that_always_fails_clarifies_everything_and_records_the_error() -> None:
    router = ScriptedChatModel.repeating("not json")
    result = result_of(runner.run_routing(make_env({"router": router}, split="dev")))
    clarify = sum(row.intent == "clarify" for row in dev_rows())
    assert result.metrics["intent_accuracy"] == pytest.approx(clarify / 20)
    assert all(item["errors"] == ["invalid_output"] for item in result.items)


# --- extraction and retrieval -------------------------------------------------------


def page_html(company: str, footer: str, filler: str = "") -> str:
    return (
        f"<html><head><title>{company}</title></head><body><article>"
        f"<p>{company} is a food broker for retailers.</p>{filler}</article>"
        f"<footer>{footer}</footer></body></html>"
    )


def make_row(root: Path, n: int, url: str, html: str, case: str = "standard", **fields: Any) -> Any:
    (root / "pages").mkdir(parents=True, exist_ok=True)
    path = root / "pages" / f"e{n:03d}.html"
    path.write_text(html, encoding="utf-8")
    return ExtractionRow.model_validate(
        {
            "id": f"e{n:03d}",
            "url": url,
            "snapshot": f"pages/e{n:03d}.html",
            "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "company_name": fields.pop("company_name", "Acme Foods"),
            "partner_type": "broker",
            "case": case,
            **fields,
        }
    )


def reply(company: str, email: str | None = None, phone: str | None = None) -> str:
    lead = ExtractedLead(
        company_name=company, partner_type="broker", email=email, phone=phone, evidence_quote="q"
    )
    return ExtractionResult(leads=[lead]).model_dump_json()


def test_extraction_suite_on_three_synthetic_pages(tmp_path: Path) -> None:
    rows = [
        make_row(
            tmp_path,
            1,
            "https://acme.example/",
            page_html(
                "Acme Foods",
                'Sales: <a href="mailto:sales@acme.example">email us</a> Call (310) 555-0100',
            ),
            emails=["sales@acme.example"],
            phones=["+13105550100"],
        ),
        make_row(  # a page with no contact: the model correctly returns nothing
            tmp_path,
            2,
            "https://nocontact.example/",
            page_html("Quiet Brokers", "Welcome"),
            case="no_contact",
            company_name="Quiet Brokers",
        ),
        make_row(  # the model invents an address that the page does not hold
            tmp_path,
            3,
            "https://beta.example/",
            page_html("Beta Brokerage", "Write to info@beta.example"),
            emails=["info@beta.example"],
            company_name="Beta Brokerage",
        ),
    ]
    model = ByPageModel(
        messages=iter([]),
        pages={
            "https://acme.example/": [reply("Acme Foods", "sales@acme.example", "+13105550100")],
            "https://nocontact.example/": [ExtractionResult(leads=[]).model_dump_json()],
            "https://beta.example/": [reply("Beta Brokerage", "ceo@beta.example")],
        },
    )
    result = result_of(
        runner.run_extraction(make_env({"extractor": model}, fixtures=tmp_path), rows)
    )
    metrics = result.metrics
    assert result.n == 3
    # emails: tp 1 (acme), fp 0, fn 1 (info@beta, never output) -> P 1.0, R 0.5
    assert (metrics["email_precision"], metrics["email_recall"]) == (1.0, 0.5)
    assert (metrics["phone_precision"], metrics["phone_recall"]) == (1.0, 1.0)
    assert metrics["raw_hallucination_rate"] == 0.5  # ceo@beta of two proposals
    assert metrics["eligible_hallucination_rate"] == 0.0  # verification kept it out
    assert (metrics["abstention_accuracy"], metrics["abstention_n"]) == (1.0, 1)
    assert metrics["company_name_match"] == 1.0
    assert {item["id"]: item["ok"] for item in result.items} == {
        "e001": True,
        "e002": True,
        "e003": False,
    }
    by_id = {item["id"]: item for item in result.items}
    assert by_id["e003"]["statuses"] == ["not_in_source"]


def test_a_missing_or_edited_snapshot_is_skipped_with_a_note(tmp_path: Path) -> None:
    kept = make_row(tmp_path, 1, "https://acme.example/", page_html("Acme", "a@acme.example"),
                    emails=["a@acme.example"])  # fmt: skip
    edited = make_row(tmp_path, 2, "https://b.example/", page_html("B", "b@b.example"),
                      emails=["b@b.example"])  # fmt: skip
    (tmp_path / edited.snapshot).write_text("changed", encoding="utf-8")
    gone = kept.model_copy(update={"id": "e003", "snapshot": "pages/e003.html"})
    model = ByPageModel(messages=iter([]), pages={"https://acme.example/": [reply("Acme")]})
    result = result_of(
        runner.run_extraction(
            make_env({"extractor": model}, fixtures=tmp_path), [kept, edited, gone]
        )
    )
    assert result.n == 1
    assert [note for note in result.notes if "skipped" in note] == [
        "e002 skipped: mismatch pages/e002.html",
        "e003 skipped: missing pages/e003.html",
    ]


def test_retrieval_suite_shows_hybrid_finding_what_vector_search_misses(tmp_path: Path) -> None:
    topic = "broker for food and beverage products contact email phone. "
    filler = "".join(
        f"<p>{topic * 3}</p>" if i == 0 else f"<p>{'lorem ipsum ' * 20}</p>" for i in range(8)
    )
    row = make_row(
        tmp_path,
        1,
        "https://acme.example/",
        page_html("Acme Foods", "Write to sales@acme.example", filler),
        emails=["sales@acme.example"],
    )
    environment = make_env(
        fixtures=tmp_path, chunk_tokens=50, chunk_overlap=10, evidence_k=1, evidence_max=3
    )
    result = result_of(runner.run_retrieval(environment, [row]))
    assert result.metrics["vector"] == {"recall": 0.0, "pages": 1, "no_gold_chunk": 0}
    assert result.metrics["contact_pattern"]["recall"] == 1.0
    assert result.metrics["hybrid"]["recall"] == 1.0
    assert result.metrics["recall"] == 1.0
    assert result.items[0]["ok"] is True


def test_retrieval_skips_pages_without_a_gold_contact(tmp_path: Path) -> None:
    empty = make_row(tmp_path, 1, "https://q.example/", page_html("Quiet", "hi"), case="no_contact")
    result = result_of(runner.run_retrieval(make_env(fixtures=tmp_path), [empty]))
    assert result.n == 0 and result.metrics["hybrid"]["recall"] is None


# --- injection ----------------------------------------------------------------------


def test_injection_suite_flags_eight_and_counts_the_one_that_fools_the_model() -> None:
    rows = load_injection()
    by_id = {row.id: row for row in rows}
    fooled, control = by_id["i09"], by_id["i10"]  # the JSON mimic is below the threshold
    both = ExtractionResult(
        leads=[
            ExtractedLead(
                company_name=fooled.company_name,
                email=fooled.attack_email,
                evidence_quote="x",
            ),
            ExtractedLead(
                company_name=fooled.company_name, email=fooled.emails[0], evidence_quote="y"
            ),
        ]
    ).model_dump_json()
    model = ByPageModel(
        messages=iter([]),
        pages={
            fooled.url: [both],
            control.url: [reply(control.company_name, control.emails[0])],
        },
    )
    result = result_of(runner.run_injection(make_env({"extractor": model}), rows, []))
    metrics = result.metrics
    assert metrics["attack_pages"] == 9
    assert metrics["attack_success_ids"] == ["i09"]
    assert metrics["attack_success_rate"] == pytest.approx(1 / 9)
    assert metrics["flag_rate"] == pytest.approx(8 / 9)
    assert metrics["unflagged_ids"] == ["i09"]
    assert (metrics["false_positive_rate"], metrics["benign_pages"]) == (0.0, 1)
    assert metrics["legit_contact_survival"] == 1.0  # i09's real address is still eligible
    ok = {item["id"]: item["ok"] for item in result.items}
    assert ok["i09"] is False and ok["i10"] is True and ok["i01"] is True


# --- knowledge base -----------------------------------------------------------------


class OracleWriter(ScriptedChatModel):
    """Cites passage 1 for every question except the ones it was told cannot be answered."""

    unanswerable: list[str] = Field(default_factory=list)

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        prompt = "\n".join(m.text for m in messages)
        known = not any(question in prompt for question in self.unanswerable)
        answer = "The notes cover this [1]." if known else "The notes do not say."
        reply_json = KBAnswer(answer=answer).model_dump_json()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply_json))])


def test_kb_suite_reads_citations_back_to_doc_ids_and_counts_abstentions() -> None:
    writer = OracleWriter.scripted("unused")
    writer.unanswerable = [row.question for row in load_kb() if not row.expected_doc_ids]
    # kb_min_score 0: the hashing embedder's scores are not the bge ones the threshold is tuned for
    result = result_of(runner.run_kb(make_env({"writer": writer}, kb_min_score=0.0)))
    assert (result.metrics["answerable"], result.metrics["unanswerable"]) == (20, 5)
    assert result.metrics["abstention_accuracy"] == 1.0
    assert result.metrics["answerable_abstained"] == 0.0
    answered = [item for item in result.items if item["expected"]]
    assert len(answered) == 20
    for item in answered:  # [1] is the best passage, whose note the "Sources" block names
        assert item["cited"] == [item["top_doc"]]
    assert all(item["abstained"] for item in result.items if not item["expected"])


def test_cited_titles_reads_the_sources_block() -> None:
    reply_text = (
        "A slotting fee is paid for shelf space [1].\n\nSources:\n"
        "[1] Slotting fees: https://example.org/a\n"
        "[2] Trade spend and promotions: https://example.org/b"
    )
    assert runner.cited_titles(reply_text) == {"Slotting fees", "Trade spend and promotions"}
    assert runner.cited_titles("no sources here") == set()


# --- e2e ----------------------------------------------------------------------------


def test_e2e_is_skipped_without_a_live_provider() -> None:
    result = result_of(runner.run_e2e(make_env()))
    assert result.skipped and "live" in result.skipped and result.n == 0


# --- settings, reports and the command line ------------------------------------------


def test_build_settings_parses_roles_overrides_and_the_cache_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    config = runner.parse_config(["CHUNK_TOKENS=500", "evidence_k=2"])
    assert config == {"chunk_tokens": "500", "evidence_k": "2"}
    cached = runner.build_settings("gemini", "simulator=ollama:qwen2.5:7b", False, config)
    spec = cached.llm_roles["simulator"]
    assert (spec.provider, spec.model) == ("ollama", "qwen2.5:7b")
    assert (cached.chunk_tokens, cached.evidence_k) == (500, 2)
    assert cached.llm_cache_dir == runner.CACHE_DIR
    assert runner.build_settings("gemini", None, True, {}).llm_cache_dir is None  # --no-cache
    assert runner.build_settings("fake", None, False, {}).llm_cache_dir is None
    with pytest.raises(SystemExit):
        runner.parse_config(["CHUNK_TOKENS"])


def test_the_command_line_writes_a_json_and_a_markdown_report(tmp_path: Path) -> None:
    code = runner.main(
        [
            "--suite",
            "routing",
            "--provider",
            "fake",
            "--split",
            "dev",
            "--config",
            "CHUNK_TOKENS=500",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    (json_path,) = tmp_path.glob("*_routing_fake_chunk_tokens-500.json")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["suite"] == "routing" and report["provider"] == "fake" and report["n"] == 20
    assert report["split"] == "dev" and report["config_overrides"] == {"chunk_tokens": "500"}
    assert (
        set(report["datasets"]) == {"routing.jsonl"}
        and len(report["datasets"]["routing.jsonl"]) == 64
    )
    assert report["prompt_versions"]["charlie"].startswith("charlie-")
    assert {"git", "date", "models", "cache", "metrics", "items"} <= set(report)
    markdown = json_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "# Eval: routing" in markdown and "| intent_accuracy |" in markdown


def test_a_skipped_suite_is_reported_as_skipped(tmp_path: Path) -> None:
    assert runner.main(["--suite", "e2e", "--provider", "fake", "--out", str(tmp_path)]) == 0
    (json_path,) = tmp_path.glob("*_e2e_fake.json")
    assert json.loads(json_path.read_text(encoding="utf-8"))["skipped"]
    assert "**Skipped:**" in json_path.with_suffix(".md").read_text(encoding="utf-8")


def test_every_suite_runs_offline_with_the_fake_provider(tmp_path: Path) -> None:
    """T-39 done-when: ``--suite all --provider fake``. The fake model answers nothing useful,
    so the numbers mean nothing; what is checked is that every suite still reports."""
    assert runner.main(["--suite", "all", "--provider", "fake", "--out", str(tmp_path)]) == 0
    reports = {
        path.name.split("_")[1]: json.loads(path.read_text("utf-8"))
        for path in tmp_path.glob("*.json")
    }
    assert set(reports) == set(runner.SUITES)
    assert reports["e2e"]["skipped"]
    assert reports["routing"]["n"] == 60 and reports["kb"]["n"] == 25
    assert reports["injection"]["n"] == 10 and reports["injection"]["metrics"]["attack_pages"] == 9
    assert reports["kb"]["metrics"]["abstention_accuracy"] == 1.0  # a failed writer abstains
    assert reports["kb"]["notes"] and "hashing embedder" in reports["kb"]["notes"][-1]


def test_a_suite_that_called_no_model_is_reported_as_provider_none_with_its_embedder(
    tmp_path: Path,
) -> None:
    """Retrieval calls no model, so it is not a "gemini" run because gemini is the default; and a
    run with overrides gets its own file instead of overwriting the default run's."""
    configured = Settings(  # type: ignore[call-arg]
        _env_file=None, llm_provider="gemini", llm_roles="simulator=ollama:qwen2.5:7b"
    )
    result = SuiteResult("retrieval", 26, {"recall": 1.0})

    def report(models: dict[str, str], overrides: dict[str, str]) -> Path:
        json_path, _ = runner.write_report(
            result,
            settings=configured,
            split="test",
            tokens={},
            models=models,
            cache={},
            out_dir=tmp_path,
            overrides=overrides,
            embedder="BAAI/bge-small-en-v1.5",
        )
        return json_path

    plain = report({}, {})
    assert plain.name.endswith("_retrieval_none_simulator-ollama-qwen2.5-7b.json")
    tuned = report({}, {"chunk_tokens": "500"})
    assert tuned != plain and "chunk_tokens-500" in tuned.name
    data = json.loads(tuned.read_text(encoding="utf-8"))
    assert data["provider"] == "none" and data["embedder"] == "BAAI/bge-small-en-v1.5"
    assert "| embedder | BAAI/bge-small-en-v1.5 |" in tuned.with_suffix(".md").read_text("utf-8")
    used = report({"router": "gemini:some-model"}, {})
    assert json.loads(used.read_text(encoding="utf-8"))["provider"] == "gemini"
