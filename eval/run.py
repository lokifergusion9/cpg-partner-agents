"""The evaluation runner (docs/EVAL.md sections 2 and 6).

    uv run python -m eval.run --suite routing --provider gemini
    uv run python -m eval.run --suite extraction --split dev
    uv run python -m eval.run --suite all --provider fake        # offline smoke, measures nothing

Each suite returns a ``SuiteResult``; ``write_report`` saves it as JSON and Markdown under
``eval/results/`` with the git SHA, the models that answered, the prompt versions, the dataset
hashes, the config overrides and the LLM cache hit rate. With ``--provider fake`` every model
reply is the stock fake reply, so the run proves that the plumbing works and nothing else.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from qdrant_client import AsyncQdrantClient

from eval.calls import run_calls
from eval.metrics import (
    E2eOutcome,
    InjectionOutcome,
    KbOutcome,
    PageScore,
    SuiteResult,
    e2e_metrics,
    evidence_recall,
    extraction_metrics,
    gold_chunks,
    injection_metrics,
    kb_metrics,
    routing_metrics,
    score_page,
    slots_correct,
)
from eval.schemas import (
    DATASETS,
    FIXTURES,
    ExtractionRow,
    InjectionRow,
    Split,
    load_e2e,
    load_extraction,
    load_injection,
    load_kb,
    load_routing,
    snapshot_problems,
)
from partner_agents.config import Role, Settings
from partner_agents.fakes import HashingEmbedder, ScriptedChatModel
from partner_agents.graph.build import PROMPT_VERSIONS, build_app_deps, build_graph, run_turn
from partner_agents.graph.charlie import NOT_IN_KB, make_charlie_route, make_kb_answer
from partner_agents.graph.scout import (
    ScoutDeps,
    build_scout_graph,
    make_scout_fetch,
    make_scout_index,
    make_scout_search,
)
from partner_agents.graph.state import AgentState
from partner_agents.kb.ingest import KB_CHUNK_OVERLAP, KB_CHUNK_TOKENS, ingest, load_notes
from partner_agents.llm import get_chat_model, track_providers, track_usage
from partner_agents.retrieval.chunking import TokenCounter, chunk_text
from partner_agents.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from partner_agents.retrieval.evidence import MAX_CONTACT_CHUNKS, build_query, select_evidence
from partner_agents.retrieval.store import KnowledgeStore, VectorStore
from partner_agents.schemas import DiscoveryIntent, Lead, RouteDecision, SearchHit
from partner_agents.storage.repo import InMemoryRepository
from partner_agents.utils.cache_stats import track_cache
from partner_agents.web.fetch import Fetcher
from partner_agents.web.search import SearchService

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE_DIR = ROOT / ".cache"
KB_NOTES = ROOT.parent / "data" / "kb"
PLAYBOOKS = ROOT.parent / "data" / "playbooks"
SUITES = ("routing", "extraction", "retrieval", "injection", "kb", "calls", "e2e")
CONCURRENCY = 4  # pages or questions in flight; LLM_MAX_CONCURRENCY bounds the model calls too
EVAL_USER = "eval"
NO_MODEL = ScriptedChatModel.repeating("this suite calls no model")


@dataclass
class Env:
    """What the suites need. Tests pass scripted models and a hashing embedder."""

    settings: Settings
    model: Callable[[Role], BaseChatModel]
    embedder: Embedder
    split: Split = "test"
    count_tokens: TokenCounter | None = None
    fixtures: Path = FIXTURES
    live: bool = False  # a real provider and the real web: the e2e suite may run
    label_sheet: Path | None = None  # calls suite: write dev transcripts here for hand-labelling


def words(text: str) -> int:
    return len(text.split())


async def bounded[T](jobs: Sequence[Awaitable[T]], limit: int = CONCURRENCY) -> list[T]:
    slots = asyncio.Semaphore(limit)

    async def run(job: Awaitable[T]) -> T:
        async with slots:
            return await job

    return list(await asyncio.gather(*(run(job) for job in jobs)))


# --- serving pages without the network ---------------------------------------------


class FixtureSearch:
    """A search provider that returns the one page a suite wants read."""

    name = "fixture"

    def __init__(self, url: str) -> None:
        self._url = url

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        return [
            SearchHit.model_validate(
                {
                    "url": self._url,
                    "title": self._url,
                    "snippet": "",
                    "rank": 1,
                    "provider": "fixture",
                }
            )
        ]


async def public_resolver(host: str) -> list[str]:
    return ["93.184.216.34"]  # every fixture host is "public" to the SSRF guard


def site_client(pages: Mapping[str, str]) -> httpx.AsyncClient:
    """An HTTP client that serves ``pages`` (url -> html), answers robots.txt with 404 and
    everything else with 404. Nothing leaves the process."""
    known = {str(httpx.URL(url)): html for url, html in pages.items()}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt" or str(request.url) not in known:
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=known[str(request.url)].encode("utf-8"),
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def page_intent(row: ExtractionRow | InjectionRow) -> DiscoveryIntent:
    """The search intent for a fixture page. Extraction pages have no category of their own, so
    they are read with a broad one (the extractor is asked for companies of the partner type)."""
    category = getattr(row, "product_category", "food and beverage products")
    location = getattr(row, "location", None)
    return DiscoveryIntent(
        partner_type=row.partner_type,
        product_category=category,
        location=location,
        max_results=5,
        raw_query=f"{row.partner_type} for {category}" + (f" in {location}" if location else ""),
    )


@dataclass
class PageRun:
    deps: ScoutDeps
    client: httpx.AsyncClient
    state: AgentState

    async def aclose(self) -> None:
        await self.deps.fetcher.aclose()
        await self.client.aclose()


def prepare_page(
    env: Env,
    qdrant: AsyncQdrantClient,
    url: str,
    pages: Mapping[str, str],
    intent: DiscoveryIntent,
    *,
    with_model: bool = True,
) -> PageRun:
    """A fresh repository, search cache and fetcher for one fixture page, and its start state.

    ``with_model=False`` is for suites that never call the extractor (retrieval, the fetch-only
    false-positive check), so that they need no API key."""
    repo = InMemoryRepository()
    client = site_client(pages)
    deps = ScoutDeps(
        settings=env.settings,
        repo=repo,
        search=SearchService([FixtureSearch(url)], repo, frozenset()),
        fetcher=Fetcher(env.settings, repo, client=client, resolver=public_resolver),
        store=VectorStore(qdrant, env.embedder),
        model=env.model("extractor") if with_model else NO_MODEL,
        count_tokens=env.count_tokens,
    )
    run_id = f"eval_{hashlib.sha256(url.encode()).hexdigest()[:8]}"
    state = AgentState(
        messages=[HumanMessage(content=intent.raw_query)],
        user_id=EVAL_USER,
        thread_id=run_id,
        run_id=run_id,
        route=RouteDecision(intent="discovery", confidence=1.0, discovery=intent, reason="eval"),
    )
    return PageRun(deps, client, state)


async def source_text(deps: ScoutDeps, state: Mapping[str, Any]) -> str:
    """The page text the verifier saw (empty when the page was not stored)."""
    for page in state.get("pages", []):
        record = await deps.repo.get_page(state["run_id"], str(page.url))
        if record is not None:
            return f"{record.text}\n{record.contact_text}"
    return ""


def extraction_pages(row: ExtractionRow, root: Path) -> dict[str, str]:
    pages = {row.url: (root / row.snapshot).read_text(encoding="utf-8")}
    if row.contact_url and row.contact_snapshot:
        pages[row.contact_url] = (root / row.contact_snapshot).read_text(encoding="utf-8")
    return pages


def usable_extraction_rows(
    env: Env, rows: Sequence[ExtractionRow]
) -> tuple[list[ExtractionRow], list[str]]:
    """The rows whose snapshots are present and unedited, and a note for each one left out."""
    usable, notes = [], []
    for row in rows:
        problems = snapshot_problems(row, env.fixtures)
        if problems:
            notes.append(f"{row.id} skipped: {', '.join(problems)}")
        else:
            usable.append(row)
    return usable, notes


# --- routing ------------------------------------------------------------------------


def seeded_leads() -> list[Lead]:
    """A fake earlier run for outreach rows, so that "the first one" has something to mean."""
    made = datetime(2026, 11, 1, tzinfo=UTC)
    names = [
        ("Golden Gate Brokerage", "Los Angeles, CA", "sales@goldengate.example"),
        ("Bay Area Foods Rep", "San Francisco, CA", "info@bayareafoods.example"),
        ("Sierra Snack Partners", "Sacramento, CA", "hello@sierrasnack.example"),
    ]
    return [
        Lead.model_validate(
            {
                "id": f"lead_{n}",
                "run_id": "eval_seed",
                "user_id": EVAL_USER,
                "company_name": name,
                "partner_type": "broker",
                "location": location,
                "website": None,
                "email": email,
                "email_status": "verified",
                "phone": None,
                "phone_status": "unchecked",
                "source_url": f"https://{email.split('@')[1]}/",
                "evidence_quote": email,
                "created_at": made,
            }
        )
        for n, (name, location, email) in enumerate(names, start=1)
    ]


async def run_routing(env: Env) -> SuiteResult:
    rows = [row for row in load_routing() if row.split == env.split]
    node = make_charlie_route(env.model("router"), env.settings)
    seed = seeded_leads()

    async def one(row: Any) -> tuple[Any, RouteDecision, list[str]]:
        state = AgentState(
            messages=[HumanMessage(content=row.query)],
            user_id=EVAL_USER,
            thread_id=row.id,
            run_id=f"eval_{row.id}",
            leads=seed if row.context_leads else [],
        )
        update = await node(state)
        return row, update["route"], [error.code for error in update.get("errors", [])]

    results = await bounded([one(row) for row in rows])
    items = [
        {
            "id": row.id,
            "query": row.query,
            "gold": row.intent,
            "predicted": decision.intent,
            "ok": row.intent == decision.intent
            and (row.intent != "discovery" or slots_correct(row, decision))
            and (row.intent != "outreach" or row.outreach_channel == decision.outreach_channel),
            "slots_ok": slots_correct(row, decision) if row.intent == "discovery" else None,
            "gold_channel": row.outreach_channel,
            "predicted_channel": decision.outreach_channel
            if decision.intent == "outreach"
            else None,
            "errors": codes,
        }
        for row, decision, codes in results
    ]
    return SuiteResult(
        "routing",
        len(rows),
        routing_metrics([(row, decision) for row, decision, _ in results]),
        items,
        [f"split: {env.split}"],
    )


# --- extraction and retrieval -------------------------------------------------------


async def run_extraction(env: Env, rows: Sequence[ExtractionRow] | None = None) -> SuiteResult:
    usable, notes = usable_extraction_rows(env, rows if rows is not None else load_extraction())
    qdrant = AsyncQdrantClient(":memory:")

    async def one(row: ExtractionRow) -> tuple[PageScore, dict[str, Any]]:
        page = prepare_page(
            env, qdrant, row.url, extraction_pages(row, env.fixtures), page_intent(row)
        )
        try:
            final: dict[str, Any] = await build_scout_graph(page.deps).ainvoke(page.state)
            text = await source_text(page.deps, final)
        finally:
            await page.aclose()
        leads: list[Lead] = final.get("leads", [])
        score = score_page(row, leads, text)
        item = {
            "id": row.id,
            "case": row.case,
            "gold_emails": sorted(score.gold_emails),
            "out_emails": sorted(score.out_emails),
            "gold_phones": sorted(score.gold_phones),
            "out_phones": sorted(score.out_phones),
            "page_status": [p.status for p in final.get("pages", [])],
            "statuses": [lead.email_status for lead in leads if lead.email],
            "errors": [error.code for error in final.get("errors", [])],
            "ok": score.out_emails == score.gold_emails and score.out_phones == score.gold_phones,
        }
        return score, item

    try:
        done = await bounded([one(row) for row in usable])
    finally:
        await qdrant.close()
    notes.append("every page is read with a generic intent for its partner type")
    return SuiteResult(
        "extraction",
        len(done),
        extraction_metrics([score for score, _ in done]),
        [item for _, item in done],
        notes,
    )


async def run_retrieval(env: Env, rows: Sequence[ExtractionRow] | None = None) -> SuiteResult:
    usable, notes = usable_extraction_rows(env, rows if rows is not None else load_extraction())
    wanted = [row for row in usable if row.emails or row.phones]
    qdrant = AsyncQdrantClient(":memory:")
    settings = env.settings

    async def one(
        row: ExtractionRow,
    ) -> tuple[dict[str, tuple[set[int], set[int]]], dict[str, Any]]:
        intent = page_intent(row)
        page = prepare_page(
            env, qdrant, row.url, extraction_pages(row, env.fixtures), intent, with_model=False
        )
        state: dict[str, Any] = dict(page.state)
        try:
            for make in (make_scout_search, make_scout_fetch, make_scout_index):
                state.update(await make(page.deps)(AgentState(**state)))  # type: ignore[typeddict-item]
            readable = [p for p in state.get("pages", []) if p.status == "ok"]
            record = (
                await page.deps.repo.get_page(state["run_id"], str(readable[0].url))
                if readable
                else None
            )
            if record is None:  # the page was not stored (blocked, flagged, unreadable)
                return (
                    {mode: (set(), set()) for mode in MODES},
                    {"id": row.id, "ok": False, "note": "page not indexed"},
                )
            url = str(readable[0].url)  # the stored form of the url (a trailing slash added)
            chunks = chunk_text(
                f"{record.text}\n\n{record.contact_text}".strip(),
                max_tokens=settings.chunk_tokens,
                overlap_tokens=settings.chunk_overlap,
                count=env.count_tokens,
            )
            gold = gold_chunks([(c.index, c.text) for c in chunks], row.emails, row.phones)
            store, run_id, query = page.deps.store, state["run_id"], build_query(intent)
            picked = {
                "vector": await store.search(query, run_id, url, settings.evidence_k),
                "contact_pattern": await store.search(
                    query, run_id, url, MAX_CONTACT_CHUNKS, contact_only=True
                ),
                "hybrid": await select_evidence(
                    store,
                    intent,
                    run_id,
                    url,
                    k=settings.evidence_k,
                    max_total=settings.evidence_max,
                ),
            }
        finally:
            await page.aclose()
        selected = {mode: {c.chunk_index for c in found} for mode, found in picked.items()}
        item = {
            "id": row.id,
            "case": row.case,
            "chunks": len(chunks),
            "gold_chunks": sorted(gold),
            "selected": {mode: sorted(found) for mode, found in selected.items()},
            "ok": bool(gold & selected["hybrid"]) if gold else None,
        }
        return {mode: (gold, found) for mode, found in selected.items()}, item

    try:
        done = await bounded([one(row) for row in wanted])
    finally:
        await qdrant.close()
    metrics = {mode: evidence_recall([pages[mode] for pages, _ in done]) for mode in MODES}
    metrics["recall"] = metrics["hybrid"]["recall"]  # the headline number (target >= 0.90)
    return SuiteResult("retrieval", len(done), metrics, [item for _, item in done], notes)


MODES = ("vector", "contact_pattern", "hybrid")


# --- injection ----------------------------------------------------------------------


async def run_injection(
    env: Env,
    rows: Sequence[InjectionRow] | None = None,
    extraction: Sequence[ExtractionRow] | None = None,
) -> SuiteResult:
    planted = list(rows if rows is not None else load_injection())
    usable, notes = usable_extraction_rows(
        env, extraction if extraction is not None else load_extraction()
    )
    qdrant = AsyncQdrantClient(":memory:")

    async def attack(row: InjectionRow) -> tuple[InjectionOutcome, dict[str, Any]]:
        html = (DATASETS / row.page).read_text(encoding="utf-8")
        page = prepare_page(env, qdrant, row.url, {row.url: html}, page_intent(row))
        try:
            final: dict[str, Any] = await build_scout_graph(page.deps).ainvoke(page.state)
        finally:
            await page.aclose()
        leads: list[Lead] = final.get("leads", [])
        flagged = any(p.status == "injection_flagged" for p in final.get("pages", []))
        outcome = InjectionOutcome(
            row,
            flagged,
            frozenset((lead.email or "").lower() for lead in leads if lead.email),
            frozenset((lead.email or "").lower() for lead in leads if lead.outreach_eligible),
        )
        item = {
            "id": row.id,
            "attack": row.attack,
            "flagged": flagged,
            "score": max((p.injection_score for p in final.get("pages", [])), default=0.0),
            "attack_in_leads": row.attack_email in outcome.lead_emails,
            "legit_survived": row.emails[0] in outcome.eligible_emails,
            "errors": [error.code for error in final.get("errors", [])],
        }
        if row.attack == "control":
            item["ok"] = not flagged  # a legitimate page must not be excluded
        else:
            item["ok"] = not item["attack_in_leads"]
        return outcome, item

    async def benign(row: ExtractionRow) -> bool:
        """Is a legitimate page wrongly excluded? Only the fetcher runs; no model is called."""
        page = prepare_page(
            env,
            qdrant,
            row.url,
            extraction_pages(row, env.fixtures),
            page_intent(row),
            with_model=False,
        )
        try:
            outcome = await page.deps.fetcher.fetch(row.url, page.state["run_id"])
        finally:
            await page.aclose()
        return outcome.page.status == "injection_flagged"

    try:
        attacked = await bounded([attack(row) for row in planted])
        flags = await bounded([benign(row) for row in usable])
    finally:
        await qdrant.close()
    outcomes = [outcome for outcome, _ in attacked]
    control = [o.flagged for o in outcomes if o.row.attack == "control"]
    metrics = injection_metrics(outcomes, [*control, *flags])
    notes.append(f"false-positive rate over {len(control)} control + {len(flags)} extraction pages")
    return SuiteResult("injection", len(outcomes), metrics, [item for _, item in attacked], notes)


# --- knowledge base -----------------------------------------------------------------

_SOURCE_LINE = re.compile(r"^\[\d+\] (.+): https?://\S+$")


def cited_titles(reply: str) -> set[str]:
    """The titles in the ``Sources:`` block that ``kb_answer`` appends to a cited answer."""
    _, _, sources = reply.partition("\nSources:\n")
    matches = (_SOURCE_LINE.match(line.strip()) for line in sources.splitlines())
    return {m.group(1) for m in matches if m}


async def run_kb(env: Env) -> SuiteResult:
    questions = load_kb()
    docs = load_notes(KB_NOTES)
    by_title = {doc.title: doc.doc_id for doc in docs}
    qdrant = AsyncQdrantClient(":memory:")
    store = KnowledgeStore(qdrant, env.embedder)
    try:
        await ingest(
            docs,
            store,
            max_tokens=KB_CHUNK_TOKENS,
            overlap_tokens=KB_CHUNK_OVERLAP,
            count=env.count_tokens,
        )
        node = make_kb_answer(store, env.model("writer"), env.settings)

        async def one(row: Any) -> tuple[KbOutcome, dict[str, Any]]:
            state = AgentState(
                messages=[HumanMessage(content=row.question)],
                user_id=EVAL_USER,
                thread_id=row.id,
                run_id=f"eval_{row.id}",
            )
            update = await node(state)
            reply = str(update["messages"][0].content)
            top = await store.search(row.question, k=1)
            cited = frozenset(by_title[t] for t in cited_titles(reply) if t in by_title)
            outcome = KbOutcome(row.id, frozenset(row.expected_doc_ids), cited, reply == NOT_IN_KB)
            item = {
                "id": row.id,
                "question": row.question,
                "expected": sorted(outcome.expected),
                "cited": sorted(cited),
                "abstained": outcome.abstained,
                "top_doc": top[0].doc_id if top else None,
                "top_score": round(top[0].score, 3) if top else None,
                "errors": [error.code for error in update.get("errors", [])],
                "ok": outcome.abstained
                if not outcome.expected
                else bool(cited) and cited <= outcome.expected,
            }
            return outcome, item

        done = await bounded([one(row) for row in questions])
    finally:
        await qdrant.close()
    return SuiteResult(
        "kb", len(done), kb_metrics([o for o, _ in done]), [item for _, item in done], []
    )


# --- e2e (live) ---------------------------------------------------------------------


async def run_e2e(env: Env) -> SuiteResult:
    if not env.live:
        return SuiteResult(
            "e2e",
            0,
            {},
            skipped="needs a real provider and the live web (use --provider gemini|groq|ollama)",
        )
    rows = load_e2e()
    deps = build_app_deps(env.settings)
    graph = build_graph(deps)
    outcomes: list[E2eOutcome] = []
    items: list[dict[str, Any]] = []
    try:
        for row in rows:  # one at a time: latency is the point
            started = time.perf_counter()
            result = await run_turn(
                graph, deps, thread_id=f"e2e_{row.id}", user_id=EVAL_USER, text=row.query
            )
            total_ms = round((time.perf_counter() - started) * 1000)
            leads: list[Lead] = result.state.get("leads", [])
            eligible = sum(lead.outreach_eligible for lead in leads)
            outcomes.append(
                E2eOutcome(
                    row.id,
                    total_ms,
                    result.run.timings_ms,
                    len(leads),
                    eligible,
                    result.run.tokens,
                    len(result.run.errors),
                )
            )
            items.append(
                {
                    "id": row.id,
                    "query": row.query,
                    "intent": result.run.intent,
                    "total_ms": total_ms,
                    "leads": len(leads),
                    "eligible": eligible,
                    "errors": [error.code for error in result.run.errors],
                    "ok": eligible > 0,
                }
            )
    finally:
        await deps.aclose()
    notes = [
        "live web: these numbers are latency and yield, not accuracy (EVAL 1, rule 6)",
        "cost is not estimated: the config has no price table yet",
    ]
    return SuiteResult("e2e", len(outcomes), e2e_metrics(outcomes), items, notes)


# --- suites, settings, reports ------------------------------------------------------

EMBEDDER_SUITES = ("extraction", "retrieval", "injection", "kb")

SUITE_FUNCTIONS: dict[str, Callable[[Env], Awaitable[SuiteResult]]] = {
    "routing": run_routing,
    "extraction": run_extraction,
    "retrieval": run_retrieval,
    "injection": run_injection,
    "kb": run_kb,
    "calls": run_calls,
    "e2e": run_e2e,
}

DATASET_FILES: dict[str, tuple[Path, ...]] = {
    "routing": (DATASETS / "routing.jsonl",),
    "extraction": (DATASETS / "extraction.jsonl",),
    "retrieval": (DATASETS / "extraction.jsonl",),
    "injection": (DATASETS / "injection.jsonl", DATASETS / "extraction.jsonl"),
    "kb": (DATASETS / "kb.jsonl", *sorted(KB_NOTES.glob("*.md"))),
    "calls": (
        DATASETS / "personas.jsonl",
        DATASETS / "call_labels.jsonl",
        PLAYBOOKS / "partner_intro.toml",
    ),
    "e2e": (DATASETS / "e2e.jsonl",),
}


def file_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dataset_hashes(suite: str) -> dict[str, str]:
    paths = DATASET_FILES.get(suite, ())
    hashes = {path.name: file_hash([path]) for path in paths if path.suffix in (".jsonl", ".toml")}
    notes = [path for path in paths if path.suffix == ".md"]
    if notes:
        hashes["kb notes"] = file_hash(notes)
    return hashes


def git_state() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        sha = subprocess.run(
            [git, "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            [git, "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        return "unknown"
    return f"{sha}{'+dirty' if dirty else ''}" if sha else "unknown"


def parse_config(pairs: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--config expects KEY=VALUE, got {pair!r}")
        out[key.strip().lower()] = value
    return out


def build_settings(
    provider: str, roles: str | None, no_cache: bool, config: Mapping[str, str]
) -> Settings:
    fields: dict[str, Any] = {"llm_provider": provider, **config}
    if roles:
        fields["llm_roles"] = roles
    # The fake provider needs no cache: its replies are free and a cached one would hide a bug.
    fields["llm_cache_dir"] = None if no_cache or provider == "fake" else CACHE_DIR
    if provider == "fake":
        return Settings(_env_file=None, **fields)  # type: ignore[call-arg]
    return Settings(**fields)  # keys come from the environment / .env


def build_env(settings: Settings, split: Split, label_sheet: Path | None = None) -> Env:
    fake = settings.llm_provider == "fake"
    return Env(
        settings=settings,
        model=lambda role: get_chat_model(role, settings=settings),
        embedder=HashingEmbedder()
        if fake
        else SentenceTransformerEmbedder(settings.embedding_model),
        split=split,
        count_tokens=words if fake else None,
        live=not fake,
        label_sheet=label_sheet,
    )


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def flatten(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key, inner in value.items():
            rows += flatten(f"{prefix}.{key}" if prefix else str(key), inner)
        return rows
    return [(prefix, value)]


def render_markdown(report: Mapping[str, Any]) -> str:
    meta_keys = ["date", "git", "provider", "split", "n"]
    if report.get("embedder"):
        meta_keys.insert(3, "embedder")
    lines = [f"# Eval: {report['suite']}", "", "| | |", "|---|---|"]
    lines += [f"| {key} | {fmt(report.get(key))} |" for key in meta_keys]
    lines += [f"| models | {fmt(report['models'] or 'none called')} |"]
    lines += [f"| prompt versions | {fmt(report['prompt_versions'])} |"]
    lines += [f"| datasets | {fmt(report['datasets'])} |"]
    lines += [f"| config overrides | {fmt(report['config_overrides'] or 'none')} |"]
    lines += [f"| cache | {fmt(report['cache'] or 'none')} |", ""]
    if report["skipped"]:
        lines += [f"**Skipped:** {report['skipped']}", ""]
    else:
        lines += ["## Metrics", "", "| metric | value |", "|---|---|"]
        lines += [f"| {key} | {fmt(value)} |" for key, value in flatten("", report["metrics"])]
        lines.append("")
    if report["notes"]:
        lines += ["## Notes", ""] + [f"- {note}" for note in report["notes"]] + [""]
    failed = [item for item in report["items"] if item.get("ok") is False]
    if failed:
        lines += [f"## Items that failed ({len(failed)} of {len(report['items'])})", ""]
        lines += ["```json"] + [json.dumps(item, ensure_ascii=False) for item in failed[:25]]
        lines += ["```", ""]
    return "\n".join(lines)


def run_tag(settings: Settings, overrides: Mapping[str, str]) -> str:
    """A short file-name tag for what differs from the default run: overrides and role models."""
    parts = [f"{key}-{value}" for key, value in sorted(overrides.items())]
    parts += [
        f"{role}-{spec.provider}-{spec.model or 'default'}"
        for role, spec in sorted(settings.llm_roles.items())
    ]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", "_".join(parts)).strip("-_")


def write_report(
    result: SuiteResult,
    *,
    settings: Settings,
    split: str,
    tokens: Mapping[str, int],
    models: Mapping[str, str],
    cache: Mapping[str, int],
    out_dir: Path,
    overrides: Mapping[str, str],
    embedder: str | None = None,
) -> tuple[Path, Path]:
    now = datetime.now(UTC)
    # A suite that called no model (retrieval) is not a "gemini" run just because gemini is the
    # configured default, and a run with overrides must not overwrite the default run's files.
    called_a_model = bool(models) or result.skipped is not None
    provider = (
        settings.llm_provider if called_a_model or settings.llm_provider == "fake" else "none"
    )
    report: dict[str, Any] = {
        "suite": result.suite,
        "date": now.astimezone().date().isoformat(),  # the local day, as the task log uses
        "generated_at": now.isoformat(timespec="seconds"),
        "git": git_state(),
        "provider": provider,
        "embedder": embedder,
        "split": split,
        "n": result.n,
        "models": dict(models),
        "tokens": dict(tokens),
        "prompt_versions": {**PROMPT_VERSIONS, **result.prompt_versions},
        "datasets": dataset_hashes(result.suite),
        "config_overrides": dict(overrides),
        "cache": dict(cache),
        "skipped": result.skipped,
        "metrics": result.metrics,
        "notes": result.notes,
        "items": result.items,
    }
    tag = run_tag(settings, overrides)
    stem = f"{report['date']}_{result.suite}_{provider}" + (f"_{tag}" if tag else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


async def run_suite(
    env: Env, suite: str, out_dir: Path, overrides: Mapping[str, str]
) -> SuiteResult:
    with track_usage() as tokens, track_providers() as models, track_cache() as cache:
        result = await SUITE_FUNCTIONS[suite](env)
    if isinstance(env.embedder, HashingEmbedder) and suite in ("extraction", "retrieval", "kb"):
        result.notes.append("embedder: the offline hashing embedder, not bge-small (fake provider)")
    embedder = None
    if suite in EMBEDDER_SUITES:
        embedder = (
            "hashing (offline, not a real embedding model)"
            if isinstance(env.embedder, HashingEmbedder)
            else env.settings.embedding_model
        )
    json_path, _ = write_report(
        result,
        settings=env.settings,
        split=env.split,
        tokens=tokens,
        models=models,
        cache=cache,
        out_dir=out_dir,
        overrides=overrides,
        embedder=embedder,
    )
    status = f"skipped ({result.skipped})" if result.skipped else f"n={result.n}"
    print(f"{suite}: {status} -> {json_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval.run", description=__doc__)
    parser.add_argument("--suite", choices=[*SUITES, "all"], required=True)
    parser.add_argument(
        "--provider", choices=["gemini", "groq", "ollama", "fake"], default="gemini"
    )
    parser.add_argument("--roles", help="per-role override, e.g. simulator=ollama:qwen2.5:7b")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--no-cache", action="store_true", help="make every model call fresh")
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=RESULTS, help="folder for the reports")
    parser.add_argument(
        "--label-sheet",
        type=Path,
        help="calls suite: write dev transcripts here, to label by hand for call_labels.jsonl",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.provider == "fake":  # every call fails on purpose: do not log each one
        for name in ("partner_agents", "eval"):
            logging.getLogger(name).setLevel(logging.ERROR)
    overrides = parse_config(args.config)
    settings = build_settings(args.provider, args.roles, args.no_cache, overrides)
    env = build_env(settings, args.split, args.label_sheet)
    names = list(SUITES) if args.suite == "all" else [args.suite]

    async def go() -> None:
        for name in names:
            await run_suite(env, name, args.out, overrides)

    asyncio.run(go())
    return 0


if __name__ == "__main__":
    sys.exit(main())
