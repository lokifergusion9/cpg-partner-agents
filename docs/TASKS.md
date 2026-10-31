# TASKS: CPG Partner Agents

**How to use this file**
- Take the **first unchecked task whose Deps are all checked**. Do one task per commit (`T-xx: summary`).
- A task is done only when its **Done when** is met and shown (test name, command output, or a Progress log note for manual checks).
- Estimates assume ~2–3 focused hours a day. The total is about 80 h, which is roughly 4–5 weeks part-time or 2 weeks full-time.
- Section references (§) point to [HLD.md](HLD.md) unless stated otherwise.

| Milestone | Goal | Est. |
|---|---|---|
| M0 | Foundation: repo, config, CI, utilities | 5 h |
| M1 | Schemas, LLM layer, repository | 6 h |
| M2 | Scout: discovery pipeline end to end (CLI) | 16 h |
| M3 | Charlie: router, knowledge base, guardrails, graph + memory | 10 h |
| M4 | Chase: policy, drafts, approval, dispatch | 9 h |
| M5 | API, UI, Mongo, demo mode → **MVP** | 10 h |
| M6 | Safety and observability | 7 h |
| M7 | Evaluation with published numbers | 12 h |
| M8 | Deploy and ship → **v1** | 8 h |
| M9 | Stretch | — |

---

## M0: Foundation

- [x] **T-01 Repo scaffold** · 1.5 h · Deps: —
  - What: uv project (`pyproject.toml`, Python 3.12), `src/partner_agents/` package skeleton per CLAUDE.md layout, ruff + mypy + pytest config (`live` marker registered), `.gitignore` (`.env`, `.qdrant/`, `eval/fixtures/pages/`, `eval/.cache/`, tokens).
  - Done when: `uv sync && uv run pytest` passes with one smoke test, and `uv run ruff check .` and `uv run mypy src` are clean.

- [ ] **T-02 Settings** · 1 h · Deps: T-01
  - What: `config.Settings` (pydantic-settings) with every variable in §10. `.env.example` mirrors it with comments.
  - Done when: tests show safe defaults (`OUTREACH_MODE=outbox`, `VOICE_PROVIDER=mock`). An empty `QDRANT_URL` / `MONGODB_URI` selects the local/in-memory backends. `DEMO_MODE=true` overrides outreach and voice.

- [ ] **T-03 CI** · 0.5 h · Deps: T-01
  - What: GitHub Actions workflow running ruff, mypy and pytest on push and PR.
  - Done when: the first push shows a green run.

- [ ] **T-04 Observability base** · 1 h · Deps: T-02
  - What: JSON log formatter, a `run_id` contextvar, the `stage_timer`/node decorator (§11) and a redaction filter (keys, emails, phones).
  - Done when: tests assert that the timing is recorded, `run_id` appears in the log record, and an email in a log message is redacted.

- [ ] **T-05 Retry utility** · 1 h · Deps: T-01
  - What: `utils/retry.py`, an async retry with exponential backoff + jitter, a retryable predicate and `Retry-After` support.
  - Done when: tests cover success after N failures, immediate raise on a non-retryable error, max attempts, and `Retry-After` honoured (time mocked).

## M1: Schemas, LLM layer, repository

- [ ] **T-06 Domain schemas** · 1.5 h · Deps: T-01
  - What: `schemas.py` exactly as in §4.
  - Done when: tests cover the `max_results` bounds, `outreach_eligible` logic, the `ExtractionResult` max length, the `EmailDraft` subject length and JSON round-trip.

- [ ] **T-07 LLM factory** · 1.5 h · Deps: T-02, T-05
  - What: `llm.get_chat_model(role)` for `gemini` / `groq` / `ollama` / `fake` with per-role temperatures (§9), plus the global concurrency semaphore.
  - Done when: offline tests with `fake` pass. `pytest -m live` returns a valid `RouteDecision` from the configured provider (note the provider and model in the Progress log).

- [ ] **T-08 Structured output helper** · 1 h · Deps: T-07
  - What: `invoke_structured()`, with one retry carrying the validation error and token usage recorded per role.
  - Done when: a test with a fake model returning invalid then valid output succeeds. Invalid twice raises a typed error. Token counts are captured.

- [ ] **T-09 Repository (in-memory)** · 2 h · Deps: T-06
  - What: `storage/repo.py`, the `Repository` protocol plus `InMemoryRepository` for leads, outreach, runs, pages (TTL), search cache and profiles (§8.1).
  - Done when: the shared repository test suite (`tests/storage/repo_contract.py`) passes against the in-memory implementation, including TTL expiry with mocked time.

## M2: Scout (discovery pipeline)

- [ ] **T-10 Search providers** · 2 h · Deps: T-06, T-09
  - What: `web/search.py` with the `SearchProvider` protocol, Tavily and DDG, fallback, query templates per partner type, domain dedupe, blocklist and cache (§6.3).
  - Done when: respx tests cover both providers, fallback on error and on empty results, dedupe by registrable domain, blocklisted domains dropped, and a second identical query served from cache.

- [ ] **T-11 Safe fetcher** · 3 h · Deps: T-05, T-09
  - What: `web/fetch.py` per §6.4: SSRF guard, robots cache, size and type limits, per-host limit, text assembly (main + footer/contact + `mailto:`/`tel:`), contact-page discovery, bodies stored to the repo.
  - Done when: respx tests cover each status (`blocked_robots`, `timeout`, `http_error`, `too_large`, `non_html`, `ssrf_blocked`), a `mailto:` captured from the footer, a contact page followed only when no contact is found on the first page, and a redirect to a private IP blocked.

- [ ] **T-12 Chunker** · 1.5 h · Deps: T-01
  - What: `retrieval/chunking.py`, token-based, with paragraph → sentence → word fallback, overlap and a `has_contact_pattern` flag (including obfuscated emails).
  - Done when: tests show every chunk ≤ `CHUNK_TOKENS`, consecutive chunks overlapping, the chunks jointly covering the text, no split words, and the flag true for `info [at] acme [dot] com`.

- [ ] **T-13 Embeddings + vector store** · 2 h · Deps: T-02, T-12
  - What: `retrieval/embeddings.py` (bge-small, batching, query prefix, LRU cache) and `retrieval/store.py` (collections and payload indexes per §8.2, upsert, filtered search, TTL cleanup).
  - Done when: tests on Qdrant `:memory:` with a deterministic fake embedder show that search filtered by `run_id` + `url` returns only that page's chunks and that cleanup removes old points. A live test embeds real text with the right dimension (384).

- [ ] **T-14 Evidence selection** · 1 h · Deps: T-13
  - What: `retrieval/evidence.py`, per-page hybrid selection (§6.6).
  - Done when: a test shows a contact-pattern chunk with a low vector score still being selected, the `EVIDENCE_MAX` cap respected, output sorted by `chunk_index`, and `selected_by` set correctly.

- [ ] **T-15 Untrusted wrapper + extraction node** · 2 h · Deps: T-08, T-14
  - What: `web/safety.wrap_untrusted()`, `prompts/scout.py` (with `PROMPT_VERSION`), and the `scout_extract` node with bounded concurrency (§6.7).
  - Done when: fake-model tests show the prompt contains the wrapper and the intent, one failing page produces a `StageError` while the others succeed, and an empty `leads` list is accepted.

- [ ] **T-16 Contact verification** · 2 h · Deps: T-06
  - What: `verify/contacts.py` per §6.8 (normalisation, de-obfuscation, email and phone checks, dedupe; MX behind a flag).
  - Done when: table-driven tests cover a hallucinated email → `not_in_source`, an obfuscated email → `verified`, a malformed email → `invalid_format`, phone format variants → E.164 `verified`, a phone not on the page → `not_in_source`, and the dedupe keeping the better status.

- [ ] **T-17 Scout subgraph + CLI** · 2.5 h · Deps: T-10, T-11, T-15, T-16
  - What: nodes `scout_search` → … → `scout_report` wired as a subgraph, and `cli.py scout "<query>"` printing a lead table plus stage timings.
  - Done when: an offline integration test (respx pages + fake LLM) produces the expected verified and rejected leads. A manual live run on 2 queries yields ≥ 3 verified leads each (record the queries, counts and timings in the Progress log).

## M3: Charlie (orchestrator)

- [ ] **T-18 Knowledge-base corpus + ingest** · 2.5 h · Deps: T-13
  - What: 15–30 short notes in `data/kb/`, **written in your own words** from public sources, with `source_url` in front matter. Topics: channel structure, broker vs distributor vs wholesaler, slotting fees, trade spend, retail calendar, private label, DSD, food-safety basics, pitching to buyers. Plus `kb/ingest.py`.
  - Done when: `uv run python -m partner_agents.kb.ingest` loads every note into `cpg_kb`, and a test checks that the front matter is present for every file.

- [ ] **T-19 Router** · 2 h · Deps: T-08
  - What: `prompts/charlie.py` and the `charlie_route` node (§6.1), with few-shot examples **from the routing dev split only**, the confidence → clarify rule and the missing-fields → clarify rule.
  - Done when: fake tests cover each of the 5 branches and both coercion rules. A live check on the 20 dev queries is noted in the Progress log.

- [ ] **T-20 Knowledge-base Q&A + guardrails** · 2 h · Deps: T-18, T-19
  - What: the `kb_answer`, `refuse` and `ask_clarification` nodes, the citation post-check and the score threshold (§6.2).
  - Done when: tests show an answer whose `[n]` markers map to sources, a low score giving the fallback template, an answer without a citation giving the fallback, and `out_of_scope` giving a refusal that lists capabilities.

- [ ] **T-21 Graph assembly + memory** · 2.5 h · Deps: T-17, T-20
  - What: `graph/build.py` per §5 with a checkpointer, message trimming, a new `run_id` per turn, and a `RunRecord` persisted at the end of each run.
  - Done when: an offline integration test shows turn 1 (discovery) followed by turn 2 ("only the ones in LA") resolving against the turn-1 leads in the same thread, and a `RunRecord` with timings for every executed node.

## M4: Chase (outreach)

- [ ] **T-22 Outreach policy** · 2 h · Deps: T-06, T-09
  - What: `outreach/policy.py` as pure functions: eligibility, the 30-day dedupe, the daily token bucket (**own implementation**), the allowlist, word limit and opt-out enforcement.
  - Done when: table-driven tests cover every skip status, the token bucket refilling with mocked time, and the opt-out line appended when missing.

- [ ] **T-23 Lead selection + drafting** · 2 h · Deps: T-21, T-22
  - What: `chase_select` (deterministic selectors with an LLM fallback) and `chase_draft` with `BrandProfile` (§6.9).
  - Done when: tests show "top 3" and "the LA ones" selected deterministically, ineligible leads skipped with their status, and drafts that respect the subject and body limits.

- [ ] **T-24 Approval gate** · 1.5 h · Deps: T-23
  - What: `approval_gate` using `interrupt()`, and resume with `Command(resume=ApprovalDecision)`.
  - Done when: an integration test shows the run pausing with `pending_approval`, resuming with 1 approve, 1 edit and 1 reject, only 2 dispatched, the edited content used, and the rejected draft's status `rejected`.

- [ ] **T-25 Dispatch: outbox + Gmail draft** · 2.5 h · Deps: T-24
  - What: `outreach/email.py` with the `outbox` mode (default) and `gmail_draft` (OAuth installed-app flow, `gmail.compose`, token outside the repo). `gmail_send` is guarded by the allowlist.
  - Done when: offline tests show the outbox statuses, `gmail_send` refusing non-allowlisted recipients, and demo mode forcing the outbox. Manually, one approved draft appears in your own Gmail Drafts (Progress log note).

- [ ] **T-26 Voice interface + mock** · 1 h · Deps: T-22
  - What: the `outreach/voice.py` protocol and `MockVoiceCaller` (script + simulated summary stored as a `voice` `OutreachDraft`).
  - Done when: a test shows the mock call stored with its script, and a non-allowlisted number refused even when `VOICE_PROVIDER=vapi`.

## M5: API, UI, storage → MVP

- [ ] **T-27 Mongo repository** · 2 h · Deps: T-09
  - What: `MongoRepository` (PyMongo async API), indexes and TTL per §8.1, and the MongoDB checkpointer wired in prod.
  - Done when: the shared repository suite passes with `pytest -m live` against Atlas M0.

- [ ] **T-28 FastAPI endpoints** · 2.5 h · Deps: T-21, T-24
  - What: every route in §7, the API key check, the error shape and `/health`.
  - Done when: `TestClient` tests cover each endpoint, including discovery → outreach → pending → approve → outboxed, a missing API key → 401, and an unknown run → 404 with the error shape.

- [ ] **T-29 Rate limiter** · 1 h · Deps: T-28
  - What: `api/ratelimit.py`, a token bucket (own implementation) per user/IP for discovery runs, returning `retry_after_s`.
  - Done when: tests with mocked time show the 6th run in an hour → 429, followed by recovery after the refill.

- [ ] **T-30 Gradio UI** · 3 h · Deps: T-28
  - What: `ui/app.py` mounted at `/ui` with these tabs: Chat (stage progress through `astream`), Leads (table + badges + source links), Approvals (approve/edit/reject), Runs (timings, errors, trace link), and a Settings panel for the brand profile.
  - Done when: the PRD §10 demo script works locally end to end (Progress log note + screenshot in `docs/img/`).

- [ ] **T-31 Demo mode** · 1 h · Deps: T-29, T-30
  - What: `DEMO_MODE` forces the outbox and mock voice, caps `max_results` at 5, uses an anonymous per-session `user_id`, pre-fills the sample brand profile and shows a banner.
  - Done when: tests prove every override, and the banner is visible in the UI.

**MVP checkpoint:** tag `v0.1.0` after T-31.

## M6: Safety and observability

- [ ] **T-32 Injection heuristics** · 2 h · Deps: T-11, T-15
  - What: `injection_score` in `web/safety.py` (instruction-like phrases, role-tag spoofing, hidden-text CSS, an unusual density of imperative sentences aimed at "the AI"), with a threshold that excludes the page.
  - Done when: unit tests over planted snippets pass, legit contact pages score below the threshold, and the false-positive rate on extraction-set pages is reported in the Progress log.

- [ ] **T-33 Langfuse tracing + run metrics** · 2 h · Deps: T-21
  - What: the callback attached when keys are present, node spans, `PROMPT_VERSION` in metadata, and the trace URL plus tokens saved to `RunRecord`.
  - Done when: a live discovery run shows a trace with every node, and `/api/runs/{id}` returns the trace URL and token counts.

- [ ] **T-34 Caches** · 1.5 h · Deps: T-10, T-13
  - What: an embedding LRU (in-process) and a search cache (repo, TTL 24 h), with hit/miss counters in `RunRecord`.
  - Done when: a test shows a repeated query making 0 search API calls and 0 re-embeddings of identical chunks.

- [ ] **T-35 TTL cleanup** · 1.5 h · Deps: T-13, T-27
  - What: a startup job deleting expired `web_chunks` points (pages already expire through Mongo TTL).
  - Done when: a test shows points older than `DATA_TTL_DAYS` removed and newer points kept.

## M7: Evaluation (see EVAL.md)

- [ ] **T-36 Routing dataset** · 1.5 h · Deps: T-19
  - What: `eval/datasets/routing.jsonl`, 80 queries (dev 20 / test 60) with the distribution in EVAL §3.1.
  - Done when: a schema test passes, there is no duplicate between splits, and the distribution matches.

- [ ] **T-37 Extraction dataset** · 4 h · Deps: T-11
  - What: 30 pages per EVAL §3.2, snapshotted with `eval/snapshot.py` (sha256 recorded) and hand-labelled per the labelling guide.
  - Done when: a schema test passes, every snapshot hash matches, and the case mix (obfuscated / contact-page / no-email) matches EVAL §3.2.

- [ ] **T-38 Injection + knowledge-base datasets** · 1.5 h · Deps: T-18, T-32
  - What: 10 planted pages (EVAL §3.4) and 25 knowledge-base questions (EVAL §3.6).
  - Done when: schema tests pass.

- [ ] **T-39 Eval runner + metrics** · 2.5 h · Deps: T-36, T-37, T-38
  - What: `eval/run.py` and `eval/metrics.py`, every suite in EVAL §2, the LLM response cache, and a JSON + Markdown report with metadata (git SHA, models, prompt versions, dataset hashes).
  - Done when: unit tests on the metrics (hand-computed cases) pass, and `--suite all --provider fake` runs offline.

- [ ] **T-40 Baseline + ablations** · 1.5 h · Deps: T-39
  - What: run every suite with the default config, plus the ablations in EVAL §7.
  - Done when: results are committed in `eval/results/` and EVAL §8 is filled in.

- [ ] **T-41 Error analysis + one improvement** · 1 h+ · Deps: T-40
  - What: categorise the top 10 failures (EVAL §9), ship one targeted fix, re-run.
  - Done when: EVAL §9 shows the categories and the before/after delta for the fix.

## M8: Deploy and ship → v1

- [ ] **T-42 Dockerfile** · 1.5 h · Deps: T-31
  - What: slim image with uv, the embedding model and tiktoken baked in, port 7860, non-root user.
  - Done when: `docker run -p 7860:7860 --env-file .env <img>` serves `/ui` and `/api/health` is ok.

- [ ] **T-43 Cloud services + Space** · 2 h · Deps: T-42, T-27
  - What: a Qdrant Cloud free cluster, Atlas M0, Langfuse Cloud, an HF Space (Docker SDK), secrets, `DEMO_MODE=true`.
  - Done when: the public URL completes the PRD §10 demo script end to end (Progress log note with the URL).

- [ ] **T-44 README** · 2 h · Deps: T-40, T-43
  - What: a one-line pitch, demo link + GIF, architecture diagram, the eval table with n/date/model, quickstart, design decisions (links to ADRs), limitations, the origin note.
  - Done when: a fresh clone following the README quickstart runs offline tests and the CLI with a single Gemini key.

- [ ] **T-45 Demo video + resume bullets** · 1.5 h · Deps: T-44
  - What: a 2-minute video of the demo script, and resume bullets using **measured** numbers from EVAL §8.
  - Done when: the video link is in the README and every number in the bullets traces to an eval result file.

**v1 checkpoint:** tag `v1.0.0` after T-45.

## M9: Stretch (only after v1)

- [ ] T-46 Vapi voice to an allowlisted, consenting number
- [ ] T-47 MX verification on by default + eval delta
- [ ] T-48 Groq for the router role, with a latency/accuracy comparison in EVAL
- [ ] T-49 Ollama comparison run (for example a friend's machine over Tailscale) added to the provider table
- [ ] T-50 Compound intents ("find and email") through a plan step
- [ ] T-51 SSE streaming endpoint for stage progress
- [ ] T-52 Semantic cache for repeated questions

---

## Progress log

<!-- One line per completed task: date · task · evidence (test/command/URL) · notes -->

| Date | Task | Evidence | Notes |
|---|---|---|---|
| 2026-10-31 | T-01 | `uv sync && uv run pytest` → 12 passed (`tests/test_smoke.py`); `ruff check .` clean; `mypy src` → no issues in 12 files | uv 0.12.5, CPython 3.12.14, pytest 9.2.0, ruff 0.17.3, mypy 2.4.1. ADR-015 records tooling; `ruff format` excludes `*.md` |

## Blockers / decisions needed

<!-- Anything waiting on a human decision. Remove once resolved (and record an ADR if it was a design choice). -->
- None yet.
