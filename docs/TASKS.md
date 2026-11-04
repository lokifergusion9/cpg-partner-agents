# TASKS: CPG Partner Agents

**How to use this file**
- Take the **first unchecked task whose Deps are all checked**. Do one task per commit (`T-xx: summary`).
- A task is done only when its **Done when** is met and shown (test name, command output, or a Progress log note for manual checks).
- **Plan: 4 weeks, 2026-11-01 → 2026-11-28, about 25 focused hours a week.** About 99 h of work remains after M0. See the schedule below.
- Task ids are stable. New tasks take the next free number, so ids are not always in order inside a milestone.
- Section references (§) point to [HLD.md](HLD.md) unless stated otherwise.

| Milestone | Goal | Est. |
|---|---|---|
| M0 | Foundation: repo, config, CI, utilities | 5 h |
| M1 | Schemas, LLM layer, repository | 6 h |
| M2 | Scout: discovery pipeline end to end (CLI) | 16 h |
| M3 | Charlie: router, knowledge base, guardrails, graph + memory | 10 h |
| M4 | Chase: policy, drafts, approval, dispatch | 9 h |
| M5 | API, UI, Mongo, demo mode → **MVP** | 10 h |
| M6 | Call agent: playbook, stage machine, checks, simulator, Call tab (+ voice, nice-to-have) | 18.5 h |
| M7 | Safety and observability | 7 h |
| M8 | Evaluation with published numbers (incl. the `calls` suite) | 15 h |
| M9 | Deploy and ship → **v1** | 7 h |
| M10 | Stretch | — |

M1 also gains T-53 (LLM cache, per-role providers, fallback chain: 2 h).

## 4-week schedule

| Week | Dates | Tasks | Hours | Done at the end of the week |
|---|---|---|---|---|
| 1 | Nov 1 – Nov 7 | M1 (T-06 → T-09, **T-53 first after T-08**) + M2 (T-10 → T-17) | ~24 h | `cli scout "cookie brokers in California"` prints verified leads |
| 2 | Nov 8 – Nov 14 | M3 (T-18 → T-21) + M4 (T-22 → T-26) + M5 (T-27 → T-31) | ~27.5 h | **MVP `v0.1.0`**: search → draft → approve in the browser |
| 3 | Nov 15 – Nov 21 | M6 text call (T-54 → T-60) + M7 (T-32 → T-35) | ~22.5 h | A simulated call in the Call tab, with stages, checks and outcome; traces in Langfuse |
| 4 | Nov 22 – Nov 28 | M8 (T-36 → T-41, T-62) + M9 (T-42 → T-45) + T-61 if time is left | ~25 h | **`v1.0.0`**: public Space, eval numbers, README, video |

**Rules for the schedule**
- **Cut rule.** If the MVP does not work in the browser on Nov 14, M6 moves to M10 (stretch) and v1 ships without the call agent.
- **Cut order** when a week overruns: T-61 (voice) → the Gmail-draft half of T-25 → T-34 / T-35 → T-41's improvement (keep the error analysis).
- **Token budget (ADR-018).** Unit tests use fake models. Daily dev runs use `LLM_CACHE_DIR` and the friend's Ollama. Spend the hosted free tiers on extraction checks, the week-4 eval run and the demo. Spread the full eval over 2–3 days.

---

## M0: Foundation

- [x] **T-01 Repo scaffold** · 1.5 h · Deps: —
  - What: uv project (`pyproject.toml`, Python 3.12), `src/partner_agents/` package skeleton per CLAUDE.md layout, ruff + mypy + pytest config (`live` marker registered), `.gitignore` (`.env`, `.qdrant/`, `eval/fixtures/pages/`, `eval/.cache/`, tokens).
  - Done when: `uv sync && uv run pytest` passes with one smoke test, and `uv run ruff check .` and `uv run mypy src` are clean.

- [x] **T-02 Settings** · 1 h · Deps: T-01
  - What: `config.Settings` (pydantic-settings) with every variable in §10. `.env.example` mirrors it with comments.
  - Done when: tests show safe defaults (`OUTREACH_MODE=outbox`, `VOICE_PROVIDER=mock`). An empty `QDRANT_URL` / `MONGODB_URI` selects the local/in-memory backends. `DEMO_MODE=true` overrides outreach and voice.

- [x] **T-03 CI** · 0.5 h · Deps: T-01
  - What: GitHub Actions workflow running ruff, mypy and pytest on push and PR.
  - Done when: the first push shows a green run.

- [x] **T-04 Observability base** · 1 h · Deps: T-02
  - What: JSON log formatter, a `run_id` contextvar, the `stage_timer`/node decorator (§11) and a redaction filter (keys, emails, phones).
  - Done when: tests assert that the timing is recorded, `run_id` appears in the log record, and an email in a log message is redacted.

- [x] **T-05 Retry utility** · 1 h · Deps: T-01
  - What: `utils/retry.py`, an async retry with exponential backoff + jitter, a retryable predicate and `Retry-After` support.
  - Done when: tests cover success after N failures, immediate raise on a non-retryable error, max attempts, and `Retry-After` honoured (time mocked).

## M1: Schemas, LLM layer, repository

- [x] **T-06 Domain schemas** · 1.5 h · Deps: T-01
  - What: `schemas.py` exactly as in §4. Then finish the §11 node decorator (`observability.timed_node`, added in T-04 with timing only): catch unexpected exceptions into a `StageError` in the update's `errors`, and re-raise when `STRICT_NODES=true` (add the setting to `Settings`, `.env.example` and §10).
  - Done when: tests cover the `max_results` bounds, `outreach_eligible` logic, the `ExtractionResult` max length, the `EmailDraft` subject length and JSON round-trip, plus a failing node returning a `StageError` (and raising under `STRICT_NODES`).

- [x] **T-07 LLM factory** · 1.5 h · Deps: T-02, T-05
  - What: `llm.get_chat_model(role)` for `gemini` / `groq` / `ollama` / `fake` with per-role temperatures (§9), plus the global concurrency semaphore.
  - Done when: offline tests with `fake` pass. `pytest -m live` returns a valid `RouteDecision` from the configured provider (note the provider and model in the Progress log).

- [x] **T-08 Structured output helper** · 1 h · Deps: T-07
  - What: `invoke_structured()`, with one retry carrying the validation error and token usage recorded per role.
  - Done when: a test with a fake model returning invalid then valid output succeeds. Invalid twice raises a typed error. Token counts are captured.

- [x] **T-09 Repository (in-memory)** · 2 h · Deps: T-06
  - What: `storage/repo.py`, the `Repository` protocol plus `InMemoryRepository` for leads, outreach, runs, pages (TTL), search cache and profiles (§8.1).
  - Done when: the shared repository test suite (`tests/storage/repo_contract.py`) passes against the in-memory implementation, including TTL expiry with mocked time.

- [x] **T-53 LLM cache, per-role providers, fallback chain** · 2 h · Deps: T-08
  - What: ADR-018 in `llm.py`: `LLM_ROLES` parsing (split on the first `:`), `LLM_FALLBACKS` via `with_fallbacks()` (only on 429 / quota / connection errors), the on-disk response cache behind `LLM_CACHE_DIR`, `OLLAMA_TIMEOUT_S`, the new roles (`classifier`, `caller`, `simulator`, `judge`) and the provider actually used recorded per role. Add the settings to `Settings`, `.env.example` and §10; add `.llm_cache/` to `.gitignore`.
  - Done when: offline tests show `simulator=ollama:qwen2.5:7b` parsed correctly, a fake 429 on the primary served by the fallback (and recorded), a validation error **not** triggering the fallback, and a second identical call served from the cache with 0 model calls. Live: one call to the friend's Ollama over Tailscale (Progress log note with the model).

## M2: Scout (discovery pipeline)

- [x] **T-10 Search providers** · 2 h · Deps: T-06, T-09
  - What: `web/search.py` with the `SearchProvider` protocol, Tavily and DDG, fallback, query templates per partner type, domain dedupe, blocklist and cache (§6.3).
  - Done when: respx tests cover both providers, fallback on error and on empty results, dedupe by registrable domain, blocklisted domains dropped, and a second identical query served from cache.

- [x] **T-11 Safe fetcher** · 3 h · Deps: T-05, T-09
  - What: `web/fetch.py` per §6.4: SSRF guard, robots cache, size and type limits, per-host limit, text assembly (main + footer/contact + `mailto:`/`tel:`), contact-page discovery, bodies stored to the repo.
  - Done when: respx tests cover each status (`blocked_robots`, `timeout`, `http_error`, `too_large`, `non_html`, `ssrf_blocked`), a `mailto:` captured from the footer, a contact page followed only when no contact is found on the first page, and a redirect to a private IP blocked.

- [x] **T-12 Chunker** · 1.5 h · Deps: T-01
  - What: `retrieval/chunking.py`, token-based, with paragraph → sentence → word fallback, overlap and a `has_contact_pattern` flag (including obfuscated emails).
  - Done when: tests show every chunk ≤ `CHUNK_TOKENS`, consecutive chunks overlapping, the chunks jointly covering the text, no split words, and the flag true for `info [at] acme [dot] com`.

- [x] **T-13 Embeddings + vector store** · 2 h · Deps: T-02, T-12
  - What: `retrieval/embeddings.py` (bge-small, batching, query prefix, LRU cache) and `retrieval/store.py` (collections and payload indexes per §8.2, upsert, filtered search, TTL cleanup).
  - Done when: tests on Qdrant `:memory:` with a deterministic fake embedder show that search filtered by `run_id` + `url` returns only that page's chunks and that cleanup removes old points. A live test embeds real text with the right dimension (384).

- [x] **T-14 Evidence selection** · 1 h · Deps: T-13
  - What: `retrieval/evidence.py`, per-page hybrid selection (§6.6).
  - Done when: a test shows a contact-pattern chunk with a low vector score still being selected, the `EVIDENCE_MAX` cap respected, output sorted by `chunk_index`, and `selected_by` set correctly.

- [x] **T-15 Untrusted wrapper + extraction node** · 2 h · Deps: T-08, T-14
  - What: `web/safety.wrap_untrusted()`, `prompts/scout.py` (with `PROMPT_VERSION`), and the `scout_extract` node with bounded concurrency (§6.7).
  - Done when: fake-model tests show the prompt contains the wrapper and the intent, one failing page produces a `StageError` while the others succeed, and an empty `leads` list is accepted.

- [x] **T-16 Contact verification** · 2 h · Deps: T-06
  - What: `verify/contacts.py` per §6.8 (normalisation, de-obfuscation, email and phone checks, dedupe; MX behind a flag).
  - Done when: table-driven tests cover a hallucinated email → `not_in_source`, an obfuscated email → `verified`, a malformed email → `invalid_format`, phone format variants → E.164 `verified`, a phone not on the page → `not_in_source`, and the dedupe keeping the better status.

- [x] **T-17 Scout subgraph + CLI** · 2.5 h · Deps: T-10, T-11, T-15, T-16
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
  - What: the `outreach/voice.py` protocol, the `call_plan` node (voice `OutreachDraft` with the call plan as its body) and a `MockVoiceCaller` stub that marks the approved voice draft `call_created` without starting anything. T-58 makes it create and run a real `CallRecord` (the schema arrives in T-54).
  - Done when: a test shows a voice draft created for "call lead 1", its status `call_created` after approval, and a non-allowlisted number refused even when `VOICE_PROVIDER=vapi`.

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
  - Done when: PRD §10 steps 1–6 and 10 work locally end to end (Progress log note + screenshot in `docs/img/`). Steps 7–9 (the call) are T-60.

- [ ] **T-31 Demo mode** · 1 h · Deps: T-29, T-30
  - What: `DEMO_MODE` forces the outbox and mock voice, caps `max_results` at 5, uses an anonymous per-session `user_id`, pre-fills the sample brand profile and shows a banner.
  - Done when: tests prove every override, and the banner is visible in the UI.

**MVP checkpoint:** tag `v0.1.0` after T-31.

## M6: Call agent (HLD §6.11, ADR-017)

- [ ] **T-54 Call schemas + playbook** · 2 h · Deps: T-06
  - What: the §4.1 models and the `BrandProfile` additions in `schemas.py`; `calls/playbook.py` loading `data/playbooks/<id>.toml` with `tomllib` and validating it (six stages in order, unique rule ids, every `check="code"` rule mapped to a check id); `data/playbooks/partner_intro.toml` **written from scratch** for the fictional sample brand; the sample brand profile with 6–10 `approved_facts`. Add `CALL_PLAYBOOK`, `CALL_MAX_TURNS`, `CALL_REPLY_MAX_WORDS` to `Settings`.
  - Done when: tests load the real playbook, reject a playbook with a missing stage, a duplicate rule id and an unknown check id, and round-trip a `CallRecord` through JSON.

- [ ] **T-55 Stage machine** · 2.5 h · Deps: T-54
  - What: `calls/stages.advance(progress, signals, playbook, brand) -> (CallProgress, Directive)`, a pure function implementing the §6.11 priority list and stage table, plus fact selection.
  - Done when: table-driven tests cover every row of the stage table, opt-out overriding everything in every stage, `asks_if_ai` → `disclose_ai`, retry-once (a second resistance exits), each stop rule → `disqualified`, an already-shared slot confirmed instead of re-asked, `in_rush` skipping optional slots, `wants_other_category` → polite exit, closing waiting on an open question, and `max_turns` → `incomplete`.

- [ ] **T-56 Scheduler** · 1 h · Deps: T-54
  - What: `calls/scheduler.py`: propose the next business day ≥ 24 h ahead on the hour inside 10:00–17:00 brand time; parse "tomorrow 3pm", "Thursday at 11", "next Monday morning"; reject "later" / "sometime".
  - Done when: tests with a fixed clock cover Friday → Monday, a 16:30 call → next day 17:00 not allowed → 10:00 the day after, each parse form, timezone conversion, and vague phrases returning `None`.

- [ ] **T-57 Classifier, caller prompt, reply checks** · 3.5 h · Deps: T-08, T-15, T-55
  - What: `prompts/caller.py` (`PROMPT_VERSION`; classifier and caller prompts built from the playbook stage + universal rules + directive + selected facts); `calls/classify.py` (partner text through `wrap_untrusted(kind="call_partner")`); `calls/checks.py` with every `C-*` check in §6.11; regenerate-once then `fallback_line`.
  - Done when: table-driven tests for each check (an invented "40%" fails `C-NUM`, "guaranteed" fails `C-BAN`, two questions fail `C-ONEQ`, an email address fails `C-PII`, a missing sign-off fails `C-SIGNOFF`); a fake-model test where the first reply fails and the regenerate passes, and one where both fail and the fallback line is used.

- [ ] **T-58 Call sub-graph + Chase wiring + API** · 2.5 h · Deps: T-24, T-26, T-56, T-57
  - What: `graph/call.py` (classify → advance → reply → check → persist, checkpointer keyed by `call_id`); `MockVoiceCaller` now starts real calls; the follow-up confirmation draft; `calls` collection in both repositories; the `/api/calls*` routes (§7). Look up current LangGraph APIs with Context7 first.
  - Done when: offline integration tests with scripted fake models reach `meeting_booked` (with `meeting_at` and a pending confirmation draft), `opted_out` mid-qualification, and `wrong_person` in the opening; `TestClient` covers start → turn → turn → ended and a 404 on an unknown call.

- [ ] **T-59 Persona simulator** · 2 h · Deps: T-58
  - What: `calls/simulator.py` alternating the `simulator` role (persona card + hidden facts + behaviour, "stay in character") with the call sub-graph until the call ends or `max_turns`; the `/simulate` route; 6 demo personas in `data/personas/` (not the eval ones).
  - Done when: an offline test with a scripted fake simulator ends with the expected outcome. Live on the friend's Ollama: 3 demo personas each end in a valid outcome (Progress log note: model, turns, outcomes).

- [ ] **T-60 Call tab (text)** · 2 h · Deps: T-30, T-58, T-59
  - What: a Call tab in `ui/app.py`: pick a lead and persona, Simulate or Live text, transcript with stage chips, per-turn checks (green / regenerated / fallback), the slots panel, the outcome badge and the follow-up draft link. Look up current Gradio APIs with Context7 first.
  - Done when: PRD §10 steps 7–9 work locally (Progress log note + screenshot in `docs/img/`).

- [ ] **T-61 Live voice (nice-to-have, cut first)** · 3 h · Deps: T-60
  - What: accept ADR-021 and add `faster-whisper` + Piper; `calls/speech.py` (transcribe, synthesise, in memory only); a mic input and audio reply in the Call tab behind `SPEECH_ENABLED`; models baked into the Docker image (update T-42).
  - Done when: a live local call by microphone reaches `meeting_booked`, an offline test with a fake speech backend passes, and no audio file is written to disk (test asserts the temp directory is unchanged).

## M7: Safety and observability

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

## M8: Evaluation (see EVAL.md)

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

- [ ] **T-62 Persona dataset + `calls` suite** · 3 h · Deps: T-39, T-59
  - What: `eval/datasets/personas.jsonl` (30, mix per EVAL §3.7), 15 hand-labelled dev transcripts in `call_labels.jsonl`, the `calls` suite with every EVAL §4.7 metric, the judge (role `judge`) with its agreement score, and the `CALL_CONTROL=prompt_only` baseline caller for ablation A8.
  - Done when: schema tests pass (no persona id shared between splits, the archetype mix matches), metric unit tests on hand-computed transcripts pass, and `--suite calls --provider fake` runs offline.

- [ ] **T-40 Baseline + ablations** · 1.5 h · Deps: T-39, T-62
  - What: run every suite with the default config, plus the ablations in EVAL §7 (including A8–A10 for calls).
  - Done when: results are committed in `eval/results/` and EVAL §8 is filled in.

- [ ] **T-41 Error analysis + one improvement** · 1 h+ · Deps: T-40
  - What: categorise the top 10 failures (EVAL §9), ship one targeted fix, re-run.
  - Done when: EVAL §9 shows the categories and the before/after delta for the fix.

## M9: Deploy and ship → v1

- [ ] **T-42 Dockerfile** · 1.5 h · Deps: T-31
  - What: slim image with uv, the embedding model and tiktoken baked in (plus the Whisper and Piper models if T-61 shipped), port 7860, non-root user.
  - Done when: `docker run -p 7860:7860 --env-file .env <img>` serves `/ui` and `/api/health` is ok.

- [ ] **T-43 Cloud services + Space** · 2 h · Deps: T-42, T-27
  - What: a Qdrant Cloud free cluster, Atlas M0, Langfuse Cloud, an HF Space (Docker SDK), secrets, `DEMO_MODE=true`.
  - Done when: the public URL completes the PRD §10 demo script end to end (Progress log note with the URL).

- [ ] **T-44 README** · 2 h · Deps: T-40, T-43
  - What: a one-line pitch, demo link + GIF (discovery **and** a simulated call), architecture diagram (main graph + call sub-graph), the eval table with n/date/model including the A8 "stage machine vs one prompt" result, quickstart, the free-tier setup (Gemini key, optional Groq, optional Ollama over Tailscale), design decisions (links to ADRs), limitations (simulated personas are not real people), the origin note.
  - Done when: a fresh clone following the README quickstart runs offline tests and the CLI with a single Gemini key.

- [ ] **T-45 Demo video + resume bullets** · 1.5 h · Deps: T-44
  - What: a 2–3-minute video of the full demo script (PRD §10, including the call), and resume bullets using **measured** numbers from EVAL §8.
  - Done when: the video link is in the README and every number in the bullets traces to an eval result file.

**v1 checkpoint:** tag `v1.0.0` after T-45.

## M10: Stretch (only after v1)

- [ ] T-46 Vapi voice to an allowlisted, consenting number
- [ ] T-47 MX verification on by default + eval delta
- [ ] T-48 Groq for the router role, with a latency/accuracy comparison in EVAL
- [ ] T-49 Ollama comparison run (for example a friend's machine over Tailscale) added to the provider table
- [ ] T-50 Compound intents ("find and email") through a plan step
- [ ] T-51 SSE streaming endpoint for stage progress
- [ ] T-52 Semantic cache for repeated questions
- [ ] T-63 Hindi / Hinglish playbook and personas (Whisper and Piper both support Hindi)
- [ ] T-64 A second playbook (for example a follow-up call after a meeting) to show the engine is domain-agnostic

---

## Progress log

<!-- One line per completed task: date · task · evidence (test/command/URL) · notes -->

| Date | Task | Evidence | Notes |
|---|---|---|---|
| 2026-10-31 | T-01 | `uv sync && uv run pytest` → 12 passed (`tests/test_smoke.py`); `ruff check .` clean; `mypy src` → no issues in 12 files | uv 0.12.5, CPython 3.12.14, pytest 9.2.0, ruff 0.17.3, mypy 2.4.1. ADR-015 records tooling; `ruff format` excludes `*.md` |
| 2026-10-31 | T-02 | `tests/test_config.py`: `test_safe_defaults`, `test_empty_store_urls_select_local_backends`, `test_demo_mode_overrides_outreach_and_voice`, `test_env_example_loads_with_safe_defaults`; 42 passed, ruff + mypy clean | pydantic 2.14.2, pydantic-settings 2.16.2 (ADR-016). HLD §10: added `QDRANT_LOCAL_PATH`, `KB_MIN_SCORE` start 0.5, Gmail path defaults, validation rules |
| 2026-10-31 | T-03 | First push green: https://github.com/lokifergusion9/cpg-partner-agents/actions/runs/41157481472 (lint, format, mypy, pytest) | `.github/workflows/ci.yml` on push to `main` + PRs, `uv sync --locked`. Actions bumped to `checkout@v7` / `setup-uv@v10.1.0` (Node 24) after a Node 20 deprecation warning |
| 2026-10-31 | T-04 | `tests/test_observability.py`: `test_stage_timer_records_timing_and_logs_it`, `test_timed_node_adds_timing_to_state_update`, `test_run_id_appears_in_log_record`, `test_contacts_are_redacted_at_info`, `test_configured_secrets_are_redacted_at_every_level`; 69 passed, ruff + mypy clean | Added `LOG_LEVEL` (HLD §10). `StageError` capture + `STRICT_NODES` moved into T-06, since `StageError` lives in `schemas.py` |
| 2026-10-31 | T-05 | `tests/utils/test_retry.py`: `test_succeeds_after_n_transient_failures`, `test_non_retryable_error_raises_immediately`, `test_gives_up_after_max_attempts_and_reraises_last_error`, `test_retry_after_seconds_header_is_honoured`, `test_retry_after_http_date_is_honoured` (sleep + clock injected); 110 passed, ruff + mypy clean | No new deps: errors classified by duck typing (`status_code` / `response.status_code`, timeouts). Retry-After above `max_retry_after_s` (60 s) gives up instead of waiting |
| 2026-11-01 | T-06 | `tests/test_schemas.py`: `test_max_results_rejects_out_of_bounds`, `test_outreach_eligible`, `test_extraction_accepts_five_leads_and_rejects_six`, `test_evidence_quote_is_capped_at_300_chars`, `test_email_subject_is_capped_at_80_chars`, `test_json_round_trip` (12 models); `tests/test_observability.py`: `test_failing_node_returns_a_stage_error`, `test_failing_node_is_reraised_under_strict_nodes`; 156 passed, ruff + mypy clean | pydantic 2.14.2. HLD §4 models only; §4.1 call models stay in T-54. New dependency `pydantic[email]` (ADR-022) for `BrandProfile.sender_email`. `STRICT_NODES` added to `Settings`, `.env.example`, HLD §10/§11. `StageError.message` is passed through `redact_contacts`; `CancelledError` is not caught. **T-24:** `GraphInterrupt` derives from `Exception`, so `approval_gate` must not be wrapped as-is (let `timed_node` re-raise it) |
| 2026-11-01 | T-07 | `tests/test_llm.py`: `test_fake_provider_needs_no_keys_or_model_ids`, `test_gemini_uses_the_role_model_and_temperature`, `test_temperature_override_beats_the_role_default`, `test_missing_api_key_is_a_config_error`, `test_llm_slot_bounds_concurrent_calls`; 176 passed, ruff + mypy clean. **Live check pending:** `tests/test_llm_live.py::test_router_returns_a_valid_route_decision` is written but not run (no `.env` / key on this machine) | langchain-core 1.7.3, langchain-google-genai 4.4.2, langchain-groq 1.1.3, langchain-ollama 1.1.0 (docs checked via Context7). Roles `router` / `extractor` / `writer` only; the rest arrive with T-53. SDK retries off (`utils/retry.py` owns them). New setting `LLM_TIMEOUT_S` (HLD §10). `fake` provider replies with a fixed string; structured-output scripting is T-08's job. Gemini 3+ models ignore an unset temperature (library default); we always pass one, so watch for odd router behaviour |
| 2026-11-01 | T-08 | `tests/test_structured.py`: `test_invalid_then_valid_succeeds_after_one_retry`, `test_invalid_twice_raises_a_typed_error`, `test_token_counts_are_captured_per_role`, `test_tokens_of_a_failed_attempt_are_counted_too`, `test_a_rate_limit_never_counts_as_the_validation_retry`; 193 passed, ruff + mypy clean | `invoke_structured(model, schema, messages, *, role)` (the `role` keyword is new, HLD §9 updated). New `fakes.ScriptedChatModel` (scripted replies, records conversations, real-provider `include_raw` contract) also backs `LLM_PROVIDER=fake`. The retry note names failing fields but never echoes input values. Usage lives in a `track_usage()` context block (ContextVar), to be read by the run wrapper in T-21/T-33. Not yet exercised against a real provider's `include_raw` output (T-07 live check will) |
| 2026-11-01 | T-09 | `tests/storage/repo_contract.py` (27 contract tests) run by `tests/storage/test_repo_memory.py::TestInMemoryRepository`, incl. `test_page_expires_after_the_ttl`, `test_search_cache_expires_after_24_hours`, `test_recent_drafts_match_the_recipient_case_insensitively`; 222 passed, ruff + mypy clean | The method-level `Repository` protocol was not in the HLD, so it is now in §8.1 (async, user-scoped reads, read-time expiry on an injectable clock; `calls` group arrives with T-58). New `PageRecord` model (HLD §4). Schema timestamps are now `AwareDatetime` (naive ones would break TTL and window comparisons), HLD §4/§4.1 updated. T-27's Mongo repo must reuse the contract class and honour the clock |
| 2026-11-01 | T-53 | `tests/test_llm_chain.py`: `test_llm_roles_parse_the_documented_example` (`simulator=ollama:qwen2.5:7b`), `test_a_429_that_outlasts_the_retries_is_served_by_the_fallback` (recorded as `b`), `test_a_validation_error_does_not_trigger_the_fallback`, `test_a_second_identical_call_is_served_from_the_cache_with_no_model_call`, `test_a_cache_hit_costs_no_tokens_and_reports_who_answered`; 272 passed, ruff + mypy clean. **Live check pending:** one call to the friend's Ollama over Tailscale (needs the machine, a model and `LLM_ROLES`) | `get_chat_model` now returns a `ModelChain` (HLD §9 rewritten). New roles `classifier` / `caller` / `simulator` / `judge`, `LLM_ROLES`, `LLM_FALLBACKS`, `LLM_CACHE_DIR`, `OLLAMA_TIMEOUT_S`, `LLM_MODEL_<ROLE>` for the four new roles; `.llm_cache/` gitignored. The fallback is a predicate walk inside `invoke_structured`, not `with_fallbacks()` (note added to ADR-018). Providers per role come from `track_providers()` next to `track_usage()`. Only structured calls use the chain and cache; a plain-text `invoke_text` with the same behaviour is to be added by the first node that needs it (kb_answer, T-20; call writer, T-57). Groq clamps temperature 0 to 1e-8 |
| 2026-11-01 | T-10 | `tests/web/test_search.py` (62 tests): `test_tavily_sends_the_documented_request_and_parses_results`, `test_falls_back_when_the_primary_errors`, `test_falls_back_when_the_primary_returns_nothing`, `test_merge_keeps_the_best_rank_per_registrable_domain`, `test_merge_drops_blocklisted_domains_and_caps_the_count`, `test_a_second_identical_query_is_served_from_the_cache`; 334 passed, ruff + mypy clean | Tavily REST docs checked via Context7 (Bearer auth, `POST /search`); `ddgs` 9.16.1. **Deviation:** `ddgs` uses `primp`, not httpx, so respx covers Tavily only; DDG is tested through an injected search function. New deps `httpx`, `ddgs`, `tldextract` (ADR-023), dev `respx`; ADR-006 note. Provider-level retry uses `search.RETRY_POLICY`. `scout_search` node wiring is T-17; this task delivers `SearchService.discover(intent)` and `build_providers(settings)`. Not run against live Tavily/DDG yet |
| 2026-11-01 | T-11 | `tests/web/test_fetch.py`: `test_robots_disallow_blocks_the_page`, `test_timeout`, `test_http_error_for_a_404`, `test_too_large_by_content_length`, `test_non_html`, `test_private_ip_literals_are_blocked`, `test_a_redirect_to_a_private_ip_is_blocked`, `test_ok_page_is_stored_with_main_and_contact_text` (footer `mailto:`), `test_contact_page_is_followed_when_the_first_page_has_no_contact`, `test_contact_page_is_not_followed_when_the_page_already_has_a_contact`, `test_at_most_two_requests_per_host_in_flight`; plus `test_html.py`, `utils/test_contacts.py`; 470 passed, ruff + mypy clean | New `web/html.py`, `utils/contacts.py` (also T-12's contact-pattern base). trafilatura 2.3.1 + lxml (ADR-024); `lxml` mypy override. Fixed a T-05 gap: `is_retryable` now matches transport errors by class name anywhere in the MRO (`httpx.ReadTimeout`). **Open:** DNS-rebinding window (resolve-then-connect) documented in HLD §12; `/security-review` is due before relying on `fetch.py` (CLAUDE.md), planned with M7. Public-suffix dedupe treats unknown TLDs such as `.example` as exact hosts, so tests use real TLDs where it matters |
| 2026-11-04 | T-12 | `tests/retrieval/test_chunking.py`: `test_every_chunk_fits_the_limit`, `test_consecutive_chunks_overlap`, `test_chunks_jointly_cover_the_text`, `test_no_word_is_split`, `test_chunks_with_a_contact_pattern_are_flagged` (`info [at] acme [dot] com`), `test_invariants_hold_for_random_texts` (25 seeds); live: `test_tiktoken_counter_chunks_real_text` passed with tiktoken 0.14.0 `cl100k_base`; 548 passed, ruff + mypy clean | tiktoken added (ADR-025). The counter is injected, so offline tests use a word-count fake. Chunks are contiguous slices with offsets; an unbroken string over the limit is the only thing cut mid-word. Contact flag reuses `utils/contacts.py` from T-11 |
| 2026-11-04 | T-13 | `tests/retrieval/test_store.py`: `test_search_filtered_by_run_and_url_returns_only_that_pages_chunks`, `test_search_never_crosses_runs`, `test_cleanup_removes_old_points_and_keeps_new_ones`, `test_indexing_a_page_twice_replaces_its_points`; `test_embeddings.py`: batches of 32, LRU cache, query prefix; live `test_real_bge_small_returns_384_dimensional_normalised_vectors` passed; 581 passed, ruff + mypy clean | qdrant-client (docs checked via Context7) + sentence-transformers 6.2.0 (torch 2.15.0 pulled in). `fakes.HashingEmbedder` keeps store tests offline. `VectorStore.cleanup(older_than)` is the T-35 hook. Local Qdrant ignores payload indexes (warning silenced). No `scroll`-by-contact-flag method yet: T-14 adds it |
| 2026-11-04 | T-14 | `tests/retrieval/test_evidence.py`: `test_a_contact_chunk_with_a_low_vector_score_is_still_selected`, `test_the_total_never_exceeds_the_cap`, `test_contact_chunks_win_when_the_cap_forces_a_choice`, `test_the_result_is_sorted_by_chunk_index`, `test_selected_by_is_set_per_chunk`; 596 passed, ruff + mypy clean | `VectorStore.search` gained `contact_only` / `exclude` instead of a scroll method, so contact chunks are ranked by similarity too. Cap rule (contact chunks survive, weakest vector chunks go first) added to HLD §6.6. A footer already inside the vector top-k is treated as a vector chunk and can be dropped by a tight cap |
| 2026-11-04 | T-15 | `tests/graph/test_scout_extract.py`: `test_the_prompt_wraps_the_page_text_and_states_the_intent`, `test_page_text_cannot_leave_the_untrusted_block`, `test_one_failing_page_is_an_error_and_the_others_succeed`, `test_an_empty_lead_list_is_accepted`, `test_calls_run_together_but_stay_inside_the_concurrency_limit`; `tests/web/test_safety.py` (marker defusing); 629 passed, ruff + mypy clean | langgraph 1.2.19 added (docs checked via Context7: `StateGraph`, `Annotated` reducers, compiled subgraph as a node). New `graph/state.py` (HLD §5.1). **Contract change:** `state["candidates"]` is `dict[url, list[ExtractedLead]]` (HLD §5.1/§6.7), since a candidate must stay tied to its source page for T-16. Node built by `make_scout_extract(model)`; the extractor model comes from `get_chat_model("extractor")` in T-17. `/security-review` is due for `web/safety.py` before relying on it (CLAUDE.md), planned with M7 / T-32. Caught and fixed: invisible characters had been written literally into the regex; they are explicit `\u` escapes now |
| 2026-11-04 | T-16 | `tests/verify/test_contacts.py` (91 tests): `test_a_hallucinated_email_is_not_in_source`, `test_an_email_on_the_page_is_verified_even_when_obfuscated`, `test_a_malformed_email_is_invalid_format`, `test_phone_format_variants_verify_and_become_e164`, `test_a_phone_not_on_the_page_is_not_in_source`, `test_the_better_verified_duplicate_wins`, `test_a_domain_without_mx_is_mx_failed`; 720 passed, ruff + mypy clean | phonenumbers 9.0.42 (ADR-026); email-validator in syntax-only mode (ADR-022). Stricter than the HLD text in two places, both recorded in HLD §6.8: whole-address matching for emails, and phone digits matched per phone-looking run. A bare `CA` in a location is not guessed as a region. Same tool quirk as T-15: `\u` escapes written through the editor become literal characters, so tests build them with `chr()` |
| 2026-11-04 | T-17 | `tests/graph/test_scout_graph.py`: `test_verified_and_rejected_leads_come_out_of_the_pipeline` (respx pages + fake search + Qdrant `:memory:` + scripted extractor: 1 verified lead, 1 invented email rejected as `not_in_source`, 1 phone-only lead), `test_pages_that_cannot_be_read_are_reported_and_skipped`, `test_every_node_records_its_timing`, `test_only_readable_pages_reach_the_model_and_page_text_is_wrapped`, `test_no_search_results_ends_early_with_an_explanation`; `tests/test_cli.py` (24); 757 passed, ruff + mypy clean. **Smoke run, real network, fake LLM** (`LLM_PROVIDER=fake`, DDG): `cli scout "cookie brokers in California" --max-results 3` → 2 pages read, 1 blocked by robots.txt (a Bing ad link), search 6.7 s, fetch 2.6 s, index 50.4 s (first load of bge-small + CPU embedding), evidence 0.2 s. **Live check pending:** the "≥ 3 verified leads on 2 queries" run needs a real extractor model (Gemini/Groq/Ollama key); record queries, counts and timings here when done | Search engines (`bing.com`, `google.com`, `duckduckgo.com`) added to the built-in blocklist after DDG returned an ad-click redirect. Node deps travel in `ScoutDeps`; `add_node` is called through a `cast(Any)` because LangGraph's overloads reject a plain async callable over a TypedDict state under strict mypy. The keyword query parser in the CLI is temporary (T-19/T-21 bring the router). First-run index time is dominated by model load: worth a warm-up in the API startup (T-28) |

## Blockers / decisions needed

<!-- Anything waiting on a human decision. Remove once resolved (and record an ADR if it was a design choice). -->
- **Next up: T-18** (knowledge-base corpus + ingest; M2 is code-complete). Run `/code-review` and `/simplify` for the M2 milestone (CLAUDE.md) before M3.
- **T-17 live check (open).** Put a real extractor key in `.env` (`LLM_PROVIDER` + `LLM_MODEL_EXTRACTOR`, plus `TAVILY_API_KEY` ideally), run `uv run python -m partner_agents.cli scout "cookie brokers in California"` and a second query, and record queries, verified-lead counts (target ≥ 3 each) and timings in the T-17 Progress log row.
- **Not pushed.** T-13 → T-17 are committed locally on `main`; push only when the user says so. Week 1 started 2026-11-01. Commits for T-13 → T-17 are local only until the user says to push.
- **T-53 live check (open).** Needs the friend's Ollama reachable over Tailscale with a pulled model: set `OLLAMA_BASE_URL`, `LLM_ROLES=simulator=ollama:<model>`, run one call and record the model in the T-53 Progress log row.
- **T-07 live check (open).** Create `.env` with `LLM_PROVIDER`, a key and `LLM_MODEL_ROUTER`, then run `uv run pytest -m live` and record the provider and model in the T-07 Progress log row.
- **Plan change 2026-10-31.** Added the call agent (M6, T-53 → T-62; ADR-017 → ADR-021), renumbered M6–M9 to M7–M10, and moved to a 4-week schedule. PRD, HLD, EVAL and CLAUDE.md updated together.
- **Friend's Ollama machine (open).** Confirm the GPU and VRAM, then pick the `simulator` / `judge` models (7B–8B for 6–8 GB VRAM, 14B for 12 GB+). Needed by T-53's live check.
- **Commit hook scope (open).** `.claude/settings.json` gates `git commit` with ruff/mypy/pytest, but its `if: Bash(git commit*)` condition is not applied, so the checks run before every Bash call (harmless while green, slower). Decide whether to move the condition into the hook script.
- **Retry-After ceiling (open).** `utils/retry.py` gives up when `Retry-After` > 60 s instead of waiting (keeps runs inside the 45 s p95 budget). Confirm or change.
