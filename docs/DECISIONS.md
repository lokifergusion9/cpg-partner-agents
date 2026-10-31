# DECISIONS: Architecture Decision Records

Short records of *why* the system is built this way. Each ADR doubles as an interview answer ("Why X over Y?").

**Rules**
- Add an ADR **before** adding a dependency or reversing a choice. Never delete an ADR. Supersede it (`Status: Superseded by ADR-0xx`).
- Keep each one short: context, decision, alternatives, consequences.
- If an eval result motivates or revisits a decision, link it.

**Template**

```markdown
## ADR-0xx: <title>
- **Status:** Proposed | Accepted | Superseded by ADR-0yy · <date>
- **Context:** what problem / constraint forces a choice
- **Decision:** what we chose
- **Alternatives:** what else, and why not
- **Consequences:** trade-offs we accept; how we'd know it was wrong
```

---

## ADR-001: LangGraph for orchestration
- **Status:** Accepted · 2026-10-30
- **Context:** Three agents need routing, shared state, per-thread memory and a human-approval pause in the middle of a run. A client-facing product needs predictable control flow.
- **Decision:** LangGraph `StateGraph` with conditional edges from Charlie, a checkpointer for memory, and `interrupt()` for approval.
- **Alternatives:**
  - CrewAI / AutoGen favour autonomous agent-to-agent conversation, which is harder to make deterministic and to test.
  - Plain LangChain agents (a ReAct loop) leave tool selection to the model each step.
  - A hand-rolled state machine would mean rebuilding checkpointing and interrupt/resume.
- **Consequences:** The control flow is explicit and testable node by node, and the graph diagram in HLD §5 is the real system. We accept the framework dependency and its API churn (pin versions).

## ADR-002: Provider-agnostic LLM layer, Gemini by default
- **Status:** Accepted · 2026-10-30
- **Context:** The project must run free. Free tiers change, and local models (Ollama) are useful for development and comparison.
- **Decision:** All calls go through `llm.get_chat_model(role)` using LangChain chat-model integrations. The default provider is Gemini (free tier, good structured output). Groq and Ollama are selectable by env. Model ids are config, not code.
- **Alternatives:**
  - Calling one provider's SDK directly gives lock-in and makes the provider ablation (EVAL A6) impossible.
  - Hugging Face Inference was rejected because its free quota is small for this call volume. HF is used for hosting instead (ADR-010).
- **Consequences:** One place to change providers, plus a quality/latency comparison table. We accept the lowest-common-denominator structured-output behaviour, mitigated by validate + retry (ADR-008).
- **Note:** Free-tier input may be used by providers to improve their products. Only public data is used (PRD non-goals).
- **Packages (added 2026-11-01, T-07):** `langchain-core`, `langchain-google-genai`, `langchain-groq`, `langchain-ollama`. Each provider SDK is imported lazily inside `llm.py` only.

## ADR-003: Local embeddings (`BAAI/bge-small-en-v1.5`)
- **Status:** Accepted · 2026-10-30
- **Context:** Each discovery run embeds 50–200 chunks. API embeddings would consume free-tier quota and add network latency and a failure point.
- **Decision:** Run bge-small (384-d) locally through sentence-transformers, baked into the Docker image. Use the BGE query prefix for queries.
- **Alternatives:**
  - Gemini / OpenAI embeddings: better quality at the margin, but quota, latency and cost.
  - Larger local models (bge-base/large): slower on free CPU hosting.
- **Consequences:** No quota or cost for embeddings, deterministic, and it works offline. If evidence recall (EVAL §4.3) misses its target, try bge-base before switching to an API.

## ADR-004: Qdrant as the vector store
- **Status:** Accepted · 2026-10-30
- **Context:** We need per-run and per-URL filtering of chunks (ADR-007), a free hosted option, and a zero-setup local mode for tests.
- **Decision:** Qdrant. Local / `:memory:` mode in development and tests, Qdrant Cloud free cluster in production. Payload indexes on `run_id` and `url`.
- **Alternatives:**
  - Pinecone is managed-only, so there is no local mode for offline tests.
  - Chroma has weaker filtering and ops story for a hosted deployment.
  - pgvector would need Postgres alongside Mongo.
  - FAISS has no payload filtering or persistence service.
- **Consequences:** The same client code runs locally and in the cloud, and payload filtering is first-class. Choosing Qdrant means operating one more service (the free tier mitigates this).

## ADR-005: MongoDB for application data and checkpoints
- **Status:** Accepted · 2026-10-30
- **Context:** Leads, drafts, runs and conversation checkpoints are document-shaped and evolve quickly. Free hosting is needed.
- **Decision:** MongoDB Atlas M0 behind a `Repository` protocol, with an in-memory implementation for development and tests. The LangGraph MongoDB checkpointer is used for thread memory. TTL indexes handle data retention.
- **Alternatives:**
  - Postgres: a stronger relational model, but more migration overhead for a fast-changing schema.
  - Redis: not durable enough on free tiers.
  - SQLite: not available as a hosted free service for HF Spaces persistence.
- **Consequences:** Schemaless storage with Pydantic as the schema, and native TTL for privacy. If the checkpointer integration proves unstable, fall back to persisting messages through the repository (record it as a new ADR).

## ADR-006: Tavily as primary search, DuckDuckGo as fallback
- **Status:** Accepted · 2026-10-30
- **Context:** Scout needs live web results through an API, at no cost.
- **Decision:** Tavily (free monthly credits, built for agents) as the primary provider. DuckDuckGo (`ddgs`, unofficial) on error or empty results. Results are cached for 24 h.
- **Alternatives:**
  - SerpAPI: the free quota is small.
  - Google Custom Search: setup friction and quota.
  - Perplexity: an answer engine, not raw results; we want URLs to fetch and verify ourselves.
- **Consequences:** Two independent sources reduce outages. The DDG library is unofficial and may rate-limit or break, which is acceptable for a fallback. Check current free limits when setting up.

## ADR-007: Per-page hybrid evidence selection
- **Status:** Accepted · 2026-10-30
- **Context:** Contact details often sit in footers or on contact pages, which semantic search ranks low. A global top-k across 10 pages lets one long page crowd out the others, and leads must be attributable to one source.
- **Decision:** For each page, select the vector top-k (filtered by `run_id` + `url`) plus up to 3 chunks with contact patterns, cap the total at 6, and order by position (HLD §6.6). Run one extraction call per page.
- **Alternatives:**
  - Global top-k: crowding, and weak attribution.
  - Whole page into the LLM: context and cost blow-up, and more hallucination.
  - Regex-only: misses company name and context.
- **Consequences:** Better recall and clean source attribution, at the cost of more LLM calls (bounded by `max_results`). Validated by ablation A1.

## ADR-008: Pydantic-structured output with validate-and-retry
- **Status:** Accepted · 2026-10-30
- **Context:** Free-text LLM output cannot safely drive automation (persisting leads, drafting outreach).
- **Decision:** `with_structured_output(PydanticModel)`, then validation. On failure, retry once with the validation error appended. A second failure becomes a `StageError` (the page is skipped, the run continues).
- **Alternatives:**
  - Regex parsing of free text: brittle.
  - JSON mode without schema validation: valid JSON with wrong shape.
  - Unlimited retries: they burn quota.
- **Consequences:** Typed objects everywhere downstream. The validation failure rate is visible in traces.

## ADR-009: Deterministic contact verification (no LLM judge)
- **Status:** Accepted · 2026-10-30
- **Context:** The worst failure is an invented email that Chase then contacts. An LLM checking another LLM has correlated errors.
- **Decision:** A contact is `verified` only if it is syntactically valid **and** literally present in the normalised (de-obfuscated) source text. Phones are matched by digit sequence. MX lookup is optional. Only verified emails are eligible for outreach.
- **Alternatives:**
  - An LLM self-check: not deterministic, and its errors correlate with the extractor's.
  - Trusting the extractor's "only from context" instruction: insufficient on its own.
- **Consequences:** Eligible-hallucination is 0 by construction, which the eval checks. Some legitimate contacts are rejected (for example ones rendered by JavaScript or shown only in images), which costs recall. This is a deliberate precision-over-recall trade.

## ADR-010: Single process (FastAPI + Gradio) on Hugging Face Spaces
- **Status:** Accepted · 2026-10-30
- **Context:** We need a free public demo and an API, without running a separate frontend.
- **Decision:** Mount Gradio on FastAPI (`/ui`, `/api`) in one Docker container on an HF Space (Docker SDK, port 7860), with one worker.
- **Alternatives:**
  - A separate frontend on Vercel + an API elsewhere: two deploys and CORS.
  - Render free: workable, but it sleeps and has a tighter build budget.
  - Azure: needs credits or a card.
- **Consequences:** One deploy and one URL. We accept cold starts on the free tier, a single-worker throughput limit (protected by rate limits), and external Qdrant/Mongo for persistence.

## ADR-011: Token-based chunking at ~1000 tokens with 150 overlap
- **Status:** Accepted · 2026-10-30
- **Context:** Chunks must keep an entity and its contact details together while staying precise for retrieval.
- **Decision:** A recursive splitter (paragraph → sentence → word) sized with tiktoken `cl100k_base` as a provider-neutral proxy, `CHUNK_TOKENS=1000`, `CHUNK_OVERLAP=150`. It never splits inside a word.
- **Alternatives:**
  - Character-based: sizes drift relative to tokens.
  - The provider's exact tokenizer: no single provider in this stack.
  - Semantic chunking: more compute for unclear benefit on short business pages.
- **Consequences:** Sizes are approximate for Gemini, which is fine for the budget. Ablation A2 picks the final size.

## ADR-012: Human approval through a LangGraph interrupt; outbox by default
- **Status:** Accepted · 2026-10-30
- **Context:** Automated outreach carries legal (anti-spam, telemarketing), reputational and safety risk. Scraped content could try to trigger it.
- **Decision:** `approval_gate` calls `interrupt()`, and the run resumes only through the approve endpoint with explicit per-draft decisions. The default dispatch is `outbox` (nothing leaves). `gmail_draft` uses user OAuth with no stored password. `gmail_send` is limited to an allowlist. Demo mode forces the outbox.
- **Alternatives:**
  - Auto-send with a daily cap: unacceptable risk.
  - SendGrid: a third-party sender lowers trust and deliverability for B2B outreach.
  - SMTP with an app password: a stored credential.
- **Consequences:** It is slower than full automation by design. The approval record satisfies PRD G3.

## ADR-013: Voice calls mocked by default
- **Status:** Superseded by ADR-017 · 2026-10-31 (the mock-by-default and Vapi-allowlist decisions carry over; the "not core" context does not)
- **Context:** Real AI voice calls cost money after trial credits and carry consent and telemarketing obligations. They are not core to the portfolio story.
- **Decision:** A `VoiceCaller` protocol with `MockVoiceCaller` as the default. `VapiVoiceCaller` is a stretch task, limited to allowlisted, consenting numbers.
- **Alternatives:**
  - Real Vapi from the start: risk to the timeline and cost.
  - Twilio + our own speech-to-text/text-to-speech: far more work.
- **Consequences:** The interface is demonstrated without risk, and the real integration can be added later without touching the graph.

## ADR-014: Offline-reproducible evaluation (snapshots + LLM response cache)
- **Status:** Accepted · 2026-10-30
- **Context:** The live web changes and free-tier quotas are limited. Eval numbers must be reproducible and cheap to re-run.
- **Decision:** Snapshot the extraction pages (gitignored, sha256 committed) and serve them through respx. Cache LLM responses by `(provider, model, prompt_version, messages hash)`. Report live-web E2E separately with no accuracy claims.
- **Alternatives:**
  - Live-only eval: not reproducible.
  - Committing third-party HTML: copyright and privacy concerns.
- **Consequences:** Reruns are deterministic except for model nondeterminism at temp 0. Others cannot fully reproduce the eval without re-snapshotting, which is acknowledged in the README.

## ADR-015: Project tooling (uv, uv_build, pytest, ruff, mypy)
- **Status:** Accepted · 2026-10-31
- **Context:** T-01 needs a package manager, a build backend and the quality gates that CLAUDE.md requires before every commit (ruff, mypy, pytest).
- **Decision:** uv manages Python 3.12 and the lockfile (`uv.lock`, committed). `uv_build` is the build backend for the `src/` layout. Dev-only group: pytest, ruff (lint + format), mypy in `strict` mode. pytest deselects `live` tests by default (`-m "not live"`), so `uv run pytest` is always offline; `pytest -m live` overrides it. `ruff format` skips `*.md`, because ruff also formats code blocks inside Markdown and the HLD's schema listings are kept exactly as written. Test-only helpers (respx, pytest-asyncio) are added in the task that first needs them.
- **Alternatives:**
  - Poetry / pip-tools: slower, and uv is already the stated stack.
  - hatchling: works, but `uv_build` needs no extra config for a plain `src/` package.
  - black + isort + flake8: three tools where ruff is one.
  - mypy non-strict: lets untyped code in, against "full type hints" in CLAUDE.md.
- **Consequences:** One tool installs, locks and runs everything. Strict mypy means third-party libraries without stubs need explicit `ignore_missing_imports` overrides, added per library with a comment.

## ADR-016: pydantic-settings for configuration
- **Status:** Accepted · 2026-10-31
- **Context:** T-02 needs one typed place that reads env vars (CLAUDE.md: `config.py` is the only reader), with validation, secret masking and `.env` support.
- **Decision:** `pydantic` v2 and `pydantic-settings` (`BaseSettings`), which brings `python-dotenv` for `.env` files. Secrets are `SecretStr`. Comma-separated lists use `NoDecode` plus a splitting validator, so they are not parsed as JSON. Safety rules live in a model validator (`DEMO_MODE` override, `gmail_send` needs an allowlist). The pydantic mypy plugin is enabled. `get_settings()` caches one instance per process; tests build `Settings(_env_file=None)` and an autouse fixture strips settings variables from the environment.
- **Alternatives:**
  - `os.environ` + hand-written parsing: no validation or typing, easy to read env vars from anywhere.
  - dynaconf / environs: another config model next to the Pydantic models used everywhere else.
- **Consequences:** Invalid config fails at startup with a clear message instead of misbehaving at run time. Every new variable must be added to `Settings`, `.env.example` and HLD §10; `test_env_example_lists_every_setting` catches a missing `.env.example` entry.

## ADR-017: Stage-controlled call agent driven by a playbook file
- **Status:** Accepted · 2026-10-31 · supersedes ADR-013
- **Context:** A text email table is a weak demo, and "an LLM that follows a sales script" is the hard, hireable problem: a free-running prompt drifts, skips steps, invents numbers and ignores opt-outs. We want Chase to run a first partner-discovery call that books a meeting with the brand founder, while staying as controllable and measurable as the rest of the system.
- **Decision:**
  - The call is a **LangGraph sub-graph** (HLD §6.11). Each counterpart turn runs `classify → advance → reply → check`.
  - The **stage machine is code, not prompt.** Six stages (opening → discovery → qualification → pitch → handoff → closing) and every transition, stop rule, retry-once rule and outcome live in a pure function (`calls/stages.py`). The LLM only classifies the counterpart's turn into typed `TurnSignals` and writes the next line for the current stage.
  - Stage goals and **must / never rules live in a playbook file** (`data/playbooks/*.toml`, every rule has a stable id). Prompts are built from the playbook, and the checker reports violations by rule id.
  - Hard rules are checked **deterministically** after every reply (numbers only from approved facts, banned phrases, one question per turn, no echo of non-disclosed data). A violating reply is regenerated once, then replaced by a safe stage fallback line.
  - The playbook uses **TOML**, read with the standard-library `tomllib`. No new dependency.
  - `VOICE_PROVIDER=mock` (default) runs the call against a simulated counterpart or a human typing/speaking in the UI. `vapi` stays a stretch goal, limited to allowlisted, consenting numbers (as in ADR-013).
- **Alternatives:**
  - One big system prompt with the whole script: no guarantee of order or stop rules, and violations are not attributable.
  - An LLM picks the next stage itself: non-deterministic and hard to test.
  - YAML playbook: friendlier syntax, but needs PyYAML. TOML is in the standard library and expressive enough for lists of rules.
  - Real telephony from the start: cost and consent risk (ADR-013).
- **Consequences:** Every call is reproducible stage by stage, the transition logic gets table-driven tests, and "rule-following rate" becomes a measured number (EVAL §4.7). The engine is domain-agnostic: a new playbook file is a new kind of call. We accept a slower, more scripted conversation than a free-form agent.

## ADR-018: Free-tier token strategy (remote Ollama, per-role providers, fallback chain, response cache)
- **Status:** Accepted · 2026-10-31
- **Context:** Free tiers hit their daily limits during development, and the persona simulator plus the call eval multiply the call volume. A friend's GPU machine running Ollama is reachable over Tailscale.
- **Decision:**
  - **Per-role provider override.** `LLM_PROVIDER` stays the default; `LLM_PROVIDER_<ROLE>` overrides it for one role (for example `simulator` and `judge` on Ollama, `extractor` on Gemini).
  - **Fallback chain.** `LLM_FALLBACKS` (for example `groq,gemini`) wraps each role's model with LangChain `with_fallbacks()`, tried on 429 / quota / connection errors after the retry budget.
  - **Dev response cache.** `LLM_CACHE_DIR` (default empty = off) caches responses on disk keyed by `(provider, model, prompt_version, sha256(messages), temperature)`; the eval cache (ADR-014) uses the same code.
  - **Remote Ollama** through `OLLAMA_BASE_URL` on the Tailscale IP. Ollama is never exposed to the public internet (it has no auth). Timeout `OLLAMA_TIMEOUT_S` (60 s).
  - Unit tests stay on fake models; daily development uses the cache and Ollama; Gemini/Groq are used for extraction quality, the final eval run and the public demo.
- **Alternatives:**
  - One paid provider: violates PRD G6.
  - A routing proxy such as LiteLLM: another service and dependency for what `with_fallbacks()` already does.
  - Local Ollama on the dev laptop: CPU-only (no discrete GPU), too slow for 7B+ models.
- **Consequences:** Daily limits stop blocking work. Results now depend on which provider answered, so every `RunRecord` and eval report records the provider and model that actually served each role. The friend's machine can be offline, so nothing may require it.

## ADR-019: The call agent always discloses that it is an AI
- **Status:** Accepted · 2026-10-31
- **Context:** Many sales-call scripts tell the caller not to reveal it is an AI unless asked. Several jurisdictions require or are moving toward AI-call disclosure, and hiding it undermines trust in a portfolio project about safe agents.
- **Decision:** The opening line states that the caller is an AI assistant calling on behalf of the brand, and any "are you a bot?" is answered truthfully. Both are playbook rules (`U-AI-1`, `U-AI-2`) checked by the call eval.
- **Alternatives:** Disclose only when asked: legally and ethically weaker, and gains little in a demo.
- **Consequences:** Some simulated counterparts may end the call earlier; that is reported, not tuned away.

## ADR-020: LLM judge for soft call rules, calibrated against human labels
- **Status:** Accepted · 2026-10-31
- **Context:** Some playbook rules can be checked in code (numbers, banned phrases, stage order, outcome); others cannot (empathy before resuming, not being pushy, answering the actual question).
- **Decision:** Deterministic checks are the primary gate. Soft rules are scored by an LLM judge (role `judge`, temp 0, a different model from the caller where possible) that returns per-rule pass/fail with a quoted turn. The judge is calibrated on 15 hand-labelled transcripts from the dev split, and its agreement with the human labels is reported next to every judged metric.
- **Alternatives:**
  - Human review of every transcript: does not scale to ablations.
  - Judge-only scoring: correlated errors with the caller, and no anchor.
- **Consequences:** Soft-rule numbers carry an explicit agreement figure. If agreement on the dev labels is < 0.8, those numbers are reported as indicative only.

## ADR-021: Local speech for the voice demo (faster-whisper + Piper)
- **Status:** Proposed · 2026-10-31 (accept when T-61 starts)
- **Context:** The call demo should work with a microphone in the browser, at zero cost, without a phone line.
- **Decision:** Speech-to-text with `faster-whisper` (MIT, `base` on CPU, `small` on the friend's GPU); text-to-speech with Piper (local, CPU). Gradio audio components handle the browser side. Audio is processed in memory and never stored.
- **Alternatives:**
  - OpenAI Whisper API / ElevenLabs / Deepgram: paid after credits, and they add keys to manage.
  - Browser Web Speech API: free, but Chrome-only and not testable server-side.
  - `edge-tts`: free but unofficial and can break without notice.
- **Consequences:** Two new dependencies and a larger image (models baked at build time). Speech is slower on the free CPU Space; the demo says so. Piper's newer releases are GPL-3.0 and each voice has its own licence; the README lists the voice and its licence.

## ADR-022: email-validator for `EmailStr` and contact verification
- **Status:** Accepted · 2026-11-01
- **Context:** `BrandProfile.sender_email` is an `EmailStr` (HLD §4), and `verify/contacts.py` rejects malformed addresses with `invalid_format` (HLD §6.8). Both need a real RFC-aware syntax check, offline. Pydantic's `EmailStr` imports the `email-validator` package.
- **Decision:** Depend on `pydantic[email]` (which pulls `email-validator`). Syntax checks only (`check_deliverability=False`), so no DNS lookup unless `VERIFY_MX` is on.
- **Alternatives:**
  - A hand-written email regex: misses quoting, IDN and length rules.
  - `str` for `sender_email`: lets a typo through to the Gmail API at dispatch time.
- **Consequences:** One small pure-Python dependency, plus `dnspython` and `idna` as its transitive dependencies.
