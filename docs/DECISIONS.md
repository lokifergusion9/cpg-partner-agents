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
- **Status:** Accepted · 2026-10-30
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
