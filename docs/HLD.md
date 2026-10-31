# HLD: CPG Partner Agents

| | |
|---|---|
| Status | Draft v1 |
| Last updated | 2026-10-30 |
| Related | [PRD](PRD.md) · [TASKS](TASKS.md) · [EVAL](EVAL.md) · [DECISIONS](DECISIONS.md) |

This document is the **contract**. Schemas (§4), graph state (§5), API (§7) and storage (§8) change only together with this file.

---

## 1. Design principles

1. **Constrain the LLM; don't trust it.** The graph does explicit routing, schemas constrain output, and contacts are verified deterministically.
2. **Ground everything.** Answers cite the knowledge base. Every lead links to a source URL and a verbatim evidence quote.
3. **Humans hold the trigger.** No outreach happens without a recorded approval.
4. **Degrade, don't crash.** Per-page and per-lead failures are recorded and the run continues.
5. **Free and portable.** Every external service has a local or in-memory fallback, and providers are switched by config.
6. **Measured.** Every quality claim maps to an eval metric.

## 2. System context

```
                    ┌──────────── Hugging Face Space (Docker, port 7860) ─────────────┐
   Browser ────────►│  Gradio UI (/ui) ──► FastAPI (/api) ──► LangGraph app            │
                    │                                         Charlie ─► Scout | Chase  │
                    └───────┬───────────┬───────────┬──────────┬──────────┬──────┬────┘
                            │           │           │          │          │      │
                     LLM provider   Search API   Websites    Qdrant    MongoDB  Langfuse
                     Gemini (def.)  Tavily       (fetch,     Cloud     Atlas    (traces,
                     Groq / Ollama  DDG fallback  robots)    (free)    M0       optional)
                                                                         │
                                                       Gmail API (optional, user OAuth)
                                                       Vapi (stretch, allowlist only)
```

Local development uses the same code with Qdrant local mode, the in-memory repository, and an optional Ollama on another machine (`OLLAMA_BASE_URL`).

## 3. Repository layout

See `CLAUDE.md` → Layout. Dependency direction:
`api/ui → graph → (web, retrieval, verify, outreach, kb) → (llm, storage, schemas, config, observability, utils)`.
Lower layers never import higher ones.

## 4. Domain models (`schemas.py`)

```python
PartnerType = Literal["retailer", "distributor", "broker", "manufacturer", "other"]
Intent = Literal["cpg_qa", "discovery", "outreach", "clarify", "out_of_scope"]
VerificationStatus = Literal["verified", "not_in_source", "invalid_format", "mx_failed", "unchecked"]

class DiscoveryIntent(BaseModel):
    partner_type: PartnerType
    product_category: str                  # "snack foods", "cookies"
    location: str | None = None            # "California, US"
    max_results: int = Field(10, ge=1, le=25)
    raw_query: str

class RouteDecision(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    discovery: DiscoveryIntent | None = None   # set iff intent == "discovery"
    outreach_goal: str | None = None           # set iff intent == "outreach"
    lead_selector: str | None = None           # "top 3", "the LA ones"; resolved by Chase
    clarifying_question: str | None = None     # set iff intent == "clarify"
    reason: str                                # short, for traces only

class SearchHit(BaseModel):
    url: HttpUrl; title: str; snippet: str; rank: int; provider: str

class PageRef(BaseModel):                  # state holds refs; page text lives in the repo (§8)
    url: HttpUrl
    title: str | None
    status: Literal["ok", "blocked_robots", "timeout", "http_error", "too_large",
                    "non_html", "ssrf_blocked", "injection_flagged"]
    chars: int = 0
    injection_score: float = 0.0
    contact_page_url: HttpUrl | None = None

class RetrievedChunk(BaseModel):
    chunk_id: str; url: HttpUrl; chunk_index: int; text: str
    score: float; selected_by: Literal["vector", "contact_pattern"]

class ExtractedLead(BaseModel):            # what the LLM returns (per page)
    company_name: str
    partner_type: PartnerType | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    website: str | None = None
    evidence_quote: str = Field(max_length=300)   # verbatim span containing the contact

class ExtractionResult(BaseModel):
    leads: list[ExtractedLead] = Field(default_factory=list, max_length=5)

class Lead(BaseModel):
    id: str; run_id: str; user_id: str
    company_name: str
    partner_type: PartnerType | None
    location: str | None
    website: HttpUrl | None
    email: str | None
    email_status: VerificationStatus
    phone: str | None                      # E.164 when parseable
    phone_status: VerificationStatus
    source_url: HttpUrl
    evidence_quote: str
    created_at: datetime

    @computed_field
    def outreach_eligible(self) -> bool:
        return self.email is not None and self.email_status == "verified"

class BrandProfile(BaseModel):
    brand_name: str; product: str; value_prop: str
    sender_name: str; sender_email: EmailStr; website: HttpUrl | None = None

class EmailDraft(BaseModel):               # LLM output
    subject: str = Field(max_length=80)
    body: str                              # ≤ 180 words, validated in outreach/policy.py

class OutreachDraft(BaseModel):
    id: str; lead_id: str; user_id: str; run_id: str
    channel: Literal["email", "voice"]
    recipient: str                         # email or E.164 phone
    subject: str | None
    body: str                              # email body or call script
    status: Literal["pending_approval", "approved", "rejected", "sent",
                    "drafted_in_gmail", "outboxed", "failed", "skipped_duplicate",
                    "skipped_ineligible", "skipped_cap"]
    prompt_version: str
    created_at: datetime
    decided_at: datetime | None = None

class ApprovalDecision(BaseModel):
    approve: list[str] = []                # draft ids
    reject: list[str] = []
    edits: dict[str, EmailDraft] = {}      # draft id -> edited content (implies approve)

class StageError(BaseModel):
    stage: str; code: str; message: str; url: str | None = None

class RunRecord(BaseModel):
    run_id: str; user_id: str; thread_id: str; intent: Intent
    started_at: datetime; finished_at: datetime | None
    timings_ms: dict[str, int]; tokens: dict[str, int]      # per role
    prompt_versions: dict[str, str]; errors: list[StageError]
    trace_url: str | None = None
```

## 5. Agent graph (LangGraph)

```
START
  │
  ▼
charlie_route ─┬─ cpg_qa ────────► kb_answer ─────────────────────────────────────────────► END
               ├─ clarify ───────► ask_clarification ─────────────────────────────────────► END
               ├─ out_of_scope ──► refuse ────────────────────────────────────────────────► END
               ├─ discovery ─────► scout_search ─► scout_fetch ─► scout_index ─► scout_evidence
               │                     ─► scout_extract ─► scout_verify ─► scout_report ────► END
               └─ outreach ──────► chase_select ─► chase_draft ─► approval_gate ⏸ ─► chase_dispatch ─► END
                                                                    (interrupt)
```

### 5.1 State (`graph/state.py`)

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # trimmed to last 20 before LLM calls
    user_id: str
    thread_id: str
    run_id: str                       # new per user turn
    route: RouteDecision
    search_hits: list[SearchHit]
    pages: list[PageRef]              # refs only; bodies in repo.pages
    evidence: dict[str, list[RetrievedChunk]]   # url -> selected chunks
    candidates: list[ExtractedLead]
    leads: list[Lead]                 # verified AND rejected, for audit + eval
    drafts: list[OutreachDraft]
    errors: Annotated[list[StageError], operator.add]
    timings_ms: Annotated[dict[str, int], merge_dicts]
```

**State holds references, not page bodies.** This keeps checkpoints small. Page text is written to `repo.pages` (TTL) and read back by `scout_verify`.

### 5.2 Nodes

| Node | Input → output | Notes |
|---|---|---|
| `charlie_route` | messages → `route` | router role, temp 0, few-shot from the eval **dev** split |
| `kb_answer` | route → AI message | §6.2 |
| `ask_clarification` | route → AI message | uses `route.clarifying_question` |
| `refuse` | → AI message | fixed template listing capabilities |
| `scout_search` | `route.discovery` → `search_hits` | §6.3 |
| `scout_fetch` | hits → `pages` (+ bodies to repo) | §6.4 |
| `scout_index` | pages → chunks in Qdrant | §6.5 |
| `scout_evidence` | pages → `evidence` | §6.6 |
| `scout_extract` | evidence → `candidates` | §6.7, one LLM call per page |
| `scout_verify` | candidates + page bodies → `leads` | §6.8, deterministic |
| `scout_report` | leads → AI message + persist leads | summary text, the table goes to the UI |
| `chase_select` | route + repo leads → selected leads | §6.9 |
| `chase_draft` | leads + brand profile → `drafts` | writer role, temp 0.7 |
| `approval_gate` | drafts → `ApprovalDecision` | `interrupt()`; resumed through the API |
| `chase_dispatch` | approved drafts → statuses | §6.9 |

The graph is compiled with a checkpointer (`InMemorySaver` in tests, MongoDB saver in prod) keyed by `thread_id`. Each user turn creates a new `run_id`.

## 6. Component design

### 6.1 Charlie: router

- The router gets the system prompt (scope, intents, rules), the last 6 messages, and the leads summary from the last run in this thread.
- It returns a `RouteDecision` through `invoke_structured`.
- If `confidence < ROUTER_MIN_CONFIDENCE` (0.6) and the intent is not `out_of_scope`, the intent is coerced to `clarify`.
- If `intent=discovery` but `product_category` or `partner_type` cannot be filled, the intent becomes `clarify`.
- Few-shot examples come from `eval/datasets/routing.jsonl` with `split=dev` only.

### 6.2 Charlie: knowledge-base Q&A and guardrails

- **Corpus.** `data/kb/*.md` holds short notes written in our own words from public sources. Front matter: `title`, `source_url`, `topic`. They are ingested into the Qdrant collection `cpg_kb` (chunk size 400 tokens).
- **Answering.**
  1. Retrieve the top 5.
  2. If the best score is below `KB_MIN_SCORE`, reply with the "not in my knowledge base" template.
  3. Otherwise the writer (temp 0.3) answers using only the numbered passages and cites them as `[n]`.
- **Post-check.** The answer must contain at least one valid `[n]`. If it does not, the canned fallback is returned.
- **Guardrail layers.**
  1. Router intent `out_of_scope` → `refuse`.
  2. The system prompt carries the scope policy.
  3. The citation post-check.
  4. Knowledge-base and scraped text is never given tool access.

  These are honest prompt-level guardrails plus a classifier and a deterministic check. They are not a separate moderation model.

### 6.3 Scout: search

- The `SearchProvider` protocol has the `tavily` (primary) and `ddg` (fallback on error or empty results) implementations.
- Query templates are chosen by `partner_type`. Example (broker): `"{product_category} food broker {location} contact"`. Up to 2 templates run per intent, and the results are merged.
- Post-processing:
  1. Normalise URLs.
  2. Dedupe by registrable domain (keep the best rank).
  3. Drop domains on `SEARCH_DOMAIN_BLOCKLIST` (social networks, marketplaces, large directories, news).
  4. Keep the top `max_results`.
- Search results are cached in `repo.search_cache` keyed by the normalised query (TTL 24 h).

### 6.4 Scout: fetch (`web/fetch.py`)

- httpx async client. `User-Agent: CPGPartnerAgents/0.1 (+repo URL)`. Timeout `FETCH_TIMEOUT_S`. Concurrency `FETCH_CONCURRENCY`, with at most 2 requests in flight per host.
- **SSRF guard.** Only `http(s)` is allowed. The host is resolved, and private, loopback, link-local and reserved IPs are rejected. Every redirect hop (max 3) is re-checked.
- **robots.txt** is cached per host. If a URL is disallowed, the page gets `status=blocked_robots`.
- **Limits.** The response must be `text/html` and under `FETCH_MAX_BYTES`.
- **Text assembly.** Page text is built from three parts:
  - (a) trafilatura main content;
  - (b) the visible text of `<footer>`, `<address>` and elements whose id or class contains "contact";
  - (c) all `mailto:` / `tel:` targets.

  Footer contacts are why (b) and (c) exist: main-content extraction drops them.
- **Contact-page discovery.** If no email or phone pattern is found, follow at most one same-domain link whose text or href matches `contact|about|reach` and append its text. The followed URL is stored as `contact_page_url`.
- Bodies are stored in `repo.pages` (TTL). `injection_score` is computed here (§12).

### 6.5 Scout: chunk and index (`retrieval/`)

- **Chunker.**
  - Token-based, with tiktoken `cl100k_base` as a provider-neutral proxy.
  - `CHUNK_TOKENS=1000` and `CHUNK_OVERLAP=150`.
  - Splits on paragraphs first, then sentences, then words. It never splits inside a word.
  - Each chunk gets `has_contact_pattern` (email/phone regex, including obfuscated forms).
- **Embeddings.**
  - `BAAI/bge-small-en-v1.5` (384-d, cosine) through sentence-transformers, in batches of 32.
  - Queries use the BGE query instruction prefix.
  - There is an in-process LRU cache keyed by `sha256(text)`.
- **Store.** Qdrant collection `web_chunks` (§8.2). Points are upserted with a payload that includes `run_id` and `url`.

### 6.6 Scout: evidence selection (`retrieval/evidence.py`), per page and hybrid

For each page with `status=ok`:
1. `q = raw_query + " " + f"{partner_type} {product_category} {location} contact email phone"`.
2. Vector search the top `EVIDENCE_K` (4), filtered by `run_id` **and** `url`.
3. Add chunks with `has_contact_pattern=true` that are not already selected, at most 3.
4. Cap the total at `EVIDENCE_MAX` (6) and sort by `chunk_index`, so the text reads in order.

The reasons for this design are in ADR-007. In short: a lead must be tied to one source; a global top-k lets one long page crowd out others; and pure semantic search misses the footer chunk that holds the email.

### 6.7 Scout: extraction

- There is one call per page. The extractor role runs at temp 0, with concurrency bounded by `LLM_MAX_CONCURRENCY`.
- The prompt contains: the extraction rules ("only values literally present; else null"; `evidence_quote` must be copied verbatim), the `DiscoveryIntent`, and the evidence wrapped by `wrap_untrusted()`.
- It returns an `ExtractionResult` (0–5 leads per page). On a validation error there is one retry with the error appended. A second failure records a `StageError` and skips the page.

### 6.8 Scout: contact verification (`verify/contacts.py`), deterministic

1. **Normalise the source text.** This uses the full stored page body plus the contact-page text: lowercase, collapse whitespace, and de-obfuscate `[at]`, `(at)`, ` at ` (only between word tokens that look like an email), `[dot]` and `(dot)`.
2. **Email.**
   - If the syntax is invalid (email-validator, no DNS), the status is `invalid_format`.
   - If the address is not a substring of the normalised text, the status is `not_in_source`.
   - Otherwise the status is `verified`.
   - If `VERIFY_MX=true`, a domain with no MX record (dnspython, 3 s timeout) gets `mx_failed`.
3. **Phone.**
   - Parse with `phonenumbers` (region from the location, default `DEFAULT_PHONE_REGION`).
   - If the national-number digits do not appear in the page's digit stream, the status is `not_in_source`.
   - Otherwise the status is `verified`, and the number is stored as E.164.
4. **Dedupe within a run** by `(normalised company_name, email or phone)`. The lead with the higher verification status is kept.
5. Rejected leads are **kept** in `state.leads`, marked as such, for audit and eval. Only eligible leads are offered to Chase.

### 6.9 Chase: outreach (`outreach/`)

- **`chase_select`.** Resolves `route.lead_selector` against the thread's latest run (or the user's recent leads). Simple selectors ("top N", by location, by name) are parsed deterministically; for anything else, the writer model picks ids from a numbered list. The result is then filtered through `policy.eligible()`.
- **Policy (`policy.py`, pure functions).** Each check maps to a skip status:
  - The email must be verified, otherwise `skipped_ineligible`.
  - No outreach to the same recipient within `OUTREACH_DEDUPE_DAYS`, otherwise `skipped_duplicate`.
  - A per-user daily token bucket of `OUTREACH_DAILY_CAP`, otherwise `skipped_cap`.
  - In `gmail_send` mode the recipient must be on `OUTREACH_ALLOWLIST`.
  - The body must be ≤ 180 words and contain the sender name and the opt-out line. If the model omits the opt-out line, it is appended.
- **`chase_draft`.** The writer role (temp 0.7) gets the `BrandProfile`, the user's goal and the lead fields, and returns an `EmailDraft`. The prompt forbids unverifiable claims and fake familiarity.
- **`approval_gate`.** Calls `interrupt({"drafts": [...]})`. The API returns `pending_approval`. `POST /api/chat/{thread_id}/approve` resumes with `Command(resume=ApprovalDecision)`. Drafts that are neither approved nor edited become `rejected`.
- **`chase_dispatch`** sends according to `OUTREACH_MODE`:
  - `outbox` (default): store the draft as `outboxed` and show it in the UI.
  - `gmail_draft`: create a draft in the user's Gmail (OAuth installed-app flow, scope `gmail.compose`, token file outside the repo). Status `drafted_in_gmail`.
  - `gmail_send`: send through the Gmail API only for allowlisted recipients. Status `sent`.
- **Voice (`voice.py`).** The `VoiceCaller` protocol has two implementations. `MockVoiceCaller` (default) produces a script and a simulated summary. `VapiVoiceCaller` (stretch) places calls only to `VOICE_ALLOWLIST` numbers.
- `DEMO_MODE=true` forces `outbox` and `mock` regardless of other settings.

### 6.10 Memory

- **Short-term.** The LangGraph checkpointer keyed by `thread_id`. Messages are trimmed to the last 20 before each LLM call.
- **Long-term.** The `leads`, `outreach` and `runs` collections are keyed by `user_id`. Chase reads these, so a request like "email the brokers you found yesterday" works across threads.

## 7. API (`api/`)

All routes are under `/api`. JSON in and out. If `APP_API_KEY` is set, the `X-API-Key` header is required (not in demo mode, which uses a per-session anonymous `user_id` plus rate limits).

| Method | Path | Body / query | Returns |
|---|---|---|---|
| POST | `/chat` | `{user_id, thread_id?, message}` | `ChatResponse` |
| POST | `/chat/{thread_id}/approve` | `ApprovalDecision` | `ChatResponse` (after dispatch) |
| GET | `/leads` | `?user_id&run_id?` | `list[Lead]` |
| GET | `/outreach` | `?user_id&status?` | `list[OutreachDraft]` |
| GET | `/runs/{run_id}` | | `RunRecord` |
| PUT | `/profile` | `BrandProfile` (+`user_id`) | `BrandProfile` |
| GET | `/health` | | `{status, llm, qdrant, mongo, search}` |

```jsonc
// POST /api/chat
{"user_id": "u_123", "thread_id": null, "message": "Find snack food brokers in California"}

// 200 ChatResponse
{
  "thread_id": "t_9f2c", "run_id": "r_51ab",
  "route": {"intent": "discovery", "confidence": 0.93},
  "reply": "Found 7 brokers; 5 have a verified email.",
  "leads": [{"id": "l_1", "company_name": "…", "email": "sales@…", "email_status": "verified",
             "phone": "+1…", "phone_status": "verified", "source_url": "https://…", "outreach_eligible": true}],
  "pending_approval": null,          // or {"drafts": [OutreachDraft, …]}
  "errors": [{"stage": "fetch", "code": "blocked_robots", "url": "https://…", "message": "…"}]
}

// error shape (4xx/5xx)
{"error": {"code": "rate_limited", "message": "…", "retry_after_s": 120}}
```

**Streaming.** The UI uses `graph.astream(stream_mode="updates")` in-process to show stage progress. A server-sent-events endpoint is a stretch task (T-51).

## 8. Data stores

### 8.1 MongoDB (Atlas M0; in-memory `Repository` for dev/tests)

| Collection | Key fields | Indexes / TTL |
|---|---|---|
| `leads` | `Lead` | `{user_id, run_id}`, `{user_id, email}` |
| `outreach` | `OutreachDraft` | `{user_id, recipient, created_at}` (dedupe), `{user_id, status}` |
| `runs` | `RunRecord` | `{run_id}` unique, `{user_id, started_at}` |
| `pages` | `{run_id, url, text, contact_text, fetched_at}` | TTL on `fetched_at` = `DATA_TTL_DAYS` |
| `search_cache` | `{query_key, hits, created_at}` | TTL 24 h |
| `profiles` | `BrandProfile + user_id` | `{user_id}` unique |
| `checkpoints*` | LangGraph MongoDB saver | managed by the library |

All access goes through the `Repository` protocol (`storage/repo.py`). There are two implementations: `InMemoryRepository` and `MongoRepository`. Both are covered by the same test suite.

### 8.2 Qdrant (Cloud free tier; local mode when `QDRANT_URL` is empty)

| Collection | Vector | Payload | Payload indexes |
|---|---|---|---|
| `web_chunks` | 384, cosine | `run_id, url, title, chunk_index, text, has_contact_pattern, fetched_at` | `run_id`, `url` |
| `cpg_kb` | 384, cosine | `doc_id, title, source_url, topic, chunk_index, text` | `topic` |

A cleanup job runs on startup and deletes `web_chunks` points older than `DATA_TTL_DAYS`.

## 9. LLM layer (`llm.py`)

| Role | Used by | Temp | Default model (set in env) |
|---|---|---|---|
| `router` | `charlie_route`, `chase_select` fallback | 0 | fastest current Gemini Flash-Lite-class model |
| `extractor` | `scout_extract` | 0 | current Gemini Flash-class model |
| `writer` | `kb_answer` (0.3), `chase_draft` (0.7) | per call | current Gemini Flash-class model |

- `get_chat_model(role)` returns a LangChain `BaseChatModel` for `LLM_PROVIDER` ∈ {`gemini`, `groq`, `ollama`, `fake`}. Model ids come from env (`LLM_MODEL_*`). Check current free-tier model names in each provider's console; do not hard-code them.
- `invoke_structured(model, schema, messages)` calls `.with_structured_output(schema)` and validates the result. On failure it retries once with the error appended. It records token usage per role into the run.
- A global `asyncio.Semaphore(LLM_MAX_CONCURRENCY)` wraps all calls. 429 and 5xx responses retry through `utils/retry.py`.

## 10. Configuration (`config.Settings`; `.env.example` mirrors this)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` \| `groq` \| `ollama` \| `fake` |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | — | provider keys |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | for example a friend's machine over Tailscale |
| `LLM_MODEL_ROUTER` / `_EXTRACTOR` / `_WRITER` | — | model ids per role |
| `LLM_MAX_CONCURRENCY` | `4` | stays under free-tier RPM |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | |
| `SEARCH_PROVIDER` / `SEARCH_FALLBACK` | `tavily` / `ddg` | |
| `TAVILY_API_KEY` | — | |
| `SEARCH_DOMAIN_BLOCKLIST` | built-in list | comma-separated additions |
| `QDRANT_URL` / `QDRANT_API_KEY` | empty → local `./.qdrant` | |
| `MONGODB_URI` | empty → in-memory | |
| `CHUNK_TOKENS` / `CHUNK_OVERLAP` | `1000` / `150` | |
| `EVIDENCE_K` / `EVIDENCE_MAX` | `4` / `6` | |
| `ROUTER_MIN_CONFIDENCE` / `KB_MIN_SCORE` | `0.6` / tuned in eval | |
| `FETCH_TIMEOUT_S` / `FETCH_MAX_BYTES` / `FETCH_CONCURRENCY` | `10` / `2000000` / `8` | |
| `VERIFY_MX` / `DEFAULT_PHONE_REGION` | `false` / `US` | |
| `OUTREACH_MODE` | `outbox` | `outbox` \| `gmail_draft` \| `gmail_send` |
| `OUTREACH_ALLOWLIST` | empty | required for `gmail_send` |
| `OUTREACH_DAILY_CAP` / `OUTREACH_DEDUPE_DAYS` | `20` / `30` | |
| `GMAIL_CREDENTIALS_PATH` / `GMAIL_TOKEN_PATH` | outside the repo | OAuth installed-app files |
| `VOICE_PROVIDER` | `mock` | `mock` \| `vapi` |
| `VAPI_API_KEY` / `VAPI_PHONE_NUMBER_ID` / `VOICE_ALLOWLIST` | — | stretch |
| `LANGFUSE_PUBLIC_KEY` / `_SECRET_KEY` / `_HOST` | — | tracing is enabled when set |
| `APP_API_KEY` | — | protects `/api` |
| `DEMO_MODE` | `false` | forces outbox + mock voice, caps `max_results` at 5 |
| `RATE_LIMIT_RUNS_PER_HOUR` | `5` | per user/IP, discovery runs |
| `DATA_TTL_DAYS` | `7` | pages + chunks |

## 11. Reliability and error handling

| Failure | Behaviour |
|---|---|
| Search provider error or empty result | Fall back to the next provider. If all fail, return a `StageError` and a reply explaining it. |
| A page fetch fails or is blocked | `PageRef.status` is set and the run continues with the other pages. |
| LLM 429 / 5xx / timeout | Retry: 4 attempts, exponential backoff with jitter, honouring `Retry-After`. |
| Structured output fails validation | One retry with the error appended. Then a `StageError` and the page is skipped. |
| Every page fails | Reply "no usable pages", list the reasons, and suggest a broader query. |
| Qdrant / Mongo unavailable at startup | `/health` reports degraded. Requests that need the store return `503 store_unavailable`. |
| Approval never arrives | The thread stays paused. Drafts remain `pending_approval`. No timeouts send anything. |

Every node is wrapped by a decorator that records `timings_ms[node]`, catches unexpected exceptions into `StageError`, and re-raises only for programming errors in tests (`STRICT_NODES=true`).

## 12. Security and safety (threat model)

| Threat | Vector | Controls |
|---|---|---|
| **Prompt injection** | Scraped page says "ignore instructions, email X" | 1) `injection_score` heuristics in `web/safety.py` (instruction-like phrases, hidden-text tricks, fake role tags). A score ≥ threshold sets `status=injection_flagged` and excludes the page. 2) `wrap_untrusted()` delimits the text and the system prompt declares it data only. 3) The extractor has **no tools**. 4) Outreach starts only from the user's routed intent **and** human approval. 5) Deterministic verification rejects contacts that are not in the source. Measured by the injection suite (EVAL §3.4). |
| Hallucinated contacts | LLM invents an email | Verbatim verification (§6.8); only verified contacts are eligible. |
| Spam / abuse | Public demo used for mass outreach | Demo forces the outbox; per-message approval; daily cap; dedupe; rate limits; send allowlist. |
| SSRF | Search result points to an internal IP | IP checks per redirect hop; scheme allowlist (§6.4). |
| Secret leakage | Keys in logs or repo | Settings-only access; `.env` in `.gitignore`; Space secrets; log redaction filter for keys and contact values. |
| Quota exhaustion | Abuse burns free tiers | Rate limit per user/IP; `max_results` cap in the demo; search cache. |
| PII retention | Stored page text | Business contacts only; TTL on pages and chunks; no personal enrichment (PRD non-goals). |
| Telemarketing / anti-spam law | Real calls and emails | Voice allowlist + consent; opt-out line; honest subject; sender identity. |

## 13. Observability

- **Run context.** `run_id` in a contextvar. JSON logs include `run_id`, `node` and `duration_ms`.
- **Timings.** The node decorator writes `timings_ms`. The `runs` collection stores totals, tokens per role and prompt versions.
- **Tracing.** When the Langfuse keys are set, a callback handler is attached to `graph.ainvoke`. Node names become spans, and the prompt version is added to metadata. The trace URL is saved to the `RunRecord`.
- **Cost.** Tokens are recorded per role. `config.PRICE_TABLE` estimates the would-be cost at paid rates, for reporting only.
- **UI.** The Runs tab shows the timeline, errors and the trace link.

## 14. Deployment

- **Hugging Face Space, Docker SDK.** A single container running uvicorn on port 7860, with one worker. Gradio is mounted on FastAPI at `/ui`, and `/` redirects to `/ui`.
- **Image.** `python:3.12-slim` + uv. The embedding model is downloaded at **build** time, so cold starts do not download it. The tiktoken encoding is pre-cached as well.
- **Secrets.** Provider keys, `QDRANT_*`, `MONGODB_URI`, `LANGFUSE_*` and `APP_API_KEY` live in the Space settings. `DEMO_MODE=true` on the public Space.
- **External services.** Qdrant Cloud free cluster, MongoDB Atlas M0, Langfuse Cloud (free). Free tiers sleep or limit; the UI shows a warm-up note.
- **CI.** GitHub Actions run ruff, mypy and pytest on every push. Deploy is a manual push to the Space remote after CI is green.

## 15. Performance budget (discovery, 10 results)

| Stage | Budget (p95) | How |
|---|---|---|
| Search | 3 s | cached; one fallback |
| Fetch + contact page | 12 s | 8 concurrent, 10 s timeout, 2 per host |
| Chunk + embed + upsert | 6 s | batched local embeddings |
| Evidence selection | 1 s | filtered vector search |
| Extraction | 18 s | ≤ 10 calls, 4 concurrent |
| Verify + persist | 1 s | deterministic |
| **Total** | **≤ 45 s** | p50 target ≤ 25 s |

## 16. Testing strategy

| Layer | Approach |
|---|---|
| Pure logic (chunking, evidence, verification, policy, rate limiter, metrics) | Table-driven unit tests; property-style checks for the chunker |
| Web | `respx` mocks for search and pages, including robots, redirects, SSRF, oversize pages, `mailto:` |
| LLM nodes | LangChain fake chat models returning scripted structured outputs, including invalid-then-valid |
| Graph | Offline integration tests: discovery end to end, two-turn memory, the interrupt/resume approval flow |
| Stores | One shared test suite run against `InMemoryRepository` (always) and Mongo (`-m live`); Qdrant `:memory:` |
| API | FastAPI `TestClient` for every endpoint, including the approve flow and the error shape |
| Live smoke | `pytest -m live`: one route, one discovery on a fixed query, one knowledge-base answer |
| Quality | The eval suites (EVAL.md), not unit tests |
