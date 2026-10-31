# PRD: CPG Partner Agents

| | |
|---|---|
| Status | Draft v1 |
| Owner | Lokesh Kumar |
| Last updated | 2026-10-30 |
| Related | [HLD](HLD.md) · [TASKS](TASKS.md) · [EVAL](EVAL.md) · [DECISIONS](DECISIONS.md) |

> **Origin.** This is a from-scratch personal rebuild of a system built during an internship. It contains no client code, client data or client branding, and it uses public web data only.

---

## 1. Summary

A chat-based assistant for Consumer Packaged Goods (CPG) brands. A user types something like *"find cookie brokers in California"*. Within seconds the system returns a table of real companies, each with a contact email or phone that has been **verified against the source page** and linked to it. The user can then ask it to draft partnership emails, review and edit them, and approve them. Nothing is sent without that approval.

Three agents split the work:
- **Charlie** (orchestrator) understands the request and routes it. It also answers general CPG questions from a cited knowledge base.
- **Scout** (discovery) searches the web, reads pages, and extracts structured contact data.
- **Chase** (outreach) writes personalised drafts and dispatches only what the user approves.

## 2. Problem

In CPG, distribution decides the business. A brand without retailers, distributors and brokers has no shelf presence. Finding and contacting these partners is manual today: hours of searching, reading websites, copying contact details into spreadsheets and writing cold emails one at a time.

The existing alternatives fall short:

| Alternative | Why it falls short |
|---|---|
| Manual research + spreadsheet/CRM | Slow and inconsistent. The data goes stale. |
| Lead databases (Apollo, ZoomInfo) | Expensive, generic, and weak on niche CPG roles such as food brokers. |
| A single chatbot prompt | Cannot reliably do multi-step work (search → read → extract → contact). It invents contact details, and its output cannot safely drive automation. |

## 3. Users

| Persona | Need | Success looks like |
|---|---|---|
| **Brand founder / sales lead** at a small CPG brand (primary) | Find the right partners in a target region and start conversations | A usable, trustworthy lead list and good first drafts in minutes, not days |
| **Distributor / broker** looking for brands (secondary) | The same pipeline in reverse (partner type = brand/manufacturer) | The same |
| **Portfolio reviewer** (recruiter, interviewer) | Judge engineering quality quickly | Opens the live demo, runs the scripted flow in under 2 minutes, and sees measured eval numbers in the README |

## 4. Goals

| # | Goal | Measured by |
|---|---|---|
| G1 | One natural-language query returns relevant partners with contact data | Lead yield ≥ 5 per query on the E2E set; p95 latency ≤ 45 s (EVAL §5) |
| G2 | **No unverified contact is ever eligible for outreach** | Eligible-hallucination rate = 0 (EVAL §4.2) |
| G3 | Outreach is personalised and **human-approved** | 100% of dispatched messages have an approval record |
| G4 | CPG questions get grounded answers with citations; off-topic requests are refused | KB citation accuracy, refusal rate (EVAL §4.6) |
| G5 | Quality is measured, not asserted | An eval suite with published, reproducible numbers |
| G6 | It costs nothing to run and anyone can try it | Free tiers only; public demo URL |

## 5. Non-goals (binding)

- No CRM features (pipelines, deal stages, reminders).
- No bulk campaigns or automated sequences, and no sending without per-message approval.
- No scraping behind logins, of LinkedIn or other social networks, or of pages disallowed by `robots.txt`.
- No personal-contact enrichment. Only business contact details that the business itself publishes (role addresses such as `sales@`, main phone lines).
- No model training or fine-tuning. This is an inference-time system.
- No multi-tenant organisations, billing or SSO. `user_id` is a simple identifier.
- No custom frontend framework. The UI is Gradio.
- No compound multi-intent requests in a single turn for the MVP (for example "find brokers and email them"). Charlie handles one intent per turn and suggests the next step. This is a stretch goal (T-50).

## 6. Features and acceptance criteria

**F1. CPG Q&A (Charlie)**
- A CPG question is answered from the knowledge base with at least one numbered citation that links to a source.
- If no knowledge-base passage clears the relevance threshold, the reply says it does not know instead of answering from model memory.

**F2. Guardrails (Charlie)**
- Off-topic requests (for example "write me a poem", "what's the weather") get a short refusal that lists what the assistant can do.
- Content retrieved from the knowledge base or from scraped pages can never trigger a tool call or outreach.

**F3. Clarification (Charlie)**
- An ambiguous request (for example "find partners" with no product or partner type) gets one clarifying question instead of a guessed search.
- This triggers when router confidence is below the threshold (HLD §6.1).

**F4. Discovery (Scout)**
- The input is a natural-language request. Charlie turns it into a structured `DiscoveryIntent` (partner type, product category, location, count).
- The output is a lead table with these columns: company, partner type, location, email, phone, a verification badge per contact, and a source link.
- Progress is shown per stage (searching → reading pages → extracting → verifying).
- A page that fails (timeout, robots disallow, injection flagged) is reported in the run's errors and does not fail the run.

**F5. Lead memory**
- Leads are saved per user and per run.
- Within the same thread, follow-ups such as "email the top 3" or "only the ones in LA" resolve against earlier results.

**F6. Outreach drafting (Chase)**
- Only leads whose email status is `verified` are eligible.
- A recipient contacted in the last 30 days is skipped and shown as skipped.
- Each draft uses the user's brand profile and goal. It has a subject of at most 80 characters, a body of at most 180 words, and a sender identity and opt-out line.
- The user can approve, edit or reject each draft individually.

**F7. Dispatch (Chase)**
- The default mode is `outbox`: approved messages are stored and shown, and nothing leaves the system.
- `gmail_draft` creates drafts in the user's own Gmail through OAuth. No password is stored.
- `gmail_send` sends only to addresses on an allowlist and is disabled in demo mode.
- A daily cap applies per user.

**F8. Voice follow-up (stretch)**
- `mock` (default) generates a call script and a simulated call summary.
- `vapi` places a real call only to allowlisted numbers that have consented.

**F9. Run inspector**
- Every run shows its stage timings, token usage, errors, prompt versions and a trace link (Langfuse) when configured.

**F10. Evaluation**
- One command per suite produces a JSON + Markdown report (EVAL §6).

## 7. Non-functional requirements

| Area | Requirement |
|---|---|
| Performance | Discovery p50 ≤ 25 s, p95 ≤ 45 s for 10 search results. Q&A p95 ≤ 8 s. (HLD §15) |
| Reliability | Any external failure degrades the result instead of crashing the run. Retries use jittered backoff. A run's partial result is always returned. |
| Cost | It runs on free tiers. LLM concurrency is bounded to stay under free-tier rate limits. Token usage is recorded per run. |
| Safety | Defence in depth against prompt injection from scraped pages (HLD §12). Contacts are deterministically verified. Human approval is required before outreach. |
| Privacy | Business contact data only. Page snapshots and chunks expire after `DATA_TTL_DAYS` (7). No contact values are logged at INFO. |
| Compliance | Outreach identifies the sender, has an honest subject and includes an opt-out line. Voice goes only to consenting allowlisted numbers. Scraping respects `robots.txt` and rate-limits itself per host. |
| Security | SSRF-safe fetcher. API key on the API when `APP_API_KEY` is set. Rate limits per user/IP. Secrets only in env / Space secrets. |
| Maintainability | Tests run offline in under 60 s. Contracts are documented in the HLD. Every design choice has an ADR. |
| Portability | LLM provider switchable by config (Gemini / Groq / Ollama). Every store has an in-memory fallback for local dev and tests. |

## 8. Release scope

| Release | Milestones (TASKS) | Contents |
|---|---|---|
| **MVP** | M0–M5 | Charlie + Scout + Chase (outbox), API, Gradio UI, local stores |
| **v1 (portfolio-ready)** | M6–M8 | Safety + tracing, the full eval with published numbers, public deploy, README + demo video |
| **Stretch** | M9 | Real Vapi calls, MX checks, Groq router, compound intents, streaming |

## 9. Success metrics

v1 ships when all of the following hold (definitions in EVAL §4, targets in EVAL §5):
- Routing accuracy ≥ 0.90 on the routing test split.
- Email precision ≥ 0.95 and recall ≥ 0.70 on the extraction set.
- Eligible-hallucination rate = 0. Injection attack success = 0.
- Discovery p95 ≤ 45 s on the E2E set.
- The public demo completes the scripted flow (§10) end to end.

## 10. Demo script (used for manual acceptance and the video)

1. Ask "What does a food broker do, and how is it different from a distributor?" → a cited answer.
2. Ask "Write me a poem" → a refusal.
3. Ask "Find partners" → a clarifying question.
4. Ask "Find snack food brokers in California" → a lead table with verified badges and source links.
5. Ask "Draft emails to the top 3 for our gluten-free cookie brand" → three drafts, pending approval.
6. Edit one, reject one, approve one → the approved message appears in the outbox and the rejected one does not.
7. Open the Runs tab → stage timings, token counts and the trace link.

## 11. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Free-tier rate limits or quota changes | Slow or failed runs | Bounded concurrency, retries, provider switch by config, eval LLM-response cache |
| Search returns directories or aggregators instead of companies | Low lead yield | Domain blocklist, partner-type-specific query templates, ablation in the eval |
| Sites block fetching or hide contacts | Low recall | Contact-page discovery, footer and `mailto:` extraction, report recall honestly |
| LLM invents contacts | Wrong outreach | Deterministic verbatim verification; only verified contacts are eligible |
| Prompt injection in scraped pages | Hijacked behaviour | HLD §12 layered defences plus the injection eval |
| Cold starts on free hosting | Slow first demo request | Bake the embedding model into the image, show a warm-up note in the UI |
| Scope creep | Never ships | Binding non-goals; stretch work only after v1 |

## 12. Open questions

- Q1. Where does the brand profile come from in the demo? Proposal: a fictional sample brand pre-filled and editable in a UI settings panel.
- Q2. What is the default phone region when the location is missing? Proposal: `US`, configurable.
- Q3. Is Gmail OAuth worth setting up on the hosted demo? Proposal: no. Demo mode uses the outbox, and `gmail_draft` is shown locally in the demo video.
