# PRD: CPG Partner Agents

| | |
|---|---|
| Status | Draft v2 (adds the call agent) |
| Owner | Lokesh Kumar |
| Last updated | 2026-10-31 |
| Related | [HLD](HLD.md) · [TASKS](TASKS.md) · [EVAL](EVAL.md) · [DECISIONS](DECISIONS.md) |

> **Origin.** This is a from-scratch personal rebuild of a system built during an internship. It contains no client code, client data or client branding, and it uses public web data only.

---

## 1. Summary

A chat-based assistant for Consumer Packaged Goods (CPG) brands. A user types something like *"find cookie brokers in California"*. Within seconds the system returns a table of real companies, each with a contact email or phone that has been **verified against the source page** and linked to it. The user can then ask it to draft partnership emails, review and edit them, and approve them. Nothing is sent without that approval.

The user can also ask it to **call** a lead. Chase then runs a scripted first partner-discovery call: it introduces the brand, learns whether the partner fits, pitches using only approved facts, and books a meeting with the brand founder. The call follows a six-stage playbook that is enforced by code, and every reply is checked against the playbook's rules. In v1 the other side of the call is a simulated partner persona or the user, typing or speaking in the browser. No real phone call is placed.

Three agents split the work:
- **Charlie** (orchestrator) understands the request and routes it. It also answers general CPG questions from a cited knowledge base.
- **Scout** (discovery) searches the web, reads pages, and extracts structured contact data.
- **Chase** (outreach) writes personalised drafts and dispatches only what the user approves. It also runs the **call agent**: a stage-controlled conversation that qualifies a partner and books a meeting (F8).

## 2. Problem

In CPG, distribution decides the business. A brand without retailers, distributors and brokers has no shelf presence. Finding and contacting these partners is manual today: hours of searching, reading websites, copying contact details into spreadsheets and writing cold emails one at a time.

The existing alternatives fall short:

| Alternative | Why it falls short |
|---|---|
| Manual research + spreadsheet/CRM | Slow and inconsistent. The data goes stale. |
| Lead databases (Apollo, ZoomInfo) | Expensive, generic, and weak on niche CPG roles such as food brokers. |
| A single chatbot prompt | Cannot reliably do multi-step work (search → read → extract → contact). It invents contact details, and its output cannot safely drive automation. |
| Generic AI calling agents | Follow a free-form prompt: they skip steps, invent prices or terms, ignore opt-outs, and nobody measures how often. |

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
| G6 | It costs nothing to run and anyone can try it | Free tiers and local models only; public demo URL |
| G7 | The call agent follows its playbook and ends every call in a valid outcome | Hard-rule violation rate = 0, outcome accuracy ≥ 0.85, opt-out compliance = 1.0 (EVAL §4.7) |

## 5. Non-goals (binding)

- No CRM features (pipelines, deal stages, reminders).
- No bulk campaigns or automated sequences, and no sending without per-message approval.
- No scraping behind logins, of LinkedIn or other social networks, or of pages disallowed by `robots.txt`.
- No personal-contact enrichment. Only business contact details that the business itself publishes (role addresses such as `sales@`, main phone lines).
- No model training or fine-tuning. This is an inference-time system.
- No multi-tenant organisations, billing or SSO. `user_id` is a simple identifier.
- No custom frontend framework. The UI is Gradio.
- No compound multi-intent requests in a single turn for the MVP (for example "find brokers and email them"). Charlie handles one intent per turn and suggests the next step. This is a stretch goal (T-50).
- No real phone calls in v1. The call agent talks to a simulated persona or to the user in the browser. Real calls (Vapi) are a stretch goal, for allowlisted and consenting numbers only.
- The call agent never negotiates terms, prices or discounts, and never closes a deal. Its job ends at a booked meeting, a booked callback or a polite exit.
- No content from third-party call scripts. Playbooks are written from scratch for the fictional sample brand.
- v1 handles the **first** call only. Call history, call types (follow-up, reminder) and previous-call summaries are out of scope (a second playbook is stretch T-64).

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

**F8. Call agent (Chase)**
- **Trigger.** "Call lead 1" (or "call the top one") routes to outreach with `channel=voice`. It goes through the same approval gate as email: the user approves the call plan before it starts. In v1 no number is dialled, so any lead can be called; a verified phone is required only for real calls (Vapi, stretch).
- **Playbook.** The call runs six stages in a fixed order: opening → discovery → qualification → pitch → handoff (scheduling) → closing. Stage goals and must / never rules come from a playbook file (`data/playbooks/partner_intro.toml`). The stage order, stop rules and outcomes are enforced in code, not by the prompt.
- **Opening.** The agent says it is an AI assistant calling on behalf of the brand, confirms it is speaking to the right person and asks whether now is a good time. If not, it offers a callback.
- **Discovery.** Learns whether the partner is taking on new lines in the category, and why or why not.
- **Qualification.** Asks one question at a time, in order: role / decision-maker, categories carried, territory, key accounts, and minimum volume or terms. Facts the partner already shared are confirmed, not re-asked. A territory or category mismatch ends qualification with a polite exit.
- **Pitch.** Uses two or three approved facts from the brand profile and knowledge base that fit what the partner said. It never invents numbers, never makes guarantees and never offers discounts.
- **Handoff.** Offers a meeting with the brand founder and gets a specific day and time (the default proposal is the next business day, at least 24 hours out, on the hour). "Later" is not a commitment.
- **Closing.** Summarises in two sentences, asks for any last questions and signs off with the brand's sign-off line.
- **Guardrails, every stage.** Honours an opt-out immediately. Retries at most once after resistance, and never re-suggests something already declined. Acknowledges distress before continuing. Books a human (the founder) if asked. Answers "are you a bot?" truthfully. Never repeats data the partner did not share on the call. If the partner is in a rush, it asks only the required qualification slots. If the partner wants a different category, it acknowledges that, offers to pass it on to the founder and exits politely. It never signs off while the partner's question is still open.
- **Outcomes.** Each call ends in exactly one of: `meeting_booked`, `callback_booked`, `not_interested`, `disqualified`, `opted_out`, `wrong_person`, `escalated_to_human`, `incomplete`.
- **After the call.** A `CallRecord` holds the transcript with stage labels, the slots collected, the outcome and every rule check. A meeting or callback creates a confirmation email draft that goes through approval like any other draft.
- **Modes.** `simulate` (an LLM plays a chosen partner persona), `live_text` (the user types as the partner) and `live_voice` (the user speaks; speech-to-text and text-to-speech run locally; nice-to-have). `vapi` is a stretch goal.

**F9. Run inspector**
- Every run shows its stage timings, token usage, errors, prompt versions and a trace link (Langfuse) when configured.

**F10. Evaluation**
- One command per suite produces a JSON + Markdown report (EVAL §6).
- The `calls` suite runs the call agent against a fixed set of partner personas and reports rule-following, outcome accuracy and opt-out compliance (EVAL §4.7).

## 7. Non-functional requirements

| Area | Requirement |
|---|---|
| Performance | Discovery p50 ≤ 25 s, p95 ≤ 45 s for 10 search results. Q&A p95 ≤ 8 s. Call agent p95 ≤ 5 s per turn in text mode on a hosted provider. (HLD §15) |
| Reliability | Any external failure degrades the result instead of crashing the run. Retries use jittered backoff. A run's partial result is always returned. |
| Cost | It runs on free tiers and local models. LLM concurrency is bounded to stay under free-tier rate limits. Roles can use different providers, with a fallback chain and a response cache (ADR-018). Token usage is recorded per run and per call. |
| Safety | Defence in depth against prompt injection from scraped pages (HLD §12). Contacts are deterministically verified. Human approval is required before outreach. |
| Privacy | Business contact data only. Page snapshots and chunks expire after `DATA_TTL_DAYS` (7). No contact values are logged at INFO. |
| Compliance | Outreach identifies the sender, has an honest subject and includes an opt-out line. The call agent discloses that it is an AI (ADR-019) and honours opt-outs immediately. Real voice calls go only to consenting allowlisted numbers. Scraping respects `robots.txt` and rate-limits itself per host. |
| Security | SSRF-safe fetcher. API key on the API when `APP_API_KEY` is set. Rate limits per user/IP. Secrets only in env / Space secrets. |
| Maintainability | Tests run offline in under 60 s. Contracts are documented in the HLD. Every design choice has an ADR. |
| Portability | LLM provider switchable by config (Gemini / Groq / Ollama). Every store has an in-memory fallback for local dev and tests. |

## 8. Release scope

The plan is four weeks. TASKS.md has the week-by-week schedule.

| Release | Milestones (TASKS) | Contents |
|---|---|---|
| **MVP** (`v0.1.0`, end of week 2) | M0–M5 | Charlie + Scout + Chase (email, outbox), API, Gradio UI, local stores |
| **v1 (portfolio-ready)** (`v1.0.0`, end of week 4) | M6–M9 | Call agent in text mode (simulate + live text), safety + tracing, the full eval with published numbers including the `calls` suite, public deploy, README + demo video |
| **Nice-to-have in v1** | T-61 | Live voice in the browser (local speech-to-text and text-to-speech). Cut first if time runs out |
| **Stretch** | M10 | Real Vapi calls, MX checks, Groq router, compound intents, streaming |

**Cut rule.** If the MVP is not working in the browser by the end of week 2, the call agent moves to stretch and v1 ships without it.

## 9. Success metrics

v1 ships when all of the following hold (definitions in EVAL §4, targets in EVAL §5):
- Routing accuracy ≥ 0.90 on the routing test split.
- Email precision ≥ 0.95 and recall ≥ 0.70 on the extraction set.
- Eligible-hallucination rate = 0. Injection attack success = 0.
- Discovery p95 ≤ 45 s on the E2E set.
- Call agent: hard-rule violation rate = 0, outcome accuracy ≥ 0.85 and opt-out compliance = 1.0 on the persona test split.
- The public demo completes the scripted flow (§10) end to end.

## 10. Demo script (used for manual acceptance and the video)

1. Ask "What does a food broker do, and how is it different from a distributor?" → a cited answer.
2. Ask "Write me a poem" → a refusal.
3. Ask "Find partners" → a clarifying question.
4. Ask "Find snack food brokers in California" → a lead table with verified badges and source links.
5. Ask "Draft emails to the top 3 for our gluten-free cookie brand" → three drafts, pending approval.
6. Edit one, reject one, approve one → the approved message appears in the outbox and the rejected one does not.
7. Ask "Call the first lead" → the call plan appears for approval. Approve it.
8. In the Call tab, pick the persona "interested broker" and press Simulate → the transcript appears with each stage labelled, the rule checks per turn and the outcome `meeting_booked`, plus a confirmation email draft pending approval.
9. Switch to live text and play the partner yourself. Say "stop calling me" → the agent apologises and ends the call, and the outcome is `opted_out`.
10. Open the Runs tab → stage timings, token counts and the trace link.

## 11. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Free-tier rate limits or quota changes | Slow or failed runs | Bounded concurrency, retries, provider switch by config, eval LLM-response cache |
| Search returns directories or aggregators instead of companies | Low lead yield | Domain blocklist, partner-type-specific query templates, ablation in the eval |
| Sites block fetching or hide contacts | Low recall | Contact-page discovery, footer and `mailto:` extraction, report recall honestly |
| LLM invents contacts | Wrong outreach | Deterministic verbatim verification; only verified contacts are eligible |
| Prompt injection in scraped pages | Hijacked behaviour | HLD §12 layered defences plus the injection eval |
| Cold starts on free hosting | Slow first demo request | Bake the embedding model into the image, show a warm-up note in the UI |
| Scope creep | Never ships | Binding non-goals; the cut rule in §8; stretch work only after v1 |
| Call agent drifts from the playbook | Wrong claims, ignored opt-outs | Code-enforced stage machine, deterministic reply checks with regenerate-once, the `calls` eval |
| Simulated personas are easier than real people | Optimistic call numbers | Adversarial personas (pushy, off-topic, injection, "are you a bot?"), live-text spot checks, and the limitation stated in the README |
| Friend's Ollama machine offline | Development stalls | Fallback chain to Groq/Gemini and the response cache (ADR-018) |

## 12. Open questions

- Q1. Where does the brand profile come from in the demo? Proposal: a fictional sample brand pre-filled and editable in a UI settings panel.
- Q2. What is the default phone region when the location is missing? Proposal: `US`, configurable.
- Q3. Is Gmail OAuth worth setting up on the hosted demo? Proposal: no. Demo mode uses the outbox, and `gmail_draft` is shown locally in the demo video.
- Q4. Should the call agent support Hindi or Hinglish? Proposal: English in v1. The playbook format allows a second language later; Whisper and Piper both support Hindi.
- Q5. Does live voice run on the public Space? Proposal: try `faster-whisper base` on CPU with a "slow" note. If it is too slow, the Space offers text mode only and the voice demo is shown in the video.
