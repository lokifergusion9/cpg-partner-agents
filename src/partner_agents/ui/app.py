"""The Gradio UI, mounted at ``/ui`` (ADR-010, PRD §10).

Tabs: Chat (with stage progress), Leads, Approvals (approve, edit or reject drafts; the outbox),
Call (simulate a persona or play the partner as live text; handlers in ``ui/calls.py``), Runs
(timings, tokens, errors, trace link) and Settings (user id and brand profile). The UI
talks to the graph in-process through the same ``run_turn`` / ``resume_turn`` the API uses.

The handlers at the top are plain functions so they can be tested without a browser; the layout
at the bottom only wires them to components.
"""

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import gradio as gr
from pydantic import ValidationError

from partner_agents.demo import SAMPLE_BRAND
from partner_agents.graph.build import Context, NothingToApprove, resume_turn, run_turn
from partner_agents.schemas import (
    ApprovalDecision,
    ApprovedFact,
    BrandProfile,
    EmailDraft,
    Lead,
    OutreachDraft,
    RunRecord,
)
from partner_agents.ui.calls import (
    CallView,
    list_call_options,
    persona_options,
    play_turn,
    simulate_selected,
    view_call,
)

USER_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

STAGE_LABELS = {
    "charlie_route": "Understanding your request",
    "rate_gate": "Checking the search limit",
    "scout_search": "Searching the web",
    "scout_fetch": "Reading the pages",
    "scout_index": "Indexing the pages",
    "scout_evidence": "Picking the relevant passages",
    "scout_extract": "Extracting contacts",
    "scout_verify": "Checking contacts against the pages",
    "scout_report": "Writing the summary",
    "kb_answer": "Looking in the knowledge base",
    "chase_select": "Choosing the leads",
    "chase_draft": "Writing the drafts",
    "call_plan": "Planning the calls",
    "approval_gate": "Waiting for your approval",
    "chase_dispatch": "Delivering the approved drafts",
}
EMAIL_BADGES = {
    "verified": "✅ verified",
    "mx_failed": "⚠️ domain has no mail server",
    "not_in_source": "❌ not on the page",
    "invalid_format": "❌ not a valid address",
    "unchecked": "— not checked",
}
LEAD_HEADERS = ["Company", "Type", "Location", "Email", "Email status", "Phone", "Source"]
DRAFT_HEADERS = ["Status", "Channel", "To", "Subject", "Decided"]
RUN_HEADERS = ["Run", "Intent", "Started", "Total (ms)", "Tokens", "Errors", "Stages", "Trace"]
APPROVE, EDIT, REJECT = "Approve", "Edit and approve", "Reject"
PENDING_NOTICE = "Some drafts are waiting for you. Open the Approvals tab and decide on them first."
NEW_USER_HELP = "Letters, digits, _ and - only (1 to 64 characters)."
DEMO_BANNER = (
    "**Demo mode.** Approved emails go to a sandbox outbox and nothing is sent, calls are "
    "simulated, a search returns at most 5 results, and your session id is anonymous. "
    "The brand profile below is a fictional sample brand."
)


def new_user_id() -> str:
    return f"u_{uuid.uuid4().hex[:10]}"


def new_thread_id() -> str:
    return f"t_{uuid.uuid4().hex[:12]}"


# --- tables -------------------------------------------------------------------------


def lead_rows(leads: Sequence[Lead]) -> list[list[str]]:
    rows = []
    for lead in leads:
        host = urlparse(str(lead.source_url)).netloc
        rows.append(
            [
                lead.company_name,
                lead.partner_type or "",
                lead.location or "",
                lead.email or "",
                EMAIL_BADGES.get(lead.email_status, lead.email_status) if lead.email else "",
                lead.phone or "",
                f"[{host}]({lead.source_url})",
            ]
        )
    return rows


def draft_rows(drafts: Sequence[OutreachDraft]) -> list[list[str]]:
    return [
        [
            d.status,
            d.channel,
            d.recipient or "(simulated)",
            d.subject or "(call plan)",
            d.decided_at.strftime("%Y-%m-%d %H:%M") if d.decided_at else "",
        ]
        for d in drafts
    ]


def run_rows(runs: Sequence[RunRecord]) -> list[list[str]]:
    rows = []
    for run in runs:
        stages = " · ".join(f"{name} {ms}" for name, ms in run.timings_ms.items())
        errors = ", ".join(f"{e.stage}: {e.code}" for e in run.errors)
        trace = f"[open]({run.trace_url})" if run.trace_url else "—"
        rows.append(
            [
                run.run_id,
                run.intent,
                run.started_at.strftime("%Y-%m-%d %H:%M:%S"),
                str(sum(run.timings_ms.values())),
                ", ".join(f"{role} {n}" for role, n in run.tokens.items()) or "—",
                errors or "—",
                stages or "—",
                trace,
            ]
        )
    return rows


# --- chat ---------------------------------------------------------------------------


@dataclass
class ChatOutcome:
    thread_id: str
    reply: str
    leads: list[Lead]
    pending: dict[str, Any] | None  # {"drafts": [...], "problems": {...}}


async def send_message(
    ctx: Context,
    *,
    user_id: str,
    thread_id: str | None,
    text: str,
    client_ip: str,
    on_node: Callable[[str], None] | None = None,
) -> ChatOutcome:
    """One chat turn. A thread that waits for approval refuses new messages."""
    thread_id = thread_id or new_thread_id()
    async with ctx.lock(thread_id):
        snapshot = await ctx.graph.aget_state({"configurable": {"thread_id": thread_id}})
        if "approval_gate" in snapshot.next:
            waiting = snapshot.interrupts[0].value if snapshot.interrupts else None
            return ChatOutcome(thread_id, PENDING_NOTICE, snapshot.values.get("leads", []), waiting)
        result = await run_turn(
            ctx.graph,
            ctx.deps,
            thread_id=thread_id,
            user_id=user_id,
            text=text,
            rate_keys=[f"user:{user_id}", f"ip:{client_ip}"],
            on_node=on_node,
        )
    return ChatOutcome(
        thread_id, result.reply, result.state.get("leads", []), result.pending_approval
    )


# --- approvals ----------------------------------------------------------------------


def pending_drafts(pending: dict[str, Any] | None) -> list[OutreachDraft]:
    raw = pending.get("drafts", []) if pending else []
    return [OutreachDraft.model_validate(d) for d in raw]


def build_decision(drafts: Sequence[OutreachDraft], fields: Sequence[str]) -> ApprovalDecision:
    """Turn the form (choice, subject, body for each draft) into an ``ApprovalDecision``.

    Raises ``ValueError`` with a readable message for an edit the schema refuses.
    """
    approve: list[str] = []
    reject: list[str] = []
    edits: dict[str, EmailDraft] = {}
    for number, draft in enumerate(drafts):
        choice, subject, body = fields[3 * number : 3 * number + 3]
        if choice == REJECT:
            reject.append(draft.id)
        elif choice == EDIT and draft.channel == "email":
            try:
                edits[draft.id] = EmailDraft(subject=subject.strip(), body=body)
            except ValidationError as exc:
                raise ValueError(f"{draft.recipient}: {exc.errors()[0]['msg']}") from exc
        else:
            approve.append(draft.id)
    return ApprovalDecision(approve=approve, reject=reject, edits=edits)


async def decide(
    ctx: Context,
    *,
    user_id: str,
    thread_id: str | None,
    drafts: Sequence[OutreachDraft],
    fields: Sequence[str],
) -> tuple[str, dict[str, Any] | None]:
    """Resume the paused run. Returns (the reply to show, the approval still pending)."""
    if not thread_id or not drafts:
        return "There is nothing to approve.", None
    try:
        decision = build_decision(drafts, fields)
    except ValueError as exc:
        return f"Could not use your edit: {exc}", {
            "drafts": [d.model_dump(mode="json") for d in drafts]
        }
    async with ctx.lock(thread_id):
        try:
            result = await resume_turn(
                ctx.graph, ctx.deps, thread_id=thread_id, user_id=user_id, decision=decision
            )
        except NothingToApprove:
            return "There is nothing to approve.", None
    return result.reply, result.pending_approval


# --- profile ------------------------------------------------------------------------

PROFILE_FIELDS = [
    "brand_name", "product", "value_prop", "category", "sender_name", "sender_email",
    "founder_name", "sign_off", "website", "target_regions", "timezone", "facts",
]  # fmt: skip


def profile_to_form(profile: BrandProfile | None) -> list[str]:
    if profile is None:
        blank = [""] * len(PROFILE_FIELDS)
        blank[PROFILE_FIELDS.index("timezone")] = "UTC"
        return blank
    return [
        profile.brand_name, profile.product, profile.value_prop, profile.category,
        profile.sender_name, str(profile.sender_email), profile.founder_name, profile.sign_off,
        str(profile.website or ""), ", ".join(profile.target_regions), profile.timezone,
        "\n".join(fact.text for fact in profile.approved_facts),
    ]  # fmt: skip


def profile_from_form(values: Sequence[str], existing: BrandProfile | None) -> BrandProfile:
    """Build a profile from the form. Facts keep their id and tags when the text is unchanged."""
    data = dict(zip(PROFILE_FIELDS, values, strict=True))
    known = {f.text: f for f in existing.approved_facts} if existing else {}
    facts = []
    for number, text in enumerate((t.strip() for t in data["facts"].splitlines()), start=1):
        if text:
            old = known.get(text)
            facts.append(old or ApprovedFact(id=f"F-{number}", text=text, source="brand_profile"))
    return BrandProfile.model_validate(
        {
            **{k: v.strip() for k, v in data.items() if k not in ("facts", "target_regions")},
            "website": data["website"].strip() or None,
            "target_regions": [r.strip() for r in data["target_regions"].split(",") if r.strip()],
            "timezone": data["timezone"].strip() or "UTC",
            "approved_facts": facts,
        }
    )


async def save_profile(ctx: Context, user_id: str, values: Sequence[str]) -> str:
    if not USER_ID.match(user_id):
        return f"Invalid user id. {NEW_USER_HELP}"
    try:
        profile = profile_from_form(values, await ctx.deps.repo.get_profile(user_id))
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return f"Not saved: {problems}"
    await ctx.deps.repo.save_profile(user_id, profile)
    return "Brand profile saved."


async def init_session(ctx: Context, demo_mode: bool) -> tuple[str, list[str]]:
    """A fresh anonymous user id and the profile form for it.

    In demo mode the fictional sample brand is saved for that id, so drafting works at once.
    """
    user_id = new_user_id()
    if not demo_mode:
        return user_id, profile_to_form(None)
    await ctx.deps.repo.save_profile(user_id, SAMPLE_BRAND)
    return user_id, profile_to_form(SAMPLE_BRAND)


# --- layout -------------------------------------------------------------------------

LABELS = {
    "brand_name": "Brand name", "product": "Product", "value_prop": "Value proposition",
    "category": "Category", "sender_name": "Sender name", "sender_email": "Sender email",
    "founder_name": "Founder (takes the meetings)", "sign_off": "Sign-off line",
    "website": "Website (optional)", "target_regions": "Target regions (comma separated)",
    "timezone": "Time zone (IANA name)", "facts": "Approved facts (one per line)",
}  # fmt: skip


def build_ui(get_ctx: Callable[[], Context], demo_mode: bool = False) -> gr.Blocks:
    """The Blocks app. ``get_ctx`` is called per event: the context exists only after startup."""

    async def drafts_of(user_id: str) -> list[list[str]]:
        if not USER_ID.match(user_id):
            return []
        return draft_rows(await get_ctx().deps.repo.list_drafts(user_id))

    async def on_send(
        text: str,
        history: list[dict[str, str]],
        user_id: str,
        thread_id: str | None,
        request: gr.Request,
    ) -> AsyncIterator[tuple[Any, ...]]:
        keep = gr.skip()
        text = text.strip()
        if not text:
            yield history, "", "", thread_id, keep, keep, keep
            return
        if not USER_ID.match(user_id):
            yield history, text, f"Invalid user id. {NEW_USER_HELP}", thread_id, keep, keep, keep
            return
        history = [*history, {"role": "user", "content": text}]
        steps: asyncio.Queue[str] = asyncio.Queue()
        client = request.client.host if request and request.client else "unknown"
        task = asyncio.ensure_future(
            send_message(
                get_ctx(), user_id=user_id, thread_id=thread_id, text=text, client_ip=client,
                on_node=steps.put_nowait,
            )
        )  # fmt: skip
        yield history, "", "Working…", thread_id, keep, keep, keep
        while not task.done():
            try:
                node = await asyncio.wait_for(steps.get(), timeout=0.2)
            except TimeoutError:
                continue
            yield history, "", f"{STAGE_LABELS.get(node, node)}…", thread_id, keep, keep, keep
        outcome = await task
        history = [*history, {"role": "assistant", "content": outcome.reply}]
        notice = "Drafts are waiting in the Approvals tab." if outcome.pending else ""
        yield (
            history, "", notice, outcome.thread_id, outcome.pending,
            lead_rows(outcome.leads), await drafts_of(user_id),
        )  # fmt: skip

    # --- call tab handlers (the logic is in ui/calls.py) ---

    async def call_update(user_id: str, call_id: str | None, view: CallView) -> tuple[Any, ...]:
        options = await list_call_options(get_ctx(), user_id)
        ids = [value for _, value in options]
        pick = gr.Dropdown(choices=options, value=call_id if call_id in ids else None)
        return view.history, view.summary, view.notice, pick

    async def on_calls_refresh(user_id: str, call_id: str | None) -> tuple[Any, ...]:
        if not USER_ID.match(user_id):
            return [], "", NEW_USER_HELP, gr.Dropdown(choices=[], value=None)
        return await call_update(user_id, call_id, await view_call(get_ctx(), user_id, call_id))

    async def on_simulate(
        user_id: str, call_id: str | None, persona: str | None
    ) -> tuple[Any, ...]:
        if not USER_ID.match(user_id):
            return await on_calls_refresh(user_id, call_id)
        view = await simulate_selected(get_ctx(), user_id, call_id, persona)
        return await call_update(user_id, call_id, view)

    async def on_live(user_id: str, call_id: str | None, text: str | None) -> tuple[Any, ...]:
        if not USER_ID.match(user_id):
            return (*await on_calls_refresh(user_id, call_id), "")
        view = await play_turn(get_ctx(), user_id, call_id, text)
        return (*await call_update(user_id, call_id, view), "")

    async def on_live_start(user_id: str, call_id: str | None) -> tuple[Any, ...]:
        update = await on_live(user_id, call_id, None)
        return update[:-1]  # no text box to clear

    with gr.Blocks(title="CPG Partner Agents") as ui:
        gr.Markdown(
            "# CPG Partner Agents\nFind partners, check their contacts against the source page, "
            "and approve every message before it leaves."
        )
        if demo_mode:
            gr.Markdown(DEMO_BANNER)
        thread_state = gr.State(None)
        pending_state = gr.State(None)

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=420, label="Conversation")
            status = gr.Markdown("")
            msg = gr.Textbox(label="Message", placeholder="Find snack food brokers in California")
            send = gr.Button("Send", variant="primary")

        with gr.Tab("Leads"):
            leads_table = gr.Dataframe(
                headers=LEAD_HEADERS,
                datatype=["str", "str", "str", "str", "str", "str", "markdown"],
                interactive=False,
                wrap=True,
                label="Leads from the last search",
            )

        with gr.Tab("Approvals"):
            approval_note = gr.Markdown("No drafts are waiting.")
            outbox_table = gr.Dataframe(
                headers=DRAFT_HEADERS, interactive=False, wrap=True, label="Outbox and history"
            )
            refresh_outbox = gr.Button("Refresh")

            @gr.render(inputs=[pending_state])  # type: ignore[untyped-decorator]
            def draw(pending: dict[str, Any] | None) -> None:
                drafts = pending_drafts(pending)
                if not drafts:
                    return
                problems: dict[str, list[str]] = (pending or {}).get("problems") or {}
                boxes: list[gr.Component] = []
                for draft in drafts:
                    is_email = draft.channel == "email"
                    with gr.Group():
                        what = "Email" if is_email else "Call plan"
                        gr.Markdown(f"**{what} for {draft.recipient or 'a simulated call'}**")
                        if draft.id in problems:
                            gr.Markdown(
                                f"⚠️ Your edit broke a rule: {', '.join(problems[draft.id])}"
                            )
                        choice = gr.Radio(
                            [APPROVE, EDIT, REJECT] if is_email else [APPROVE, REJECT],
                            value=REJECT, label="Decision",
                        )  # fmt: skip
                        subject = gr.Textbox(
                            value=draft.subject or "", label="Subject", visible=is_email
                        )
                        body = gr.Textbox(value=draft.body, label="Body", lines=7)
                    boxes += [choice, subject, body]
                submit = gr.Button("Submit decisions", variant="primary")

                async def on_submit(
                    user_id: str, thread_id: str | None, history: list[dict[str, str]], *fields: str
                ) -> tuple[Any, ...]:
                    reply, still = await decide(
                        get_ctx(),
                        user_id=user_id,
                        thread_id=thread_id,
                        drafts=drafts,
                        fields=fields,
                    )
                    history = [*history, {"role": "assistant", "content": reply}]
                    return history, still, await drafts_of(user_id)

                submit.click(
                    on_submit,
                    inputs=[user_box, thread_state, chatbot, *boxes],
                    outputs=[chatbot, pending_state, outbox_table],
                )

        with gr.Tab("Call") as call_tab:
            gr.Markdown(
                "Approve a call plan in the Approvals tab (say *call the first lead* in Chat), "
                "then play the call here. **Nobody is dialled.** *Simulate* lets a demo persona "
                "answer. *Start live text call* lets you type the partner's lines."
            )
            with gr.Row():
                call_pick = gr.Dropdown(choices=[], label="Call", interactive=True, scale=4)
                refresh_calls = gr.Button("Refresh", scale=1)
            persona_pick = gr.Dropdown(
                choices=persona_options(), label="Persona (for Simulate)", interactive=True
            )
            with gr.Row():
                simulate_btn = gr.Button("Simulate", variant="primary")
                live_btn = gr.Button("Start live text call")
            call_notice = gr.Markdown("")
            call_chat = gr.Chatbot(height=420, label="Transcript")
            with gr.Row():
                call_msg = gr.Textbox(label="Your line as the partner", scale=4)
                call_send = gr.Button("Send", scale=1)
            call_panel = gr.Markdown("")

        with gr.Tab("Runs"):
            runs_table = gr.Dataframe(
                headers=RUN_HEADERS,
                datatype=["str", "str", "str", "str", "str", "str", "str", "markdown"],
                interactive=False,
                wrap=True,
                label="Runs (stage timings in ms)",
            )
            refresh_runs = gr.Button("Refresh")

        with gr.Tab("Settings"):
            user_box = gr.Textbox(label="User id", info=NEW_USER_HELP, visible=not demo_mode)
            gr.Markdown("### Brand profile\nEvery email and call plan is written from this.")
            fields = [
                gr.Textbox(label=LABELS[name], lines=4 if name == "facts" else 1)
                for name in PROFILE_FIELDS
            ]
            with gr.Row():
                load_btn = gr.Button("Load")
                save_btn = gr.Button("Save", variant="primary")
            profile_status = gr.Markdown("")

        async def runs_of(user_id: str) -> list[list[str]]:
            if not USER_ID.match(user_id):
                return []
            return run_rows(await get_ctx().deps.repo.list_runs(user_id))

        async def load_profile(user_id: str) -> list[str]:
            ok = USER_ID.match(user_id)
            profile = await get_ctx().deps.repo.get_profile(user_id) if ok else None
            return profile_to_form(profile)

        async def on_save(user_id: str, *values: str) -> str:
            return await save_profile(get_ctx(), user_id, values)

        async def on_load() -> tuple[Any, ...]:
            user_id, values = await init_session(get_ctx(), demo_mode)
            return user_id, *values

        ui.load(on_load, None, [user_box, *fields])

        for trigger in (send.click, msg.submit):
            trigger(
                on_send,
                inputs=[msg, chatbot, user_box, thread_state],
                outputs=[
                    chatbot,
                    msg,
                    status,
                    thread_state,
                    pending_state,
                    leads_table,
                    outbox_table,
                ],
            )
        call_outputs = [call_chat, call_panel, call_notice, call_pick]
        for trigger in (call_tab.select, refresh_calls.click, call_pick.change):
            trigger(on_calls_refresh, [user_box, call_pick], call_outputs)
        simulate_btn.click(on_simulate, [user_box, call_pick, persona_pick], call_outputs)
        live_btn.click(on_live_start, [user_box, call_pick], call_outputs)
        for trigger in (call_send.click, call_msg.submit):
            trigger(on_live, [user_box, call_pick, call_msg], [*call_outputs, call_msg])
        refresh_outbox.click(drafts_of, user_box, outbox_table)
        refresh_runs.click(runs_of, user_box, runs_table)
        load_btn.click(load_profile, user_box, fields)
        save_btn.click(on_save, [user_box, *fields], profile_status)
        pending_state.change(
            lambda p: (
                "A decision is needed on the drafts below." if p else "No drafts are waiting."
            ),
            pending_state,
            approval_note,
        )
    return ui  # type: ignore[no-any-return]
