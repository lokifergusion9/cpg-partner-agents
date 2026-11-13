"""Domain models shared across modules (HLD §4, §4.1).

This module imports nothing from the rest of the package, so every layer can depend on it.
"""

from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    EmailStr,
    Field,
    HttpUrl,
    PlainSerializer,
    computed_field,
)

# An http(s) URL that dumps as a plain string, also in python mode. LangGraph's checkpoint
# serializer msgpacks `model_dump()` output and cannot encode pydantic's Url objects.
Url = Annotated[HttpUrl, PlainSerializer(str, return_type=str)]

PartnerType = Literal["retailer", "distributor", "broker", "manufacturer", "other"]
Intent = Literal["cpg_qa", "discovery", "outreach", "clarify", "out_of_scope"]
VerificationStatus = Literal[
    "verified", "not_in_source", "invalid_format", "mx_failed", "unchecked"
]


class DiscoveryIntent(BaseModel):
    partner_type: PartnerType
    product_category: str  # "snack foods", "cookies"
    location: str | None = None  # "California, US"
    max_results: int = Field(10, ge=1, le=25)
    raw_query: str


class RouteDecision(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    discovery: DiscoveryIntent | None = None  # set iff intent == "discovery"
    outreach_goal: str | None = None  # set iff intent == "outreach"
    outreach_channel: Literal["email", "voice"] = "email"  # "call lead 1" -> "voice"
    lead_selector: str | None = None  # "top 3", "the LA ones"; resolved by Chase
    clarifying_question: str | None = None  # set iff intent == "clarify"
    reason: str  # short, for traces only


class SearchHit(BaseModel):
    url: Url
    title: str
    snippet: str
    rank: int
    provider: str


class PageRef(BaseModel):  # state holds refs; page text lives in the repo (§8)
    url: Url
    title: str | None
    status: Literal[
        "ok",
        "blocked_robots",
        "timeout",
        "http_error",
        "too_large",
        "non_html",
        "ssrf_blocked",
        "injection_flagged",
    ]
    chars: int = 0
    injection_score: float = 0.0
    contact_page_url: Url | None = None


class PageRecord(BaseModel):  # the `pages` collection (§8.1); TTL runs from fetched_at
    run_id: str
    url: str
    text: str
    contact_text: str = ""  # footer / contact-element text and mailto:/tel: targets
    fetched_at: AwareDatetime


class RetrievedChunk(BaseModel):
    chunk_id: str
    url: Url
    chunk_index: int
    text: str
    score: float
    selected_by: Literal["vector", "contact_pattern"]


class KBPassage(BaseModel):  # one `cpg_kb` search hit (§6.2)
    doc_id: str
    title: str
    source_url: str
    chunk_index: int
    text: str
    score: float


class KBAnswer(BaseModel):  # writer output for kb_answer; cites passages as [n]
    answer: str


class ExtractedLead(BaseModel):  # what the LLM returns (per page)
    company_name: str
    partner_type: PartnerType | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    website: str | None = None
    evidence_quote: str = Field(max_length=300)  # verbatim span containing the contact


class ExtractionResult(BaseModel):
    leads: list[ExtractedLead] = Field(default_factory=list, max_length=5)


class Lead(BaseModel):
    id: str
    run_id: str
    user_id: str
    company_name: str
    partner_type: PartnerType | None
    location: str | None
    website: Url | None
    email: str | None
    email_status: VerificationStatus
    phone: str | None  # E.164 when parseable
    phone_status: VerificationStatus
    source_url: Url
    evidence_quote: str
    created_at: AwareDatetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def outreach_eligible(self) -> bool:
        return self.email is not None and self.email_status == "verified"


class ApprovedFact(BaseModel):  # the only claims the call agent may make
    id: str  # "F-3"
    text: str  # "Our cookies are certified gluten-free."
    source: str  # "brand_profile" or a cpg_kb doc_id
    tags: list[str] = []  # matched against partner slots to pick 2-3 facts


class BrandProfile(BaseModel):
    brand_name: str
    product: str
    value_prop: str
    sender_name: str
    sender_email: EmailStr
    website: Url | None = None
    founder_name: str  # the human the call agent books meetings for
    category: str  # "gluten-free cookies"
    target_regions: list[str] = []  # territory stop rule; empty = anywhere
    timezone: str = "UTC"  # IANA name, used by the scheduler
    sign_off: str  # closing line, e.g. "Thanks from all of us at …"
    approved_facts: list[ApprovedFact] = []


class LeadSelection(BaseModel):  # LLM output: chase_select's fallback for free-form selectors
    numbers: list[int] = Field(default_factory=list, max_length=25)  # 1-based list positions


class EmailDraft(BaseModel):  # LLM output
    subject: str = Field(max_length=80)
    body: str  # ≤ 180 words, validated in outreach/policy.py


class OutreachDraft(BaseModel):
    id: str
    lead_id: str
    user_id: str
    run_id: str
    channel: Literal["email", "voice"]
    recipient: str  # email or E.164 phone
    subject: str | None
    body: str  # email body, or the call plan for channel "voice"
    status: Literal[
        "pending_approval",
        "approved",
        "rejected",
        "sent",
        "drafted_in_gmail",
        "outboxed",
        "failed",
        "skipped_duplicate",
        "skipped_ineligible",
        "skipped_cap",
        "call_created",  # voice only
    ]
    prompt_version: str
    created_at: AwareDatetime
    decided_at: AwareDatetime | None = None


class ApprovalDecision(BaseModel):
    approve: list[str] = []  # draft ids
    reject: list[str] = []
    edits: dict[str, EmailDraft] = {}  # draft id -> edited content (implies approve)


class StageError(BaseModel):
    stage: str
    code: str
    message: str
    url: str | None = None


class RunRecord(BaseModel):
    run_id: str
    user_id: str
    thread_id: str
    intent: Intent
    started_at: AwareDatetime
    finished_at: AwareDatetime | None
    timings_ms: dict[str, int]
    tokens: dict[str, int]  # per role
    prompt_versions: dict[str, str]
    errors: list[StageError]
    trace_url: str | None = None
    providers: dict[str, str] = {}  # role -> "provider:model" that actually answered (ADR-018)


# --- Call agent (HLD §4.1, §6.11) ---------------------------------------------------------

CallStage = Literal["opening", "discovery", "qualification", "pitch", "handoff", "closing", "ended"]
CallOutcome = Literal[
    "meeting_booked",
    "callback_booked",
    "not_interested",
    "disqualified",
    "opted_out",
    "wrong_person",
    "escalated_to_human",
    "incomplete",
]
CallMode = Literal["simulate", "live_text", "live_voice", "vapi"]


class PlaybookRule(BaseModel):
    id: str  # stable, e.g. "U-OPT-1", "Q-2"; used in checks and eval
    kind: Literal["must", "never"]
    text: str  # shown to the caller model
    check: Literal["code", "judge", "none"]  # who verifies it (ADR-020)
    check_id: str | None = None  # check="code" only: the C-* check that enforces it


class PlaybookStage(BaseModel):
    name: CallStage
    goal: str
    rules: list[PlaybookRule]
    slots: list[str] = []  # qualification only, in asking order
    required_slots: list[str] = []  # asked even when the partner is in a rush
    fallback_line: str  # used when a reply fails its checks twice


class CallPlaybook(BaseModel):  # loaded from data/playbooks/<id>.toml (calls/playbook.py)
    id: str
    version: str
    language: str = "en"
    universal_rules: list[PlaybookRule]
    stages: list[PlaybookStage]  # validated: the six stages, in order, unique rule ids
    banned_phrases: list[str]  # "guaranteed", "100%", "discount", ...
    max_turns: int = Field(30, ge=1)


class TurnSignals(BaseModel):  # classifier output for one partner turn
    right_person: bool | None = None
    good_time: bool | None = None
    interest: Literal["positive", "negative", "unsure"] | None = None
    opt_out: bool = False  # "don't call me again"
    wants_human: bool = False
    asks_if_ai: bool = False
    distressed: bool = False
    resistance: bool = False  # refuses to share / rejects the suggestion
    wants_callback: bool = False
    in_rush: bool = False  # ask only the required slots
    wants_other_category: str | None = None  # acknowledge, offer to pass on, exit politely
    agrees_to_meeting: bool | None = None
    proposed_time: str | None = None  # raw text; parsed by calls/scheduler.py
    question: str | None = None  # a question the partner asked
    slot_values: dict[str, str] = {}  # qualification slots answered in this turn


DirectiveAction = Literal[
    "open",
    "confirm_person",
    "discover",  # ask the discovery question: taking on new lines in the category?
    "offer_callback",
    "ask_slot",
    "confirm_slot",
    "acknowledge",
    "answer_question",
    "pitch",
    "retry_once",
    "propose_slot",
    "confirm_booking",
    "disclose_ai",
    "close",
    "exit_politely",
]


class Directive(BaseModel):  # what the stage machine tells the writer to do next
    action: DirectiveAction
    slot: str | None = None
    fact_ids: list[str] = []  # approved facts the pitch may use
    proposed_slot: AwareDatetime | None = None


class RuleCheck(BaseModel):
    rule_id: str
    passed: bool
    by: Literal["code", "judge"]
    detail: str | None = None


class CallTurn(BaseModel):
    index: int
    speaker: Literal["agent", "partner"]
    text: str
    stage: CallStage
    signals: TurnSignals | None = None  # partner turns only
    directive: Directive | None = None  # agent turns only
    checks: list[RuleCheck] = []  # agent turns only
    regenerated: bool = False
    used_fallback: bool = False


class CallProgress(BaseModel):  # the stage machine's state (pure data)
    stage: CallStage = "opening"
    slots: dict[str, str] = {}
    retries_used: dict[str, int] = {}  # per stage; at most 1 (rule U-RETRY-1)
    proposed_slot: AwareDatetime | None = None
    turns: int = 0  # partner turns handled
    outcome: CallOutcome | None = None
    booking: Literal["meeting", "callback", "founder"] = "meeting"  # what an agreed time books
    asked_slots: list[str] = []  # qualification slots already asked or confirmed, in order
    in_rush: bool = False  # from now on ask only the required slots
    recovering: bool = False  # distress was acknowledged; a negative next turn ends the call
    closing_outcome: CallOutcome | None = None  # the outcome the call ends with after closing
    open_question: bool = False  # a partner question still to answer before signing off
    partner_time: AwareDatetime | None = None  # a time the partner named that was not bookable


class CallRecord(BaseModel):
    id: str
    user_id: str
    lead_id: str
    run_id: str
    draft_id: str
    playbook_id: str
    playbook_version: str
    prompt_version: str
    mode: CallMode
    persona_id: str | None = None
    progress: CallProgress
    turns: list[CallTurn] = []
    meeting_at: AwareDatetime | None = None
    followup_draft_id: str | None = None  # confirmation email, pending approval
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    tokens: dict[str, int] = {}
    providers: dict[str, str] = {}
