"""Domain models shared across modules (HLD §4).

The call-agent models of HLD §4.1 arrive with T-54. This module imports nothing from the
rest of the package, so every layer can depend on it.
"""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, EmailStr, Field, HttpUrl, computed_field

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
    url: HttpUrl
    title: str
    snippet: str
    rank: int
    provider: str


class PageRef(BaseModel):  # state holds refs; page text lives in the repo (§8)
    url: HttpUrl
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
    contact_page_url: HttpUrl | None = None


class PageRecord(BaseModel):  # the `pages` collection (§8.1); TTL runs from fetched_at
    run_id: str
    url: str
    text: str
    contact_text: str = ""  # footer / contact-element text and mailto:/tel: targets
    fetched_at: AwareDatetime


class RetrievedChunk(BaseModel):
    chunk_id: str
    url: HttpUrl
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
    website: HttpUrl | None
    email: str | None
    email_status: VerificationStatus
    phone: str | None  # E.164 when parseable
    phone_status: VerificationStatus
    source_url: HttpUrl
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
    website: HttpUrl | None = None
    founder_name: str  # the human the call agent books meetings for
    category: str  # "gluten-free cookies"
    target_regions: list[str] = []  # territory stop rule; empty = anywhere
    timezone: str = "UTC"  # IANA name, used by the scheduler
    sign_off: str  # closing line, e.g. "Thanks from all of us at …"
    approved_facts: list[ApprovedFact] = []


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
