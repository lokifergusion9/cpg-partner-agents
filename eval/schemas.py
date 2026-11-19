"""Row models and loaders for the evaluation datasets (docs/EVAL.md section 3).

Every dataset row is validated on load, so a typo in a label fails a test instead of silently
changing a metric. Rows forbid unknown keys.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from partner_agents.schemas import Intent, PartnerType

DATASETS = Path(__file__).resolve().parent / "datasets"
Split = Literal["dev", "test"]


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldDiscovery(Row):
    partner_type: PartnerType
    product_category: str = Field(min_length=1)
    location: str | None = None


class RoutingRow(Row):
    """One routing query and its gold route (EVAL section 3.1)."""

    id: str = Field(pattern=r"^r\d{3}$")
    split: Split
    query: str = Field(min_length=1)
    intent: Intent
    discovery: GoldDiscovery | None = None  # set iff intent == "discovery"
    outreach_channel: Literal["email", "voice"] | None = None  # set iff intent == "outreach"
    lead_selector: str | None = None  # set iff intent == "outreach"
    context_leads: bool = False  # the runner seeds a prior run with leads (outreach only)

    @model_validator(mode="after")
    def _fields_match_the_intent(self) -> Self:
        if (self.discovery is not None) != (self.intent == "discovery"):
            raise ValueError("discovery is set exactly when the intent is discovery")
        outreach = self.intent == "outreach"
        if outreach and not (self.outreach_channel and self.lead_selector and self.context_leads):
            raise ValueError("an outreach row needs a channel, a selector and context_leads")
        if not outreach and (self.outreach_channel or self.lead_selector or self.context_leads):
            raise ValueError("only an outreach row has a channel, a selector or context_leads")
        return self


FIXTURES = Path(__file__).resolve().parent / "fixtures"  # snapshots live under pages/ (gitignored)
ExtractionCase = Literal["standard", "obfuscated", "contact_page", "no_contact"]


class ExtractionRow(Row):
    """One snapshotted page and the contacts a human sees on it (EVAL section 3.2)."""

    id: str = Field(pattern=r"^e\d{3}$")
    url: str = Field(pattern=r"^https?://\S+$")
    snapshot: str = Field(pattern=r"^pages/e\d{3}\.html$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contact_url: str | None = Field(default=None, pattern=r"^https?://\S+$")
    contact_snapshot: str | None = Field(default=None, pattern=r"^pages/e\d{3}_contact\.html$")
    contact_snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    company_name: str = Field(min_length=1)
    partner_type: PartnerType
    emails: list[str] = []  # lowercase, as a human reads them on the page and the contact page
    phones: list[str] = []  # E.164
    case: ExtractionCase
    notes: str = ""
    ambiguous: bool = False  # excluded from scoring

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        contact = (self.contact_url, self.contact_snapshot, self.contact_snapshot_sha256)
        if any(contact) and not all(contact):
            raise ValueError("contact_url, contact_snapshot and its sha256 go together")
        if self.case == "contact_page" and not all(contact):
            raise ValueError("a contact_page row needs its contact snapshot")
        if any(contact) and self.case not in ("contact_page", "no_contact"):
            raise ValueError("only a contact_page or no_contact row has a contact snapshot")
        for email in self.emails:
            if email != email.strip().lower() or "@" not in email:
                raise ValueError(f"email {email!r} must be lowercase")
        for phone in self.phones:
            if not re.fullmatch(r"\+[1-9]\d{6,14}", phone):
                raise ValueError(f"phone {phone!r} must be E.164")
        if len(set(self.emails)) != len(self.emails) or len(set(self.phones)) != len(self.phones):
            raise ValueError("a contact is listed twice")
        if self.case == "no_contact" and (self.emails or self.phones):
            raise ValueError("a no_contact page has no gold contact")
        if self.case != "no_contact" and not (self.emails or self.phones) and not self.ambiguous:
            raise ValueError("only a no_contact page may have no gold contact")
        return self


def snapshot_problems(row: ExtractionRow, root: Path = FIXTURES) -> list[str]:
    """``missing <file>`` or ``mismatch <file>`` for each snapshot of ``row`` that is not intact.

    The snapshots are third-party HTML and are not committed, so ``missing`` is normal on a
    fresh clone; ``mismatch`` means a snapshot was edited or re-fetched after labelling.
    """
    files = [(row.snapshot, row.snapshot_sha256)]
    if row.contact_snapshot and row.contact_snapshot_sha256:
        files.append((row.contact_snapshot, row.contact_snapshot_sha256))
    problems = []
    for name, expected in files:
        path = root / name
        if not path.exists():
            problems.append(f"missing {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"mismatch {name}")
    return problems


def load_extraction(path: Path = DATASETS / "extraction.jsonl") -> list[ExtractionRow]:
    return [ExtractionRow.model_validate(row) for row in read_jsonl(path)]


InjectionAttack = Literal[
    "plain_text",
    "role_tags",
    "hidden_text",
    "ai_directive",
    "reveal_prompt",
    "mark_verified",
    "multilingual",
    "json_mimic",
    "control",
]


class InjectionRow(Row):
    """A planted page: a legitimate contact plus one attack, or a benign control (EVAL 3.4).

    The pages are written by us (``eval/datasets/injection/``) and are committed.
    """

    id: str = Field(pattern=r"^i\d{2}$")
    url: str = Field(pattern=r"^https://\S+$")
    page: str = Field(pattern=r"^injection/i\d{2}\.html$")
    company_name: str = Field(min_length=1)
    partner_type: PartnerType
    product_category: str = Field(min_length=1)
    location: str | None = None
    emails: list[str] = Field(min_length=1)  # the legitimate contact
    phones: list[str] = []
    attack: InjectionAttack
    attack_email: str | None = None  # must never reach a lead; None only for the control

    @model_validator(mode="after")
    def _control_has_no_attack(self) -> Self:
        if (self.attack == "control") != (self.attack_email is None):
            raise ValueError("the control page has no attack_email and every attack page has one")
        if self.attack_email in self.emails:
            raise ValueError("the attack email cannot also be the legitimate contact")
        return self


class KbRow(Row):
    """A knowledge-base question; no expected docs means the base cannot answer it (EVAL 3.6)."""

    id: str = Field(pattern=r"^k\d{3}$")
    question: str = Field(min_length=1)
    expected_doc_ids: list[str] = []
    gist: str = ""  # what a correct answer says, for the human reading a report

    @property
    def answerable(self) -> bool:
        return bool(self.expected_doc_ids)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_routing(path: Path = DATASETS / "routing.jsonl") -> list[RoutingRow]:
    return [RoutingRow.model_validate(row) for row in read_jsonl(path)]


def load_injection(path: Path = DATASETS / "injection.jsonl") -> list[InjectionRow]:
    return [InjectionRow.model_validate(row) for row in read_jsonl(path)]


def load_kb(path: Path = DATASETS / "kb.jsonl") -> list[KbRow]:
    return [KbRow.model_validate(row) for row in read_jsonl(path)]
