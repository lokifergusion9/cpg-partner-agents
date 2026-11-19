"""Row models and loaders for the evaluation datasets (docs/EVAL.md section 3).

Every dataset row is validated on load, so a typo in a label fails a test instead of silently
changing a metric. Rows forbid unknown keys.
"""

import json
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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_routing(path: Path = DATASETS / "routing.jsonl") -> list[RoutingRow]:
    return [RoutingRow.model_validate(row) for row in read_jsonl(path)]
