from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from partner_agents import schemas as s

NOW = datetime(2026, 11, 1, 9, 30, tzinfo=UTC)


def make_lead(**overrides: Any) -> s.Lead:
    fields: dict[str, Any] = {
        "id": "l_1",
        "run_id": "r_1",
        "user_id": "u_1",
        "company_name": "Acme Foods",
        "partner_type": "broker",
        "location": "Los Angeles, CA",
        "website": "https://acme-foods.example",
        "email": "sales@acme-foods.example",
        "email_status": "verified",
        "phone": "+13105550100",
        "phone_status": "verified",
        "source_url": "https://acme-foods.example/contact",
        "evidence_quote": "Email sales@acme-foods.example",
        "created_at": NOW,
    }
    return s.Lead(**{**fields, **overrides})


def make_brand(**overrides: Any) -> s.BrandProfile:
    fields: dict[str, Any] = {
        "brand_name": "Crumb & Co",
        "product": "Gluten-free cookies",
        "value_prop": "Small-batch cookies for health-minded shoppers",
        "sender_name": "Asha Rao",
        "sender_email": "asha@crumbandco.example",
        "founder_name": "Asha Rao",
        "category": "gluten-free cookies",
        "sign_off": "Thanks from all of us at Crumb & Co",
    }
    return s.BrandProfile(**{**fields, **overrides})


def make_draft(**overrides: Any) -> s.OutreachDraft:
    fields: dict[str, Any] = {
        "id": "d_1",
        "lead_id": "l_1",
        "user_id": "u_1",
        "run_id": "r_1",
        "channel": "email",
        "recipient": "sales@acme-foods.example",
        "subject": "Gluten-free cookies for Acme",
        "body": "Hello",
        "status": "pending_approval",
        "prompt_version": "chase-v1",
        "created_at": NOW,
    }
    return s.OutreachDraft(**{**fields, **overrides})


def make_run_record(**overrides: Any) -> s.RunRecord:
    fields: dict[str, Any] = {
        "run_id": "r_1",
        "user_id": "u_1",
        "thread_id": "t_1",
        "intent": "discovery",
        "started_at": NOW,
        "finished_at": None,
        "timings_ms": {"scout_search": 420},
        "tokens": {"extractor": 1200},
        "prompt_versions": {"scout": "scout-v1"},
        "errors": [s.StageError(stage="scout_fetch", code="timeout", message="slow", url=None)],
    }
    return s.RunRecord(**{**fields, **overrides})


def discovery(**overrides: Any) -> s.DiscoveryIntent:
    fields: dict[str, Any] = {
        "partner_type": "broker",
        "product_category": "cookies",
        "raw_query": "cookie brokers in California",
    }
    return s.DiscoveryIntent(**{**fields, **overrides})


# --- DiscoveryIntent ----------------------------------------------------------------


def test_max_results_defaults_to_ten() -> None:
    assert discovery().max_results == 10


@pytest.mark.parametrize("value", [1, 10, 25])
def test_max_results_accepts_bounds(value: int) -> None:
    assert discovery(max_results=value).max_results == value


@pytest.mark.parametrize("value", [0, -1, 26, 100])
def test_max_results_rejects_out_of_bounds(value: int) -> None:
    with pytest.raises(ValidationError):
        discovery(max_results=value)


def test_unknown_partner_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        discovery(partner_type="influencer")


# --- RouteDecision ------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_route_confidence_is_bounded(confidence: float) -> None:
    with pytest.raises(ValidationError):
        s.RouteDecision(intent="cpg_qa", confidence=confidence, reason="x")


def test_route_defaults_to_the_email_channel() -> None:
    route = s.RouteDecision(intent="outreach", confidence=0.9, reason="x")
    assert route.outreach_channel == "email"
    assert route.discovery is None


def test_unknown_intent_is_rejected() -> None:
    with pytest.raises(ValidationError):
        s.RouteDecision(intent="chitchat", confidence=0.9, reason="x")  # type: ignore[arg-type]


# --- Lead ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("email", "status", "eligible"),
    [
        ("sales@acme-foods.example", "verified", True),
        ("sales@acme-foods.example", "not_in_source", False),
        ("sales@acme-foods.example", "invalid_format", False),
        ("sales@acme-foods.example", "mx_failed", False),
        ("sales@acme-foods.example", "unchecked", False),
        (None, "verified", False),
        (None, "unchecked", False),
    ],
)
def test_outreach_eligible(email: str | None, status: str, eligible: bool) -> None:
    assert make_lead(email=email, email_status=status).outreach_eligible is eligible


def test_phone_status_does_not_make_a_lead_eligible() -> None:
    lead = make_lead(email=None, email_status="unchecked", phone_status="verified")
    assert lead.outreach_eligible is False


def test_outreach_eligible_is_serialised() -> None:
    assert make_lead().model_dump()["outreach_eligible"] is True


# --- ExtractionResult / ExtractedLead ------------------------------------------------


def extracted(**overrides: Any) -> s.ExtractedLead:
    fields: dict[str, Any] = {"company_name": "Acme", "evidence_quote": "sales@acme.example"}
    return s.ExtractedLead(**{**fields, **overrides})


def test_empty_extraction_is_accepted() -> None:
    assert s.ExtractionResult().leads == []
    assert s.ExtractionResult(leads=[]).leads == []


def test_extraction_accepts_five_leads_and_rejects_six() -> None:
    assert len(s.ExtractionResult(leads=[extracted()] * 5).leads) == 5
    with pytest.raises(ValidationError):
        s.ExtractionResult(leads=[extracted()] * 6)


def test_evidence_quote_is_capped_at_300_chars() -> None:
    assert extracted(evidence_quote="x" * 300)
    with pytest.raises(ValidationError):
        extracted(evidence_quote="x" * 301)


# --- EmailDraft / BrandProfile ------------------------------------------------------


def test_email_subject_is_capped_at_80_chars() -> None:
    assert s.EmailDraft(subject="s" * 80, body="b").subject == "s" * 80
    with pytest.raises(ValidationError):
        s.EmailDraft(subject="s" * 81, body="b")


def test_brand_profile_validates_the_sender_email() -> None:
    assert make_brand().timezone == "UTC"
    with pytest.raises(ValidationError):
        make_brand(sender_email="not-an-email")


def test_brand_profile_carries_approved_facts() -> None:
    fact = s.ApprovedFact(id="F-1", text="Certified gluten-free.", source="brand_profile")
    brand = make_brand(approved_facts=[fact])
    assert brand.approved_facts[0].tags == []


# --- defaults are not shared between instances --------------------------------------


def test_mutable_defaults_are_per_instance() -> None:
    first, second = make_brand(), make_brand()
    first.target_regions.append("CA")
    assert second.target_regions == []
    decision_a, decision_b = s.ApprovalDecision(), s.ApprovalDecision()
    decision_a.approve.append("d_1")
    assert decision_b.approve == []


# --- JSON round-trip ----------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        discovery(location="California, US", max_results=5),
        s.RouteDecision(
            intent="discovery", confidence=0.92, discovery=discovery(), reason="named a region"
        ),
        s.SearchHit(url="https://a.example/x", title="A", snippet="s", rank=1, provider="tavily"),  # type: ignore[arg-type]
        s.PageRef(url="https://a.example/x", title=None, status="ok", chars=900),  # type: ignore[arg-type]
        s.RetrievedChunk(
            chunk_id="c1",
            url="https://a.example/x",  # type: ignore[arg-type]
            chunk_index=2,
            text="t",
            score=0.4,
            selected_by="contact_pattern",
        ),
        s.ExtractionResult(leads=[extracted(partner_type="retailer")]),
        make_lead(),
        make_brand(approved_facts=[s.ApprovedFact(id="F-1", text="t", source="brand_profile")]),
        s.EmailDraft(subject="Hi", body="Hello"),
        make_draft(),
        s.ApprovalDecision(
            approve=["d_1"], reject=["d_2"], edits={"d_3": s.EmailDraft(subject="s", body="b")}
        ),
        make_run_record(finished_at=NOW, trace_url="https://trace.example/1"),
    ],
    ids=lambda model: type(model).__name__,
)
def test_json_round_trip(model: BaseModel) -> None:
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model


def test_lead_round_trip_ignores_the_computed_field() -> None:
    """The serialised ``outreach_eligible`` is derived, so a stored copy cannot override it."""
    payload = make_lead(email_status="not_in_source").model_dump_json()
    assert '"outreach_eligible":false' in payload
    assert s.Lead.model_validate_json(payload).outreach_eligible is False
