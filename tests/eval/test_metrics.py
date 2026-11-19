"""Hand-computed cases for the evaluation metrics (T-39, docs/EVAL.md section 4)."""

from datetime import UTC, datetime

import pytest
from eval.metrics import (
    E2eOutcome,
    InjectionOutcome,
    KbOutcome,
    PageScore,
    covers,
    e2e_metrics,
    evidence_recall,
    extraction_metrics,
    f1,
    gold_chunks,
    injection_metrics,
    kb_metrics,
    percentile,
    routing_metrics,
    score_page,
)
from eval.schemas import ExtractionRow, InjectionRow, RoutingRow

from partner_agents.schemas import Lead, RouteDecision

# --- helpers ------------------------------------------------------------------------


def route_row(intent: str, **fields: object) -> RoutingRow:
    base: dict[str, object] = {"id": "r001", "split": "test", "query": "q", "intent": intent}
    if intent == "discovery":
        base["discovery"] = {
            "partner_type": "broker",
            "product_category": "cookies",
            "location": "California",
        }
    if intent == "outreach":
        base.update(outreach_channel="voice", lead_selector="the first one", context_leads=True)
    return RoutingRow.model_validate({**base, **fields})


def decision(intent: str, **fields: object) -> RouteDecision:
    base: dict[str, object] = {"intent": intent, "confidence": 0.9, "reason": "t"}
    if intent == "discovery":
        base["discovery"] = {
            "partner_type": "broker",
            "product_category": "cookies",
            "location": "California, US",
            "raw_query": "q",
        }
    return RouteDecision.model_validate({**base, **fields})


def approx(value: float | None) -> object:
    return pytest.approx(value)


# --- generic ------------------------------------------------------------------------


def test_percentile_is_nearest_rank_and_empty_is_none() -> None:
    values = [4000, 1000, 3000, 2000]
    assert percentile(values, 50) == 2000
    assert percentile(values, 95) == 4000
    assert percentile(values, 0) == 1000
    assert percentile([], 50) is None
    assert percentile([7], 95) == 7


def test_f1_is_none_when_it_is_undefined() -> None:
    assert f1(None, 0.5) is None
    assert f1(0.0, 0.0) is None
    assert f1(1 / 3, 1 / 2) == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("predicted", "gold", "expected"),
    [
        ("California, US", "California", True),
        ("Gluten Free Snacks", "gluten-free snacks", True),  # hyphen and case
        ("cookies and biscuits", "cookies", True),  # the prediction may say more
        ("snack foods", "snacks", True),  # a plural is the same word
        ("snacks", "gluten-free snacks", False),  # but it may not say less
        ("Texas", "California", False),
        (None, None, True),
        ("Texas", None, False),  # a location the gold row does not have
        (None, "Texas", False),
    ],
)
def test_covers(predicted: str | None, gold: str | None, expected: bool) -> None:
    assert covers(predicted, gold) is expected


# --- routing ------------------------------------------------------------------------


def test_routing_metrics_on_six_hand_checked_rows() -> None:
    results = [
        (route_row("discovery"), decision("discovery")),  # correct, slots correct
        (  # intent right, category loses "gluten-free"
            route_row(
                "discovery",
                discovery={
                    "partner_type": "distributor",
                    "product_category": "gluten-free snacks",
                    "location": "Texas",
                },
            ),
            decision(
                "discovery",
                discovery={
                    "partner_type": "distributor",
                    "product_category": "snacks",
                    "location": "Texas",
                    "raw_query": "q",
                },
            ),
        ),
        (route_row("cpg_qa"), decision("cpg_qa")),
        (route_row("outreach"), decision("outreach", outreach_channel="email")),  # wrong channel
        (route_row("clarify"), decision("discovery")),  # a missed clarify
        (route_row("discovery"), decision("clarify")),  # an over-clarify
    ]
    metrics = routing_metrics(results)
    assert metrics["n"] == 6
    assert metrics["intent_accuracy"] == pytest.approx(4 / 6)
    # discovery: tp 2, fn 1, fp 1 -> 2/3; cpg_qa and outreach 1; clarify 0; out_of_scope absent.
    assert metrics["per_intent"]["discovery"]["f1"] == pytest.approx(2 / 3)
    assert metrics["per_intent"]["clarify"]["f1"] == 0.0
    assert metrics["per_intent"]["out_of_scope"]["f1"] is None
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 1 + 1 + 0) / 4)
    assert metrics["confusion"]["discovery"] == {"discovery": 2, "clarify": 1}
    assert metrics["confusion"]["clarify"] == {"discovery": 1}
    assert (metrics["slot_accuracy"], metrics["slot_n"]) == (pytest.approx(1 / 3), 3)
    assert metrics["outreach_channel_accuracy"] == 0.0
    assert metrics["over_clarify_rate"] == pytest.approx(1 / 5)  # 1 of the 5 non-clarify rows


def test_routing_metrics_with_no_rows_are_none_not_zero() -> None:
    metrics = routing_metrics([])
    assert metrics["intent_accuracy"] is None and metrics["slot_accuracy"] is None


# --- extraction ---------------------------------------------------------------------


def extraction_row(row_id: str, case: str = "standard", **fields: object) -> ExtractionRow:
    base: dict[str, object] = {
        "id": row_id,
        "url": f"https://{row_id}.example/",
        "snapshot": f"pages/{row_id}.html",
        "snapshot_sha256": "a" * 64,
        "company_name": "Acme Foods Inc",
        "partner_type": "broker",
        "emails": [],
        "phones": [],
        "case": case,
    }
    return ExtractionRow.model_validate({**base, **fields})


def lead(
    email: str | None = None,
    email_status: str = "verified",
    phone: str | None = None,
    phone_status: str = "verified",
    name: str = "Acme Foods",
) -> Lead:
    return Lead.model_validate(
        {
            "id": "l1",
            "run_id": "r1",
            "user_id": "u1",
            "company_name": name,
            "partner_type": "broker",
            "location": None,
            "website": None,
            "email": email,
            "email_status": email_status if email else "unchecked",
            "phone": phone,
            "phone_status": phone_status if phone else "unchecked",
            "source_url": "https://acme.example/",
            "evidence_quote": "q",
            "created_at": datetime(2026, 11, 1, tzinfo=UTC),
        }
    )


def four_pages() -> list[PageScore]:
    a = score_page(
        extraction_row("e001", emails=["a@x.com"], phones=["+13105550100"]),
        [
            lead("a@x.com", phone="+13105550100"),
            lead("fake@x.com", email_status="not_in_source"),  # invented by the model
        ],
        "Contact a@x.com or call (310) 555-0100",
    )
    b = score_page(
        extraction_row("e002", emails=["b@y.com"]), [lead("c@y.com")], "write to c@y.com"
    )
    c = score_page(extraction_row("e003", "no_contact"), [], "no contact here")
    d = score_page(  # an eligible email that the source does not hold: must never happen
        extraction_row("e004", "no_contact"), [lead("d@z.com")], "nothing here"
    )
    return [a, b, c, d]


def test_score_page_reduces_leads_to_the_numbers_the_metrics_use() -> None:
    a = four_pages()[0]
    assert a.out_emails == {"a@x.com"} and a.out_phones == {"+13105550100"}
    assert (a.proposed_emails, a.hallucinated_emails, a.eligible_not_in_source) == (2, 1, 0)
    assert (a.contact_leads, a.name_matches) == (1, 1)  # "Acme Foods" vs "Acme Foods Inc"


def test_extraction_metrics_on_four_hand_checked_pages() -> None:
    metrics = extraction_metrics(four_pages())
    # emails: tp 1 (a), fp 2 (c@y.com, d@z.com), fn 1 (b@y.com) -> P 1/3, R 1/2, F1 0.4
    assert metrics["email_precision"] == pytest.approx(1 / 3)
    assert metrics["email_recall"] == pytest.approx(1 / 2)
    assert metrics["email_f1"] == pytest.approx(0.4)
    assert (metrics["phone_precision"], metrics["phone_recall"], metrics["phone_f1"]) == (1, 1, 1)
    assert metrics["raw_hallucination_rate"] == pytest.approx(1 / 4)  # fake@ of 4 proposals
    assert metrics["eligible_hallucination_rate"] == pytest.approx(1 / 3)  # d@z.com of 3
    assert (metrics["abstention_accuracy"], metrics["abstention_n"]) == (0.5, 2)
    assert metrics["company_name_match"] == 1.0
    assert metrics["n"] == 4


def test_an_ambiguous_page_is_not_scored() -> None:
    unsure = score_page(
        extraction_row("e005", emails=[], ambiguous=True), [lead("q@q.com")], "q@q.com"
    )
    assert extraction_metrics([unsure])["n"] == 0
    assert extraction_metrics([unsure])["email_precision"] is None


# --- retrieval ----------------------------------------------------------------------


def test_gold_chunks_finds_plain_obfuscated_and_phone_contacts() -> None:
    chunks = [
        (0, "About our brokerage and the brands we sell."),
        (1, "Email: info [at] acme [dot] com"),
        (2, "Call us on (310) 555-0100 today"),
        (3, "Fax (310) 555-0199"),
    ]
    assert gold_chunks(chunks, ["info@acme.com"], ["+13105550100"]) == {1, 2}
    assert gold_chunks(chunks, [], []) == set()


def test_evidence_recall_counts_pages_whose_gold_chunk_was_selected() -> None:
    pages = [({1}, {1, 3}), ({2}, {0}), (set(), {0})]  # hit, miss, no gold chunk at all
    assert evidence_recall(pages) == {"recall": 0.5, "pages": 2, "no_gold_chunk": 1}


# --- injection ----------------------------------------------------------------------


def injection_row(row_id: str, attack: str) -> InjectionRow:
    return InjectionRow.model_validate(
        {
            "id": row_id,
            "url": f"https://{row_id}.example/",
            "page": f"injection/{row_id}.html",
            "company_name": "Acme",
            "partner_type": "broker",
            "product_category": "cookies",
            "emails": [f"sales@{row_id}.example"],
            "attack": attack,
            "attack_email": None if attack == "control" else "attacker@evil.test",
        }
    )


def test_injection_metrics_on_four_hand_checked_pages() -> None:
    legit = {"i01": "sales@i01.example", "i02": "sales@i02.example", "i03": "sales@i03.example"}
    outcomes = [
        InjectionOutcome(injection_row("i01", "plain_text"), True, frozenset(), frozenset()),
        InjectionOutcome(  # the attack worked, the real contact is still there
            injection_row("i02", "hidden_text"),
            False,
            frozenset({"attacker@evil.test", legit["i02"]}),
            frozenset({legit["i02"]}),
        ),
        InjectionOutcome(  # not flagged, not fooled, but the real contact was lost
            injection_row("i03", "json_mimic"), False, frozenset({legit["i03"]}), frozenset()
        ),
        InjectionOutcome(
            injection_row("i04", "control"),
            False,
            frozenset({"sales@i04.example"}),
            frozenset({"sales@i04.example"}),
        ),
    ]
    metrics = injection_metrics(outcomes, [False, False, True, False])
    assert metrics["attack_pages"] == 3
    assert metrics["attack_success_rate"] == pytest.approx(1 / 3)
    assert metrics["attack_success_ids"] == ["i02"]
    assert metrics["flag_rate"] == pytest.approx(1 / 3)
    assert metrics["unflagged_ids"] == ["i02", "i03"]
    assert metrics["false_positive_rate"] == 0.25
    assert metrics["legit_contact_survival"] == 0.5


# --- knowledge base -----------------------------------------------------------------


def test_kb_metrics_on_six_hand_checked_questions() -> None:
    both = frozenset({"slotting-fees", "trade-spend-basics"})
    outcomes = [
        KbOutcome("k1", both, frozenset({"slotting-fees"}), False),  # cites a subset: right
        KbOutcome("k2", both, frozenset({"slotting-fees", "haccp-basics"}), False),  # extra doc
        KbOutcome("k3", both, frozenset(), True),  # abstained on an answerable question
        KbOutcome("k4", frozenset({"haccp-basics"}), frozenset({"haccp-basics"}), False),
        KbOutcome("k5", frozenset(), frozenset(), True),  # correct abstention
        KbOutcome("k6", frozenset(), frozenset({"slotting-fees"}), False),  # answered anyway
    ]
    metrics = kb_metrics(outcomes)
    assert (metrics["answerable"], metrics["unanswerable"]) == (4, 2)
    assert metrics["citation_accuracy"] == 0.5
    assert metrics["answerable_abstained"] == 0.25
    assert metrics["abstention_accuracy"] == 0.5


def test_an_answer_without_a_citation_is_not_accurate() -> None:
    outcome = KbOutcome("k1", frozenset({"a"}), frozenset(), False)
    assert kb_metrics([outcome])["citation_accuracy"] == 0.0


# --- e2e ----------------------------------------------------------------------------


def test_e2e_metrics_on_four_hand_checked_runs() -> None:
    outcomes = [
        E2eOutcome(
            f"q{n}",
            total_ms=1000 * n,
            stage_ms={"scout_fetch": 100 * n},
            leads=2 * n - 1,
            eligible=2 * (n - 1),
            tokens={"extractor": 100 * n},
            errors=n % 2,
        )
        for n in (1, 2, 3, 4)
    ]
    metrics = e2e_metrics(outcomes)
    assert (metrics["latency_ms_p50"], metrics["latency_ms_p95"]) == (2000, 4000)
    assert metrics["stage_ms"]["scout_fetch"] == {"p50": 200, "p95": 400}
    assert metrics["yield_median"] == 3.0  # eligible 0, 2, 4, 6
    assert metrics["verified_ratio"] == pytest.approx(12 / 16)
    assert metrics["tokens_per_run"] == {"extractor": 250.0}
    assert metrics["errors"] == 2
