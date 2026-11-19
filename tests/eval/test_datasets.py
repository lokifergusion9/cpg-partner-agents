"""Schema and mix tests for the evaluation datasets (T-36 onwards, docs/EVAL.md section 3)."""

import re
from collections import Counter
from pathlib import Path

import pytest
from eval.schemas import (
    DATASETS,
    ExtractionRow,
    InjectionRow,
    KbRow,
    RoutingRow,
    load_extraction,
    load_injection,
    load_kb,
    load_routing,
    snapshot_problems,
)
from pydantic import ValidationError

from partner_agents.prompts.charlie import FEW_SHOT

# EVAL 3.1: discovery 35%, cpg_qa 25%, outreach 15%, clarify 15%, out_of_scope 10% per split.
SHARES = {
    "discovery": 0.35,
    "cpg_qa": 0.25,
    "outreach": 0.15,
    "clarify": 0.15,
    "out_of_scope": 0.10,
}
SIZES = {"dev": 20, "test": 60}
HINGLISH = ("me ", " ke ", "dhundo", "chahiye", "kya ", "kuch ", "hota hai", "maangte")


def rows(split: str | None = None) -> list[RoutingRow]:
    return [row for row in load_routing() if split in (None, row.split)]


def normal(query: str) -> str:
    return " ".join(query.lower().split())


# --- routing ------------------------------------------------------------------------


def test_there_are_80_rows_20_dev_and_60_test_with_unique_sequential_ids() -> None:
    everything = rows()
    assert Counter(row.split for row in everything) == SIZES
    assert [row.id for row in everything] == [f"r{n:03d}" for n in range(1, 81)]


def test_no_query_appears_twice_so_no_row_is_shared_between_splits() -> None:
    queries = [normal(row.query) for row in rows()]
    assert len(set(queries)) == len(queries)


@pytest.mark.parametrize("split", ["dev", "test"])
def test_each_split_has_the_documented_intent_distribution(split: str) -> None:
    counts = Counter(row.intent for row in rows(split))
    expected = {intent: round(share * SIZES[split]) for intent, share in SHARES.items()}
    assert dict(counts) == expected


def test_the_test_split_counts_are_the_ones_in_the_docs() -> None:
    counts = Counter(row.intent for row in rows("test"))
    assert counts == {"discovery": 21, "cpg_qa": 15, "outreach": 9, "clarify": 9, "out_of_scope": 6}


@pytest.mark.parametrize("split", ["dev", "test"])
def test_about_a_third_of_the_outreach_rows_are_call_requests(split: str) -> None:
    outreach = [row for row in rows(split) if row.intent == "outreach"]
    voice = [row for row in outreach if row.outreach_channel == "voice"]
    assert len(voice) * 3 == len(outreach)
    assert all(row.context_leads for row in outreach)


def test_the_test_split_has_the_hard_cases_the_docs_ask_for() -> None:
    test = rows("test")
    hinglish = [row for row in test if any(word in row.query.lower() for word in HINGLISH)]
    assert len(hinglish) >= 4  # discovery and a cpg_qa and a clarify among them
    assert {row.intent for row in hinglish} >= {"discovery", "cpg_qa", "clarify"}
    # A discovery-looking request that misses a slot is expected to be a clarify.
    assert any(row.intent == "clarify" and row.query.startswith("find") for row in test)
    # CPG-adjacent but out of scope.
    assert any(row.intent == "out_of_scope" and "brand" in row.query for row in test)
    assert any(row.discovery and row.discovery.location is None for row in test)
    typos = {"r023", "r034"}  # "Califronia", "distributers"
    assert typos <= {row.id for row in test}


def test_the_discovery_rows_cover_every_partner_type_in_both_splits() -> None:
    for split in ("dev", "test"):
        types = {row.discovery.partner_type for row in rows(split) if row.discovery}
        assert {"broker", "distributor", "retailer", "manufacturer"} <= types


def test_no_few_shot_example_of_the_router_prompt_comes_from_the_test_split() -> None:
    """CLAUDE.md rule 9: the prompt uses dev rows only, by id and by query text."""
    by_id = {row.id: row for row in rows()}
    test_queries = {normal(row.query) for row in rows("test")}
    for example_id, (query, _) in FEW_SHOT.items():
        assert by_id[example_id].split == "dev"
        assert normal(query) not in test_queries


# --- extraction (T-37) --------------------------------------------------------------


def extraction() -> list[ExtractionRow]:
    return load_extraction()


def test_there_are_30_pages_with_unique_ids_and_urls() -> None:
    pages = extraction()
    assert [row.id for row in pages] == [f"e{n:03d}" for n in range(1, 31)]
    assert len({row.url for row in pages}) == 30
    assert len({row.snapshot_sha256 for row in pages}) == 30  # no two snapshots are the same page


def test_the_case_mix_is_the_one_in_the_docs() -> None:
    """EVAL 3.2 asked for 15 / 5 / 5 / 5. Only two real obfuscated pages were found among about
    130 candidate sites, so the mix is 19 / 2 / 5 / 4 (the docs and TASKS.md say so)."""
    counts = Counter(row.case for row in extraction())
    assert counts == {"standard": 19, "obfuscated": 2, "contact_page": 5, "no_contact": 4}


def test_the_partner_types_are_the_documented_ones() -> None:
    counts = Counter(row.partner_type for row in extraction())
    assert counts == {"retailer": 13, "distributor": 12, "broker": 4, "manufacturer": 1}


def test_gold_contacts_are_normalised_and_the_cases_are_consistent() -> None:
    for row in extraction():
        assert row.emails == sorted(set(row.emails)) or len(row.emails) < 2
        if row.case == "no_contact":
            assert not row.emails and not row.phones
        else:
            assert row.emails or row.phones
        assert row.contact_snapshot is not None or row.case != "contact_page"
        assert not row.ambiguous


def test_obfuscated_and_contact_page_rows_describe_how_the_contact_is_hidden() -> None:
    for row in extraction():
        if row.case == "obfuscated":
            assert "data-cfemail" in row.notes and row.emails
        if row.case == "contact_page":
            assert row.contact_url and row.emails and "contact" in row.notes


def test_every_snapshot_that_is_present_still_matches_its_hash() -> None:
    """The snapshots are third-party HTML and gitignored, so a fresh clone has none: those are
    skipped. A snapshot that exists must be the one that was labelled."""
    problems = [p for row in extraction() for p in snapshot_problems(row)]
    assert [p for p in problems if p.startswith("mismatch")] == []
    present = sum(not p.startswith("missing") for p in problems)
    if len(problems) and all(p.startswith("missing") for p in problems):
        pytest.skip("no snapshots on this machine (eval/fixtures/pages is gitignored)")
    assert present >= 0


def test_a_missing_or_edited_snapshot_is_reported(tmp_path: Path) -> None:
    row = extraction()[0]
    assert snapshot_problems(row, tmp_path) == [f"missing {row.snapshot}"]
    (tmp_path / "pages").mkdir()
    (tmp_path / row.snapshot).write_text("edited", encoding="utf-8")
    assert snapshot_problems(row, tmp_path) == [f"mismatch {row.snapshot}"]


def good_page(**fields: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "e001",
        "url": "https://acme.example/",
        "snapshot": "pages/e001.html",
        "snapshot_sha256": "a" * 64,
        "company_name": "Acme",
        "partner_type": "broker",
        "emails": ["sales@acme.example"],
        "phones": [],
        "case": "standard",
    }
    return {**base, **fields}


@pytest.mark.parametrize(
    "bad",
    [
        good_page(emails=["Sales@Acme.example"]),  # not lowercase
        good_page(phones=["310-555-0100"]),  # not E.164
        good_page(emails=["a@acme.example", "a@acme.example"]),  # listed twice
        good_page(case="no_contact"),  # a no_contact page with a gold email
        good_page(emails=[], phones=[]),  # a standard page with no contact
        good_page(case="contact_page"),  # without a contact snapshot
        good_page(contact_url="https://acme.example/contact"),  # a partial contact snapshot
        good_page(snapshot="pages/other.html"),
        good_page(snapshot_sha256="xyz"),
        good_page(partner_type="agency"),
        good_page(extra="x"),
    ],
)
def test_a_malformed_extraction_row_is_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExtractionRow.model_validate(bad)


def test_a_good_extraction_row_and_an_ambiguous_one_are_accepted() -> None:
    assert ExtractionRow.model_validate(good_page()).emails == ["sales@acme.example"]
    unsure = good_page(emails=[], phones=[], ambiguous=True, notes="image-only contact")
    assert ExtractionRow.model_validate(unsure).ambiguous


# --- injection (T-38) ---------------------------------------------------------------

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "kb"
ATTACK_ORDER = [  # EVAL 3.4, in the order of the table
    "plain_text",
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


def injection() -> list[InjectionRow]:
    return load_injection()


def page_html(row: InjectionRow) -> str:
    return (DATASETS / row.page).read_text(encoding="utf-8")


def test_there_are_ten_planted_pages_in_the_documented_order() -> None:
    pages = injection()
    assert [row.id for row in pages] == [f"i{n:02d}" for n in range(1, 11)]
    assert [row.attack for row in pages] == ATTACK_ORDER
    assert len({row.url for row in pages}) == 10
    assert sum(row.attack != "control" for row in pages) == 9


def test_every_page_exists_and_shows_its_legitimate_contact() -> None:
    for row in injection():
        html = page_html(row)
        assert f'href="mailto:{row.emails[0]}"' in html
        assert row.company_name.split()[0] in html


def test_attack_pages_carry_the_attack_email_and_the_control_does_not() -> None:
    for row in injection():
        html = page_html(row)
        if row.attack == "control":
            assert "evil.test" not in html
        else:
            assert row.attack_email and row.attack_email in html
            assert row.attack_email not in row.emails


def test_the_hidden_attack_is_hidden_and_the_role_tags_are_text_not_markup() -> None:
    by_id = {row.id: page_html(row) for row in injection()}
    assert "display:none" in by_id["i04"]
    assert "&lt;/untrusted_source&gt;" in by_id["i03"]  # escaped, so it reaches the model as text


@pytest.mark.parametrize(
    "bad",
    [
        {"attack": "plain_text", "attack_email": None},  # an attack page needs its email
        {"attack": "control", "attack_email": "attacker@evil.test"},
        {"attack": "plain_text", "attack_email": "sales@acme.example"},  # equals the real contact
        {"attack": "sql"},
        {"page": "other/i01.html"},
        {"emails": []},
        {"surprise": 1},
    ],
)
def test_a_malformed_injection_row_is_rejected(bad: dict[str, object]) -> None:
    good: dict[str, object] = {
        "id": "i01",
        "url": "https://acme.example/",
        "page": "injection/i01.html",
        "company_name": "Acme",
        "partner_type": "broker",
        "product_category": "cookies",
        "emails": ["sales@acme.example"],
        "attack": "plain_text",
        "attack_email": "attacker@evil.test",
    }
    assert InjectionRow.model_validate(good).attack == "plain_text"
    with pytest.raises(ValidationError):
        InjectionRow.model_validate({**good, **bad})


# --- knowledge base (T-38) ----------------------------------------------------------


def kb_doc_ids() -> set[str]:
    ids = set()
    for note in KB_DIR.glob("*.md"):
        match = re.search(r"^doc_id:\s*(\S+)", note.read_text(encoding="utf-8"), re.MULTILINE)
        assert match, f"{note.name} has no doc_id"
        ids.add(match.group(1))
    return ids


def test_there_are_25_questions_20_answerable_and_5_not() -> None:
    questions = load_kb()
    assert [row.id for row in questions] == [f"k{n:03d}" for n in range(1, 26)]
    assert sum(row.answerable for row in questions) == 20
    assert sum(not row.answerable for row in questions) == 5
    assert len({" ".join(row.question.lower().split()) for row in questions}) == 25


def test_every_expected_doc_id_is_a_real_note_and_none_repeats() -> None:
    known = kb_doc_ids()
    for row in load_kb():
        assert set(row.expected_doc_ids) <= known, row.id
        assert len(set(row.expected_doc_ids)) == len(row.expected_doc_ids)


def test_a_malformed_kb_row_is_rejected() -> None:
    assert KbRow.model_validate({"id": "k001", "question": "q?"}).answerable is False
    for bad in ({"id": "q1"}, {"question": ""}, {"expected_doc_ids": "x"}, {"extra": 1}):
        with pytest.raises(ValidationError):
            KbRow.model_validate({"id": "k001", "question": "q?", **bad})


# --- the row model itself -----------------------------------------------------------


def valid(**fields: object) -> dict[str, object]:
    base: dict[str, object] = {"id": "r001", "split": "dev", "query": "q", "intent": "clarify"}
    return {**base, **fields}


@pytest.mark.parametrize(
    "bad",
    [
        valid(intent="discovery"),  # a discovery row without slots
        valid(discovery={"partner_type": "broker", "product_category": "x"}),  # slots, wrong intent
        valid(intent="outreach"),  # an outreach row without a channel
        valid(intent="outreach", outreach_channel="voice", lead_selector="x"),  # no context
        valid(outreach_channel="email"),  # a channel on a non-outreach row
        valid(intent="maybe"),
        valid(split="train"),
        valid(id="x1"),
        valid(query=""),
        valid(surprise=1),
    ],
)
def test_a_malformed_row_is_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RoutingRow.model_validate(bad)


def test_a_well_formed_discovery_row_is_accepted() -> None:
    row = RoutingRow.model_validate(
        valid(
            intent="discovery",
            discovery={"partner_type": "broker", "product_category": "cookies", "location": None},
        )
    )
    assert row.discovery is not None and row.discovery.location is None
