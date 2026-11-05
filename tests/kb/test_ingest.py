import asyncio
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient

from partner_agents.fakes import HashingEmbedder
from partner_agents.kb import ingest as kb
from partner_agents.retrieval.store import KnowledgeStore

BODY = " ".join(f"word{n}" for n in range(60))
URL = "https://a.example/"


def note(**overrides: str | None) -> str:
    meta = {
        "doc_id": "slotting-fees",
        "title": "Slotting fees",
        "topic": "retail-economics",
        "source_url": "https://en.wikipedia.org/wiki/Slotting_fee",
    }
    meta.update({k: v for k, v in overrides.items() if k != "body" and v is not None})
    for key, value in overrides.items():
        if value is None:
            meta.pop(key, None)
    body = overrides.get("body", BODY) or BODY
    front = "\n".join(f"{k}: {v}" for k, v in meta.items())
    return f"---\n{front}\n---\n{body}\n"


def run[T](coro: Any) -> T:
    return asyncio.run(coro)  # type: ignore[no-any-return]


def words(text: str) -> int:
    return len(text.split())


# --- the real notes -----------------------------------------------------------------


def test_the_shipped_notes_are_valid_and_the_right_size() -> None:
    docs = kb.load_notes()
    assert 15 <= len(docs) <= 30
    assert len({d.doc_id for d in docs}) == len(docs)


def test_every_shipped_note_has_its_front_matter() -> None:
    for path in sorted(kb.DEFAULT_KB_DIR.glob("*.md")):
        doc = kb.parse_note(path.read_text(encoding="utf-8"), path.name)
        assert doc.doc_id == path.stem
        assert doc.title and doc.topic
        assert doc.source_url.startswith("https://")
        assert doc.source_title, f"{path.name} should name its source"


def test_the_shipped_notes_cover_the_planned_topics() -> None:
    ids = {d.doc_id for d in kb.load_notes()}
    for expected in [
        "channel-structure-overview",
        "broker-vs-distributor-vs-wholesaler",
        "slotting-fees",
        "trade-spend-basics",
        "retail-calendar-454",
        "private-label-basics",
        "direct-store-delivery",
        "food-safety-fsma-overview",
        "pitching-to-buyers",
    ]:
        assert expected in ids


def test_shipped_notes_are_short_and_not_copied_text() -> None:
    for doc in kb.load_notes():
        assert 60 <= words(doc.body) <= 350, doc.doc_id


# --- parsing ------------------------------------------------------------------------


def test_a_valid_note_parses() -> None:
    doc = kb.parse_note(note(source_title="Slotting fee, Wikipedia"), "slotting-fees.md")
    assert (doc.doc_id, doc.topic, doc.source_title) == (
        "slotting-fees",
        "retail-economics",
        "Slotting fee, Wikipedia",
    )
    assert doc.body == BODY


def test_quotes_around_values_are_removed() -> None:
    doc = kb.parse_note(note(title='"Slotting fees"'), "slotting-fees.md")
    assert doc.title == "Slotting fees"


def test_a_byte_order_mark_is_tolerated() -> None:
    assert kb.parse_note(kb.BOM + note(), "slotting-fees.md").doc_id == "slotting-fees"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("no front matter\n" + BODY, "must start with"),
        ("---\ndoc_id: x\n", "not closed"),
        (note(doc_id=None), "missing front matter"),
        (note(title=None), "missing front matter"),
        (note(topic=None), "missing front matter"),
        (note(source_url=None), "missing front matter"),
        (note(source_url="not a url"), "not an http(s) URL"),
        (note(source_url="ftp://example.com/x"), "not an http(s) URL"),
        (note(doc_id="other-name"), "must match the file name"),
        (note(topic="Retail Economics"), "lower-case words and dashes"),
        (note(body="too short"), "fewer than"),
        (note().replace("title:", "titel:"), "unknown front matter keys"),
        ("---\njust text\n---\n" + BODY, "not 'key: value'"),
    ],
)
def test_malformed_notes_are_rejected_with_a_clear_message(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)) as caught:
        kb.parse_note(text, "slotting-fees.md")
    assert "slotting-fees.md" in str(caught.value)


def test_load_notes_reports_every_problem_at_once(tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text(note(doc_id="good"), encoding="utf-8")
    (tmp_path / "bad-one.md").write_text(note(doc_id="bad-one", topic="NOPE"), encoding="utf-8")
    (tmp_path / "bad-two.md").write_text("nothing", encoding="utf-8")
    with pytest.raises(kb.KBError) as caught:
        kb.load_notes(tmp_path)
    assert len(caught.value.problems) == 2


def test_an_empty_folder_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(kb.KBError, match="no notes found"):
        kb.load_notes(tmp_path)


def test_notes_are_loaded_in_a_stable_order(tmp_path: Path) -> None:
    for name in ("b-note", "a-note", "c-note"):
        (tmp_path / f"{name}.md").write_text(note(doc_id=name), encoding="utf-8")
    assert [d.doc_id for d in kb.load_notes(tmp_path)] == ["a-note", "b-note", "c-note"]


# --- ingestion ----------------------------------------------------------------------


@pytest.fixture
def store() -> Iterator[KnowledgeStore]:
    client = AsyncQdrantClient(":memory:")
    yield KnowledgeStore(client, HashingEmbedder())
    run(client.close())


def go(store: KnowledgeStore, docs: list[kb.KBDoc], **kwargs: Any) -> kb.IngestReport:
    options: dict[str, Any] = {"max_tokens": 1000, "overlap_tokens": 150, "count": words}
    return run(kb.ingest(docs, store, **{**options, **kwargs}))


def test_every_shipped_note_is_loaded_into_the_collection(store: KnowledgeStore) -> None:
    docs = kb.load_notes()
    report = go(store, docs)
    assert report.docs == len(docs)
    assert report.chunks == run(store.count()) >= len(docs)
    assert run(store.doc_ids()) == {d.doc_id for d in docs}
    assert report.removed == 0


def test_ingesting_twice_does_not_duplicate(store: KnowledgeStore) -> None:
    docs = kb.load_notes()
    first = go(store, docs)
    second = go(store, docs)
    assert first.chunks == second.chunks == run(store.count())


def test_a_changed_note_is_replaced_and_a_removed_note_is_pruned(store: KnowledgeStore) -> None:
    a = kb.KBDoc(doc_id="a-note", title="A", topic="t", source_url="https://a.example/", body=BODY)
    b = kb.KBDoc(doc_id="b-note", title="B", topic="t", source_url="https://b.example/", body=BODY)
    go(store, [a, b])
    shorter = a.model_copy(update={"body": "short replacement " + BODY[:200]})
    report = go(store, [shorter])
    assert report.removed == 1
    assert run(store.doc_ids()) == {"a-note"}


def test_a_note_that_shrinks_loses_its_old_chunks(store: KnowledgeStore) -> None:
    long_body = " ".join(f"w{n}" for n in range(300))
    doc = kb.KBDoc(
        doc_id="a-note", title="A", topic="t", source_url="https://a.example/", body=long_body
    )
    many = go(store, [doc], max_tokens=60, overlap_tokens=10).chunks
    assert many > 3
    few = go(
        store, [doc.model_copy(update={"body": BODY})], max_tokens=60, overlap_tokens=10
    ).chunks
    assert run(store.count()) == few < many


def test_payload_matches_the_documented_fields(store: KnowledgeStore) -> None:
    go(store, kb.load_notes()[:1])

    async def payloads() -> list[dict[str, Any]]:
        points, _ = await store._client.scroll("cpg_kb", limit=10, with_payload=True)
        return [p.payload or {} for p in points]

    [first, *_] = run(payloads())
    assert set(first) == {"doc_id", "title", "source_url", "topic", "chunk_index", "text"}
    assert first["text"].startswith(first["title"])


def test_the_title_is_part_of_the_embedded_text(store: KnowledgeStore) -> None:
    doc = kb.KBDoc(
        doc_id="a-note",
        title="Zebra crossing",
        topic="t",
        source_url="https://a.example/",
        body=BODY,
    )
    go(store, [doc])

    async def text() -> str:
        points, _ = await store._client.scroll("cpg_kb", limit=1, with_payload=True)
        return str((points[0].payload or {})["text"])

    assert run(text()).startswith("Zebra crossing\n\n")


# --- the command --------------------------------------------------------------------


def test_main_reports_problems_and_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bad.md").write_text("nothing here", encoding="utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert kb.main(["--dir", str(tmp_path)]) == 1
    assert "bad.md" in capsys.readouterr().err


def test_main_loads_notes_and_prints_a_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("one-note", "two-note"):
        (tmp_path / f"{name}.md").write_text(note(doc_id=name), encoding="utf-8")

    class FakeEmbedder(HashingEmbedder):
        def __init__(self, *_: Any, **__: Any) -> None:
            super().__init__()

    monkeypatch.setattr(kb, "SentenceTransformerEmbedder", FakeEmbedder)
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "q"))
    assert kb.main(["--dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.startswith("Loaded 2 notes (2 chunks) into cpg_kb")
