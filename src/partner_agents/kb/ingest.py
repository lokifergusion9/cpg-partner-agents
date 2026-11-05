"""Load the CPG knowledge-base notes in ``data/kb/`` into the ``cpg_kb`` collection (HLD §6.2).

Run it with ``uv run python -m partner_agents.kb.ingest``. Each note is a Markdown file with a
small front matter block (``doc_id``, ``title``, ``topic``, ``source_url``) followed by the
text. Notes are written in our own words from public sources and cite their source. Running
it again replaces changed notes and removes deleted ones.
"""

import argparse
import asyncio
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, HttpUrl, TypeAdapter, ValidationError

from partner_agents.config import Settings, get_settings
from partner_agents.observability import configure_logging
from partner_agents.retrieval.chunking import TokenCounter, chunk_text
from partner_agents.retrieval.embeddings import SentenceTransformerEmbedder
from partner_agents.retrieval.store import KnowledgeStore, make_client

logger = logging.getLogger(__name__)

# data/kb at the repository root (src/partner_agents/kb/ingest.py is three levels below it).
DEFAULT_KB_DIR = Path(__file__).resolve().parents[3] / "data" / "kb"
REQUIRED_KEYS = ("doc_id", "title", "topic", "source_url")
OPTIONAL_KEYS = ("source_title",)
# Notes are short and focused, so chunks are smaller than the web pages' (HLD §6.2).
KB_CHUNK_TOKENS = 400
KB_CHUNK_OVERLAP = 60
MIN_BODY_WORDS = 40
BOM = chr(0xFEFF)  # a byte-order mark some editors put at the start of a file
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_URL = TypeAdapter(HttpUrl)


class KBError(ValueError):
    """One or more notes are malformed. ``problems`` lists every issue, one per line."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = list(problems)


class KBDoc(BaseModel):
    doc_id: str
    title: str
    topic: str
    source_url: str
    source_title: str | None = None
    body: str


class IngestReport(BaseModel):
    docs: int
    chunks: int
    removed: int


def parse_note(text: str, filename: str) -> KBDoc:
    """Parse one note. Raises ``ValueError`` with a message that names the file."""
    lines = text.lstrip(BOM).splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{filename}: front matter must start with '---'")
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise ValueError(f"{filename}: front matter is not closed with '---'") from None
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep or not key.strip():
            raise ValueError(f"{filename}: front matter line {line!r} is not 'key: value'")
        meta[key.strip()] = value.strip().strip("\"'")
    unknown = set(meta) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
    if unknown:
        raise ValueError(f"{filename}: unknown front matter keys {sorted(unknown)}")
    missing = [key for key in REQUIRED_KEYS if not meta.get(key)]
    if missing:
        raise ValueError(f"{filename}: missing front matter {missing}")
    stem = Path(filename).stem
    if meta["doc_id"] != stem:
        raise ValueError(f"{filename}: doc_id {meta['doc_id']!r} must match the file name")
    for key in ("doc_id", "topic"):
        if not _SLUG.match(meta[key]):
            raise ValueError(f"{filename}: {key} {meta[key]!r} must be lower-case words and dashes")
    try:
        _URL.validate_python(meta["source_url"])
    except ValidationError:
        raise ValueError(
            f"{filename}: source_url {meta['source_url']!r} is not an http(s) URL"
        ) from None
    body = "\n".join(lines[end + 1 :]).strip()
    if len(body.split()) < MIN_BODY_WORDS:
        raise ValueError(f"{filename}: the note has fewer than {MIN_BODY_WORDS} words")
    return KBDoc(body=body, **meta)


def load_notes(directory: Path = DEFAULT_KB_DIR) -> list[KBDoc]:
    """Every ``*.md`` note in ``directory``, sorted by doc_id. Reports all problems at once."""
    files = sorted(directory.glob("*.md"))
    if not files:
        raise KBError([f"no notes found in {directory}"])
    docs: list[KBDoc] = []
    problems: list[str] = []
    for path in files:
        try:
            docs.append(parse_note(path.read_text(encoding="utf-8"), path.name))
        except ValueError as exc:
            problems.append(str(exc))
    seen: set[str] = set()
    for doc in docs:
        if doc.doc_id in seen:
            problems.append(f"{doc.doc_id}: duplicate doc_id")
        seen.add(doc.doc_id)
    if problems:
        raise KBError(problems)
    return docs


async def ingest(
    docs: Sequence[KBDoc],
    store: KnowledgeStore,
    *,
    max_tokens: int,
    overlap_tokens: int,
    count: TokenCounter | None = None,
) -> IngestReport:
    """Chunk, embed and store ``docs``; remove notes that are no longer present."""
    total = 0
    for doc in docs:
        chunks = await asyncio.to_thread(
            chunk_text,
            f"{doc.title}\n\n{doc.body}",
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            count=count,
        )
        total += await store.upsert_doc(doc.doc_id, doc.title, doc.source_url, doc.topic, chunks)
    removed = await store.prune({doc.doc_id for doc in docs})
    logger.info("knowledge base ingested", extra={"docs": len(docs), "chunks": total})
    return IngestReport(docs=len(docs), chunks=total, removed=removed)


async def run(directory: Path, settings: Settings) -> IngestReport:
    docs = load_notes(directory)
    client = make_client(settings)
    store = KnowledgeStore(client, SentenceTransformerEmbedder(settings.embedding_model))
    try:
        return await ingest(
            docs, store, max_tokens=KB_CHUNK_TOKENS, overlap_tokens=KB_CHUNK_OVERLAP
        )
    finally:
        await store.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="partner_agents.kb.ingest", description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_KB_DIR, help="folder of notes")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    try:
        report = asyncio.run(run(args.dir, settings))
    except KBError as exc:
        print("Knowledge base notes have problems:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"Loaded {report.docs} notes ({report.chunks} chunks) into cpg_kb"
        f"; removed {report.removed} stale"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
