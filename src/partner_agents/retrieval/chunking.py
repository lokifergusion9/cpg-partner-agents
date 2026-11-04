"""Token-based chunker (HLD §6.5, ADR-011, ADR-025).

``chunk_text`` cuts page text into chunks of at most ``max_tokens`` tokens. It prefers to
break between paragraphs, then between sentences, then between words, so a chunk is always
a contiguous slice of the input and a word is never split. The one exception is a single
"word" longer than the limit on its own (a base64 blob, say), which is cut by characters.

Consecutive chunks overlap by up to ``overlap_tokens`` tokens, starting on a word boundary.
Each chunk is flagged with ``has_contact_pattern`` so evidence selection can keep chunks that
look like they hold an email or phone number even when they score low on similarity.
"""

import re
from collections.abc import Callable
from functools import lru_cache

from pydantic import BaseModel

from partner_agents.utils.contacts import has_contact_pattern

TokenCounter = Callable[[str], int]

DEFAULT_CHUNK_TOKENS = 1000
DEFAULT_OVERLAP_TOKENS = 150

# Where to cut, coarse to fine. A span ends after each match, so the separator stays with
# the text before it and the spans cover the input without gaps.
_SEPARATORS = (
    re.compile(r"\n[ \t]*\n\s*"),  # paragraph
    re.compile(r"(?<=[.!?])\s+"),  # sentence
    re.compile(r"\s+"),  # word
)
_WORD = re.compile(r"\S+")
# Characters per attempt when a single word is too long (a guess; the loop corrects it).
_CHARS_PER_TOKEN = 4

Span = tuple[int, int]
Atom = tuple[int, int, int]  # start, end, tokens


class Chunk(BaseModel):
    index: int
    text: str
    start: int  # offsets into the input: text == input[start:end]
    end: int
    tokens: int
    has_contact_pattern: bool


@lru_cache(maxsize=1)
def tiktoken_counter() -> TokenCounter:
    """Count tokens with tiktoken ``cl100k_base``. Loads the vocabulary on first use."""
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")

    def count(text: str) -> int:
        return len(encoding.encode(text, disallowed_special=()))

    return count


def _split(text: str, start: int, end: int, level: int) -> list[Span]:
    spans: list[Span] = []
    cursor = start
    for match in _SEPARATORS[level].finditer(text, start, end):
        if match.end() > cursor and match.end() < end:
            spans.append((cursor, match.end()))
            cursor = match.end()
    spans.append((cursor, end))
    return spans


def _split_long_word(
    text: str, start: int, end: int, max_tokens: int, count: TokenCounter
) -> list[Atom]:
    atoms: list[Atom] = []
    cursor = start
    while cursor < end:
        size = min(end - cursor, max_tokens * _CHARS_PER_TOKEN)
        while size > 1 and count(text[cursor : cursor + size]) > max_tokens:
            size = max(1, size // 2)
        atoms.append((cursor, cursor + size, count(text[cursor : cursor + size])))
        cursor += size
    return atoms


def _atoms(
    text: str, start: int, end: int, level: int, max_tokens: int, count: TokenCounter
) -> list[Atom]:
    """The largest pieces of ``text[start:end]`` that each fit in ``max_tokens``."""
    tokens = count(text[start:end])
    if tokens <= max_tokens:
        return [(start, end, tokens)]
    for lv in range(level, len(_SEPARATORS)):
        spans = _split(text, start, end, lv)
        if len(spans) > 1:
            return [
                atom for a, b in spans for atom in _atoms(text, a, b, lv + 1, max_tokens, count)
            ]
    return _split_long_word(text, start, end, max_tokens, count)


def _overlap_start(
    text: str, start: int, end: int, overlap_tokens: int, count: TokenCounter
) -> int | None:
    """Start of the last whole words of ``text[start:end]`` that fit in ``overlap_tokens``.

    Never the first word of the chunk, so the next chunk always moves forward.
    """
    if overlap_tokens <= 0:
        return None
    best: int | None = None
    for match in reversed(list(_WORD.finditer(text, start, end))):
        if match.start() <= start or count(text[match.start() : end]) > overlap_tokens:
            break
        best = match.start()
    return best


def chunk_text(
    text: str,
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    count: TokenCounter | None = None,
) -> list[Chunk]:
    """Cut ``text`` into overlapping chunks of at most ``max_tokens`` tokens."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be in [0, max_tokens)")
    if not text.strip():
        return []
    counter = count or tiktoken_counter()
    atoms = _atoms(text, 0, len(text), 0, max_tokens, counter)
    chunks: list[Chunk] = []
    i = 0
    carry: int | None = None  # where the overlap with the previous chunk starts
    while i < len(atoms):
        start = atoms[i][0] if carry is None else carry
        used = 0 if carry is None else counter(text[start : atoms[i][0]])
        j = i
        while j < len(atoms) and used + atoms[j][2] <= max_tokens:
            used += atoms[j][2]
            j += 1
        if j == i:
            if carry is not None:
                carry = None  # no room for the overlap: start this chunk clean
                continue
            j = i + 1  # a lone atom that the counter rates over the limit: take it anyway
        # Token counts are not exactly additive across pieces, so confirm the real size.
        while j - 1 > i and counter(text[start : atoms[j - 1][1]].strip()) > max_tokens:
            j -= 1
        if carry is not None and counter(text[start : atoms[j - 1][1]].strip()) > max_tokens:
            carry = None
            continue
        end = atoms[j - 1][1]
        piece = text[start:end]
        lead = len(piece) - len(piece.lstrip())
        body = piece.strip()
        chunks.append(
            Chunk(
                index=len(chunks),
                text=body,
                start=start + lead,
                end=start + lead + len(body),
                tokens=counter(body),
                has_contact_pattern=has_contact_pattern(body),
            )
        )
        i = j
        carry = (
            _overlap_start(text, start, end, overlap_tokens, counter) if i < len(atoms) else None
        )
    return chunks
