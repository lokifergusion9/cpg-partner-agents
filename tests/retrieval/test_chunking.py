import itertools
import random
import re

import pytest

from partner_agents.retrieval.chunking import Chunk, chunk_text

MAX = 50
OVERLAP = 10


def words(text: str) -> int:
    """A deterministic counter: one token per whitespace-separated word."""
    return len(text.split())


def chars(text: str) -> int:
    """One token per 4 characters, so a long unbroken string costs many tokens."""
    return -(-len(text) // 4)


def cut(text: str, **kwargs: object) -> list[Chunk]:
    options: dict[str, object] = {"max_tokens": MAX, "overlap_tokens": OVERLAP, "count": words}
    return chunk_text(text, **{**options, **kwargs})  # type: ignore[arg-type]


def prose(sentences: int, per_sentence: int = 8, seed: int = 1) -> str:
    rng = random.Random(seed)
    vocab = [f"w{n}" for n in range(200)]
    return " ".join(
        " ".join(rng.choice(vocab) for _ in range(per_sentence)) + "." for _ in range(sentences)
    )


def paragraphs(count: int, words_each: int) -> str:
    return "\n\n".join(" ".join(f"p{p}w{n}" for n in range(words_each)) for p in range(count))


SAMPLES = {
    "sentences": prose(60),
    "one long sentence": " ".join(f"word{n}" for n in range(400)),
    "paragraphs": paragraphs(8, 30),
    "mixed": paragraphs(3, 70) + "\n\n" + prose(15) + "\n\n" + "tail " * 20,
    "newlines": "\n".join(f"line {n} of the page text here" for n in range(80)),
}


# --- basics -------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t "])
def test_blank_text_has_no_chunks(text: str) -> None:
    assert cut(text) == []


def test_short_text_is_one_chunk() -> None:
    [only] = cut("  We broker cookies in California.  ")
    assert only.text == "We broker cookies in California."
    assert (only.index, only.tokens) == (0, 5)


@pytest.mark.parametrize("name", SAMPLES)
def test_every_chunk_fits_the_limit(name: str) -> None:
    chunks = cut(SAMPLES[name])
    assert len(chunks) > 1
    assert all(c.tokens <= MAX for c in chunks)
    assert all(words(c.text) == c.tokens for c in chunks)


@pytest.mark.parametrize("name", SAMPLES)
def test_a_chunk_is_a_slice_of_the_input(name: str) -> None:
    text = SAMPLES[name]
    for chunk in cut(text):
        assert text[chunk.start : chunk.end] == chunk.text


@pytest.mark.parametrize("name", SAMPLES)
def test_chunks_jointly_cover_the_text(name: str) -> None:
    text = SAMPLES[name]
    covered = [False] * len(text)
    for chunk in cut(text):
        for position in range(chunk.start, chunk.end):
            covered[position] = True
    missing = [i for i, ch in enumerate(text) if not ch.isspace() and not covered[i]]
    assert missing == []


@pytest.mark.parametrize("name", SAMPLES)
def test_no_word_is_split(name: str) -> None:
    text = SAMPLES[name]
    for chunk in cut(text):
        assert chunk.start == 0 or text[chunk.start - 1].isspace()
        assert chunk.end == len(text) or text[chunk.end].isspace()


@pytest.mark.parametrize("name", SAMPLES)
def test_indexes_are_sequential(name: str) -> None:
    assert [c.index for c in cut(SAMPLES[name])] == list(range(len(cut(SAMPLES[name]))))


# --- overlap ------------------------------------------------------------------------


@pytest.mark.parametrize("name", SAMPLES)
def test_consecutive_chunks_overlap(name: str) -> None:
    text = SAMPLES[name]
    chunks = cut(text)
    for before, after in itertools.pairwise(chunks):
        assert after.start < before.end, "no overlap"
        assert after.start > before.start, "the next chunk must move forward"
        shared = text[after.start : before.end]
        assert 0 < words(shared) <= OVERLAP


def test_the_overlap_is_the_last_words_of_the_previous_chunk() -> None:
    text = SAMPLES["one long sentence"]
    chunks = cut(text)
    for before, after in itertools.pairwise(chunks):
        shared = text[after.start : before.end]
        assert before.text.endswith(shared)
        assert after.text.startswith(shared)


def test_zero_overlap_chunks_are_back_to_back() -> None:
    text = SAMPLES["one long sentence"]
    chunks = cut(text, overlap_tokens=0)
    assert all(a.end <= b.start for a, b in itertools.pairwise(chunks))
    assert " ".join(c.text for c in chunks) == text


def test_the_overlap_is_dropped_when_the_next_piece_needs_the_room() -> None:
    # A 45-word paragraph, then a 48-word one: with 10 words of overlap the second cannot fit.
    first = " ".join(f"a{n}" for n in range(45))
    second = " ".join(f"b{n}" for n in range(48))
    chunks = cut(f"{first}\n\n{second}")
    assert [c.tokens for c in chunks] == [45, 48]
    assert chunks[1].text == second


# --- where it cuts ------------------------------------------------------------------


def test_it_prefers_paragraph_boundaries() -> None:
    first = " ".join(f"a{n}" for n in range(30))
    second = " ".join(f"b{n}" for n in range(30))
    chunks = cut(f"{first}\n\n{second}", max_tokens=40)
    assert chunks[0].text == first


def test_it_prefers_sentence_boundaries_within_a_paragraph() -> None:
    chunks = cut(SAMPLES["sentences"], overlap_tokens=0)
    assert all(c.text.endswith(".") for c in chunks)
    assert all(re.match(r"w\d+", c.text) for c in chunks)


def test_it_falls_back_to_words_when_there_is_no_punctuation() -> None:
    chunks = cut(SAMPLES["one long sentence"])
    assert len(chunks) > 8
    assert all(c.tokens <= MAX for c in chunks)


def test_paragraphs_that_fit_are_kept_whole() -> None:
    text = paragraphs(4, 20)
    chunks = cut(text, overlap_tokens=0)
    assert all(c.text.count("\n\n") in (0, 1) for c in chunks)
    assert chunks[0].text == "\n\n".join(text.split("\n\n")[:2])


def test_a_single_oversized_word_is_the_only_thing_that_gets_cut() -> None:
    blob = "A" * 400
    text = f"before the blob {blob} after the blob"
    chunks = cut(text, max_tokens=20, overlap_tokens=0, count=chars)
    assert all(c.tokens <= 20 for c in chunks)
    assert "".join(c.text.replace(" ", "") for c in chunks) == text.replace(" ", "")
    assert sum(len(c.text) for c in chunks if set(c.text) == {"A"}) == 400


# --- contact flag -------------------------------------------------------------------


def test_chunks_with_a_contact_pattern_are_flagged() -> None:
    filler = " ".join(f"filler{n}" for n in range(60))
    text = f"{filler}\n\nReach us: info [at] acme [dot] com or 310-555-0100\n\n{filler}"
    flags = {c.has_contact_pattern for c in cut(text)}
    assert flags == {True, False}
    [flagged] = [c for c in cut(text, overlap_tokens=0) if c.has_contact_pattern]
    assert "info [at] acme [dot] com" in flagged.text


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        ("write to sales@acme.com", True),
        ("info [at] acme [dot] com", True),
        ("call (310) 555 0100", True),
        ("we sell cookies", False),
    ],
)
def test_contact_flag_examples(text: str, flagged: bool) -> None:
    assert cut(text)[0].has_contact_pattern is flagged


# --- arguments ----------------------------------------------------------------------


@pytest.mark.parametrize(("max_tokens", "overlap"), [(0, 0), (-5, 0), (10, 10), (10, 11), (10, -1)])
def test_bad_limits_are_rejected(max_tokens: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        cut("some text", max_tokens=max_tokens, overlap_tokens=overlap)


def test_chunking_is_deterministic() -> None:
    assert cut(SAMPLES["mixed"]) == cut(SAMPLES["mixed"])


@pytest.mark.parametrize("seed", range(25))
def test_invariants_hold_for_random_texts(seed: int) -> None:
    rng = random.Random(seed)
    pieces: list[str] = []
    for _ in range(rng.randint(1, 60)):
        sentence = " ".join(f"t{rng.randint(0, 99)}" for _ in range(rng.randint(1, 40))) + "."
        pieces.append(sentence)
        pieces.append(rng.choice([" ", " ", "\n", "\n\n", "  \n \n  "]))
    text = "".join(pieces)
    chunks = cut(text)
    assert all(c.tokens <= MAX for c in chunks)
    assert all(text[c.start : c.end] == c.text for c in chunks)
    covered = {i for c in chunks for i in range(c.start, c.end)}
    assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())


# --- the real tokenizer (needs the vocabulary file: network on first use) ------------


@pytest.mark.live
def test_tiktoken_counter_chunks_real_text() -> None:
    from partner_agents.retrieval.chunking import tiktoken_counter

    count = tiktoken_counter()
    text = prose(400, per_sentence=15)
    chunks = chunk_text(text, max_tokens=200, overlap_tokens=30, count=count)
    assert len(chunks) > 3
    assert all(count(c.text) <= 200 for c in chunks)
    assert count("hello world") == 2
    assert count("<|endoftext|>") > 0  # special-token text must not raise
