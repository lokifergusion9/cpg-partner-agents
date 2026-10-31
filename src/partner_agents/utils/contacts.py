"""Patterns that spot an email address or phone number in text.

These only *detect* that a contact is probably present (contact-page discovery in the
fetcher, the ``has_contact_pattern`` chunk flag in T-12). Whether a value is real is decided
by ``verify/contacts.py``.
"""

import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

# ``info [at] acme [dot] com``, ``info(at)acme.com``. The bare word form (``info at acme dot
# com``) needs the word "dot" too, so ordinary prose like "look at example.com" is not a hit.
_OBFUSCATED_BRACKETED = (
    r"[\w.+-]+\s*[\[\(\{]\s*at\s*[\]\)\}]\s*[\w-]+(?:\s*(?:[\[\(\{]\s*dot\s*[\]\)\}]|\.)\s*[\w-]+)+"
)
_OBFUSCATED_WORDS = r"[\w.+-]+\s+at\s+[\w-]+(?:\s+dot\s+[\w-]+)+"
OBFUSCATED_EMAIL_RE = re.compile(f"{_OBFUSCATED_BRACKETED}|{_OBFUSCATED_WORDS}", re.IGNORECASE)

# "+<country> ..." international numbers, NANP numbers with separators, or 10 bare digits.
# Dates (2026-10-31) and short ids do not match.
PHONE_RE = re.compile(
    r"(?<![\w+.])(?:"
    r"\+\d{1,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?){2,4}\d{2,4}"
    r"|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"
    r"|\d{10}"
    r")(?!\w)"
)


def has_email(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or OBFUSCATED_EMAIL_RE.search(text))


def has_phone(text: str) -> bool:
    return bool(PHONE_RE.search(text))


def has_contact_pattern(text: str) -> bool:
    return has_email(text) or has_phone(text)
