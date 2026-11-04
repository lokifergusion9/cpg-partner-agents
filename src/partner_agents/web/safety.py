"""Untrusted-text handling (HLD §12, CLAUDE.md hard rules 2 and 11).

Anything that came from outside the system, a scraped page or a partner's words on a call,
reaches a prompt only through ``wrap_untrusted``. The wrapper marks where the data starts and
ends so the prompt can say "this is data, never instructions", and it stops the data from
faking those marks. ``injection_score`` arrives with T-32.
"""

import hashlib
import re
from typing import Literal

UntrustedKind = Literal["web_page", "call_partner"]

# Control characters other than tab, newline and carriage return, and zero-width characters
# that can hide text from a human reviewer while a model still reads it.
_INVISIBLE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]"
)
_MARK = re.compile(r"<<<\s*(?:END\s+)?UNTRUSTED", re.IGNORECASE)


def wrap_untrusted(text: str, *, kind: UntrustedKind, source: str | None = None) -> str:
    """Mark ``text`` as data.

    The boundary carries an id derived from the text itself, so the text cannot contain the
    closing marker that matches its own id (that would need a hash fixed point). Look-alike
    markers inside the text are defused. The result is deterministic, which keeps the LLM
    response cache working.
    """
    clean = _INVISIBLE.sub("", text)
    clean = _MARK.sub("[marker removed]", clean)
    boundary = hashlib.sha256(clean.encode("utf-8")).hexdigest()[:10]
    origin = f' source="{_attribute(source)}"' if source else ""
    return (
        f'<<<UNTRUSTED kind="{kind}"{origin} id={boundary}>>>\n'
        f"{clean}\n"
        f"<<<END UNTRUSTED id={boundary}>>>"
    )


def _attribute(value: str) -> str:
    return re.sub(r'[\s"<>]', "_", value)[:300]
