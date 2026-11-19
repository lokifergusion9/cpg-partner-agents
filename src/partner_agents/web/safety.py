"""Untrusted-text handling (HLD §12, CLAUDE.md hard rules 2 and 11).

Anything that came from outside the system, a scraped page or a partner's words on a call,
reaches a prompt only through ``wrap_untrusted``. The wrapper marks where the data starts and
ends so the prompt can say "this is data, never instructions", and it stops the data from
faking those marks.

``injection_score`` is the first line of defence for scraped pages (HLD §12, T-32): a cheap
heuristic score in [0, 1] that excludes a page which tries to instruct the model. It is not a
guarantee. The other layers (``wrap_untrusted``, an extractor with no tools, verbatim contact
verification and the approval gate) hold even when a page scores low.
"""

import hashlib
import re
from re import Pattern
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


# --- injection heuristics (T-32) ----------------------------------------------------

DEFAULT_INJECTION_THRESHOLD = 0.6  # Settings.injection_threshold; a score at or above it flags
_FLAGS = re.IGNORECASE


def _rule(pattern: str, flags: int = _FLAGS) -> Pattern[str]:
    return re.compile(pattern, flags)


# name -> (weight, patterns). A signal counts once however often it matches. The weights are
# combined as 1 - prod(1 - w), so two weak signals can flag a page that neither would alone.
_SIGNALS: dict[str, tuple[float, tuple[Pattern[str], ...]]] = {
    "override": (
        0.7,
        (
            _rule(
                r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
                r"\b(previous|prior|above|earlier|all|any|your|these|those)\b[^.\n]{0,30}"
                r"\b(instructions?|rules?|prompts?|guidelines?|directions?)\b"
            ),
            _rule(r"\bnew (instructions?|system prompt)\s*:"),
            _rule(
                r"\b(reveal|show|print|repeat|output)\b[^.\n]{0,30}"
                r"\b(system prompt|your (instructions|prompt|rules))\b"
            ),
            _rule(r"\byou are now (in )?(admin|developer|debug|god|dan|unrestricted|jailbroken)\b"),
            _rule(r"\b(admin|developer|debug|god) mode\b"),
            _rule(r"\bjailbreak(ed)?\b"),
        ),
    ),
    "role_tag": (
        0.6,
        (
            _rule(r"</?\s*(untrusted\w*|system|assistant|instructions?)\s*>"),
            _rule(r"<\|\s*(im_start|im_end|system|assistant)\s*\|>"),
            _rule(r"\[/?(INST|SYS)\]", 0),
            _rule(r"^\s*(system|assistant|developer)\s*:\s*\S", _FLAGS | re.MULTILINE),
            _rule(r"<<<\s*(END\s+)?UNTRUSTED"),
        ),
    ),
    "ai_directive": (
        0.6,
        (
            _rule(
                r"\b(ai|a\.i\.|llm|language models?|chat ?bots?|ai assistants?|ai agents?"
                r"|automated agents?|web crawlers?)\b[^.\n]{0,20}"
                r"\b(must|are required to|have to|need to|shall)\b"
            ),
            _rule(r"\bif you are (an? )?(ai|llm|language model|chat ?bot|ai assistant|bot)\b"),
            _rule(
                r"\b(attention|note to|message for|dear)\s+"
                r"(ai|llms?|ai assistants?|language models?|bots?|crawlers?)\b"
            ),
        ),
    ),
    "action": (
        0.6,
        (
            _rule(
                r"\b(email|send|forward)\b[^.\n]{0,40}"
                r"\b(everyone|all contacts|all leads|every contact|every lead)\b"
            ),
            _rule(
                r"\bmark\b[^.\n]{0,30}\b(contacts?|emails?|leads?|addresses)\b"
                r"[^.\n]{0,20}\bverified\b"
            ),
            _rule(r"\bdo not (tell|inform|mention)\b[^.\n]{0,30}\b(the )?(user|human|operator)\b"),
            _rule(r"\b(call|invoke|run|execute)\b[^.\n]{0,20}\b(tool|function|command)\b"),
        ),
    ),
    "schema_mimicry": (
        0.4,
        (_rule(r'"(email_status|evidence_quote|verified|outreach_eligible)"\s*:'),),
    ),
    "multilingual": (
        0.7,
        (
            _rule(
                r"(अनदेखा|नज़रअंदाज़|नजरअंदाज|भूल\s*जाओ|भूल\s*जाइए)[^.\n]{0,30}(निर्देश|आदेश)"
                r"|(निर्देश|आदेश)\w*[^.\n]{0,30}(अनदेखा|नज़रअंदाज़|नजरअंदाज|भूल)"
            ),
            _rule(r"\bignora (todas )?las instrucciones\b"),
            _rule(r"\bignore[zr]? (toutes )?les instructions\b"),
            _rule(r"\bignorier(e|en)? (alle )?(vorherigen |bisherigen )?anweisungen\b"),
        ),
    ),
}
_DIRECTIVE_SENTENCE = _rule(r"\byou (must|should|will|are required to|have to)\b")
DIRECTIVE_DENSITY = 4  # sentences telling "you" what to do, from which a page looks suspicious
_DENSITY_WEIGHT = 0.3
_HIDDEN_WEIGHT = 0.7


def injection_signals(text: str, hidden_text: str = "") -> dict[str, float]:
    """The signals found in ``text`` (and in text hidden by CSS), by name, with their weights.

    ``hidden_text`` is text that a reader cannot see but a model reads. Hidden text alone is
    normal (menus, screen-reader labels); hidden text that carries an instruction is not.
    """
    clean = _INVISIBLE.sub("", text)
    found = {
        name: weight
        for name, (weight, patterns) in _SIGNALS.items()
        if any(pattern.search(clean) for pattern in patterns)
    }
    if len(_DIRECTIVE_SENTENCE.findall(clean)) >= DIRECTIVE_DENSITY:
        found["directive_density"] = _DENSITY_WEIGHT
    if hidden_text.strip():
        hidden = _INVISIBLE.sub("", hidden_text)
        hidden_hits = [
            name
            for name, (_, patterns) in _SIGNALS.items()
            if name != "schema_mimicry" and any(pattern.search(hidden) for pattern in patterns)
        ]
        if hidden_hits:
            found["hidden_instructions"] = _HIDDEN_WEIGHT
    return found


def injection_score(text: str, hidden_text: str = "") -> float:
    """0.0 for a page that does not try to instruct the model, up to 1.0 for one that does."""
    quiet = 1.0
    for weight in injection_signals(text, hidden_text).values():
        quiet *= 1.0 - weight
    return round(1.0 - quiet, 3)
