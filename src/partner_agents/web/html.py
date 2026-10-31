"""Turn an HTML page into the text Scout reads (HLD §6.4, text assembly).

Three parts are kept apart because main-content extraction drops the places contacts
live:

(a) ``main_text``: trafilatura's main content;
(b) the visible text of ``<footer>``, ``<address>`` and elements whose id or class contains
    "contact";
(c) every ``mailto:`` / ``tel:`` target, in ``contact_text`` together with (b).

``contact_link`` is the best same-company link to a contact/about page, used when the page
itself shows no email or phone.
"""

import re
from dataclasses import dataclass
from urllib.parse import unquote, urldefrag, urljoin, urlsplit

import trafilatura
from lxml import html as lxml_html

from partner_agents.web.search import registrable_domain

MAX_MAIN_CHARS = 100_000
MAX_CONTACT_ELEMENT_CHARS = 2_000
MAX_CONTACT_CHARS = 6_000

_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"
_CONTACT_ELEMENTS = (
    "//footer | //address | "
    f"//*[contains(translate(@id,'{_UPPER}','{_LOWER}'),'contact') "
    f"or contains(translate(@class,'{_UPPER}','{_LOWER}'),'contact')]"
)
_LINK_SCORES = (("contact", 3), ("reach", 2), ("about", 1))
_NOT_NAVIGABLE = ("javascript:", "mailto:", "tel:", "#", "data:")


@dataclass(frozen=True)
class ExtractedPage:
    title: str | None
    main_text: str
    contact_text: str
    contact_link: str | None


def _words(text: str) -> str:
    return " ".join(text.split())


def _parse(html: str) -> lxml_html.HtmlElement | None:
    if not html.strip():
        return None
    try:
        doc = lxml_html.fromstring(html)
    except (lxml_html.etree.ParserError, ValueError):
        return None
    for noise in doc.xpath("//script | //style | //noscript | //template"):
        noise.drop_tree()
    return doc


def _main_text(html: str, doc: lxml_html.HtmlElement) -> str:
    extracted = trafilatura.extract(html, include_comments=False, favor_recall=True)
    text = extracted or _words(doc.text_content())
    return text[:MAX_MAIN_CHARS]


def _contact_elements_text(doc: lxml_html.HtmlElement) -> list[str]:
    matched = doc.xpath(_CONTACT_ELEMENTS)
    matched_set = set(matched)
    chunks: list[str] = []
    for element in matched:
        # A matched element inside another matched element would repeat its text.
        if any(parent in matched_set for parent in element.iterancestors()):
            continue
        text = _words(element.text_content())[:MAX_CONTACT_ELEMENT_CHARS]
        if text and text not in chunks:
            chunks.append(text)
    return chunks


def _link_targets(doc: lxml_html.HtmlElement) -> list[str]:
    targets: list[str] = []
    for href in doc.xpath("//a/@href"):
        value = href.strip()
        lowered = value.lower()
        if lowered.startswith("mailto:"):
            for address in unquote(value[7:].split("?", 1)[0]).split(","):
                line = f"mailto: {address.strip()}"
                if address.strip() and line not in targets:
                    targets.append(line)
        elif lowered.startswith("tel:"):
            line = f"tel: {unquote(value[4:]).strip()}"
            if len(line) > 5 and line not in targets:
                targets.append(line)
    return targets


def _contact_link(doc: lxml_html.HtmlElement, base_url: str) -> str | None:
    base_domain = registrable_domain(base_url)
    base_page = urldefrag(base_url)[0]
    best: tuple[int, str] | None = None
    for anchor in doc.xpath("//a[@href]"):
        href = anchor.get("href", "").strip()
        if not href or href.lower().startswith(_NOT_NAVIGABLE):
            continue
        haystack = f"{_words(anchor.text_content())} {href}".lower()
        score = next((points for word, points in _LINK_SCORES if word in haystack), 0)
        if score == 0 or (best is not None and score <= best[0]):
            continue
        target = urldefrag(urljoin(base_url, href))[0]
        parts = urlsplit(target)
        if parts.scheme not in {"http", "https"} or target == base_page:
            continue
        if registrable_domain(target) != base_domain:
            continue
        best = (score, target)
    return best[1] if best else None


def extract_page(html: str, base_url: str) -> ExtractedPage:
    """Split ``html`` into main text, contact text and the best contact-page link."""
    doc = _parse(html)
    if doc is None:
        return ExtractedPage(None, "", "", None)
    title = _words(doc.findtext(".//title") or "") or None
    contact_parts = [*_contact_elements_text(doc), *_link_targets(doc)]
    contact_text = "\n".join(contact_parts)[:MAX_CONTACT_CHARS]
    return ExtractedPage(
        title=title,
        main_text=_main_text(html, doc),
        contact_text=contact_text,
        contact_link=_contact_link(doc, base_url),
    )


_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([\w.:-]+)", re.IGNORECASE)


def decode_html(body: bytes, content_type: str) -> str:
    """Decode with the charset from ``Content-Type``, else UTF-8; never raises."""
    match = _CHARSET_RE.search(content_type)
    for encoding in ([match.group(1)] if match else []) + ["utf-8"]:
        try:
            return body.decode(encoding, errors="replace")
        except LookupError:
            continue
    return body.decode("utf-8", errors="replace")
