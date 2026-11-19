"""Save a web page as an evaluation snapshot (docs/EVAL.md section 3.2).

    uv run python -m eval.snapshot https://example-broker.com/ e014

fetches the page through the app's own fetcher (SSRF guard, robots.txt, size limit), follows the
contact page when the page has no email or phone, writes the HTML under
``eval/fixtures/pages/`` (gitignored: third-party content) and prints the sha256 of each file
with the contacts a regex finds, as an aid for hand-labelling. It never labels anything itself.
"""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from eval.schemas import FIXTURES
from partner_agents.config import get_settings
from partner_agents.storage.repo import InMemoryRepository
from partner_agents.utils.contacts import EMAIL_RE, PHONE_RE, has_contact_pattern
from partner_agents.web.fetch import Fetcher, FetchFailure
from partner_agents.web.html import extract_page


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates(html: str, url: str) -> dict[str, list[str]]:
    """Emails and phone-like strings the regexes find in the page text (not a label)."""
    page = extract_page(html, url)
    text = f"{page.main_text}\n{page.contact_text}"
    return {
        "emails": sorted({m.group(0).lower() for m in EMAIL_RE.finditer(text)}),
        "phones": sorted({m.group(0).strip() for m in PHONE_RE.finditer(text)}),
    }


async def take(
    url: str, snapshot_id: str, *, follow_contact: bool, root: Path
) -> dict[str, object]:
    (root / "pages").mkdir(parents=True, exist_ok=True)
    async with Fetcher(get_settings(), InMemoryRepository()) as fetcher:
        final_url, html = await fetcher.fetch_html(url)
        page = extract_page(html, final_url)
        main = root / "pages" / f"{snapshot_id}.html"
        main.write_text(html, encoding="utf-8")
        row: dict[str, object] = {
            "id": snapshot_id,
            "url": final_url,
            "snapshot": f"pages/{snapshot_id}.html",
            "snapshot_sha256": sha256_of(main),
            "found": candidates(html, final_url),
        }
        needs_contact = not has_contact_pattern(f"{page.main_text}\n{page.contact_text}")
        if follow_contact and needs_contact and page.contact_link:
            try:
                contact_url, contact_html = await fetcher.fetch_html(page.contact_link)
            except FetchFailure as failure:
                row["contact_error"] = failure.status
            else:
                contact = root / "pages" / f"{snapshot_id}_contact.html"
                contact.write_text(contact_html, encoding="utf-8")
                row["contact_url"] = contact_url
                row["contact_snapshot"] = f"pages/{snapshot_id}_contact.html"
                row["contact_snapshot_sha256"] = sha256_of(contact)
                row["found_on_contact_page"] = candidates(contact_html, contact_url)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot a page for the extraction eval set.")
    parser.add_argument("url")
    parser.add_argument("id", help="row id, for example e014")
    parser.add_argument("--no-contact", action="store_true", help="do not follow a contact page")
    parser.add_argument("--root", type=Path, default=FIXTURES)
    args = parser.parse_args(argv)
    try:
        row = asyncio.run(
            take(args.url, args.id, follow_contact=not args.no_contact, root=args.root)
        )
    except FetchFailure as failure:
        print(f"not saved: {failure.status}: {failure.message}", file=sys.stderr)
        return 1
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
