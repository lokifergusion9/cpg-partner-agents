"""The snapshot tool (T-37): it uses the app's fetcher and writes what a human then labels."""

import functools
import hashlib
from pathlib import Path

import pytest
import respx
from eval.schemas import ExtractionRow, snapshot_problems

from eval import snapshot
from partner_agents.web.fetch import Fetcher
from tests.web.test_fetch import HOME, MAIN, html_page, robots


@pytest.fixture(autouse=True)
def public_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolve(host: str) -> list[str]:
        return ["93.184.216.34"]

    monkeypatch.setattr(snapshot, "Fetcher", functools.partial(Fetcher, resolver=resolve))


def run(args: list[str]) -> int:
    return snapshot.main(args)


@respx.mock
def test_a_page_is_saved_with_its_hash_and_the_contacts_the_regexes_find(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    robots()
    body = (
        MAIN + '<footer>Email <a href="mailto:sales@acme.example">us</a> or 310-555-0100</footer>'
    )
    respx.get(HOME).respond(200, html=html_page(body))
    assert run([HOME, "e001", "--root", str(tmp_path)]) == 0
    saved = tmp_path / "pages" / "e001.html"
    assert "broker cookies" in saved.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert hashlib.sha256(saved.read_bytes()).hexdigest() in out
    assert "sales@acme.example" in out and "contact_snapshot" not in out


@respx.mock
def test_the_contact_page_is_followed_when_the_page_has_no_contact(tmp_path: Path) -> None:
    robots()
    respx.get(HOME).respond(200, html=html_page(MAIN + '<a href="/contact">Contact us</a>'))
    respx.get("https://acme.example/contact").respond(
        200, html=html_page("<p>Write to buyers@acme.example</p>")
    )
    assert run([HOME, "e002", "--root", str(tmp_path)]) == 0
    assert (tmp_path / "pages" / "e002_contact.html").exists()


@respx.mock
def test_no_contact_flag_leaves_the_contact_page_alone(tmp_path: Path) -> None:
    robots()
    respx.get(HOME).respond(200, html=html_page(MAIN + '<a href="/contact">Contact us</a>'))
    assert run([HOME, "e003", "--no-contact", "--root", str(tmp_path)]) == 0
    assert not (tmp_path / "pages" / "e003_contact.html").exists()


@respx.mock
def test_a_page_that_robots_txt_disallows_is_not_saved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    robots("User-agent: *\nDisallow: /\n")
    assert run([HOME, "e004", "--root", str(tmp_path)]) == 1
    assert "blocked_robots" in capsys.readouterr().err
    assert not (tmp_path / "pages" / "e004.html").exists()


@respx.mock
def test_a_snapshot_round_trips_through_the_row_model(tmp_path: Path) -> None:
    robots()
    respx.get(HOME).respond(200, html=html_page(MAIN + "<footer>sales@acme.example</footer>"))
    run([HOME, "e005", "--root", str(tmp_path)])
    digest = hashlib.sha256((tmp_path / "pages" / "e005.html").read_bytes()).hexdigest()
    row = ExtractionRow(
        id="e005",
        url=HOME,
        snapshot="pages/e005.html",
        snapshot_sha256=digest,
        company_name="Acme",
        partner_type="broker",
        emails=["sales@acme.example"],
        case="standard",
    )
    assert snapshot_problems(row, tmp_path) == []
