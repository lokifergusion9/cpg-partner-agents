import pytest

from partner_agents.web.html import decode_html, extract_page

BASE = "https://www.acme.com/products/"


def page(body: str, head: str = "<title>Acme</title>") -> str:
    return f"<html><head>{head}</head><body>{body}</body></html>"


ARTICLE = (
    "<article><h1>Acme Brokers</h1><p>We broker cookies across California for "
    "retailers and distributors of all sizes.</p></article>"
)


def test_main_text_comes_from_the_article() -> None:
    result = extract_page(page("<nav>Menu Home Login</nav>" + ARTICLE), BASE)
    assert "broker cookies across California" in result.main_text


def test_title_is_whitespace_normalised() -> None:
    assert (
        extract_page(page(ARTICLE, "<title>  Acme \n Brokers </title>"), BASE).title
        == "Acme Brokers"
    )
    assert extract_page(page(ARTICLE, ""), BASE).title is None


def test_footer_address_and_contact_elements_go_to_contact_text() -> None:
    body = (
        ARTICLE
        + "<footer>Call 310-555-0100</footer>"
        + "<address>1 Main St, Los Angeles</address>"
        + '<div id="Contact-Us">Write to sales team</div>'
        + '<div class="site-CONTACT box">Hours 9-5</div>'
        + "<div class='unrelated'>Not me</div>"
    )
    contact = extract_page(page(body), BASE).contact_text
    for expected in ["Call 310-555-0100", "1 Main St", "Write to sales team", "Hours 9-5"]:
        assert expected in contact
    assert "Not me" not in contact


def test_nested_contact_elements_are_not_repeated() -> None:
    body = '<footer><address>1 Main St</address><div class="contact">Hours 9-5</div></footer>'
    contact = extract_page(page(body), BASE).contact_text
    assert contact.count("1 Main St") == 1
    assert contact.count("Hours 9-5") == 1


def test_mailto_and_tel_targets_are_collected() -> None:
    body = (
        ARTICLE
        + '<a href="mailto:sales@acme.com?subject=Hi%20there">Email</a>'
        + '<a href="MAILTO:a@x.example,b@y.example">Both</a>'
        + '<a href="tel:+1-310-555-0100">Call</a>'
        + '<a href="mailto:sales@acme.com">Again</a>'
        + '<a href="mailto:">Empty</a>'
    )
    contact = extract_page(page(body), BASE).contact_text.splitlines()
    assert contact == [
        "mailto: sales@acme.com",
        "mailto: a@x.example",
        "mailto: b@y.example",
        "tel: +1-310-555-0100",
    ]


def test_hidden_script_and_style_text_is_ignored() -> None:
    body = (
        ARTICLE
        + "<footer>Visible<script>var e='hidden@evil.example'</script><style>.x{}</style></footer>"
    )
    result = extract_page(page(body), BASE)
    assert "hidden@evil.example" not in result.contact_text + result.main_text
    assert "Visible" in result.contact_text


def test_a_page_trafilatura_cannot_read_falls_back_to_the_body_text() -> None:
    result = extract_page(page("<div>Short page text</div>"), BASE)
    assert "Short page text" in result.main_text


@pytest.mark.parametrize("html", ["", "   ", "\n"])
def test_empty_html_gives_an_empty_page(html: str) -> None:
    result = extract_page(html, BASE)
    assert (result.title, result.main_text, result.contact_text, result.contact_link) == (
        None,
        "",
        "",
        None,
    )


# --- contact link -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        ('<a href="/contact-us">Contact us</a>', "https://www.acme.com/contact-us"),
        (
            '<a href="/about">About</a><a href="/contact">Contact</a>',
            "https://www.acme.com/contact",
        ),
        ('<a href="/team">Reach us</a><a href="/about">About</a>', "https://www.acme.com/team"),
        (
            '<a href="/get-in-touch/contact.html">Talk</a>',
            "https://www.acme.com/get-in-touch/contact.html",
        ),
        ('<a href="contact">Contact</a>', "https://www.acme.com/products/contact"),
        ('<a href="//www.acme.com/c">Contact</a>', "https://www.acme.com/c"),
        (
            '<a href="https://shop.acme.com/contact">Contact</a>',
            "https://shop.acme.com/contact",
        ),
        ('<a href="/contact#form">Contact</a>', "https://www.acme.com/contact"),
    ],
)
def test_contact_link_choice(links: str, expected: str) -> None:
    assert extract_page(page(ARTICLE + links), BASE).contact_link == expected


@pytest.mark.parametrize(
    "links",
    [
        '<a href="https://other.example/contact">Contact</a>',
        '<a href="mailto:sales@acme.com">Contact</a>',
        '<a href="tel:5550100">Contact</a>',
        '<a href="javascript:void(0)">Contact</a>',
        '<a href="#contact">Contact</a>',
        '<a href="/careers">Jobs</a>',
        '<a href="/products/#contact">Contact</a>',  # the page itself
        "<a>Contact</a>",
        '<a href="">Contact</a>',
    ],
)
def test_links_that_are_not_a_same_company_contact_page_are_ignored(links: str) -> None:
    assert extract_page(page(ARTICLE + links), BASE).contact_link is None


def test_the_first_link_wins_among_equals() -> None:
    links = '<a href="/contact-a">Contact</a><a href="/contact-b">Contact</a>'
    assert (
        extract_page(page(ARTICLE + links), BASE).contact_link == "https://www.acme.com/contact-a"
    )


# --- decoding -----------------------------------------------------------------------


def test_decode_uses_the_declared_charset() -> None:
    body = "Café".encode("latin-1")
    assert decode_html(body, "text/html; charset=ISO-8859-1") == "Café"
    assert decode_html(body, 'text/html; charset="latin-1"') == "Café"


def test_decode_defaults_to_utf8_and_never_raises() -> None:
    assert decode_html("Café".encode(), "text/html") == "Café"
    assert chr(0xFFFD) in decode_html(b"\xff\xfe bad", "text/html")
    assert decode_html(b"plain", "text/html; charset=no-such-codec") == "plain"
